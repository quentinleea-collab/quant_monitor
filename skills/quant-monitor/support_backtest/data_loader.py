"""
Data loader using urllib + system proxy for Eastmoney API.
Avoids akshare (which uses requests/urllib3 — incompatible with Python 3.14 proxy).

Supports daily K-line and minute-level K-line (60/30/15/5/1min).
Weekly/monthly resampled from daily; 120min resampled from 60min.
"""
import json
import time
import logging
from typing import Optional
from urllib.request import ProxyHandler, build_opener, Request

import pandas as pd

logger = logging.getLogger(__name__)

# ═══ Proxy setup via urllib (works where requests/urllib3 fails) ═══
import urllib.request as _ur
_SYS_PROXIES = _ur.getproxies()
_PROXY_URL = _SYS_PROXIES.get("https") or _SYS_PROXIES.get("http")
if _PROXY_URL:
    _PROXY = ProxyHandler({"https": _PROXY_URL, "http": _PROXY_URL})
    _OPENER = build_opener(_PROXY)
    logger.debug(f"Using proxy: {_PROXY_URL}")
else:
    _OPENER = build_opener()

# Standard column names
STD_COLS = ["date", "open", "high", "low", "close", "volume", "amount"]

# Eastmoney API
_EASTMONEY_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
_EASTMONEY_UT = "7eea3edcaed734bea9cbfc24409ed989"
_EASTMONEY_FIELDS1 = "f1,f2,f3,f4,f5,f6"
_EASTMONEY_FIELDS2 = "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"


class DataLoader:
    """Load ETF/stock K-line data from Eastmoney API via urllib (proxy-safe)."""

    def fetch_daily(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """
        Fetch daily K-line.

        Args:
            symbol: ETF/stock code, e.g. "588170"
            start: "YYYYMMDD"
            end: "YYYYMMDD"

        Returns:
            DataFrame with columns: date, open, high, low, close, volume, amount
        """
        logger.info(f"Fetching daily K-line for {symbol}: {start} -> {end}")

        # Shanghai: secid=1.XXXXXX, Shenzhen: secid=0.XXXXXX
        secid = f"1.{symbol}"  # Most ETFs are Shanghai; a few Shenzhen
        df = self._fetch_eastmoney(secid, "101", start, end)

        # If Shanghai code fails, try Shenzhen
        if df.empty:
            secid = f"0.{symbol}"
            df = self._fetch_eastmoney(secid, "101", start, end)

        if df.empty:
            raise RuntimeError(
                f"Cannot fetch daily data for {symbol}. "
                f"Check proxy ({_PROXY_URL}) and network."
            )

        logger.info(f"Fetched {len(df)} daily bars for {symbol}")
        return df

    def fetch_minute(
        self, symbol: str, period: str, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        """
        Fetch minute-level K-line.

        Supported periods: "1", "5", "15", "30", "60"
        ("120" not supported natively — resample from 60min)
        """
        if period not in ("1", "5", "15", "30", "60"):
            logger.warning(f"Unsupported minute period: {period}, skipping")
            return None

        logger.info(f"Fetching {period}min K-line for {symbol}")

        secid = f"1.{symbol}"
        for attempt in range(3):
            try:
                df = self._fetch_eastmoney(secid, period, start, end)
                if not df.empty:
                    logger.info(f"Fetched {len(df)} {period}min bars for {symbol}")
                    return df
            except Exception as e:
                logger.warning(f"Minute fetch attempt {attempt+1} failed: {e}")
                time.sleep(2)

        # Try Shenzhen
        secid = f"0.{symbol}"
        for attempt in range(3):
            try:
                df = self._fetch_eastmoney(secid, period, start, end)
                if not df.empty:
                    return df
            except Exception:
                time.sleep(2)

        logger.warning(f"No {period}min data available for {symbol}")
        return None

    def load_all(self, symbol: str, start: str, end: str) -> dict:
        """
        Load all available timeframes.

        Returns dict with keys: daily, weekly, monthly, min60, min120
        """
        result = {}

        daily = self.fetch_daily(symbol, start, end)
        daily["date"] = pd.to_datetime(daily["date"])
        result["daily"] = daily

        result["weekly"] = self._resample_ohlcv(daily, "W")
        result["monthly"] = self._resample_ohlcv(daily, "ME")

        min60 = self.fetch_minute(symbol, "60", start, end)
        if min60 is not None and not min60.empty:
            min60["date"] = pd.to_datetime(min60["date"])
            result["min60"] = min60
            result["min120"] = self._resample_ohlcv(min60, "2h")
        else:
            logger.warning("No 60min data available; skipping intraday MAs")

        logger.info(
            f"Data loaded: daily={len(daily)}, weekly={len(result['weekly'])}, "
            f"monthly={len(result['monthly'])}, "
            f"60min={len(result.get('min60', pd.DataFrame()))}, "
            f"120min={len(result.get('min120', pd.DataFrame()))}"
        )
        return result

    # ── Eastmoney API via urllib ────────────────────────

    def _fetch_eastmoney(
        self, secid: str, klt: str, start: str, end: str
    ) -> pd.DataFrame:
        """
        Fetch K-line data from Eastmoney API using urllib (proxy-safe).

        Args:
            secid: "1.588170" (Shanghai) or "0.XXXXXX" (Shenzhen)
            klt: "101"=daily, "60"=60min, "30"=30min, etc.
        """
        params = (
            f"fields1={_EASTMONEY_FIELDS1}"
            f"&fields2={_EASTMONEY_FIELDS2}"
            f"&ut={_EASTMONEY_UT}"
            f"&klt={klt}"
            f"&fqt=1"
            f"&secid={secid}"
            f"&beg={start}"
            f"&end={end}"
        )
        url = f"{_EASTMONEY_URL}?{params}"
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})

        for attempt in range(5):
            try:
                with _OPENER.open(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                klines = data.get("data", {}).get("klines", [])
                if not klines:
                    return pd.DataFrame()
                return self._parse_klines(klines, klt)
            except Exception as e:
                if attempt < 4:
                    time.sleep(2)
                else:
                    raise RuntimeError(
                        f"Eastmoney API failed after 5 attempts for {secid} klt={klt}: {e}"
                    ) from e

    # ── helpers ─────────────────────────────────────────

    @staticmethod
    def _parse_klines(klines: list[str], klt: str) -> pd.DataFrame:
        """
        Parse Eastmoney kline strings to DataFrame.

        Format: "date,open,close,high,low,volume,amount,amplitude,change%,change, turnover"
        (10 or 11 fields depending on API response)
        """
        rows = []
        for line in klines:
            parts = line.split(",")
            if len(parts) < 7:
                continue
            rows.append({
                "date": parts[0],
                "open": float(parts[1]),
                "close": float(parts[2]),
                "high": float(parts[3]),
                "low": float(parts[4]),
                "volume": float(parts[5]),
                "amount": float(parts[6]),
            })

        df = pd.DataFrame(rows, columns=STD_COLS)
        # Fix decimal: Eastmoney sometimes returns X.000 for volume/amount
        for c in ["open", "high", "low", "close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        return df.dropna(subset=["open", "close"])

    @staticmethod
    def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
        """Resample OHLCV to a coarser timeframe."""
        df = df.set_index("date").sort_index()
        ohlcv = {
            "open": ("open", "first"),
            "high": ("high", "max"),
            "low": ("low", "min"),
            "close": ("close", "last"),
            "volume": ("volume", "sum"),
            "amount": ("amount", "sum"),
        }
        result = df.resample(rule).agg({v: m for v, m in ohlcv.values()})
        result.columns = list(ohlcv.keys())
        return result.dropna().reset_index()
