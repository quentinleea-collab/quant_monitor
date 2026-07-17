"""
Data loader using urllib + system proxy for Eastmoney API.
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

import urllib.request as _ur
_SYS_PROXIES = _ur.getproxies()
_PROXY_URL = _SYS_PROXIES.get("https") or _SYS_PROXIES.get("http")
if _PROXY_URL:
    _PROXY = ProxyHandler({"https": _PROXY_URL, "http": _PROXY_URL})
    _OPENER = build_opener(_PROXY)
else:
    _OPENER = build_opener()

STD_COLS = ["date", "open", "high", "low", "close", "volume", "amount"]
_EASTMONEY_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
_EASTMONEY_UT = "7eea3edcaed734bea9cbfc24409ed989"
_EASTMONEY_FIELDS1 = "f1,f2,f3,f4,f5,f6"
_EASTMONEY_FIELDS2 = "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"


class DataLoader:
    """Load ETF/stock K-line data from Eastmoney API via urllib (proxy-safe)."""

    def fetch_daily(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        logger.info(f"Fetching daily K-line for {symbol}: {start} -> {end}")
        secid = f"1.{symbol}"
        df = self._fetch_eastmoney(secid, "101", start, end)
        if df.empty:
            df = self._fetch_eastmoney(f"0.{symbol}", "101", start, end)
        if df.empty:
            raise RuntimeError(f"Cannot fetch daily data for {symbol}")
        logger.info(f"Fetched {len(df)} daily bars for {symbol}")
        return df

    def fetch_minute(self, symbol: str, period: str, start: str, end: str) -> Optional[pd.DataFrame]:
        if period not in ("1", "5", "15", "30", "60"):
            return None
        logger.info(f"Fetching {period}min K-line for {symbol}")
        try:
            df = self._fetch_eastmoney(f"1.{symbol}", period, start, end)
            if not df.empty:
                logger.info(f"Fetched {len(df)} {period}min bars for {symbol}")
                return df
        except Exception as e:
            logger.warning(f"Minute fetch failed: {e}")
        try:
            df = self._fetch_eastmoney(f"0.{symbol}", period, start, end)
            if not df.empty:
                return df
        except Exception:
            pass
        logger.warning(f"No {period}min data available for {symbol}")
        return None

    def load_all(self, symbol: str, start: str, end: str) -> dict:
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
        logger.info(f"Data loaded: daily={len(daily)}, weekly={len(result['weekly'])}, monthly={len(result['monthly'])}")
        return result

    def _fetch_eastmoney(self, secid: str, klt: str, start: str, end: str) -> pd.DataFrame:
        params = (f"fields1={_EASTMONEY_FIELDS1}&fields2={_EASTMONEY_FIELDS2}"
                  f"&ut={_EASTMONEY_UT}&klt={klt}&fqt=1&secid={secid}&beg={start}&end={end}")
        url = f"{_EASTMONEY_URL}?{params}"
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        for attempt in range(3):
            try:
                with _OPENER.open(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                klines = data.get("data", {}).get("klines", [])
                if not klines:
                    return pd.DataFrame()
                return self._parse_klines(klines)
            except Exception as e:
                if attempt == 2:
                    raise RuntimeError(f"Eastmoney API failed after 3 attempts for {secid} klt={klt}: {e}")
                time.sleep(2)

    @staticmethod
    def _parse_klines(klines: list[str]) -> pd.DataFrame:
        rows = []
        for line in klines:
            parts = line.split(",")
            if len(parts) < 7:
                continue
            rows.append({"date": parts[0], "open": float(parts[1]), "close": float(parts[2]),
                         "high": float(parts[3]), "low": float(parts[4]),
                         "volume": float(parts[5]), "amount": float(parts[6])})
        df = pd.DataFrame(rows, columns=STD_COLS)
        for c in ["open", "high", "low", "close", "volume", "amount"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna(subset=["open", "close"])

    @staticmethod
    def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
        df = df.set_index("date").sort_index()
        ohlcv = {"open": ("open", "first"), "high": ("high", "max"),
                 "low": ("low", "min"), "close": ("close", "last"),
                 "volume": ("volume", "sum"), "amount": ("amount", "sum")}
        result = df.resample(rule).agg({v: m for v, m in ohlcv.values()})
        result.columns = list(ohlcv.keys())
        return result.dropna().reset_index()
