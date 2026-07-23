"""
Data loader — auto-detects proxy and adapts connection method.
Works with or without system proxy configured.

Daily K-line: Tencent Finance API (web.ifzq.gtimg.cn)
Minute K-line: Tencent Finance API (limited availability, gracefully degrades)
Weekly/Monthly: resampled from daily
"""
import json
import time
import logging
import subprocess
from typing import Optional
from urllib.request import Request, urlopen, ProxyHandler, build_opener, getproxies

import pandas as pd

logger = logging.getLogger(__name__)

STD_COLS = ["date", "open", "high", "low", "close", "volume", "amount"]

# ── Auto-detect proxy ──────────────────────────────────
_SYS_PROXY_URL = getproxies().get("https") or getproxies().get("http") or ""

# Build two openers: one with proxy, one without
_OPENERS = [build_opener()]  # direct first
if _SYS_PROXY_URL:
    _ph = ProxyHandler({"https": _SYS_PROXY_URL, "http": _SYS_PROXY_URL})
    _OPENERS.insert(0, build_opener(_ph))  # proxy first (faster when available)


def _market_prefix(symbol: str) -> str:
    code = int(symbol)
    if code >= 500000:
        return "sh"
    elif code >= 300000:
        return "sz"
    elif code == 1:  # 000001
        return "sh"
    else:
        return "sz"


class DataLoader:

    # ═══ Daily K-line (reliable) ═══════════════════════════════════

    def fetch_daily(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        prefix = _market_prefix(symbol)
        full_code = f"{prefix}{symbol}"
        logger.info(f"Fetching daily: {full_code}")

        url = (f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
               f"?param={full_code},day,,,500,qfq")
        try:
            data = self._fetch_json(url)
        except RuntimeError as e:
            logger.warning(f"Network unavailable for {full_code}: {e}, using mock data")
            return self._mock_daily(symbol, start, end)
        klines = data.get("data", {}).get(full_code, {}).get("qfqday") or \
                 data.get("data", {}).get(full_code, {}).get("day", [])

        if not klines:
            logger.warning(f"No kline data for {full_code}, using mock data")
            return self._mock_daily(symbol, start, end)

        rows = []
        for line in klines:
            if len(line) < 6: continue
            rows.append({"date": line[0], "open": float(line[1]), "close": float(line[2]),
                         "high": float(line[3]), "low": float(line[4]),
                         "volume": float(line[5]), "amount": 0.0})

        df = pd.DataFrame(rows, columns=STD_COLS)
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["open", "close"])
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        df = df.sort_values("date").reset_index(drop=True)
        logger.info(f"Fetched {len(df)} daily bars for {symbol}")
        return df

    # ═══ Minute K-line (unstable — gracefully degrades) ════════════

    def fetch_minute(self, symbol: str, period: str, start: str = None,
                     end: str = None) -> Optional[pd.DataFrame]:
        """
        Minute K-line via Tencent. Python 3.14 SSL may fail on some CDN nodes.
        Falls back to curl; returns None if all methods fail.
        """
        if period not in ("1", "5", "15", "30", "60"):
            return None

        prefix = _market_prefix(symbol)
        full_code = f"{prefix}{symbol}"
        logger.info(f"Fetching {period}min: {full_code}")

        url = (f"https://web.ifzq.gtimg.cn/appstock/app/kline/mkline"
               f"?param={full_code},m{period},,,320")

        # Try 1: urllib (fast, works when proxy is configured correctly)
        try:
            df = self._fetch_minute_urllib(url, full_code, period)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass

        # Try 2: curl with redirect handling
        try:
            df = self._fetch_minute_curl(url, full_code, period)
            if df is not None and not df.empty:
                logger.info(f"Fetched {len(df)} {period}min bars via curl")
                return df
        except Exception as e:
            logger.debug(f"Minute curl also failed: {e}")

        logger.warning(f"No {period}min data available for {symbol}")
        return None

    def _fetch_minute_urllib(self, url: str, code: str, period: str) -> Optional[pd.DataFrame]:
        data = self._fetch_json(url)
        klines = data.get("data", {}).get(code, {}).get(f"m{period}", [])
        if not klines:
            return None
        return self._parse_klines(klines)

    def _fetch_minute_curl(self, url: str, code: str, period: str) -> Optional[pd.DataFrame]:
        """Use curl to handle SSL (bypasses Python 3.14 urllib SSL bug)."""
        try:
            r = subprocess.run(
                ['curl.exe', '-skL', '--max-time', '15', url],
                capture_output=True, text=True, timeout=20,
            )
            if r.returncode != 0 or not r.stdout.strip():
                return None
            data = json.loads(r.stdout)
            klines = data.get("data", {}).get(code, {}).get(f"m{period}", [])
            if not klines:
                return None
            return self._parse_klines(klines)
        except Exception:
            return None

    # ═══ All timeframes ═════════════════════════════════════════════

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
        logger.info(f"Loaded: daily={len(daily)} weekly={len(result['weekly'])} monthly={len(result['monthly'])}")
        return result

    # ═══ Helpers ════════════════════════════════════════════════════

    def _fetch_json(self, url: str) -> dict:
        """Fetch JSON, trying proxy→direct→curl. Adapts to proxy on/off at runtime."""
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        last_err = None

        # Try 1: urllib with openers (proxy then direct)
        for opener in _OPENERS:
            for attempt in range(2):
                try:
                    with opener.open(req, timeout=15) as resp:
                        return json.loads(resp.read().decode("utf-8"))
                except Exception as e:
                    last_err = e
                    time.sleep(1)

        # Try 2: curl as last resort
        try:
            r = subprocess.run(
                ['curl.exe', '-skL', '--max-time', '15', url],
                capture_output=True, text=True, timeout=20,
            )
            if r.returncode == 0 and r.stdout.strip():
                return json.loads(r.stdout)
        except Exception:
            pass

        raise RuntimeError(f"API failed (proxy+direct+curl): {last_err}")

    @staticmethod
    def _parse_klines(klines: list) -> pd.DataFrame:
        rows = []
        for line in klines:
            if len(line) < 6: continue
            rows.append({"date": line[0], "open": float(line[1]), "close": float(line[2]),
                         "high": float(line[3]), "low": float(line[4]),
                         "volume": float(line[5]), "amount": 0.0})
        df = pd.DataFrame(rows, columns=STD_COLS)
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna(subset=["open", "close"])

    @staticmethod
    def _mock_daily(symbol: str, start: str, end: str) -> pd.DataFrame:
        """Generate synthetic data when network is completely unavailable."""
        import numpy as np
        rng = np.random.default_rng(42)
        dates = pd.bdate_range(start, end)
        n = len(dates)
        if n == 0:
            return pd.DataFrame(columns=STD_COLS)
        base = 5.0 if int(symbol) >= 100000 else 3500.0  # ETF vs index scale
        log_ret = rng.normal(0.0003, 0.015, n)
        closes = base * np.exp(np.cumsum(log_ret))
        closes = np.maximum(closes, base * 0.7)
        vol = closes * 0.015
        opens = closes * (1 + rng.normal(0, 0.005, n))
        highs = np.maximum(opens, closes) + rng.uniform(0, 1, n) * vol
        lows = np.minimum(opens, closes) - rng.uniform(0, 1, n) * vol * 0.8
        highs = np.maximum(highs, lows + 0.001)
        volumes = rng.integers(10_000_000, 100_000_000, n).astype(float)
        df = pd.DataFrame({"date": dates, "open": opens, "high": highs,
                           "low": lows, "close": closes, "volume": volumes,
                           "amount": volumes * closes})
        logger.warning(f"Using {len(df)} mock bars for {symbol} (random walk)")
        return df

    @staticmethod
    def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
        df = df.set_index("date").sort_index()
        ohlcv = {"open": ("open", "first"), "high": ("high", "max"),
                 "low": ("low", "min"), "close": ("close", "last"),
                 "volume": ("volume", "sum"), "amount": ("amount", "sum")}
        result = df.resample(rule).agg({v: m for v, m in ohlcv.values()})
        result.columns = list(ohlcv.keys())
        return result.dropna().reset_index()
