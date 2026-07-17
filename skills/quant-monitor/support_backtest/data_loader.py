"""
Data loader using Tencent Finance API (urllib).
Eastmoney is blocked; Tencent ifzq.gtimg.cn works reliably.

Supports daily K-line. Weekly/monthly resampled from daily.
"""
import json
import time
import logging
from typing import Optional
from urllib.request import Request, urlopen

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

STD_COLS = ["date", "open", "high", "low", "close", "volume", "amount"]

# Market prefix: sh=Shanghai, sz=Shenzhen
def _market_prefix(symbol: str) -> str:
    """Determine exchange prefix from symbol code."""
    code = int(symbol)
    if code >= 500000 and code < 600000:
        return "sh"  # Shanghai ETF/stocks 5xxxxx
    elif code >= 600000:
        return "sh"  # Shanghai stocks 6xxxxx
    elif code == 1:  # 000001 上证指数
        return "sh"
    elif code >= 300000:
        return "sz"  # Shenzhen (创业板 3xxxxx, 深证 399xxx)
    else:
        return "sz"  # Default Shenzhen

class DataLoader:
    """Load ETF/stock/index K-line data from Tencent Finance API."""

    def fetch_daily(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """
        Fetch daily K-line from Tencent Finance.

        Args:
            symbol: code, e.g. "588170", "000001"
            start/end: "YYYYMMDD" (start is ignored — API returns all available)
        """
        prefix = _market_prefix(symbol)
        full_code = f"{prefix}{symbol}"
        logger.info(f"Fetching daily K-line for {full_code}")

        # Tencent daily K-line API (qfq=前复权)
        url = (f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
               f"?param={full_code},day,,,500,qfq")
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})

        for attempt in range(3):
            try:
                with urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                break
            except Exception as e:
                if attempt == 2:
                    raise RuntimeError(f"Tencent API failed for {full_code}: {e}")
                time.sleep(2)

        # Parse response
        stock_data = data.get("data", {}).get(full_code, {})
        klines = stock_data.get("qfqday") or stock_data.get("day", [])

        if not klines:
            raise RuntimeError(f"No kline data for {full_code}")

        rows = []
        for line in klines:
            # Format: [date, open, close, high, low, volume]
            if len(line) < 6:
                continue
            rows.append({
                "date": line[0],
                "open": float(line[1]),
                "close": float(line[2]),
                "high": float(line[3]),
                "low": float(line[4]),
                "volume": float(line[5]),
                "amount": 0.0,  # Tencent free API doesn't include amount
            })

        df = pd.DataFrame(rows, columns=STD_COLS)
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["open", "close"])

        # Filter by date range
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        df = df.sort_values("date").reset_index(drop=True)

        logger.info(f"Fetched {len(df)} daily bars for {symbol}")
        return df

    def fetch_minute(self, symbol: str, period: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """Minute-level K-line. Limited availability (~1-2 months). Returns None if unavailable."""
        prefix = _market_prefix(symbol)
        full_code = f"{prefix}{symbol}"
        logger.info(f"Fetching {period}min K-line for {full_code}")

        url = (f"https://web.ifzq.gtimg.cn/appstock/app/kline/mkline"
               f"?param={full_code},m{period},,,320")
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})

        try:
            with urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.warning(f"Minute API failed for {full_code}: {e}")
            return None

        stock_data = data.get("data", {}).get(full_code, {})
        klines = stock_data.get(f"m{period}", [])

        if not klines:
            return None

        rows = []
        for line in klines:
            if len(line) < 6:
                continue
            rows.append({
                "date": line[0],
                "open": float(line[1]),
                "close": float(line[2]),
                "high": float(line[3]),
                "low": float(line[4]),
                "volume": float(line[5]),
                "amount": 0.0,
            })

        df = pd.DataFrame(rows, columns=STD_COLS)
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["open", "close"])
        logger.info(f"Fetched {len(df)} {period}min bars for {symbol}")
        return df

    def load_all(self, symbol: str, start: str, end: str) -> dict:
        """Load daily + resampled weekly/monthly."""
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

        logger.info(f"Loaded: daily={len(daily)}, weekly={len(result['weekly'])}, monthly={len(result['monthly'])}")
        return result

    @staticmethod
    def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
        df = df.set_index("date").sort_index()
        ohlcv = {"open": ("open", "first"), "high": ("high", "max"),
                 "low": ("low", "min"), "close": ("close", "last"),
                 "volume": ("volume", "sum"), "amount": ("amount", "sum")}
        result = df.resample(rule).agg({v: m for v, m in ohlcv.values()})
        result.columns = list(ohlcv.keys())
        return result.dropna().reset_index()
