"""
Data loader with proxy bypass for akshare.
Supports daily K-line and minute-level K-line (60min).
Weekly/monthly resampled from daily; 120min resampled from 60min.
"""
import os
import logging
from typing import Optional

# ═══ CRITICAL: clear proxy BEFORE akshare import ═══
for _key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_key, None)
os.environ["no_proxy"] = "*"
# ══════════════════════════════════════════════════════

import pandas as pd
import akshare as ak

logger = logging.getLogger(__name__)

# Standard column names
STD_COLS = ["date", "open", "high", "low", "close", "volume", "amount"]


class DataLoader:
    """Load ETF/stock K-line data with proxy-safe akshare access."""

    def fetch_daily(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """
        Fetch daily K-line from Eastmoney via akshare.

        Args:
            symbol: ETF/stock code, e.g. "588170"
            start: start date "YYYYMMDD"
            end: end date "YYYYMMDD"

        Returns:
            DataFrame with columns: date, open, high, low, close, volume, amount
        """
        logger.info(f"Fetching daily K-line for {symbol}: {start} → {end}")
        try:
            df = ak.fund_etf_hist_em(
                symbol=symbol, period="daily",
                start_date=start, end_date=end, adjust="qfq"
            )
        except Exception:
            logger.info("ETF API failed, trying stock API...")
            df = ak.stock_zh_a_hist(
                symbol=symbol, period="daily",
                start_date=start, end_date=end, adjust="qfq"
            )

        return self._standardize(df, date_col="日期")

    def fetch_minute(
        self, symbol: str, period: str, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        """
        Fetch minute-level K-line. Only periods "1","5","15","30","60" supported.

        Returns None if no data available.
        """
        period_map = {
            "60": "60", "30": "30", "15": "15", "5": "5", "1": "1",
        }
        klt = period_map.get(period, period)
        logger.info(f"Fetching {period}min K-line for {symbol}: {start} → {end}")

        import time
        for attempt in range(3):
            try:
                df = ak.fund_etf_hist_min_em(
                    symbol=symbol, period=klt,
                    start_date=f"{start[:4]}-{start[4:6]}-{start[6:]} 09:30:00",
                    end_date=f"{end[:4]}-{end[4:6]}-{end[6:]} 15:00:00",
                    adjust="qfq",
                )
                return self._standardize(df, date_col="时间")
            except Exception as e:
                logger.warning(f"Minute fetch attempt {attempt+1} failed: {e}")
                time.sleep(2)
        return None

    def load_all(self, symbol: str, start: str, end: str) -> dict:
        """
        Load all available timeframes.

        Returns dict with keys: daily, weekly, monthly, min60, min120
        - daily/weekly/monthly: from daily K-line + resampling
        - min60: from 60-minute K-line API
        - min120: resampled from 60-minute (2 bars → 1)
        """
        result = {}

        # Daily
        daily = self.fetch_daily(symbol, start, end)
        if daily.empty:
            raise RuntimeError(f"No daily data for {symbol}")
        daily["date"] = pd.to_datetime(daily["date"])
        result["daily"] = daily

        # Weekly resampling
        weekly = self._resample_ohlcv(daily, "W")
        result["weekly"] = weekly

        # Monthly resampling
        monthly = self._resample_ohlcv(daily, "ME")
        result["monthly"] = monthly

        # 60-minute
        min60 = self.fetch_minute(symbol, "60", start, end)
        if min60 is not None and not min60.empty:
            min60["date"] = pd.to_datetime(min60["date"])
            result["min60"] = min60
            # 120-minute resampled
            min120 = self._resample_ohlcv(min60, "2h")
            result["min120"] = min120
        else:
            logger.warning("No 60min data available; skipping intraday MAs")

        logger.info(
            f"Data loaded: daily={len(daily)}, weekly={len(weekly)}, "
            f"monthly={len(monthly)}, "
            f"60min={len(result.get('min60', pd.DataFrame()))}, "
            f"120min={len(result.get('min120', pd.DataFrame()))}"
        )
        return result

    # ── helpers ───────────────────────────────────────

    def _standardize(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Map akshare columns to standard English names."""
        col_map = {
            date_col: "date",
            "开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
            "成交量": "volume", "成交额": "amount",
        }
        df = df.rename(columns=col_map)
        keep = [c for c in STD_COLS if c in df.columns]
        df = df[keep].copy()
        # Ensure numeric types
        for c in ["open", "high", "low", "close", "volume", "amount"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
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
