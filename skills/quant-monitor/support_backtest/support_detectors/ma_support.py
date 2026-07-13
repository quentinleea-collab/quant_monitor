"""
MA support detector — works on any timeframe.
Each MA column in the DataFrame becomes a support level.

Usage:
    det = MASupport(periods=[5,10,20,30,60], timeframe="日")
    supports = det.detect(df_daily)
"""
import pandas as pd

from support_detectors.base import BaseSupportDetector


class MASupport(BaseSupportDetector):
    """Detect support levels from Moving Average columns in the DataFrame."""

    def __init__(self, periods: list[int], timeframe: str):
        """
        Args:
            periods: MA periods to look for, e.g. [5, 10, 20, 30, 60]
            timeframe: label, e.g. "日", "周", "月", "60分钟", "120分钟"
        """
        self._periods = periods
        self._timeframe = timeframe
        self._ma_cols = [f"MA{p}" for p in periods]

    @property
    def name(self) -> str:
        return f"ma_{self._timeframe}"

    @property
    def category(self) -> str:
        return "ma"

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """For each date and each MA column, emit one support row."""
        rows = []
        for col in self._ma_cols:
            if col not in df.columns:
                continue
            period = col.replace("MA", "")
            for _, row in df.iterrows():
                price = row[col]
                if pd.isna(price) or price <= 0:
                    continue
                rows.append({
                    "date": row["date"],
                    "support_type": f"MA{period}",
                    "support_price": price,
                    "timeframe": self._timeframe,
                })
        return pd.DataFrame(rows, columns=["date", "support_type", "support_price", "timeframe"])
