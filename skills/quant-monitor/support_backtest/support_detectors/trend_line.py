"""
Trend line support from swing lows.

Algorithm:
  1. Find swing lows (local minima in 10-day windows)
  2. For each pair of consecutive swing lows, fit a line
  3. If R² > threshold AND slope is positive → ascending trend line
  4. Extrapolate to current day → support price
"""
import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import Polynomial

from support_detectors.base import BaseSupportDetector
from config import config as default_config, SupportBacktestConfig
from typing import Optional


class TrendLineSupport(BaseSupportDetector):

    def __init__(self, cfg: Optional[SupportBacktestConfig] = None):
        self.cfg = cfg or default_config

    @property
    def name(self) -> str:
        return "trend_line"

    @property
    def category(self) -> str:
        return "price_structure"

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("date").reset_index(drop=True)
        n = len(df)
        swing_win = self.cfg.trend_swing_window

        # Find swing lows
        lows = df["low"].values
        swing_idxs = []
        for i in range(swing_win, n - swing_win):
            if lows[i] <= lows[i - swing_win:i].min() and lows[i] <= lows[i + 1:i + swing_win + 1].min():
                swing_idxs.append(i)

        if len(swing_idxs) < 2:
            return pd.DataFrame(columns=["date", "support_type", "support_price", "timeframe"])

        rows = []
        # For each consecutive pair of swing lows, fit a trend line
        for j in range(1, len(swing_idxs)):
            idx_a, idx_b = swing_idxs[j - 1], swing_idxs[j]
            x_vals = np.array([idx_a, idx_b], dtype=float)
            y_vals = np.array([lows[idx_a], lows[idx_b]], dtype=float)

            # Linear regression
            slope = (y_vals[1] - y_vals[0]) / (x_vals[1] - x_vals[0])
            intercept = y_vals[0] - slope * x_vals[0]

            # R²
            y_pred = slope * x_vals + intercept
            ss_res = np.sum((y_vals - y_pred) ** 2)
            ss_tot = np.sum((y_vals - y_vals.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            if r2 >= self.cfg.trend_min_r_squared and slope > 0:
                # Extrapolate from idx_b to end
                for i in range(idx_b, n):
                    support_price = intercept + slope * i
                    if support_price > 0:
                        rows.append({
                            "date": df["date"].iloc[i],
                            "support_type": "趋势线",
                            "support_price": support_price,
                            "timeframe": "日",
                        })

        return pd.DataFrame(rows, columns=["date", "support_type", "support_price", "timeframe"])
