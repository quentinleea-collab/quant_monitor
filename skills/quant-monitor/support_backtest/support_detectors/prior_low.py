"""
Prior swing low support detector.

Algorithm:
  1. In rolling window of N days, find local minima (low[i] < all neighbors)
  2. The most recent prior swing low = support level for the current day
  3. Repeat for multiple window sizes (20, 40, 60)
"""
import pandas as pd
import numpy as np

from support_detectors.base import BaseSupportDetector
from config import config as default_config
from typing import Optional


class PriorLowSupport(BaseSupportDetector):

    def __init__(self, windows: Optional[tuple[int, ...]] = None):
        self._windows = windows or default_config.prior_low_windows

    @property
    def name(self) -> str:
        return "prior_low"

    @property
    def category(self) -> str:
        return "price_structure"

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("date").reset_index(drop=True)
        lows = df["low"].values
        n = len(df)
        rows = []

        for w in self._windows:
            # Find all swing lows within window w
            swing_low_mask = self._find_swing_lows(lows, w)
            # For each day, the most recent swing low before it = support
            last_swing_idx = -1
            for i in range(n):
                if swing_low_mask[i]:
                    last_swing_idx = i
                if last_swing_idx >= 0 and last_swing_idx < i:
                    rows.append({
                        "date": df["date"].iloc[i],
                        "support_type": f"前低{w}日",
                        "support_price": lows[last_swing_idx],
                        "timeframe": "日",
                    })

        return pd.DataFrame(rows, columns=["date", "support_type", "support_price", "timeframe"])

    @staticmethod
    def _find_swing_lows(lows: np.ndarray, window: int) -> np.ndarray:
        """Find local minima: a point is lower than all points within ±window/2."""
        n = len(lows)
        mask = np.zeros(n, dtype=bool)
        half = max(1, window // 2)
        for i in range(half, n - half):
            left = lows[i - half:i]
            right = lows[i + 1:i + half + 1]
            if lows[i] < left.min() and lows[i] <= right.min():
                mask[i] = True
        return mask
