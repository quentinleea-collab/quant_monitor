"""
Box range (consolidation zone) support detector.

Algorithm:
  1. Rolling N-day window (default 60)
  2. If (max_high - min_low) / midpoint < consolidation_pct → box active
  3. Box lower bound = support until price breaks below by > breakout_pct
"""
import pandas as pd
import numpy as np

from support_detectors.base import BaseSupportDetector
from config import config as default_config, SupportBacktestConfig
from typing import Optional


class BoxRangeSupport(BaseSupportDetector):

    def __init__(self, cfg: Optional[SupportBacktestConfig] = None):
        self.cfg = cfg or default_config

    @property
    def name(self) -> str:
        return "box_range"

    @property
    def category(self) -> str:
        return "price_structure"

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("date").reset_index(drop=True)
        n = len(df)
        lookback = self.cfg.box_lookback
        rows = []
        active_boxes: list[dict] = []  # [{lower, upper, start_idx}]

        for i in range(lookback, n):
            window = df.iloc[i - lookback:i]
            max_high = window["high"].max()
            min_low = window["low"].min()
            midpoint = (max_high + min_low) / 2

            # Check if in consolidation
            range_pct = (max_high - min_low) / midpoint if midpoint > 0 else 999
            if range_pct < self.cfg.box_consolidation_pct:
                # New box found
                active_boxes.append({
                    "lower": min_low,
                    "upper": max_high,
                    "start_idx": i,
                })

            # Emit active box supports (keep only recent, remove broken ones)
            current_price = df["close"].iloc[i]
            for box in active_boxes[:]:
                # Check if box is broken (price below lower by > breakout_pct)
                if current_price < box["lower"] * (1 - self.cfg.box_breakout_pct):
                    active_boxes.remove(box)
                    continue
                # Don't emit for the first few days after box formation
                if i - box["start_idx"] < 3:
                    continue
                rows.append({
                    "date": df["date"].iloc[i],
                    "support_type": "箱体下沿",
                    "support_price": box["lower"],
                    "timeframe": "日",
                })

            # Keep at most 5 active boxes
            if len(active_boxes) > 5:
                active_boxes = active_boxes[-5:]

        return pd.DataFrame(rows, columns=["date", "support_type", "support_price", "timeframe"])
