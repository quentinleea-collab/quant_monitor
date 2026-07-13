"""
Round number (整数关口) psychological support detector.

Generates static levels based on the ETF's price range:
  - Price < 1.0: 0.05 intervals
  - Price 1.0-10.0: 0.10 intervals
  - Price 10.0-50.0: 0.50 intervals
  - Price > 50.0: 1.00 intervals
"""
import pandas as pd
import numpy as np

from support_detectors.base import BaseSupportDetector
from config import config as default_config


class RoundNumberSupport(BaseSupportDetector):

    @property
    def name(self) -> str:
        return "round_number"

    @property
    def category(self) -> str:
        return "psychological"

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("date").reset_index(drop=True)
        price_min = df["low"].min()
        price_max = df["high"].max()

        # Determine step size based on price range midpoint
        midpoint = (price_min + price_max) / 2
        if midpoint < 1.0:
            step = 0.05
        elif midpoint < 10.0:
            step = 0.10
        elif midpoint < 50.0:
            step = 0.50
        else:
            step = 1.00

        # Generate ALL round numbers in the price range
        start = np.floor(price_min / step) * step
        end = np.ceil(price_max / step) * step
        levels = np.arange(start, end + step / 2, step)

        # Also add half-step levels for finer granularity
        half_levels = levels + step / 2

        all_levels = sorted(set(levels) | set(half_levels))

        rows = []
        for _, row in df.iterrows():
            current_close = row["close"]
            for level in all_levels:
                if level < current_close:
                    rows.append({
                        "date": row["date"],
                        "support_type": f"关口{level:.3g}",
                        "support_price": level,
                        "timeframe": "日",
                    })

        return pd.DataFrame(rows, columns=["date", "support_type", "support_price", "timeframe"])
