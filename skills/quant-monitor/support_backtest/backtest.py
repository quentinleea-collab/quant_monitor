"""
Touch detection and rebound evaluation engine.

For each day, checks if the day's low touched any active support
(priced within tolerance). If touched, evaluates rebound across
multiple (period, target) combinations.
"""
import logging
from typing import Optional
from dataclasses import dataclass, field
from itertools import product

import pandas as pd
import numpy as np

from config import config as default_config, SupportBacktestConfig

logger = logging.getLogger(__name__)


@dataclass
class TouchEvent:
    """A single support touch event."""
    date: object
    support_type: str          # e.g. "MA20"
    support_price: float
    actual_low: float
    touch_depth: float         # positive = pierced support by this % below
    close_price: float
    timeframe: str = "日"
    # {(period, target): {"rebounded": bool, "days": int}}
    rebound_results: dict = field(default_factory=dict)


class BacktestEngine:
    """Detect touches and evaluate rebounds."""

    def __init__(self, cfg: Optional[SupportBacktestConfig] = None):
        self.cfg = cfg or default_config
        self._rebound_combos = list(
            product(self.cfg.rebound_periods, self.cfg.rebound_targets)
        )

    def run(self, df: pd.DataFrame, supports_df: pd.DataFrame) -> list[TouchEvent]:
        """
        Args:
            df: Daily OHLCV + indicators, sorted by date
            supports_df: All support levels from all detectors,
                         columns: [date, support_type, support_price, timeframe]

        Returns:
            List of TouchEvent objects
        """
        df = df.sort_values("date").reset_index(drop=True)
        dates = df["date"].values
        lows = df["low"].values
        closes = df["close"].values
        n = len(df)

        # Group supports by date for fast lookup
        supports_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
        for d, grp in supports_df.groupby("date"):
            supports_by_date[d] = grp

        events: list[TouchEvent] = []
        tolerance = self.cfg.touch_tolerance

        for i in range(n):
            d = dates[i]
            if d not in supports_by_date:
                continue

            day_supports = supports_by_date[d]
            day_low = lows[i]
            day_close = closes[i]

            for _, sup in day_supports.iterrows():
                s_price = sup["support_price"]
                if pd.isna(s_price) or s_price <= 0:
                    continue

                # Touch check
                if day_low <= s_price * (1 + tolerance):
                    depth = (s_price - day_low) / s_price
                    event = TouchEvent(
                        date=d,
                        support_type=str(sup["support_type"]),
                        support_price=float(s_price),
                        actual_low=float(day_low),
                        touch_depth=float(depth),
                        close_price=float(day_close),
                        timeframe=str(sup.get("timeframe", "日")),
                    )
                    # Check rebounds
                    for period, target in self._rebound_combos:
                        rebounded, days = self._check_rebound(
                            closes, i, n, s_price, period, target
                        )
                        event.rebound_results[(period, target)] = {
                            "rebounded": rebounded,
                            "days": days,
                        }
                    events.append(event)

        logger.info(
            f"Backtest: {len(events)} touch events across "
            f"{supports_df['support_type'].nunique()} support types"
        )
        return events

    def _check_rebound(
        self, closes: np.ndarray, touch_idx: int, n_days: int,
        support_price: float, period: int, target: float
    ) -> tuple[bool, int]:
        """
        Check if price rebounds from support within `period` days.

        Returns: (rebounded: bool, days_to_rebound: int)
        """
        target_price = support_price * (1 + target)
        end_idx = min(touch_idx + period + 1, n_days)
        for offset in range(1, end_idx - touch_idx):
            idx = touch_idx + offset
            if closes[idx] >= target_price:
                return True, offset
        return False, period
