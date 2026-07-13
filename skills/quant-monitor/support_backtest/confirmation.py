"""
Top 5 confirming indicators at support touch points.

1. Extreme volume shrinkage: volume / vol_ma5 < 0.5
2. Long lower shadow: lower_shadow >= 0.6
3. Quick recovery: days to reclaim support
4. K-line pattern: hammer, morning star, etc. on touch day
5. Multi-dimension resonance: ≥2 support types converge within ±1%
"""
import logging
from typing import Optional

import pandas as pd
import numpy as np

from backtest import TouchEvent
from config import config as default_config, SupportBacktestConfig

logger = logging.getLogger(__name__)


class ConfirmationAnalyzer:
    """Enrich touch events with Top 5 confirming indicators."""

    def __init__(self, cfg: Optional[SupportBacktestConfig] = None):
        self.cfg = cfg or default_config

    def analyze(
        self, events: list[TouchEvent], df: pd.DataFrame,
        supports_df: pd.DataFrame
    ) -> list[TouchEvent]:
        """
        Compute confirmation scores for each event.

        Returns the same list with events enriched (mutated in-place).
        """
        df = df.sort_values("date").reset_index(drop=True)
        # Build date → row index map
        date_to_idx = {d: i for i, d in enumerate(df["date"])}

        # For resonance: count supports per date
        resonance_counts = self._compute_resonance(supports_df)

        for event in events:
            idx = date_to_idx.get(event.date)
            if idx is None:
                event.confirmation_score = 0
                event.confirmation_detail = {}
                continue

            row = df.iloc[idx]
            detail = {}

            # 1. Volume shrinkage
            vol_ratio = row["volume"] / row["vol_ma5"] if row["vol_ma5"] > 0 else 999
            detail["shrink"] = vol_ratio < self.cfg.shrink_vol_ratio
            detail["shrink_detail"] = {
                "vol_ratio": round(float(vol_ratio), 3),
                "threshold": self.cfg.shrink_vol_ratio,
            }

            # 2. Long lower shadow
            ls = row.get("lower_shadow", 0)
            detail["long_shadow"] = ls >= self.cfg.long_shadow_min
            detail["shadow_detail"] = {
                "lower_shadow": round(float(ls), 3),
                "threshold": self.cfg.long_shadow_min,
            }

            # 3. Recovery speed (from rebound results, shortest successful recovery)
            recovery_days = []
            for (period, target), result in event.rebound_results.items():
                if result["rebounded"]:
                    recovery_days.append(result["days"])
            detail["quick_recovery"] = len(recovery_days) > 0
            detail["recovery_detail"] = {
                "fastest_days": int(min(recovery_days)) if recovery_days else None,
                "recovery_count": len(recovery_days),
                "total_combos": len(event.rebound_results),
            }

            # 4. K-line pattern
            patterns = {
                "hammer": row.get("pat_hammer", False),
                "morning_star": row.get("pat_morning_star", False),
                "bullish_engulf": row.get("pat_bullish_engulf", False),
                "piercing": row.get("pat_piercing", False),
                "doji": row.get("pat_doji", False),
            }
            detail["pattern"] = any(patterns.values())
            detail["pattern_detail"] = {k: bool(v) for k, v in patterns.items()}

            # 5. Multi-dimension resonance
            day_resonance = resonance_counts.get(event.date, 0)
            detail["resonance"] = day_resonance >= 2
            detail["resonance_detail"] = {
                "convergent_count": int(day_resonance),
                "is_strong": day_resonance >= 3,
            }

            # Composite score
            score = sum([
                1 if detail["shrink"] else 0,
                1 if detail["long_shadow"] else 0,
                1 if detail["quick_recovery"] else 0,
                1 if detail["pattern"] else 0,
                1 if detail["resonance"] else 0,
            ])

            event.confirmation_score = score
            event.confirmation_detail = detail

        # Log distribution
        scores = [e.confirmation_score for e in events]
        if scores:
            logger.info(
                f"Confirmation scores: mean={np.mean(scores):.2f}, "
                f"max={max(scores)}, distribution: "
                f"{dict(zip(*np.unique(scores, return_counts=True)))}"
            )

        return events

    def _compute_resonance(self, supports_df: pd.DataFrame) -> dict:
        """
        Count unique support types that converge within ±1% on each day.
        Resonance = multiple different support types pointing to similar price.
        """
        if supports_df.empty:
            return {}

        df = supports_df.copy()
        resonance: dict[pd.Timestamp, int] = {}

        for date, grp in df.groupby("date"):
            prices = grp["support_price"].values
            types_all = grp["support_type"].values  # per-row, length = len(prices)
            types_unique = grp["support_type"].unique()
            n = len(prices)

            # Count how many support types have prices within 1% of each other
            # Simple approach: count unique support type categories per day
            # "Resonance" means ≥2 DIFFERENT support types point to similar price zone
            convergent_groups = 0
            used = set()
            for i in range(n):
                if i in used:
                    continue
                group = [i]
                for j in range(i + 1, n):
                    if j in used:
                        continue
                    if abs(prices[i] - prices[j]) / prices[i] < self.cfg.resonance_price_pct:
                        # Check if different support types
                        if types_all[i] != types_all[j]:
                            group.append(j)
                            used.add(j)
                if len(set(types_all[g] for g in group)) >= 2:
                    convergent_groups += 1
                used.add(i)

            # Count total unique support types on this day
            resonance[date] = len(types_unique)

        return resonance
