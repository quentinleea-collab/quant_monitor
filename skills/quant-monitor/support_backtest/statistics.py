"""
Statistical analysis and composite scoring.

Ranks support types by:
  composite_score = touch_weight * log(1+touches)
                  + rebound_weight * rebound_rate
                  + confirmation_weight * avg_confirmation

Outputs two views:
  1. By support type — e.g. "MA20" aggregated across all dates
  2. By category — e.g. "ma" aggregated
"""
import logging
from typing import Optional

import pandas as pd
import numpy as np

from backtest import TouchEvent
from config import config as default_config, SupportBacktestConfig

logger = logging.getLogger(__name__)


class StatisticsAnalyzer:
    """Compute rankings and statistics from touch events."""

    def __init__(self, cfg: Optional[SupportBacktestConfig] = None):
        self.cfg = cfg or default_config

    def analyze(self, events: list[TouchEvent]) -> dict:
        """
        Args:
            events: List of TouchEvent with confirmation scores

        Returns dict with keys:
          - type_ranking: DataFrame (ranked by composite score)
          - category_ranking: DataFrame
          - summary: dict with overall stats
        """
        if not events:
            return self._empty_result()

        # Build per-type aggregation
        rows = []
        for e in events:
            # Best rebound result
            best_rebound = False
            best_days = None
            for (period, target), res in e.rebound_results.items():
                if res["rebounded"]:
                    best_rebound = True
                    if best_days is None or res["days"] < best_days:
                        best_days = res["days"]

            rows.append({
                "support_type": e.support_type,
                "category": self._classify_category(e.support_type),
                "timeframe": e.timeframe,
                "date": e.date,
                "support_price": e.support_price,
                "touched": True,
                "rebounded": best_rebound,
                "fastest_recovery_days": best_days,
                "confirmation_score": e.confirmation_score,
                "touch_depth": e.touch_depth,
            })

        df = pd.DataFrame(rows)

        # ── By support type ──────────────────────────
        type_agg = df.groupby(["support_type", "timeframe"]).agg(
            touch_count=("touched", "sum"),
            rebound_count=("rebounded", "sum"),
            avg_recovery_days=("fastest_recovery_days", "mean"),
            avg_confirmation=("confirmation_score", "mean"),
            avg_touch_depth=("touch_depth", "mean"),
        ).reset_index()

        type_agg["rebound_rate"] = np.where(
            type_agg["touch_count"] > 0,
            type_agg["rebound_count"] / type_agg["touch_count"],
            0.0,
        )

        # Composite score
        tw = self.cfg.touch_count_weight
        rw = self.cfg.rebound_weight
        cw = self.cfg.confirmation_weight

        max_confirmation = 5.0
        type_agg["composite_score"] = (
            tw * np.log1p(type_agg["touch_count"]) / np.log1p(type_agg["touch_count"].max())
            + rw * type_agg["rebound_rate"]
            + cw * type_agg["avg_confirmation"] / max_confirmation
        )

        type_agg = type_agg.sort_values("composite_score", ascending=False)
        type_agg["rank"] = range(1, len(type_agg) + 1)

        # ── By category ──────────────────────────────
        cat_agg = df.groupby("category").agg(
            touch_count=("touched", "sum"),
            rebound_count=("rebounded", "sum"),
            avg_recovery_days=("fastest_recovery_days", "mean"),
            avg_confirmation=("confirmation_score", "mean"),
            unique_types=("support_type", "nunique"),
        ).reset_index()

        cat_agg["rebound_rate"] = np.where(
            cat_agg["touch_count"] > 0,
            cat_agg["rebound_count"] / cat_agg["touch_count"],
            0.0,
        )
        cat_agg["composite_score"] = (
            tw * np.log1p(cat_agg["touch_count"]) / np.log1p(cat_agg["touch_count"].max())
            + rw * cat_agg["rebound_rate"]
            + cw * cat_agg["avg_confirmation"] / max_confirmation
        )
        cat_agg = cat_agg.sort_values("composite_score", ascending=False)

        # ── Summary ──────────────────────────────────
        summary = {
            "total_touch_events": len(df),
            "unique_support_types": df["support_type"].nunique(),
            "overall_rebound_rate": df["rebounded"].mean() if len(df) > 0 else 0,
            "avg_confirmation_score": df["confirmation_score"].mean() if len(df) > 0 else 0,
            "top_support": type_agg.iloc[0]["support_type"] if len(type_agg) > 0 else None,
            "top_rebound_rate": type_agg["rebound_rate"].max() if len(type_agg) > 0 else 0,
        }

        logger.info(
            f"Stats: {len(df)} touches, overall rebound rate={summary['overall_rebound_rate']:.1%}, "
            f"top={summary['top_support']}"
        )

        return {
            "type_ranking": type_agg,
            "category_ranking": cat_agg,
            "event_details": df,
            "summary": summary,
        }

    def _empty_result(self) -> dict:
        return {
            "type_ranking": pd.DataFrame(),
            "category_ranking": pd.DataFrame(),
            "event_details": pd.DataFrame(),
            "summary": {"total_touch_events": 0, "overall_rebound_rate": 0},
        }

    @staticmethod
    def _classify_category(support_type: str) -> str:
        if support_type.startswith("MA"):
            return "ma"
        if "前低" in support_type:
            return "price_structure"
        if "箱体" in support_type:
            return "price_structure"
        if "趋势" in support_type:
            return "price_structure"
        if "BB" in support_type:
            return "volatility"
        if "密集" in support_type:
            return "volume"
        if "关口" in support_type:
            return "psychological"
        return "other"
