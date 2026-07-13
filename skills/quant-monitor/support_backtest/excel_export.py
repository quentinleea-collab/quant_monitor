"""
Multi-sheet Excel report generator.

Sheets:
  1. Summary — Top N supports ranked
  2. By Category — One table per support category
  3. Touch Events — All events with details
  4. Top 5 Indicators — Distribution per support type
  5. Config — Parameters used
"""
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd

from config import config as default_config, SupportBacktestConfig

logger = logging.getLogger(__name__)


class ExcelExporter:
    """Generate formatted multi-sheet Excel report."""

    def __init__(self, cfg: Optional[SupportBacktestConfig] = None):
        self.cfg = cfg or default_config

    def export(self, stats: dict, output_path: Optional[str] = None) -> str:
        """
        Args:
            stats: Output from StatisticsAnalyzer.analyze()
            output_path: Optional custom path

        Returns:
            Path to generated Excel file
        """
        if output_path is None:
            out_dir = Path(self.cfg.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(
                out_dir / f"{self.cfg.symbol}_support_backtest_{ts}.xlsx"
            )

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            # Sheet 1: Summary
            type_rank = stats["type_ranking"]
            top_n = min(self.cfg.top_n_supports, len(type_rank))
            top_n_df = type_rank.head(top_n)
            summary_data = {
                "指标": ["总触及次数", "支撑位类型数", "整体反弹率",
                       "平均确认得分", "最强支撑位", "最高反弹率"],
                "数值": [
                    stats["summary"]["total_touch_events"],
                    stats["summary"]["unique_support_types"],
                    f"{stats['summary']['overall_rebound_rate']:.1%}",
                    f"{stats['summary']['avg_confirmation_score']:.2f}/5",
                    stats["summary"]["top_support"],
                    f"{stats['summary']['top_rebound_rate']:.1%}",
                ],
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False, startrow=0)

            top_n_df.to_excel(writer, sheet_name="Summary", index=False, startrow=10)

            # Sheet 2: By Category
            stats["category_ranking"].to_excel(writer, sheet_name="By Category", index=False)

            # Sheet 3: Touch Events
            events = stats["event_details"]
            # Select key columns
            event_cols = [
                "date", "support_type", "support_price", "touched",
                "rebounded", "fastest_recovery_days", "confirmation_score",
                "touch_depth", "timeframe",
            ]
            event_df = events[event_cols] if not events.empty else pd.DataFrame()
            event_df.to_excel(writer, sheet_name="Touch Events", index=False)

            # Sheet 4: Top 5 Indicators Detail
            if not events.empty:
                indicator_dist = events.groupby("support_type").agg(
                    touch_count=("touched", "sum"),
                    rebound_rate=("rebounded", "mean"),
                    avg_confirmation=("confirmation_score", "mean"),
                ).reset_index()
                indicator_dist["rebound_rate"] = indicator_dist["rebound_rate"].apply(
                    lambda x: f"{x:.1%}"
                )
                indicator_dist.to_excel(writer, sheet_name="Indicators Detail", index=False)

            # Sheet 5: Config
            config_data = {
                "参数": ["标的代码", "数据开始", "数据结束", "触及容差",
                       "反弹周期", "反弹目标", "输出Top N", "前低窗口",
                       "布林带周期", "箱体回溯期"],
                "值": [
                    self.cfg.symbol, self.cfg.start_date, self.cfg.end_date,
                    self.cfg.touch_tolerance,
                    str(self.cfg.rebound_periods), str(self.cfg.rebound_targets),
                    self.cfg.top_n_supports,
                    str(self.cfg.prior_low_windows),
                    f"{self.cfg.bollinger_period}/{self.cfg.bollinger_std}σ",
                    self.cfg.box_lookback,
                ],
            }
            pd.DataFrame(config_data).to_excel(writer, sheet_name="Config", index=False)

        logger.info(f"Excel report saved to {output_path}")
        return output_path
