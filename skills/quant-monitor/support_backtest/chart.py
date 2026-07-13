"""
Chart generation for support backtest results.

Generates 5 charts:
  1. Support Map — K-line with support lines overlaid (last 120 days)
  2. Probability Ranking — Horizontal bar chart of top N supports
  3. Rebound Rate Heatmap — Support type x Rebound period
  4. Touch Distribution Pie — By support category
  5. Top 5 Indicator Radar — Average confirmation scores per category
"""
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from config import config as default_config, SupportBacktestConfig

# Chinese font setup
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

logger = logging.getLogger(__name__)


class ChartGenerator:

    def __init__(self, cfg: Optional[SupportBacktestConfig] = None):
        self.cfg = cfg or default_config

    def generate_all(
        self, stats: dict, df: pd.DataFrame, supports_df: pd.DataFrame
    ) -> list[str]:
        """Generate all charts. Returns list of saved file paths."""
        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        code = self.cfg.symbol
        paths = []

        paths.append(self._chart_support_map(df, supports_df, out_dir, code, ts))
        paths.append(self._chart_ranking(stats, out_dir, code, ts))
        paths.append(self._chart_heatmap(stats, out_dir, code, ts))
        paths.append(self._chart_pie(stats, out_dir, code, ts))
        paths.append(self._chart_radar(stats, out_dir, code, ts))

        logger.info(f"{len(paths)} charts saved to {out_dir}")
        return paths

    def _chart_support_map(self, df, supports_df, out_dir, code, ts):
        """K-line chart with key support lines overlaid."""
        fig, ax = plt.subplots(figsize=(16, 8))

        # Last 120 days for readability
        plot_df = df.tail(120).copy()
        plot_df = plot_df.reset_index(drop=True)
        x = range(len(plot_df))

        # Candlestick approximation: color bars
        colors = np.where(
            plot_df["close"] >= plot_df["open"], "red", "green"
        )
        ax.bar(x, plot_df["high"] - plot_df["low"], bottom=plot_df["low"],
               color=colors, width=0.6, linewidth=0.5)
        ax.bar(x, abs(plot_df["close"] - plot_df["open"]),
               bottom=plot_df[["open", "close"]].min(axis=1),
               color=colors, width=0.6, linewidth=0.5)

        # Overlay top 5 supports (if they exist in the plot range)
        type_rank = stats.get("type_ranking", pd.DataFrame())
        if not type_rank.empty:
            top5_types = type_rank.head(5)["support_type"].tolist()
            plot_supports = supports_df[
                supports_df["support_type"].isin(top5_types)
            ]
            plot_dates = set(plot_df["date"].values)
            for sup_type in top5_types[:3]:
                sup_data = plot_supports[plot_supports["support_type"] == sup_type]
                sup_data = sup_data[sup_data["date"].isin(plot_dates)]
                if sup_data.empty:
                    continue
                # Average support price
                avg_price = sup_data["support_price"].mean()
                ax.axhline(y=avg_price, linestyle="--", alpha=0.6,
                          label=f"{sup_type}: {avg_price:.3f}")

        ax.set_title(f"{code} Support Map (Last 120 Days)", fontsize=14)
        ax.set_xlabel("Trading Days (recent)")
        ax.set_ylabel("Price")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)

        path = str(out_dir / f"{code}_01_support_map_{ts}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def _chart_ranking(self, stats, out_dir, code, ts):
        """Horizontal bar chart of top N support types by composite score."""
        type_rank = stats.get("type_ranking", pd.DataFrame())
        if type_rank.empty:
            return ""

        fig, ax = plt.subplots(figsize=(10, 8))
        top_n = min(15, len(type_rank))
        data = type_rank.head(top_n).iloc[::-1]  # Reverse for horizontal bar

        labels = [f"{r['support_type']}({r['timeframe']})" for _, r in data.iterrows()]
        scores = data["composite_score"].values
        rates = data["rebound_rate"].values * 100

        bars = ax.barh(labels, scores, color=plt.cm.RdYlGn(rates / 100))
        ax.set_xlabel("Composite Score")
        ax.set_title(f"{code} Support Level Probability Ranking (Top {top_n})")
        ax.grid(axis="x", alpha=0.3)

        # Add rebound rate annotations
        for i, (score, rate) in enumerate(zip(scores, rates)):
            ax.text(score + 0.01, i, f"{rate:.0f}%", va="center", fontsize=9)

        path = str(out_dir / f"{code}_02_ranking_{ts}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def _chart_heatmap(self, stats, out_dir, code, ts):
        """Rebound rate heatmap: support type x rebound period."""
        events = stats.get("event_details", pd.DataFrame())
        if events.empty:
            return ""

        # Use the first target (1%) for heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        # Simplified: aggregate by support type
        pivot = events.pivot_table(
            values="rebounded", index="support_type", aggfunc="mean"
        )
        pivot = pivot.sort_values("rebounded", ascending=False).head(15)

        heatmap_data = pivot.values
        im = ax.imshow(heatmap_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        ax.set_yticks(range(len(pivot)))
        ax.set_yticklabels(pivot.index.tolist())
        ax.set_xticks([0])
        ax.set_xticklabels(["Overall Rebound Rate"])
        ax.set_title(f"{code} Rebound Rate by Support Type")

        plt.colorbar(im, ax=ax)
        # Annotate cells
        for i in range(len(pivot)):
            val = heatmap_data[i][0]
            ax.text(0, i, f"{val:.1%}", ha="center", va="center", fontsize=10)

        path = str(out_dir / f"{code}_03_heatmap_{ts}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def _chart_pie(self, stats, out_dir, code, ts):
        """Touch distribution by support category."""
        events = stats.get("event_details", pd.DataFrame())
        if events.empty:
            return ""

        fig, ax = plt.subplots(figsize=(8, 8))
        cat_counts = events["category"].value_counts()
        colors = plt.cm.Set3(range(len(cat_counts)))
        wedges, texts, autotexts = ax.pie(
            cat_counts.values, labels=cat_counts.index,
            autopct="%1.1f%%", colors=colors, startangle=90,
        )
        ax.set_title(f"{code} Touch Distribution by Support Category")

        path = str(out_dir / f"{code}_04_pie_{ts}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def _chart_radar(self, stats, out_dir, code, ts):
        """Radar chart of Top 5 indicator scores by category."""
        events = stats.get("event_details", pd.DataFrame())
        if events.empty:
            return ""

        # Average confirmation per category
        cat_scores = events.groupby("category")["confirmation_score"].mean()

        categories = list(cat_scores.index)
        values = cat_scores.values
        n = len(categories)

        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        values = values.tolist()
        angles += angles[:1]
        values += values[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.fill(angles, values, alpha=0.25, color="steelblue")
        ax.plot(angles, values, "o-", color="steelblue", linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_title(f"{code} Avg Confirmation Score by Category (Top 5 Indicators)")

        path = str(out_dir / f"{code}_05_radar_{ts}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
