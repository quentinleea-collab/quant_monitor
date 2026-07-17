"""
Multi-outcome label generator for market bottom detection.

Labels:
  rebound_3pct: 1 if future 10d max return >= 3%    (original)
  tp_win:       1 if price hit +5% before hitting -3% in next 20d
  sl_loss:      1 if price hit -3% before hitting +5% in next 20d
  final_profit: 1 if final outcome after 20d is positive
"""
import pandas as pd
import numpy as np
from typing import Optional
from ma_config import config as default_config, MarketAnalyzerConfig


class LabelGenerator:
    """Generate multi-outcome labels for bottom detection."""

    def __init__(self, cfg: Optional[MarketAnalyzerConfig] = None):
        self.cfg = cfg or default_config

    def generate_all(self, df: pd.DataFrame, horizon: int = None) -> pd.DataFrame:
        """
        Generate all label types.

        Args:
            df: DataFrame with 'close' column, sorted by date
            horizon: forward-looking days (default: config.label_horizon)

        Returns DataFrame with columns:
            rebound_3pct, tp_win, sl_loss, final_profit
        """
        horizon = horizon or self.cfg.label_horizon
        close = df['close'].values
        n = len(close)

        labels = pd.DataFrame(index=df.index)
        labels['rebound_3pct'] = np.nan
        labels['tp_win'] = np.nan
        labels['sl_loss'] = np.nan
        labels['final_profit'] = np.nan

        for i in range(n - horizon):
            entry = close[i]
            future = close[i+1 : i+horizon+1]
            tp_target = entry * 1.05   # +5% take profit
            sl_target = entry * 0.97   # -3% stop loss

            # 1. Rebound >= 3%
            max_ret = np.max(future) / entry - 1
            labels.iloc[i, labels.columns.get_loc('rebound_3pct')] = 1 if max_ret >= 0.03 else 0

            # 2. Which hits first: TP or SL?
            hit_tp = False
            hit_sl = False
            tp_day = None
            sl_day = None
            for j, price in enumerate(future):
                if price >= tp_target:
                    hit_tp = True
                    tp_day = j + 1
                    break
                if price <= sl_target:
                    hit_sl = True
                    sl_day = j + 1
                    break

            labels.iloc[i, labels.columns.get_loc('tp_win')] = 1 if hit_tp else 0
            labels.iloc[i, labels.columns.get_loc('sl_loss')] = 1 if hit_sl else 0

            # 3. Final profit (after 20d or at exit)
            if hit_tp or hit_sl:
                exit_price = tp_target if hit_tp else sl_target
            else:
                exit_price = close[i + horizon]
            labels.iloc[i, labels.columns.get_loc('final_profit')] = 1 if exit_price > entry else 0

        return labels

    def generate(self, df: pd.DataFrame, horizon: int = None, threshold: float = None) -> pd.Series:
        """Legacy: single rebound label (backward compat)."""
        horizon = horizon or self.cfg.label_horizon
        threshold = threshold or self.cfg.label_threshold
        close = df['close'].values
        n = len(close)
        labels = np.full(n, np.nan)
        for i in range(n - horizon):
            ret = np.max(close[i+1 : i+horizon+1]) / close[i] - 1
            labels[i] = 1 if ret >= threshold else 0
        return pd.Series(labels, index=df.index, name='label')

    def get_label_stats(self, labels) -> dict:
        """Class balance stats."""
        if isinstance(labels, pd.DataFrame):
            stats = {}
            for col in labels.columns:
                valid = labels[col].dropna()
                stats[col] = {
                    'total': len(valid), 'positive': int(valid.sum()),
                    'positive_rate': round(float(valid.mean()) * 100, 1),
                }
            return stats
        valid = labels.dropna()
        return {
            'total': len(valid), 'positive': int(valid.sum()),
            'negative': int((1 - valid).sum()),
            'positive_rate': float(valid.mean()),
        }
