"""
Label generator for market bottom detection.
Label = 1 if future N-day max return >= threshold (i.e., a bottom was formed).
"""
import pandas as pd
import numpy as np
from typing import Optional
from config import config as default_config, MarketAnalyzerConfig


class LabelGenerator:
    """Generate binary labels: 1 = bottom found (future return >= threshold)."""

    def __init__(self, cfg: Optional[MarketAnalyzerConfig] = None):
        self.cfg = cfg or default_config

    def generate(self, df: pd.DataFrame, horizon: int = None, threshold: float = None) -> pd.Series:
        """
        Args:
            df: DataFrame with 'close' column, sorted by date ascending
            horizon: forward-looking days (default: config.label_horizon)
            threshold: return threshold (default: config.label_threshold)

        Returns:
            pd.Series with 1 (bottom) or 0 (no bottom), aligned to df index.
            Last `horizon` rows are NaN (no forward data).
        """
        horizon = horizon or self.cfg.label_horizon
        threshold = threshold or self.cfg.label_threshold

        close = df['close'].values
        n = len(close)
        labels = np.full(n, np.nan)

        for i in range(n - horizon):
            future_max = np.max(close[i + 1: i + horizon + 1])
            ret = (future_max - close[i]) / close[i]
            labels[i] = 1 if ret >= threshold else 0

        return pd.Series(labels, index=df.index, name='label')

    def get_label_stats(self, labels: pd.Series) -> dict:
        """Return class balance and other stats."""
        valid = labels.dropna()
        return {
            'total': len(valid),
            'positive': int(valid.sum()),
            'negative': int((1 - valid).sum()),
            'positive_rate': float(valid.mean()),
        }
