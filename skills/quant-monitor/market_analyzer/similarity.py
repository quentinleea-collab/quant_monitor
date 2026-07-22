"""
Historical similarity analysis — finds time periods most similar to current market.
Uses rolling window feature vectors + cosine similarity.
"""
import logging
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional

logger = logging.getLogger(__name__)


class SimilarityAnalyzer:
    """Find historical periods that look most like today."""

    def __init__(self, lookback_window: int = 10, top_k: int = 10):
        """
        Args:
            lookback_window: how many recent days to use as the "pattern"
            top_k: number of top matches to return
        """
        self.lookback = lookback_window
        self.top_k = top_k

    def find_similar(self, features: pd.DataFrame, close: pd.Series) -> list[dict]:
        """
        Find historical periods similar to the most recent pattern.

        Args:
            features: feature matrix (date index, feature columns)
            close: close price series (same date index)

        Returns list of dicts sorted by similarity (high→low):
            {start_date, end_date, similarity, fwd_3d, fwd_5d, fwd_10d}
        """
        n = len(features)
        if n < self.lookback + 20:
            return []

        # ── 1. Current pattern: last `lookback` days flattened ──
        current_window = features.iloc[-self.lookback:]
        current_vec = self._flatten_normalize(current_window)

        if current_vec is None:
            return []

        # ── 2. Slide through history, compute similarity ──
        matches = []
        # Minimum gap: exclude last 3 days (too close to current)
        for i in range(0, n - self.lookback - 3):
            hist_window = features.iloc[i:i + self.lookback]
            hist_vec = self._flatten_normalize(hist_window)
            if hist_vec is None:
                continue

            sim = float(cosine_similarity([current_vec], [hist_vec])[0][0])

            # Forward returns from end of historical window
            end_idx = i + self.lookback
            if end_idx + 10 >= n:
                continue
            entry_price = float(close.iloc[end_idx])

            fwd_3d = float(close.iloc[end_idx + 3] / entry_price - 1) * 100 if end_idx + 3 < n else None
            fwd_5d = float(close.iloc[end_idx + 5] / entry_price - 1) * 100 if end_idx + 5 < n else None
            fwd_10d = float(close.iloc[end_idx + 10] / entry_price - 1) * 100 if end_idx + 10 < n else None

            matches.append({
                'start_date': str(features.index[i])[:10],
                'end_date': str(features.index[i + self.lookback - 1])[:10],
                'similarity': round(sim * 100, 1),  # 0-100 scale
                'fwd_3d': round(fwd_3d, 1) if fwd_3d is not None else None,
                'fwd_5d': round(fwd_5d, 1) if fwd_5d is not None else None,
                'fwd_10d': round(fwd_10d, 1) if fwd_10d is not None else None,
            })

        # ── 3. Sort by similarity descending ──
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        return matches[:self.top_k]

    def _flatten_normalize(self, window: pd.DataFrame) -> np.ndarray | None:
        """Flatten a feature window into a normalized vector."""
        if window.isna().any().any():
            return None
        vec = window.values.flatten().astype(np.float64)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return None
        return vec / norm
