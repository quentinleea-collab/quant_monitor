"""
Pattern miner: discovers high-probability feature combinations for bottom detection.

Algorithm: binarize each feature at its median, test all 2/3/4-feature combos,
score by win_rate * log(sample_count), return top patterns.
"""
import logging
import numpy as np
import pandas as pd
from itertools import combinations
from typing import Optional
from ma_config import config as default_config, MarketAnalyzerConfig

logger = logging.getLogger(__name__)


class PatternMiner:
    """Mine high-probability bottom signal patterns from feature data."""

    def __init__(self, cfg: Optional[MarketAnalyzerConfig] = None):
        self.cfg = cfg or default_config

    def mine(self, X: pd.DataFrame, y: pd.Series, max_features: int = None, min_support: int = None) -> list[dict]:
        """
        Find feature combination patterns with highest win rates.

        Args:
            X: feature matrix
            y: labels (0/1)
            max_features: max features per combo (default from config)
            min_support: min occurrences for a valid pattern

        Returns: sorted list of {features, sample_count, win_rate, score}
        """
        max_features = max_features or self.cfg.max_pattern_features
        min_support = min_support or self.cfg.min_pattern_support

        # Binarize features at median
        X_bin = pd.DataFrame(index=X.index)
        for col in X.columns:
            median = X[col].median()
            X_bin[f'{col}_high'] = (X[col] >= median).astype(int)

        # Get all feature binary columns
        bin_cols = X_bin.columns.tolist()

        patterns = []

        # Test all combinations of size 2, 3, 4
        for size in [2, 3, 4]:
            if size > max_features:
                break
            for combo in combinations(bin_cols, size):
                # Pattern: all features in combo are True (1)
                mask = X_bin[list(combo)].all(axis=1)
                if mask.sum() < min_support:
                    continue

                win_rate = y[mask].mean() if mask.sum() > 0 else 0
                sample_count = int(mask.sum())
                score = win_rate * np.log1p(sample_count)  # reward both win_rate and frequency

                patterns.append({
                    'features': [c.replace('_high', '') for c in combo],
                    'sample_count': sample_count,
                    'win_rate': round(float(win_rate), 4),
                    'score': round(float(score), 4),
                })

        # Sort by score descending
        patterns.sort(key=lambda x: x['score'], reverse=True)

        logger.info(f"Mined {len(patterns)} patterns (max_features={max_features}, min_support={min_support})")
        if patterns:
            top = patterns[0]
            logger.info(f"Top pattern: {top['features']} -> win_rate={top['win_rate']:.1%}, n={top['sample_count']}")

        return patterns[:50]  # return top 50
