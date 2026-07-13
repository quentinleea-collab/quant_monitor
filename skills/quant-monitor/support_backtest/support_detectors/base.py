"""Abstract base class for support level detectors."""
from abc import ABC, abstractmethod

import pandas as pd


class BaseSupportDetector(ABC):
    """
    All support detectors inherit from this.

    The detect() method takes a DataFrame with OHLCV + indicators
    and returns a DataFrame with one row per (date x support_price):
      date | support_type | support_price | timeframe
    """

    @abstractmethod
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Args:
            df: OHLCV + indicator DataFrame (from indicator.py)

        Returns:
            DataFrame with columns:
              - date: datetime
              - support_type: str, e.g. "MA20", "PriorLow40"
              - support_price: float
              - timeframe: str, e.g. "日", "周", "月", "60分钟", "120分钟"
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short unique name, e.g. 'ma_support'"""
        ...

    @property
    @abstractmethod
    def category(self) -> str:
        """Category for grouping: 'ma' | 'price_structure' | 'volume' | 'volatility' | 'psychological'"""
        ...


def combine_detections(detectors: list[BaseSupportDetector],
                       df: pd.DataFrame) -> pd.DataFrame:
    """Run all detectors and concatenate results."""
    frames = []
    for det in detectors:
        result = det.detect(df)
        if not result.empty:
            frames.append(result)
    if not frames:
        return pd.DataFrame(columns=["date", "support_type", "support_price", "timeframe"])
    return pd.concat(frames, ignore_index=True).sort_values(["date", "support_type"])
