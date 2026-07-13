"""
Volume cluster (成交密集区) support detector.

Algorithm:
  1. Divide the full price range into N bins
  2. Histogram total volume in each price bin
  3. Top (1 - threshold) percentile bins = high-volume zones
  4. Each high-volume zone is a support when price is above it
"""
import pandas as pd
import numpy as np

from support_detectors.base import BaseSupportDetector
from config import config as default_config, SupportBacktestConfig
from typing import Optional


class VolumeClusterSupport(BaseSupportDetector):

    def __init__(self, cfg: Optional[SupportBacktestConfig] = None):
        self.cfg = cfg or default_config

    @property
    def name(self) -> str:
        return "volume_cluster"

    @property
    def category(self) -> str:
        return "volume"

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("date").reset_index(drop=True)
        n = len(df)
        nbins = self.cfg.volume_profile_bins
        threshold_pct = self.cfg.volume_cluster_threshold

        # Build volume profile over full history
        price_min, price_max = df["low"].min(), df["high"].max()
        bins = np.linspace(price_min, price_max, nbins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        volume_per_bin = np.zeros(nbins)

        for _, row in df.iterrows():
            # Distribute volume across the price range for each day
            day_range = row["high"] - row["low"]
            if day_range <= 0:
                continue
            for b in range(nbins):
                bin_low, bin_high = bins[b], bins[b + 1]
                overlap = max(0, min(row["high"], bin_high) - max(row["low"], bin_low))
                if overlap > 0:
                    volume_per_bin[b] += (overlap / day_range) * row["volume"]

        # Top percentile threshold
        vol_cutoff = np.percentile(volume_per_bin[volume_per_bin > 0], threshold_pct * 100)
        cluster_bins = np.where(volume_per_bin >= vol_cutoff)[0]

        if len(cluster_bins) == 0:
            return pd.DataFrame(columns=["date", "support_type", "support_price", "timeframe"])

        # Merge adjacent cluster bins into zones
        zones = self._merge_adjacent(cluster_bins)
        zone_prices = [bin_centers[z[0]:z[-1] + 1].mean() for z in zones]

        # For each day, emit zone prices below current price as supports
        rows = []
        for _, row in df.iterrows():
            current_close = row["close"]
            for z_price in zone_prices:
                if z_price < current_close:
                    rows.append({
                        "date": row["date"],
                        "support_type": "成交密集区",
                        "support_price": z_price,
                        "timeframe": "日",
                    })

        return pd.DataFrame(rows, columns=["date", "support_type", "support_price", "timeframe"])

    @staticmethod
    def _merge_adjacent(indices: np.ndarray) -> list[list[int]]:
        """Merge consecutive bin indices into zones."""
        if len(indices) == 0:
            return []
        zones = []
        current = [indices[0]]
        for i in range(1, len(indices)):
            if indices[i] == indices[i - 1] + 1:
                current.append(indices[i])
            else:
                zones.append(current)
                current = [indices[i]]
        zones.append(current)
        return zones
