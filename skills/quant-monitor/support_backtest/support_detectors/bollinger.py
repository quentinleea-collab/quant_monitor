"""Bollinger Band lower band support detector."""
import pandas as pd

from support_detectors.base import BaseSupportDetector


class BollingerSupport(BaseSupportDetector):
    """BB lower band = MA20 - 2*std. Simple, one value per day."""

    @property
    def name(self) -> str:
        return "bollinger"

    @property
    def category(self) -> str:
        return "volatility"

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        if "BOLL_lower" not in df.columns:
            raise ValueError("BOLL_lower column missing; run indicator.py first")

        rows = []
        for _, row in df.iterrows():
            price = row["BOLL_lower"]
            if pd.isna(price) or price <= 0:
                continue
            rows.append({
                "date": row["date"],
                "support_type": "BB下轨",
                "support_price": price,
                "timeframe": "日",
            })
        return pd.DataFrame(rows, columns=["date", "support_type", "support_price", "timeframe"])
