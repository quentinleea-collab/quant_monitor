"""
Technical indicators: MA, BOLL, ATR, RSI, volume MA,
lower shadow ratio, and Japanese candlestick pattern detection.
"""
import logging
from typing import Optional, Tuple

import pandas as pd
import numpy as np

from config import config as default_config, SupportBacktestConfig

logger = logging.getLogger(__name__)


class IndicatorCalculator:
    """Compute all technical indicators. Appends columns, never mutates input."""

    def __init__(self, cfg: Optional[SupportBacktestConfig] = None):
        self.cfg = cfg or default_config

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all indicators and return new DataFrame."""
        df = df.copy()
        df = df.sort_values("date").reset_index(drop=True)

        # MA — daily periods
        for p in self.cfg.daily_ma_periods:
            df[f"MA{p}"] = df["close"].rolling(p, min_periods=p).mean()

        # BOLL
        df["BOLL_mid"] = df["close"].rolling(self.cfg.bollinger_period,
                                              min_periods=self.cfg.bollinger_period).mean()
        boll_std = df["close"].rolling(self.cfg.bollinger_period,
                                        min_periods=self.cfg.bollinger_period).std()
        df["BOLL_upper"] = df["BOLL_mid"] + self.cfg.bollinger_std * boll_std
        df["BOLL_lower"] = df["BOLL_mid"] - self.cfg.bollinger_std * boll_std

        # ATR
        df["ATR14"] = self._atr(df, 14)

        # RSI
        df["RSI6"] = self._rsi(df, 6)

        # Volume MA
        df["vol_ma5"] = df["volume"].rolling(5, min_periods=5).mean()
        df["vol_ma20"] = df["volume"].rolling(20, min_periods=20).mean()

        # Lower shadow ratio
        body_low = df[["open", "close"]].min(axis=1)
        total_range = df["high"] - df["low"]
        df["lower_shadow"] = np.where(
            total_range > 0,
            (body_low - df["low"]) / total_range,
            0.0,
        )
        # Upper shadow ratio
        body_high = df[["open", "close"]].max(axis=1)
        df["upper_shadow"] = np.where(
            total_range > 0,
            (df["high"] - body_high) / total_range,
            0.0,
        )
        # Body ratio
        df["body_ratio"] = np.where(
            total_range > 0,
            (body_high - body_low) / total_range,
            0.0,
        )

        # Previous close
        df["prev_close"] = df["close"].shift(1)

        logger.info(f"Indicators computed: {len(df.columns)} columns, {len(df)} rows")
        return df

    def detect_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect K-line patterns. Appends boolean columns to df.

        Patterns: doji, hammer, inverted_hammer, bullish_engulfing,
                  bearish_engulfing, morning_star, piercing_line
        """
        df = df.copy()
        o, h, l, c = df["open"], df["high"], df["low"], df["close"]
        body = (c - o).abs()
        total_range = h - l
        prev_o = df["open"].shift(1)
        prev_c = df["close"].shift(1)

        # Doji: tiny body relative to range
        df["pat_doji"] = (total_range > 0) & (body / total_range < self.cfg.doji_body_ratio)

        # Hammer: small body, long lower shadow, tiny upper shadow
        lower_shadow = np.where(total_range > 0, (np.minimum(o, c) - l) / total_range, 0)
        upper_shadow = np.where(total_range > 0, (h - np.maximum(o, c)) / total_range, 0)
        df["pat_hammer"] = (
            (lower_shadow >= 0.6) &
            (upper_shadow <= self.cfg.hammer_upper_max) &
            (body > 0)
        )

        # Inverted hammer (shooting star mirror)
        df["pat_inverted_hammer"] = (
            (upper_shadow >= 0.6) &
            (lower_shadow <= self.cfg.hammer_upper_max) &
            (body > 0)
        )

        # Bullish engulfing
        df["pat_bullish_engulf"] = (
            (prev_c < prev_o) &           # prior bear candle
            (c > o) &                      # current bull candle
            (o <= prev_c) & (c >= prev_o)  # engulfs
        )

        # Morning star: 3-candle pattern
        df["pat_morning_star"] = (
            (prev_c.shift(1) < prev_o.shift(1)) &     # day-2: bear
            (body.shift(1) < body.shift(2) * 0.5) &    # day-1: small body
            (c > o) &                                   # today: bull
            (c > (prev_c.shift(2) + prev_o.shift(2)) / 2)  # closes above midpoint of day-2
        )

        # Piercing line
        df["pat_piercing"] = (
            (prev_c < prev_o) &           # prior bear
            (c > o) &                      # today bull
            (o < prev_c) &                 # gap down open
            (c > (prev_o + prev_c) / 2)    # closes above midpoint of prior
        )

        logger.info("K-line patterns detected")
        return df

    # ── private ───────────────────────────────────────

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    @staticmethod
    def _rsi(df: pd.DataFrame, period: int = 6) -> pd.Series:
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100.0 - (100.0 / (1.0 + rs))


def compute_indicators(df: pd.DataFrame,
                       cfg: Optional[SupportBacktestConfig] = None) -> pd.DataFrame:
    """Convenience: compute all indicators + patterns in one call."""
    calc = IndicatorCalculator(cfg)
    df = calc.compute_all(df)
    df = calc.detect_patterns(df)
    return df
