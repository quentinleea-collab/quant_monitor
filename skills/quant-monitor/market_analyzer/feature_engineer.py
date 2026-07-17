"""
Feature engineering pipeline for market bottom detection.

Produces ~23 features from multi-timeframe OHLCV data:
  - Price position (MA deviation, BB position, N-day low ratio)
  - Momentum (RSI, MACD, returns)
  - Volume (ratios and trend)
  - K-line (lower shadow, body ratio, hammer)
  - Multi-TF (weekly MA20, 60min RSI/MACD)

Usage:
    from feature_engineer import FeatureEngineer
    fe = FeatureEngineer()
    df = fe.build_features("588170", "20240101", "20260716")
"""

import sys
import os
import logging
from typing import Optional

import pandas as pd
import numpy as np

# ── Ensure support_backtest is importable ────────────────────────────────
_sb_path = os.path.join(os.path.dirname(__file__), "..", "support_backtest")
if _sb_path not in sys.path:
    sys.path.insert(0, _sb_path)

from data_loader import DataLoader
from indicator import IndicatorCalculator, compute_indicators

# ── Load market_analyzer config under a unique module name ───────────────
# We use importlib here to avoid a sys.modules collision with
# support_backtest/config.py (which indicator.py imports as 'config').
import importlib.util as _util

_ma_cfg_path = os.path.join(os.path.dirname(__file__), "ma_config.py")
_ma_cfg_spec = _util.spec_from_file_location("_market_analyzer_config_mod", _ma_cfg_path)
_ma_cfg_mod = _util.module_from_spec(_ma_cfg_spec)
_ma_cfg_spec.loader.exec_module(_ma_cfg_mod)

# 'config' at module level is the MarketAnalyzerConfig singleton.
# This does NOT shadow support_backtest/config — that module lives under
# a different sys.modules key loaded by indicator.py internally.
config = _ma_cfg_mod.config

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
#  Indicator helpers (extend what support_backtest.indicator provides)
# ═══════════════════════════════════════════════════════════════════════


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index for an arbitrary period."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD histogram: (DIF - DEA) * 2."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    return (dif - dea) * 2


# ═══════════════════════════════════════════════════════════════════════

class FeatureEngineer:
    """Build feature matrix for market bottom detection."""

    def __init__(self, cfg=None):
        self.cfg = cfg or config
        self.loader = DataLoader()

    # ── public API ─────────────────────────────────────────────────

    def build_features(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """
        Build feature matrix from multi-timeframe data.

        Steps:
          1. Load daily + weekly (resampled) + 60min data
          2. Compute indicators on each timeframe
          3. Build feature columns across all categories
          4. Return clean DataFrame (date index, feature columns, no NaN)

        Returns:
            DataFrame with columns = features, index = date
        """
        logger.info("Building features for %s [%s -> %s]", symbol, start, end)

        # ── 1. Daily data ──────────────────────────────────────────
        try:
            daily = self.loader.fetch_daily(symbol, start, end)
        except RuntimeError as e:
            logger.warning("Real API failed: %s — using mock data", e)
            daily = self._mock_daily(start, end)
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.sort_values("date").reset_index(drop=True)

        # ── 2. Daily indicators (MA, BOLL, RSI6, vol_ma, shadows) ──
        daily = compute_indicators(daily)

        # RSI14 — indicator module only computes RSI6
        daily["RSI14"] = _rsi(daily["close"], 14)

        # MACD histogram
        daily["MACD_hist"] = _macd_hist(daily["close"])

        # ── 3. Weekly MA20 ────────────────────────────────────────
        weekly = self._resample_weekly(daily)
        weekly["MA20"] = weekly["close"].rolling(20, min_periods=20).mean()
        weekly_col = weekly[["date", "MA20"]].rename(columns={"MA20": "w_MA20"}).dropna()

        daily = pd.merge_asof(
            daily.sort_values("date"),
            weekly_col.sort_values("date"),
            on="date",
            direction="backward",
            allow_exact_matches=False,  # don't match same day (week ends before current bar)
        )

        # ── 4. 60-minute data ─────────────────────────────────────
        m60 = self.loader.fetch_minute(symbol, "60", start, end)
        if m60 is not None and not m60.empty:
            m60["date"] = pd.to_datetime(m60["date"])
            m60 = m60.sort_values("date").reset_index(drop=True)
            m60["rsi14"] = _rsi(m60["close"], 14)
            m60["macd_hist"] = _macd_hist(m60["close"])

            # Last bar per calendar day
            m60["day"] = m60["date"].dt.date
            m60_daily = (
                m60.groupby("day")
                .agg({"rsi14": "last", "macd_hist": "last"})
                .reset_index()
            )
            m60_daily["day"] = pd.to_datetime(m60_daily["day"])
            m60_daily = m60_daily.rename(
                columns={"day": "date", "rsi14": "m60_rsi", "macd_hist": "m60_macd_hist"}
            )
            daily = daily.merge(m60_daily, on="date", how="left")

        # ── 5. Build feature columns ──────────────────────────────
        self._add_price_position_features(daily)
        self._add_momentum_features(daily)
        self._add_volume_features(daily)
        self._add_kline_features(daily)
        self._add_multitf_features(daily)

        # ── 6. Select and clean final feature matrix ──────────────
        feature_cols = [
            # Price Position
            "ma5_dev",
            "ma10_dev",
            "ma20_dev",
            "ma30_dev",
            "ma60_dev",
            "bb_position",
            "n_day_low_ratio",
            # Momentum
            "rsi_6",
            "rsi_14",
            "macd_hist",
            "macd_hist_direction",
            "ret_5d",
            "ret_10d",
            "ret_20d",
            # Volume
            "vol_ratio_5",
            "vol_ratio_20",
            "vol_trend_5",
            # K-line
            "lower_shadow",
            "body_ratio",
            "is_hammer",
            # Multi-TF
            "w_ma20_dev",
            "m60_rsi",
            "m60_macd_hist",
        ]

        # Ensure all columns exist (fill missing with sensible defaults)
        for col in feature_cols:
            if col not in daily.columns:
                daily[col] = 0.0  # default: no deviation, no signal
        # Fill missing multi-TF features (no 60min data = neutral signal)
        daily["m60_rsi"] = daily.get("m60_rsi", 50).fillna(50)
        daily["m60_macd_hist"] = daily.get("m60_macd_hist", 0).fillna(0)
        daily["w_ma20_dev"] = daily.get("w_ma20_dev", 0).fillna(0)

        result = daily[["date", "close"] + feature_cols].copy()
        # Only drop rows where core price/momentum features are NaN (not multi-TF)
        core_cols = ["ma5_dev","ma10_dev","ma20_dev","rsi_6","rsi_14","ret_5d"]
        result = result.dropna(subset=core_cols).reset_index(drop=True)
        result = result.set_index("date")
        result.index.name = "date"

        logger.info(
            "Features built: %d columns, %d rows, NaN count = %d",
            len(result.columns),
            len(result),
            result.isna().sum().sum(),
        )
        return result

    # ── private feature builders ──────────────────────────────────

    @staticmethod
    def _add_price_position_features(df: pd.DataFrame) -> None:
        """Price-position features: MA deviations, BB position, N-day low ratio."""
        # MA deviations  (close - MA) / MA * 100
        for p in (5, 10, 20, 30, 60):
            col = f"MA{p}"
            if col in df.columns:
                df[f"ma{p}_dev"] = (df["close"] / df[col] - 1.0) * 100.0

        # Bollinger Band position  [0, 1]  — 0 = at lower, 1 = at upper
        bb_range = df["BOLL_upper"] - df["BOLL_lower"]
        df["bb_position"] = np.where(
            bb_range > 0,
            (df["close"] - df["BOLL_lower"]) / bb_range,
            0.5,  # degenerate case: bands squeezed
        )

        # Distance from 20-day low  [0 = at the low]
        df["n_day_low_ratio"] = (
            df["close"] / df["low"].rolling(20, min_periods=20).min() - 1.0
        )

    @staticmethod
    def _add_momentum_features(df: pd.DataFrame) -> None:
        """Momentum features: RSI, MACD, price returns."""
        df["rsi_6"] = df["RSI6"]
        df["rsi_14"] = df["RSI14"]
        df["macd_hist"] = df["MACD_hist"]

        # MACD histogram direction: 1 = expanding (green bars), -1 = shrinking
        hist_diff = df["MACD_hist"].diff()
        df["macd_hist_direction"] = np.sign(hist_diff).fillna(0)

        # N-day returns
        for n in (5, 10, 20):
            df[f"ret_{n}d"] = df["close"].pct_change(n)

    @staticmethod
    def _add_volume_features(df: pd.DataFrame) -> None:
        """Volume features: ratios vs MA and trend."""
        df["vol_ratio_5"] = df["volume"] / df["vol_ma5"]
        df["vol_ratio_20"] = df["volume"] / df["vol_ma20"]
        df["vol_trend_5"] = df["vol_ma5"] / df["vol_ma5"].shift(5) - 1.0

    @staticmethod
    def _add_kline_features(df: pd.DataFrame) -> None:
        """K-line features: shadows, body, pattern flags."""
        # Already computed by indicator module — just rename or pass through
        df["lower_shadow"] = df["lower_shadow"]
        df["body_ratio"] = df["body_ratio"]
        df["is_hammer"] = df.get("pat_hammer", pd.Series(False, index=df.index)).astype(int)

    @staticmethod
    def _add_multitf_features(df: pd.DataFrame) -> None:
        """Multi-timeframe features."""
        df["w_ma20_dev"] = (df["close"] / df["w_MA20"] - 1.0) * 100.0

    # ── mock data ────────────────────────────────────────────────

    @staticmethod
    def _mock_daily(start: str, end: str) -> pd.DataFrame:
        """Generate synthetic OHLCV data for offline testing."""
        rng = np.random.default_rng(42)
        dates = pd.bdate_range(start, end)
        n = len(dates)
        if n == 0:
            return pd.DataFrame(columns=["date","open","high","low","close","volume","amount"])
        base = 1.0
        log_ret = rng.normal(0.0003, 0.015, n)
        log_ret = np.where(rng.random(n) < 0.02, rng.normal(-0.025, 0.01, n), log_ret)
        closes = base * np.exp(np.cumsum(log_ret))
        closes = np.maximum(closes, base * 0.7)
        vol = closes * 0.015
        opens = closes * (1 + rng.normal(0, 0.005, n))
        highs = np.maximum(opens, closes) + rng.uniform(0, 1, n) * vol
        lows = np.minimum(opens, closes) - rng.uniform(0, 1, n) * vol * 0.8
        highs = np.maximum(highs, lows + 0.001)
        volumes = rng.integers(10_000_000, 100_000_000, n).astype(float)
        return pd.DataFrame({
            "date": dates, "open": opens, "high": highs, "low": lows,
            "close": closes, "volume": volumes, "amount": volumes * closes,
        })

    # ── helpers ───────────────────────────────────────────────────

    @staticmethod
    def _resample_weekly(daily: pd.DataFrame) -> pd.DataFrame:
        """Resample daily OHLCV to weekly frequency, returning DataFrame indexed by date."""
        df = daily.set_index("date").sort_index()
        weekly = df.resample("W").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "amount": "sum",
            }
        ).dropna()
        return weekly.reset_index()
