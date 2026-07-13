"""Configuration for support level backtest system."""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class SupportBacktestConfig:
    """All configurable parameters with sensible defaults."""

    # ── Data ──────────────────────────────────────────
    symbol: str = "588170"
    start_date: str = "20000101"       # full history
    end_date: str = "20260708"

    # ── Multi-timeframe MA periods ────────────────────
    # (period, timeframe_label)
    ma_configs: Tuple[Tuple[int, str], ...] = (
        # Daily
        (5, "日"), (10, "日"), (20, "日"), (30, "日"), (60, "日"),
        # Weekly (resampled)
        (10, "周"), (20, "周"), (30, "周"),
        # Monthly (resampled)
        (5, "月"), (10, "月"), (20, "月"),
        # 60-minute
        (20, "60分钟"), (60, "60分钟"), (120, "60分钟"),
        # 120-minute (resampled from 60min)
        (20, "120分钟"), (60, "120分钟"),
    )

    # ── Touch detection ───────────────────────────────
    touch_tolerance: float = 0.005     # ±0.5% of support price = touch zone

    # ── Rebound evaluation ────────────────────────────
    rebound_periods: Tuple[int, ...] = (1, 2, 3, 5, 10, 20)
    rebound_targets: Tuple[float, ...] = (0.01, 0.02, 0.03, 0.05)

    # ── Detector params ───────────────────────────────
    prior_low_windows: Tuple[int, ...] = (20, 40, 60)
    box_lookback: int = 60
    box_consolidation_pct: float = 0.15
    box_breakout_pct: float = 0.03
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    volume_profile_bins: int = 50
    volume_cluster_threshold: float = 0.70
    round_intervals: Tuple[float, ...] = (0.05, 0.10, 0.50, 1.00)
    trend_swing_window: int = 10
    trend_min_r_squared: float = 0.7

    # ── K-line pattern ────────────────────────────────
    doji_body_ratio: float = 0.001     # body / (high-low) < this → doji
    hammer_shadow_ratio: float = 2.0   # lower_shadow >= body * 2
    hammer_upper_max: float = 0.3      # upper shadow / total ≤ 30%

    # ── Confirmation thresholds ───────────────────────
    shrink_vol_ratio: float = 0.5      # vol / vol_ma5 < 0.5 = severe shrinkage
    long_shadow_min: float = 0.6       # lower shadow >= 60% of range
    resonance_price_pct: float = 0.01  # ±1% convergence

    # ── Scoring weights ───────────────────────────────
    touch_count_weight: float = 0.25
    rebound_weight: float = 0.50
    confirmation_weight: float = 0.25

    # ── Output ────────────────────────────────────────
    output_dir: str = "support_backtest_results"
    generate_excel: bool = True
    generate_charts: bool = True
    top_n_supports: int = 30

    # ── Derived ───────────────────────────────────────
    @property
    def daily_ma_periods(self) -> Tuple[int, ...]:
        return tuple(p for p, t in self.ma_configs if t == "日")

    @property
    def weekly_ma_periods(self) -> Tuple[int, ...]:
        return tuple(p for p, t in self.ma_configs if t == "周")

    @property
    def monthly_ma_periods(self) -> Tuple[int, ...]:
        return tuple(p for p, t in self.ma_configs if t == "月")

    @property
    def min60_ma_periods(self) -> Tuple[int, ...]:
        return tuple(p for p, t in self.ma_configs if t == "60分钟")

    @property
    def min120_ma_periods(self) -> Tuple[int, ...]:
        return tuple(p for p, t in self.ma_configs if t == "120分钟")


# Singleton
config = SupportBacktestConfig()
