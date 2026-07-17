"""Configuration for market bottom detector."""
from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class MarketAnalyzerConfig:
    # Symbols to analyze (index codes for Eastmoney API)
    symbols: List[str] = field(default_factory=lambda: ["000001", "399001", "399006", "159915"])
    symbol_names: dict = field(default_factory=lambda: {
        "000001": "上证指数", "399001": "深证成指", "399006": "创业板指", "159915": "创业板ETF",
    })

    # Data
    start_date: str = "20000101"
    end_date: str = "20260717"

    # Label: future N-day max return >= threshold -> label=1 (bottom found)
    label_horizon: int = 10       # days forward
    label_threshold: float = 0.03 # 3% return = bottom

    # XGBoost params
    xgb_max_depth: int = 4
    xgb_learning_rate: float = 0.05
    xgb_n_estimators: int = 200
    xgb_early_stopping: int = 20

    # TimeSeriesSplit
    tss_n_splits: int = 5

    # Output
    output_dir: str = "market_analyzer_results"
    model_dir: str = "market_analyzer_models"

    # Bottom score thresholds for historical stats
    score_thresholds: Tuple[int, ...] = (50, 60, 70, 80, 90)

    # Pattern mining
    min_pattern_support: int = 10   # minimum occurrences for a pattern
    max_pattern_features: int = 4   # max features in a combo


config = MarketAnalyzerConfig()
