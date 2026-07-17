"""
Daily scanner: load models, compute features, predict bottom probability for each symbol.

Usage:
    from daily_scanner import DailyScanner
    scanner = DailyScanner()
    df = scanner.scan()              # scan all configured symbols
    df = scanner.scan(["000001"])    # scan a subset
"""
import sys, os, logging
import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# ── Ensure support_backtest is importable ────────────────────────────────
_sb_path = os.path.join(os.path.dirname(__file__), "..", "support_backtest")
if _sb_path not in sys.path:
    sys.path.insert(0, _sb_path)

# ── pipeline modules ─────────────────────────────────────────────────────
from ma_config import config as default_config, MarketAnalyzerConfig
from feature_engineer import FeatureEngineer
from model_trainer import ModelTrainer
from label_generator import LabelGenerator
from data_loader import DataLoader

config = default_config


class DailyScanner:
    """Daily market bottom scanner.

    Orchestrates feature engineering, model loading/prediction, and
    result formatting for all configured symbols.

    Typical usage:
        scanner = DailyScanner()
        results = scanner.scan()
        print(results.to_string(index=False))
    """

    def __init__(self, cfg: Optional[MarketAnalyzerConfig] = None):
        self.cfg = cfg or config
        self.engineer = FeatureEngineer(cfg)
        self.trainer = ModelTrainer(cfg)

    # ── public API ─────────────────────────────────────────────────────────

    def scan(self, symbols: list[str] = None) -> pd.DataFrame:
        """Scan all symbols and return bottom probabilities.

        Args:
            symbols: List of symbols to scan. Defaults to config.symbols.

        Returns:
            DataFrame with columns:
                symbol, name, date, bottom_prob, trend_state, recommendation
            On errors, bottom_prob is None and recommendation describes
            the issue.
        """
        symbols = symbols or self.cfg.symbols
        today = datetime.now().strftime('%Y%m%d')
        results = []

        for symbol in symbols:
            try:
                row = self._scan_one(symbol, today)
                results.append(row)
            except FileNotFoundError:
                logger.warning(
                    "No trained model for %s. Run 'python main.py train' first.",
                    symbol,
                )
                results.append({
                    'symbol': symbol,
                    'name': self.cfg.symbol_names.get(symbol, symbol),
                    'date': today,
                    'bottom_prob': None,
                    'trend_state': '未训练',
                    'recommendation': '需先训练模型',
                })
            except Exception as e:
                logger.error("Error scanning %s: %s", symbol, e)
                results.append({
                    'symbol': symbol,
                    'name': self.cfg.symbol_names.get(symbol, symbol),
                    'date': today,
                    'bottom_prob': None,
                    'trend_state': '数据错误',
                    'recommendation': str(e)[:50],
                })

        result_df = pd.DataFrame(results)
        logger.info("Scan complete: %d symbols processed", len(result_df))
        return result_df

    # ── per-symbol logic ──────────────────────────────────────────────────

    def _scan_one(self, symbol: str, end_date: str) -> dict:
        """Scan a single symbol and return its bottom probability dict.

        Steps:
            1. Build feature matrix from full history.
            2. Load or train an XGBoost model.
            3. Predict bottom probability for the latest bar.
            4. Classify trend state and generate a recommendation.
        """
        start_date = '20000101'

        # ── 1. Build features ──────────────────────────────────────────────
        data = self.engineer.build_features(symbol, start_date, end_date)
        if data.empty:
            raise RuntimeError(f"No features generated for {symbol}")

        # Remove close to prevent data leakage (model was trained without it)
        close = data["close"] if "close" in data.columns else None
        features = data.drop(columns=["close"]) if "close" in data.columns else data

        # ── 2. Load or auto-train model ────────────────────────────────────
        try:
            self.trainer.load(symbol)
        except FileNotFoundError:
            logger.info("No saved model for %s — training on the fly", symbol)
            self._auto_train(symbol, features, start_date, end_date)

        # ── 3. Predict latest bar ──────────────────────────────────────────
        # (close is already extracted from data before features was built)
        try:
            proba = self.trainer.predict_proba(features, symbol)
        except KeyError:
            raise RuntimeError(f"Model for {symbol} failed to load")

        latest_prob = float(proba[-1]) * 100.0  # 0-100 scale

        # ── 4. Compute risk metrics ─────────────────────────────────────────
        close_series = close if close is not None else None
        stop_loss = self._calc_stop_loss(features, close_series)
        hist_dd = self._calc_historical_drawdown(features, close_series)

        # ── 5. Format result ──────────────────────────────────────────────
        last_date = str(features.index[-1])[:10]
        trend_state = self._classify_trend(features, latest_prob)
        recommendation = self._get_recommendation(latest_prob)

        return {
            'symbol': symbol,
            'name': self.cfg.symbol_names.get(symbol, symbol),
            'date': last_date,
            'bottom_prob': round(latest_prob, 1),     # 中期反弹概率
            'trend_state': trend_state,
            'recommendation': recommendation,
            'stop_loss': stop_loss,
            'hist_max_dd': hist_dd,
        }

    def _auto_train(self, symbol: str, features: pd.DataFrame,
                    start_date: str, end_date: str) -> None:
        """Train a model on the fly when no saved model exists.

        Loads raw daily data for label generation, aligns labels with
        the feature matrix, and trains then saves the model.
        """
        # Load raw OHLCV data for label generation
        loader = DataLoader()
        daily = loader.fetch_daily(symbol, start_date, end_date)
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.sort_values("date").reset_index(drop=True)
        daily = daily.set_index("date")

        # Generate labels from daily close
        label_gen = LabelGenerator(self.cfg)
        labels = label_gen.generate(daily)

        # Align labels with features by date index
        common_idx = features.index.intersection(labels.index)
        if len(common_idx) == 0:
            raise RuntimeError(
                f"Date alignment failed for {symbol}: "
                f"no overlap between features "
                f"({features.index[0]}..{features.index[-1]}) "
                f"and labels ({labels.index[0]}..{labels.index[-1]})"
            )

        X_train = features.loc[common_idx]
        y_train = labels.loc[common_idx].dropna()
        X_train = X_train.loc[y_train.index]

        # Drop close to prevent data leakage
        if "close" in X_train.columns:
            X_train = X_train.drop(columns=["close"])

        if len(X_train) < 10:
            raise RuntimeError(
                f"Too few training samples for {symbol}: {len(X_train)} "
                f"(need >= 10)"
            )

        logger.info(
            "Auto-training %s: %d samples with %.1f%% positive rate",
            symbol, len(y_train), y_train.mean() * 100,
        )

        self.trainer.train(X_train, y_train, symbol)
        self.trainer.save(symbol)

    # ── classification helpers ─────────────────────────────────────────────

    @staticmethod
    def _classify_trend(features: pd.DataFrame, bottom_prob: float) -> str:
        """Classify market trend state from bottom probability and features."""
        if bottom_prob >= 70:
            return '底部构建'
        elif bottom_prob >= 50:
            return '下跌趋势(接近底部)'

        last = features.iloc[-1]

        # Feature-based trend classification
        below_ma20 = last.get('ma20_dev', 0) < 0
        below_ma60 = last.get('ma60_dev', 0) < 0
        negative_momentum = last.get('ret_10d', 0) < 0

        if below_ma20 and below_ma60 and negative_momentum:
            rsi_low = last.get('rsi_14', 50) < 25
            return '加速下跌' if rsi_low else '下跌趋势'
        elif last.get('ret_10d', 0) > 0 and last.get('ma20_dev', 0) > 0:
            return '上升趋势'
        elif abs(last.get('ma20_dev', 0)) < 2:
            return '横盘震荡'
        else:
            return '下跌趋势'

    @staticmethod
    def _get_recommendation(bottom_prob: float) -> str:
        """Trading recommendation based on bottom probability."""
        if bottom_prob >= 80:
            return '试探仓 20%'
        elif bottom_prob >= 70:
            return '试探仓 15%'
        elif bottom_prob >= 60:
            return '观察(暂不建仓)'
        else:
            return '继续等待'

    @staticmethod
    def _calc_stop_loss(features: pd.DataFrame, close: pd.Series = None) -> str:
        """
        Calculate recommended stop-loss level.
        Uses: recent 20-day low, MA60, or 2x ATR below current price.
        """
        if close is None:
            return "N/A"
        try:
            current_price = float(close.iloc[-1])
            recent_low = float(close.iloc[-20:].min()) if len(close) >= 20 else current_price * 0.95
            # Use MA60 if available
            if 'ma60_dev' in features.columns:
                ma60_dev = float(features['ma60_dev'].iloc[-1])
                ma60 = current_price / (1 + ma60_dev / 100) if ma60_dev != -100 else recent_low
            else:
                ma60 = recent_low
            # Stop loss: 2% below the stronger support level
            stop_price = min(recent_low, ma60) * 0.98
            return f"{stop_price:.1f} (MA60:{ma60:.1f}, 前低:{recent_low:.1f})"
        except Exception:
            return "N/A"

    @staticmethod
    def _calc_historical_drawdown(features: pd.DataFrame, close: pd.Series = None) -> str:
        """
        Historical max further drawdown for similar MA60 deviation patterns.
        Searches history for days with similar MA60偏离度, computes max further drop.
        """
        if close is None or 'ma60_dev' not in features.columns:
            return "N/A"
        try:
            current_dev = float(features['ma60_dev'].iloc[-1])
            # Find similar patterns (MA60 dev within 5% of current), exclude last 20 days
            similar_mask = (abs(features['ma60_dev'] - current_dev) < 5)
            # Only consider days with 20+ future bars available
            max_dds = []
            n = len(close)
            for i in range(n - 20):
                if similar_mask.iloc[i]:
                    entry_price = float(close.iloc[i])
                    future_low = float(close.iloc[i+1:i+21].min())
                    dd = (future_low - entry_price) / entry_price * 100
                    max_dds.append(dd)
            if len(max_dds) < 3:
                # Try wider tolerance
                similar_mask2 = (abs(features['ma60_dev'] - current_dev) < 10)
                for i in range(n - 20):
                    if similar_mask2.iloc[i]:
                        entry_price = float(close.iloc[i])
                        future_low = float(close.iloc[i+1:i+21].min())
                        dd = (future_low - entry_price) / entry_price * 100
                        max_dds.append(dd)
            if len(max_dds) < 3:
                return f"样本不足(仅{len(max_dds)}个)"
            avg_dd = sum(max_dds) / len(max_dds)
            worst_dd = min(max_dds)
            return f"均值{avg_dd:+.1f}% 最差{worst_dd:+.1f}%"
        except Exception as e:
            return f"计算失败:{e}"
