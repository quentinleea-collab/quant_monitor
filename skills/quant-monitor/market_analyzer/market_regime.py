"""
Market Regime Detector — trend + bottom + top probability.

Three-in-one model:
  1. Trend classifier: down / sideways / up (multi-class XGBoost)
  2. Bottom detector: P(future 10d max return >= 3%)  [existing, reused]
  3. Top detector:    P(future 10d max drop >= 3%)     [new, mirrored]

All share the same feature matrix, trained with TimeSeriesSplit.
"""
import sys, os, pickle, logging
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'support_backtest'))
from data_loader import DataLoader
from indicator import compute_indicators

logger = logging.getLogger(__name__)

TREND_LABELS = {0: '下跌', 1: '横盘', 2: '上涨'}


class MarketRegime:
    """Unified trend + bottom + top prediction."""

    def __init__(self, model_dir: str = "market_analyzer_models"):
        self.model_dir = model_dir
        self.loader = DataLoader()
        self.models: dict[str, xgb.XGBClassifier] = {}

    # ═══ Training ══════════════════════════════════════════════════

    def train_all(self, symbol: str, start: str = '20240101',
                  end: str = None) -> dict:
        """Train trend + top models. Bottom model is trained separately."""
        end = end or __import__('datetime').datetime.now().strftime('%Y%m%d')

        # Build features once
        X, close = self._build_features(symbol, start, end)
        if X is None or len(X) < 100:
            return {'error': 'Not enough data'}

        results = {}

        # 1. Trend model (multi-class)
        y_trend = self._make_trend_labels(close)
        results['trend'] = self._train_one(X, y_trend, symbol, 'trend',
                                           multi_class=3)

        # 2. Top model (binary: future max drop >= 3%)
        y_top = self._make_top_labels(close)
        results['top'] = self._train_one(X, y_top, symbol, 'top',
                                         multi_class=None)

        logger.info(f"Trained trend+top models for {symbol}")
        return results

    def _train_one(self, X, y, symbol, label_type, multi_class=None):
        """Train a single XGBoost model."""
        valid = y.notna()
        Xt, yt = X[valid], y[valid]
        pos_rate = yt.mean() if multi_class is None else None
        logger.info(f"Training {symbol}/{label_type}: {len(Xt)} samples"
                    + (f", {pos_rate:.1%} positive" if pos_rate is not None else ""))

        params = dict(max_depth=2, learning_rate=0.05, n_estimators=100,
                      subsample=0.8, reg_lambda=1.0,
                      eval_metric='mlogloss' if multi_class else 'logloss',
                      random_state=42)
        if multi_class:
            params['num_class'] = multi_class

        model = xgb.XGBClassifier(**params)
        tss = TimeSeriesSplit(n_splits=2)
        splits = list(tss.split(Xt))
        train_idx, eval_idx = splits[0]
        model.fit(Xt.iloc[train_idx], yt.iloc[train_idx],
                  eval_set=[(Xt.iloc[eval_idx], yt.iloc[eval_idx])],
                  verbose=False)

        # Save
        path = os.path.join(self.model_dir, f'{symbol}_{label_type}.pkl')
        os.makedirs(self.model_dir, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(model, f)

        self.models[f'{symbol}_{label_type}'] = model
        return {'samples': len(Xt), 'accuracy': self._eval_accuracy(model, Xt, yt, multi_class)}

    # ═══ Prediction ════════════════════════════════════════════════

    def predict_all(self, symbol: str, start: str = '20240101',
                    end: str = None) -> dict:
        """Predict trend, bottom, and top probabilities for the latest bar."""
        end = end or __import__('datetime').datetime.now().strftime('%Y%m%d')
        X, close = self._build_features(symbol, start, end)
        if X is None or len(X) < 5:
            return {}

        latest = X.iloc[-1:]

        # Trend
        try:
            model_t = self._load(symbol, 'trend')
            trend_proba = model_t.predict_proba(latest)[0]  # [P(down), P(sideways), P(up)]
            trend_idx = int(np.argmax(trend_proba))
            trend_label = TREND_LABELS.get(trend_idx, '未知')
            trend_conf = float(trend_proba[trend_idx]) * 100
        except FileNotFoundError:
            trend_label = '未训练'; trend_conf = 0; trend_proba = [0,0,0]

        # Top
        try:
            model_top = self._load(symbol, 'top')
            top_prob = float(model_top.predict_proba(latest)[0][1]) * 100
        except FileNotFoundError:
            top_prob = None

        # Bottom model uses separate feature set — run 'python main.py scan' separately

        # Current trend (descriptive, rule-based — what IS happening now)
        current_trend = self._current_trend(close)

        result = {
            'symbol': symbol,
            'current_trend': current_trend,
            'forward_trend': trend_label,
            'trend_confidence': round(trend_conf, 1),
            'trend_detail': {
                '下跌': round(float(trend_proba[0]) * 100, 1),
                '横盘': round(float(trend_proba[1]) * 100, 1),
                '上涨': round(float(trend_proba[2]) * 100, 1),
            },
            'top_prob': round(top_prob, 1) if top_prob is not None else None,
        }

        # Action based on CURRENT trend + top/bottom signals
        if current_trend == '下跌':
            result['action'] = '当前下跌 → 关注底部信号(python main.py scan)'
        elif current_trend == '上涨':
            if top_prob and top_prob >= 60:
                result['action'] = f'当前上涨, 顶部风险{top_prob:.0f}% → 注意减仓'
            else:
                result['action'] = '当前上涨, 未见顶 → 持有'
        else:
            result['action'] = '当前横盘 → 波段操作或等待方向'

        return result

    # ═══ Label Generation ══════════════════════════════════════════

    @staticmethod
    def _make_trend_labels(close: pd.Series) -> pd.Series:
        """Multi-class trend labels from forward 20-day return."""
        c = close.values
        n = len(c)
        labels = pd.Series(np.full(n, np.nan), index=close.index)
        for i in range(n - 20):
            fwd_ret = (c[i + 20] / c[i] - 1) * 100
            if fwd_ret > 5:
                labels.iloc[i] = 2   # up
            elif fwd_ret < -5:
                labels.iloc[i] = 0   # down
            else:
                labels.iloc[i] = 1   # sideways
        return labels

    @staticmethod
    def _make_top_labels(close: pd.Series, horizon: int = 10,
                         threshold: float = 0.03) -> pd.Series:
        """Top label: 1 if future 10d max DROP >= 3%."""
        c = close.values
        n = len(c)
        labels = pd.Series(np.full(n, np.nan), index=close.index)
        for i in range(n - horizon):
            future_min = np.min(c[i+1 : i+horizon+1])
            drop = (c[i] - future_min) / c[i]
            labels.iloc[i] = 1 if drop >= threshold else 0
        return labels

    # ═══ Feature Engineering ═══════════════════════════════════════

    def _build_features(self, symbol, start, end):
        """Build feature matrix (same as bottom detector)."""
        daily = self.loader.fetch_daily(symbol, start, end)
        if daily.empty or len(daily) < 100:
            return None, None
        daily = daily.sort_values('date').reset_index(drop=True)
        daily = compute_indicators(daily)

        df = daily.copy()
        c, v = df['close'], df['volume']

        # MA deviations
        for p in [5, 10, 20, 30, 60]:
            ma = c.rolling(p).mean()
            df[f'ma{p}_dev'] = (c - ma) / ma * 100

        # RSI
        df['rsi_14'] = self._rsi(c, 14)

        # MACD
        df['macd_hist'] = self._macd_hist(c)
        df['macd_hist_chg'] = df['macd_hist'].diff(2)

        # Volume
        df['vol_ratio'] = v / v.rolling(5).mean()
        df['vol_trend'] = v.rolling(3).mean() / v.rolling(10).mean()

        # Returns
        for p in [5, 10, 20]:
            df[f'ret_{p}d'] = c.pct_change(p) * 100

        # K-line
        body_low = np.minimum(df.get('open', c), c)
        total_range = df['high'] - df['low']
        df['lower_shadow'] = np.where(total_range > 0,
                                       (body_low - df['low']) / total_range, 0)

        feature_cols = [c for c in df.columns if c.startswith('ma') or
                        c.startswith('ret_') or c.startswith('vol_') or
                        c.startswith('rsi') or c.startswith('macd') or
                        c == 'lower_shadow']
        df = df.dropna(subset=feature_cols)
        return df[feature_cols], df['close']

    # ═══ Current Trend (descriptive — what IS happening) ═══════════

    @staticmethod
    def _current_trend(close: pd.Series) -> str:
        """
        Classify current market trend based on actual price action.
        Uses MA alignment + recent price position (descriptive, not predictive).
        """
        c = close.values
        if len(c) < 60:
            return '数据不足'

        ma5 = pd.Series(c).rolling(5).mean().iloc[-1]
        ma10 = pd.Series(c).rolling(10).mean().iloc[-1]
        ma20 = pd.Series(c).rolling(20).mean().iloc[-1]
        ma60 = pd.Series(c).rolling(60).mean().iloc[-1]
        current = c[-1]

        # Days below MA5
        below_ma5 = sum(1 for i in range(len(c)-15, len(c)) if c[i] < pd.Series(c).rolling(5).mean().iloc[i])

        # MA alignment
        mas_above = sum(1 for m in [ma5, ma10, ma20, ma60] if not np.isnan(m) and current < m)

        # 20-day return
        ret20 = (c[-1] / c[-20] - 1) * 100 if len(c) >= 20 else 0

        # Classification
        if mas_above >= 3 and below_ma5 >= 8 and ret20 < -5:
            return '下跌'  # All MAs overhead, price persistently weak
        elif mas_above >= 2 and ret20 < -3:
            return '下跌'
        elif mas_above == 0 and ret20 > 3:
            return '上涨'  # Price above all MAs, strong
        elif mas_above <= 1 and ret20 > 0:
            return '上涨'
        elif abs(ret20) < 3 and mas_above <= 1:
            return '横盘'
        elif mas_above >= 2:
            return '下跌'  # Default: most MAs overhead = weakness
        else:
            return '横盘'

    # ═══ Helpers ═══════════════════════════════════════════════════

    def _load(self, symbol, label_type):
        # Try both naming conventions (with and without _xgb suffix)
        for name in [f'{symbol}_{label_type}', f'{symbol}_{label_type}_xgb']:
            path = os.path.join(self.model_dir, f'{name}.pkl')
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    model = pickle.load(f)
                self.models[f'{symbol}_{label_type}'] = model
                return model
        raise FileNotFoundError(f"No model for {symbol}/{label_type}")

    @staticmethod
    def _eval_accuracy(model, X, y, multi_class=None):
        preds = model.predict(X)
        if multi_class:
            return round(float((preds == y).mean()), 3)
        return round(float((preds == y).mean()), 3)

    @staticmethod
    def _rsi(close, period=14):
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta).clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _macd_hist(close, fast=12, slow=26, signal=9):
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=signal, adjust=False).mean()
        return 2 * (dif - dea)
