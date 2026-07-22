"""
XGBoost model trainer with multi-label support.
Trains separate models for: rebound_3pct, tp_win (hit +5% before -3%), sl_loss.
"""
import os, pickle, logging
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Optional
from ma_config import config as default_config, MarketAnalyzerConfig

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train XGBoost classifiers for multi-outcome bottom detection."""

    def __init__(self, cfg: Optional[MarketAnalyzerConfig] = None):
        self.cfg = cfg or default_config
        self.models: dict[str, xgb.XGBClassifier] = {}  # key: "{symbol}_{label_type}"

    def train(self, X: pd.DataFrame, y: pd.Series, symbol: str = None,
              label_type: str = "rebound_3pct") -> xgb.XGBClassifier:
        """Train one model for one label type."""
        valid = y.notna()
        X_train, y_train = X[valid], y[valid]

        pos_rate = y_train.mean()
        logger.info(f"Training {symbol}/{label_type}: {len(X_train)} samples, {pos_rate:.1%} positive")

        model = xgb.XGBClassifier(
            max_depth=self.cfg.xgb_max_depth,
            learning_rate=self.cfg.xgb_learning_rate,
            n_estimators=self.cfg.xgb_n_estimators,
            early_stopping_rounds=self.cfg.xgb_early_stopping,
            subsample=self.cfg.xgb_subsample,
            reg_lambda=self.cfg.xgb_reg_lambda,
            eval_metric='logloss', random_state=42,
        )

        # Hold out last 20% as test set for honest evaluation
        tss = TimeSeriesSplit(n_splits=2)
        splits = list(tss.split(X_train))
        train_idx, test_idx = splits[0]
        # Use second-to-last fold for early stopping during training
        eval_idx = train_idx[-len(test_idx):]

        model.fit(
            X_train.iloc[train_idx], y_train.iloc[train_idx],
            eval_set=[(X_train.iloc[eval_idx], y_train.iloc[eval_idx])],
            verbose=False,
        )

        key = f"{symbol}_{label_type}" if symbol else label_type
        self.models[key] = model

        importance = model.feature_importances_
        for idx in np.argsort(importance)[-3:][::-1]:
            logger.info(f"  {X.columns[idx]}: {importance[idx]:.4f}")

        return model

    def train_all(self, X: pd.DataFrame, labels: pd.DataFrame, symbol: str) -> dict:
        """Train models for all label types. Returns dict of models."""
        models = {}
        for col in labels.columns:
            if col in ['rebound_3pct', 'tp_win', 'sl_loss', 'final_profit']:
                m = self.train(X, labels[col], symbol, col)
                models[col] = m
        return models

    def evaluate(self, model, X, y):
        valid = y.notna()
        X_eval, y_eval = X[valid], y[valid]
        proba = model.predict_proba(X_eval)[:, 1]
        results = {'num_samples': len(y_eval), 'positive_rate': float(y_eval.mean())}
        for threshold in self.cfg.score_thresholds:
            t = threshold / 100.0
            preds = (proba >= t).astype(int)
            if preds.sum() == 0:
                results[f'thresh_{threshold}'] = {'signals': 0, 'win_rate': 0}
                continue
            win_rate = y_eval[preds == 1].mean()
            results[f'thresh_{threshold}'] = {
                'signals': int(preds.sum()),
                'win_rate': round(float(win_rate), 4),
                'precision': round(float(precision_score(y_eval, preds, zero_division=0)), 4),
            }
        preds_default = (proba >= 0.5).astype(int)
        results['overall'] = {
            'accuracy': round(float(accuracy_score(y_eval, preds_default)), 3),
            'f1': round(float(f1_score(y_eval, preds_default, zero_division=0)), 3),
        }
        return results

    def predict_proba(self, X: pd.DataFrame, symbol: str, label_type: str = "rebound_3pct") -> np.ndarray:
        key = f"{symbol}_{label_type}"
        if key not in self.models:
            raise KeyError(f"Model '{key}' not trained")
        return self.models[key].predict_proba(X)[:, 1]

    def save(self, symbol: str, label_type: str = "rebound_3pct") -> str:
        os.makedirs(self.cfg.model_dir, exist_ok=True)
        key = f"{symbol}_{label_type}"
        path = os.path.join(self.cfg.model_dir, f'{key}_xgb.pkl')
        with open(path, 'wb') as f:
            pickle.dump(self.models[key], f)
        return path

    def load(self, symbol: str, label_type: str = "rebound_3pct") -> xgb.XGBClassifier:
        key = f"{symbol}_{label_type}"
        path = os.path.join(self.cfg.model_dir, f'{key}_xgb.pkl')
        with open(path, 'rb') as f:
            model = pickle.load(f)
        self.models[key] = model
        return model

    def save_all(self, symbol: str, label_types: list[str] = None):
        for lt in (label_types or ['rebound_3pct', 'tp_win', 'sl_loss', 'final_profit']):
            try:
                self.save(symbol, lt)
            except KeyError:
                pass
