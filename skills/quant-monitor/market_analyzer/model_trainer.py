"""
XGBoost model trainer with TimeSeriesSplit cross-validation.
Trains one model per symbol, evaluates with forward-looking metrics.
"""
import os, pickle, logging
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Optional, Tuple
from config import config as default_config, MarketAnalyzerConfig

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train XGBoost classifier for bottom detection with temporal CV."""

    def __init__(self, cfg: Optional[MarketAnalyzerConfig] = None):
        self.cfg = cfg or default_config
        self.models: dict[str, xgb.XGBClassifier] = {}

    def train(self, X: pd.DataFrame, y: pd.Series, symbol: str = None) -> xgb.XGBClassifier:
        """
        Train XGBoost on full dataset (model saved for later prediction).

        Args:
            X: feature matrix (pd.DataFrame, date index)
            y: labels (pd.Series, aligned index, 0/1 with NaN at end)

        Returns: trained XGBClassifier
        """
        # Drop rows where y is NaN
        valid = y.notna()
        X_train = X[valid]
        y_train = y[valid]

        logger.info(f"Training {symbol or 'model'}: {len(X_train)} samples, {X_train.shape[1]} features")
        logger.info(f"Class balance: {y_train.mean():.1%} positive")

        model = xgb.XGBClassifier(
            max_depth=self.cfg.xgb_max_depth,
            learning_rate=self.cfg.xgb_learning_rate,
            n_estimators=self.cfg.xgb_n_estimators,
            early_stopping_rounds=self.cfg.xgb_early_stopping,
            eval_metric='logloss',
            random_state=42,
            enable_categorical=False,
        )

        # Use TimeSeriesSplit for eval during training
        tss = TimeSeriesSplit(n_splits=self.cfg.tss_n_splits)
        splits = list(tss.split(X_train))
        # Use last split as eval set
        _, eval_idx = splits[-1]

        model.fit(
            X_train, y_train,
            eval_set=[(X_train.iloc[eval_idx], y_train.iloc[eval_idx])],
            verbose=False,
        )

        key = symbol or 'default'
        self.models[key] = model

        # Log feature importance
        importance = model.feature_importances_
        top_idx = np.argsort(importance)[-5:][::-1]
        for idx in top_idx:
            logger.info(f"  Feature: {X.columns[idx]}: {importance[idx]:.4f}")

        return model

    def evaluate(self, model: xgb.XGBClassifier, X: pd.DataFrame, y: pd.Series) -> dict:
        """Evaluate model at multiple probability thresholds."""
        valid = y.notna()
        X_eval, y_eval = X[valid], y[valid]
        proba = model.predict_proba(X_eval)[:, 1]

        results = {'num_samples': len(y_eval), 'positive_rate': float(y_eval.mean())}

        for threshold in self.cfg.score_thresholds:
            t = threshold / 100.0
            preds = (proba >= t).astype(int)
            if preds.sum() == 0:
                results[f'thresh_{threshold}'] = {'signals': 0, 'win_rate': 0, 'avg_return': 0}
                continue

            # For bottom detection: "win" = actual label was 1
            win_rate = y_eval[preds == 1].mean() if preds.sum() > 0 else 0
            results[f'thresh_{threshold}'] = {
                'signals': int(preds.sum()),
                'win_rate': round(float(win_rate), 4),
                'precision': round(float(precision_score(y_eval, preds, zero_division=0)), 4),
                'recall': round(float(recall_score(y_eval, preds, zero_division=0)), 4),
            }

        # Overall metrics at default 0.5 threshold
        preds_default = (proba >= 0.5).astype(int)
        results['overall'] = {
            'accuracy': round(float(accuracy_score(y_eval, preds_default)), 4),
            'precision': round(float(precision_score(y_eval, preds_default, zero_division=0)), 4),
            'recall': round(float(recall_score(y_eval, preds_default, zero_division=0)), 4),
            'f1': round(float(f1_score(y_eval, preds_default, zero_division=0)), 4),
        }

        return results

    def predict_proba(self, X: pd.DataFrame, symbol: str = None) -> np.ndarray:
        """Return bottom probability [0, 1] for each row in X."""
        key = symbol or 'default'
        if key not in self.models:
            raise KeyError(f"Model for '{key}' not trained. Call train() first.")
        return self.models[key].predict_proba(X)[:, 1]

    def save(self, symbol: str) -> str:
        """Save model to disk. Returns path."""
        os.makedirs(self.cfg.model_dir, exist_ok=True)
        path = os.path.join(self.cfg.model_dir, f'{symbol}_xgb.pkl')
        with open(path, 'wb') as f:
            pickle.dump(self.models[symbol], f)
        logger.info(f"Model saved: {path}")
        return path

    def load(self, symbol: str) -> xgb.XGBClassifier:
        """Load model from disk."""
        path = os.path.join(self.cfg.model_dir, f'{symbol}_xgb.pkl')
        with open(path, 'rb') as f:
            model = pickle.load(f)
        self.models[symbol] = model
        return model
