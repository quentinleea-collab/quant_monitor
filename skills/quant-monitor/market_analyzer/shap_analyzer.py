"""SHAP-based feature importance and interaction analysis."""
import logging
import numpy as np
import pandas as pd
import shap
from typing import Optional

logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    """Analyze feature contributions using SHAP values."""

    def analyze(self, model, X: pd.DataFrame, top_k: int = 20) -> dict:
        """
        Compute SHAP values and feature importance.

        Args:
            model: trained xgboost.XGBClassifier
            X: feature matrix
            top_k: number of top features to return

        Returns dict:
            - feature_importance: list of {feature, importance, rank}
            - shap_values: ndarray (n_samples, n_features)
            - top_features: list of feature names in importance order
        """
        logger.info(f"Computing SHAP for {X.shape[0]} samples, {X.shape[1]} features")

        # Use TreeExplainer (fast for XGBoost)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Mean absolute SHAP = feature importance
        importance = np.abs(shap_values).mean(axis=0)
        sorted_idx = np.argsort(importance)[::-1]

        feature_importance = []
        for rank, idx in enumerate(sorted_idx[:top_k]):
            feature_importance.append({
                'rank': rank + 1,
                'feature': X.columns[idx],
                'importance': round(float(importance[idx]), 6),
                'importance_pct': 0.0,  # filled below
            })

        # Normalize to percentages
        total = sum(f['importance'] for f in feature_importance)
        for f in feature_importance:
            f['importance_pct'] = round(f['importance'] / total * 100, 1) if total > 0 else 0

        top_features = [f['feature'] for f in feature_importance]

        logger.info(f"Top 5 features: {[f['feature'] for f in feature_importance[:5]]}")

        return {
            'feature_importance': feature_importance,
            'shap_values': shap_values,
            'top_features': top_features,
        }

    def get_interactions(self, model, X: pd.DataFrame, top_k: int = 5) -> list[dict]:
        """
        Get top SHAP interaction pairs (feature combinations that jointly affect predictions).
        """
        explainer = shap.TreeExplainer(model)
        shap_interaction = explainer.shap_interaction_values(X)

        # shap_interaction shape: (n_samples, n_features, n_features)
        # Diagonal = main effects, off-diagonal = interactions
        n_features = X.shape[1]
        interactions = []

        for i in range(n_features):
            for j in range(i + 1, n_features):
                interaction_strength = np.abs(shap_interaction[:, i, j]).mean()
                interactions.append({
                    'feature_1': X.columns[i],
                    'feature_2': X.columns[j],
                    'strength': float(interaction_strength),
                })

        interactions.sort(key=lambda x: x['strength'], reverse=True)
        return interactions[:top_k]
