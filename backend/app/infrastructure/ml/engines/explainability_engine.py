"""
ExplainabilityEngine — feature importance and SHAP explanations.
Gracefully degrades: TreeExplainer → LinearExplainer → permutation importance.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.core.constants import FEATURE_IMPORTANCE_TOP_N, SHAP_MAX_BACKGROUND_SAMPLES
from app.core.logging import get_logger
from app.domain.entities.model_result import ModelResult

logger = get_logger(__name__)


@dataclass
class FeatureImportanceEntry:
    feature: str
    importance: float
    rank: int


@dataclass
class ExplainabilityResult:
    """All explainability outputs for a single model."""

    model_result_id: str
    feature_importances: list[FeatureImportanceEntry] = field(default_factory=list)
    shap_values: list[list[float]] | None = None  # shape [n_samples, n_features]
    shap_base_value: float | None = None
    top_features: list[str] = field(default_factory=list)
    method_used: str = "unknown"
    error: str | None = None


class ExplainabilityEngine:
    """
    Produces feature importance and SHAP explanations.
    Falls back gracefully when methods are not supported.
    """

    def explain(
        self,
        estimator: Any,
        x_train: np.ndarray | None = None,
        x_test: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        model_result: ModelResult | None = None,
        max_shap_samples: int = SHAP_MAX_BACKGROUND_SAMPLES,
        **kwargs: Any,
    ) -> ExplainabilityResult:
        """
        Compute feature importances and optionally SHAP values.
        Never raises — errors are captured in the result.
        """
        if x_train is None and "x_train" in kwargs:
            x_train = kwargs.pop("x_train")
        if x_train is None and "X_train" in kwargs:
            x_train = kwargs.pop("X_train")
        if x_test is None and "x_test" in kwargs:
            x_test = kwargs.pop("x_test")
        if x_test is None and "X_test" in kwargs:
            x_test = kwargs.pop("X_test")
        if feature_names is None and "feature_names" in kwargs:
            feature_names = kwargs.pop("feature_names")
        if feature_names is None and "featureNames" in kwargs:
            feature_names = kwargs.pop("featureNames")
        if model_result is None and "model_result" in kwargs:
            model_result = kwargs.pop("model_result")

        if model_result is None:
            raise TypeError("model_result is required")
        if x_train is None or x_test is None or feature_names is None:
            raise TypeError("X_train, X_test, and feature_names are required")

        result = ExplainabilityResult(model_result_id=model_result.id)

        # 1. Feature importance
        importances = self._get_feature_importance(estimator, x_train, x_test, feature_names)
        result.feature_importances = importances
        result.top_features = [fi.feature for fi in importances[:FEATURE_IMPORTANCE_TOP_N]]
        result.method_used = self._detect_method(estimator)

        # 2. SHAP (only if supported and enabled on config)
        if model_result.supports_shap:
            try:
                shap_vals, base_val = self._compute_shap(
                    estimator, x_train, x_test, max_shap_samples
                )
                result.shap_values = shap_vals
                result.shap_base_value = base_val
            except Exception as exc:
                logger.warning("shap_failed", model=model_result.config_name, error=str(exc))
                result.error = f"SHAP computation failed: {exc}"

        logger.info(
            "explanation_complete",
            model=model_result.config_name,
            method=result.method_used,
            top_features=result.top_features[:3],
        )
        return result

    def _get_feature_importance(
        self,
        estimator: Any,
        x_train: np.ndarray,
        x_test: np.ndarray,
        feature_names: list[str],
    ) -> list[FeatureImportanceEntry]:
        """
        Try three methods in priority order:
        1. Native feature_importances_ (tree/ensemble models)
        2. coef_ magnitude (linear models)
        3. Permutation importance (universal fallback)
        """
        raw_importances: np.ndarray | None = None

        # Method 1: tree-based feature importances
        if hasattr(estimator, "feature_importances_"):
            raw_importances = np.array(estimator.feature_importances_)

        # Method 2: linear model coefficients
        elif hasattr(estimator, "coef_"):
            coef = np.array(estimator.coef_)
            if coef.ndim > 1:
                raw_importances = np.abs(coef).mean(axis=0)
            else:
                raw_importances = np.abs(coef)

        # Method 3: permutation importance (slow but universal)
        if raw_importances is None:
            raw_importances = self._permutation_importance(
                estimator, x_train, x_test, feature_names
            )

        # Normalize to [0, 1]
        if raw_importances is not None and raw_importances.sum() > 0:
            raw_importances = raw_importances / raw_importances.sum()

        n = min(len(feature_names), len(raw_importances) if raw_importances is not None else 0)
        entries = [
            FeatureImportanceEntry(
                feature=feature_names[i],
                importance=round(float(raw_importances[i]), 6),
                rank=0,
            )
            for i in range(n)
        ]

        # Sort by importance descending and assign ranks
        entries.sort(key=lambda e: e.importance, reverse=True)
        for rank, entry in enumerate(entries, 1):
            entry.rank = rank

        return entries

    def _permutation_importance(
        self,
        estimator: Any,
        x_train: np.ndarray,
        x_test: np.ndarray,
        feature_names: list[str],
    ) -> np.ndarray:
        """Compute permutation importance as fallback."""
        try:
            from sklearn.inspection import permutation_importance as pi

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = pi(
                    estimator,
                    x_test,
                    estimator.predict(x_test),
                    n_repeats=5,
                    random_state=42,
                )
            return np.abs(result.importances_mean)
        except Exception as exc:
            logger.warning("permutation_importance_failed", error=str(exc))
            return np.zeros(len(feature_names))

    def _compute_shap(
        self,
        estimator: Any,
        x_train: np.ndarray,
        x_test: np.ndarray,
        max_samples: int,
    ) -> tuple[list[list[float]], float]:
        """
        Compute SHAP values using the most appropriate explainer.
        Uses a sample of X_test to keep computation tractable.
        """
        import shap

        sample_size = min(max_samples, len(x_test))
        x_sample = x_test[:sample_size]

        # Choose explainer based on model type
        try:
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(x_sample)
            base_value = float(np.mean(explainer.expected_value))
        except Exception:
            try:
                background = x_train[: min(50, len(x_train))]
                explainer = shap.LinearExplainer(estimator, background)
                shap_values = explainer.shap_values(x_sample)
                base_value = float(np.mean(explainer.expected_value))
            except Exception:
                # Universal fallback: KernelExplainer (slow)
                background = shap.sample(x_train, min(25, len(x_train)))
                explainer = shap.KernelExplainer(estimator.predict, background)
                shap_values = explainer.shap_values(x_sample, nsamples=50)
                base_value = float(np.mean(explainer.expected_value))

        # Normalize multi-class SHAP to per-class mean absolute values
        if isinstance(shap_values, list):
            shap_array = np.abs(np.array(shap_values)).mean(axis=0)
        else:
            shap_array = np.array(shap_values)

        return shap_array.tolist(), base_value

    @staticmethod
    def _detect_method(estimator: Any) -> str:
        if hasattr(estimator, "feature_importances_"):
            return "feature_importances"
        if hasattr(estimator, "coef_"):
            return "coefficients"
        return "permutation_importance"
