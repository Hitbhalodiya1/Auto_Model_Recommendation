"""
EvaluationEngine — computes task-appropriate metrics and detects overfitting.
Uses a Strategy pattern: one evaluator class per task type.
"""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.model_selection import cross_val_score

from app.core.config import get_settings
from app.core.logging import get_logger
from app.domain.entities.model_result import ModelResult
from app.domain.value_objects.task_type import TaskType
from app.infrastructure.ml.engines.training_engine import TrainingResult

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Full evaluation outcome for a single trained model."""

    training_result: TrainingResult
    metrics: dict[str, float] = field(default_factory=dict)
    cv_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    cv_mean: float = 0.0
    cv_std: float = 0.0
    train_score: float = 0.0
    is_overfitting: bool = False
    confusion_matrix: list[list[int]] | None = None
    roc_auc: float | None = None


# ── Per-task Evaluators ───────────────────────────────────────────────────────


class BaseEvaluator(ABC):
    @abstractmethod
    def compute(
        self,
        estimator: Any,
        predictions: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict[str, float]: ...

    @property
    @abstractmethod
    def cv_scoring(self) -> str:
        """Scoring metric name for cross_val_score."""
        ...

    @property
    @abstractmethod
    def primary_metric(self) -> str:
        """The single metric used for ranking and comparison."""
        ...


class BinaryClassificationEvaluator(BaseEvaluator):
    @property
    def cv_scoring(self) -> str:
        return "f1"

    @property
    def primary_metric(self) -> str:
        return "f1_score"

    def compute(self, estimator, predictions, x_test, y_test) -> dict[str, float]:
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        metrics: dict[str, float] = {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        }

        # ROC-AUC requires probability estimates
        try:
            if hasattr(estimator, "predict_proba"):
                proba = estimator.predict_proba(x_test)[:, 1]
                metrics["roc_auc"] = round(float(roc_auc_score(y_test, proba)), 4)
            elif hasattr(estimator, "decision_function"):
                scores = estimator.decision_function(x_test)
                metrics["roc_auc"] = round(float(roc_auc_score(y_test, scores)), 4)
        except Exception:
            pass

        # Confusion matrix as flat list
        cm = confusion_matrix(y_test, predictions).tolist()
        metrics["cm_tn"] = float(cm[0][0]) if len(cm) >= 2 else 0
        metrics["cm_fp"] = float(cm[0][1]) if len(cm) >= 2 else 0
        metrics["cm_fn"] = float(cm[1][0]) if len(cm) >= 2 else 0
        metrics["cm_tp"] = float(cm[1][1]) if len(cm) >= 2 else 0

        return metrics


class MulticlassEvaluator(BaseEvaluator):
    @property
    def cv_scoring(self) -> str:
        return "f1_weighted"

    @property
    def primary_metric(self) -> str:
        return "f1_weighted"

    def compute(self, estimator, predictions, x_test, y_test) -> dict[str, float]:
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        return {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision_macro": round(
                float(precision_score(y_test, predictions, average="macro", zero_division=0)), 4
            ),
            "recall_macro": round(
                float(recall_score(y_test, predictions, average="macro", zero_division=0)), 4
            ),
            "f1_macro": round(
                float(f1_score(y_test, predictions, average="macro", zero_division=0)), 4
            ),
            "f1_weighted": round(
                float(f1_score(y_test, predictions, average="weighted", zero_division=0)), 4
            ),
        }


class RegressionEvaluator(BaseEvaluator):
    @property
    def cv_scoring(self) -> str:
        return "r2"

    @property
    def primary_metric(self) -> str:
        return "r2_score"

    def compute(self, estimator, predictions, x_test, y_test) -> dict[str, float]:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        mae = float(mean_absolute_error(y_test, predictions))
        mse = float(mean_squared_error(y_test, predictions))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, predictions))

        return {
            "mae": round(mae, 4),
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "r2_score": round(r2, 4),
        }


# ── Evaluation Engine ─────────────────────────────────────────────────────────


class EvaluationEngine:
    """
    Evaluates all trained models and computes task-appropriate metrics.
    """

    _EVALUATORS = {
        TaskType.BINARY_CLASSIFICATION: BinaryClassificationEvaluator(),
        TaskType.MULTICLASS_CLASSIFICATION: MulticlassEvaluator(),
        TaskType.REGRESSION: RegressionEvaluator(),
    }

    def __init__(self) -> None:
        self._settings = get_settings()

    def evaluate_all(
        self,
        training_results: list[TrainingResult],
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        task_type: TaskType,
    ) -> list[EvaluationResult]:
        """
        Evaluate all training results. Skips failed trainings.
        Returns EvaluationResult objects sorted by primary metric descending.
        """
        evaluations = []
        for tr in training_results:
            if not tr.succeeded:
                continue
            try:
                ev = self._evaluate_one(tr, x_train, y_train, x_test, y_test, task_type)
                evaluations.append(ev)
            except Exception as exc:
                logger.warning(
                    "evaluation_failed",
                    config=tr.config.name,
                    error=str(exc),
                )

        evaluator = self._EVALUATORS.get(task_type)
        if evaluator:
            evaluations.sort(
                key=lambda e: e.metrics.get(evaluator.primary_metric, 0),
                reverse=True,
            )

        return evaluations

    def _evaluate_one(
        self,
        tr: TrainingResult,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        task_type: TaskType,
    ) -> EvaluationResult:

        evaluator = self._EVALUATORS.get(task_type)
        if not evaluator:
            raise ValueError(f"No evaluator for task type: {task_type}")

        metrics = evaluator.compute(tr.estimator, tr.predictions, x_test, y_test)

        # Cross-validation
        cv_scores = np.array([0.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                cv_scores = cross_val_score(
                    tr.estimator,
                    x_train,
                    y_train,
                    cv=self._settings.CV_FOLDS,
                    scoring=evaluator.cv_scoring,
                    n_jobs=1,
                )
            except Exception as exc:
                logger.warning("cv_failed", config=tr.config.name, error=str(exc))

        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())
        train_score = tr.train_score or 0.0

        # Overfitting detection: train score significantly better than cv score
        primary = metrics.get(evaluator.primary_metric, 0.0)
        is_overfitting = (train_score - primary) > self._settings.OVERFITTING_THRESHOLD

        # Add timing to metrics
        metrics["training_time_s"] = round(tr.training_time_s, 4)
        metrics["prediction_time_s"] = round(tr.prediction_time_s, 6)
        metrics["cv_score"] = round(cv_mean, 4)
        metrics["cv_std"] = round(cv_std, 4)
        metrics["train_score"] = round(train_score, 4)

        return EvaluationResult(
            training_result=tr,
            metrics=metrics,
            cv_scores=cv_scores,
            cv_mean=cv_mean,
            cv_std=cv_std,
            train_score=train_score,
            is_overfitting=is_overfitting,
        )

    def to_model_result(
        self,
        ev: EvaluationResult,
        experiment_id: str,
    ) -> ModelResult:
        """Convert an EvaluationResult to a persisted ModelResult entity."""
        config = ev.training_result.config
        return ModelResult(
            experiment_id=experiment_id,
            algorithm_name=config.algorithm_family,
            config_name=config.name,
            display_name=config.display_name,
            configuration=config.params,
            metrics=ev.metrics,
            cv_score=ev.cv_mean,
            cv_std=ev.cv_std,
            is_overfitting=ev.is_overfitting,
            training_time_s=ev.training_result.training_time_s,
            prediction_time_s=ev.training_result.prediction_time_s,
            requires_scaling=config.requires_scaling,
            supports_feature_importance=config.supports_feature_importance,
            supports_shap=config.supports_shap,
            interpretability_score=config.interpretability_score,
        )
