"""
RecommendationEngine — ranks models using weighted multi-criteria scoring
and generates human-readable explanations for every recommendation.

Refactored to use modular recommendation components while maintaining
backward compatibility with existing APIs.
"""

from dataclasses import dataclass, field
from typing import Any

from app.core.constants import (
    DEFAULT_WEIGHT_GENERALIZATION,
    DEFAULT_WEIGHT_INTERPRETABILITY,
    DEFAULT_WEIGHT_OVERFITTING,
    DEFAULT_WEIGHT_PERFORMANCE,
    DEFAULT_WEIGHT_PRED_SPEED,
    DEFAULT_WEIGHT_TRAIN_SPEED,
)
from app.core.logging import get_logger
from app.core.recommendation_config import RecommendationConfig
from app.domain.entities.model_result import ModelResult, Recommendation
from app.domain.value_objects.task_type import TaskType
from app.infrastructure.ml.engines.evaluation_engine import EvaluationResult
from app.infrastructure.ml.recommendation.recommendation_result_builder import (
    RecommendationResultBuilder,
)

logger = get_logger(__name__)


@dataclass
class ScoringWeights:
    """Weights for the multi-criteria scoring model. Must sum to 1.0."""

    performance: float = DEFAULT_WEIGHT_PERFORMANCE
    generalization: float = DEFAULT_WEIGHT_GENERALIZATION
    train_speed: float = DEFAULT_WEIGHT_TRAIN_SPEED
    pred_speed: float = DEFAULT_WEIGHT_PRED_SPEED
    interpretability: float = DEFAULT_WEIGHT_INTERPRETABILITY
    overfitting_penalty: float = DEFAULT_WEIGHT_OVERFITTING


@dataclass
class ScoredResult:
    model_result: ModelResult
    evaluation: EvaluationResult
    composite_score: float
    breakdown: dict[str, float] = field(default_factory=dict)


class RecommendationEngine:
    """
    Scores and ranks all evaluated models, selects the best,
    and produces a structured Recommendation with explanation text.
    
    Refactored to use modular recommendation components while maintaining
    backward compatibility with the existing API.
    """

    def __init__(
        self,
        weights: ScoringWeights | None = None,
        config: RecommendationConfig | None = None,
    ) -> None:
        self._weights = weights or ScoringWeights()
        self._config = config or RecommendationConfig()
        self._result_builder = RecommendationResultBuilder(self._config)

    def recommend(
        self,
        evaluations: list[EvaluationResult],
        model_results: list[ModelResult],
        task_type: TaskType,
        experiment_id: str,
        dataset_analysis: dict[str, Any] | None = None,
        model_configs: dict[str, Any] | None = None,
    ) -> Recommendation:
        """
        Score all models, select the best, and return a Recommendation entity.
        
        Refactored to use the new modular recommendation pipeline while
        maintaining backward compatibility with the existing API.
        
        Args:
            evaluations: List of evaluation results (kept for backward compatibility)
            model_results: List of model results from training
            task_type: The ML task type
            experiment_id: Experiment ID
            dataset_analysis: Optional dataset analysis results for compatibility analysis
            model_configs: Optional mapping of model_id to ModelConfig
            
        Returns:
            Recommendation entity with the recommended model
        """
        if not model_results:
            raise ValueError("No model results to recommend from.")

        # Use the new recommendation pipeline
        if model_configs is None:
            # Build model configs from model results for backward compatibility
            model_configs = self._build_model_configs_from_results(model_results)

        pipeline_result = self._result_builder.build_recommendation(
            model_results=model_results,
            model_configs=model_configs,
            dataset_analysis=dataset_analysis,
            task_type=task_type,
            experiment_id=experiment_id,
        )

        # Format for backward compatibility
        formatted = pipeline_result.formatted_recommendations

        # Use best_overall as the primary recommendation for backward compatibility
        primary = formatted.best_overall

        if not primary:
            # Fallback to legacy scoring if pipeline fails
            return self._legacy_recommend(
                evaluations, model_results, task_type, experiment_id
            )

        # Build backward-compatible Recommendation entity
        all_rankings = [
            {
                "rank": r.rank,
                "config_name": r.config_name,
                "display_name": r.display_name,
                "composite_score": r.composite_score,
                "primary_metric": r.primary_metric,
                "cv_score": r.cv_score,
                "is_overfitting": r.is_overfitting,
            }
            for r in formatted.all_rankings
        ]

        logger.info(
            "recommendation_generated",
            experiment_id=experiment_id,
            best_model=primary.config_name,
            score=primary.overall_score,
            total_ranked=len(all_rankings),
        )

        return Recommendation(
            experiment_id=experiment_id,
            model_result_id=primary.model_id,
            composite_score=primary.overall_score,
            score_breakdown=primary.score_breakdown,
            rationale=primary.rationale,
            explanation_text=primary.explanation_text,
            all_rankings=all_rankings,
        )

    # ── Backward Compatibility Helpers ─────────────────────────────────────────

    def _build_model_configs_from_results(
        self, model_results: list[ModelResult]
    ) -> dict[str, Any]:
        """Build model configs from model results for backward compatibility."""
        from app.domain.interfaces.registry.model_registry import ModelConfig

        configs = {}
        for mr in model_results:
            configs[mr.id] = ModelConfig(
                name=mr.config_name,
                display_name=mr.display_name,
                algorithm_family=mr.algorithm_name,
                params=mr.configuration,
                task_types=[],  # Not available from model results
                requires_scaling=mr.requires_scaling,
                supports_feature_importance=mr.supports_feature_importance,
                supports_shap=mr.supports_shap,
                interpretability_score=mr.interpretability_score,
            )
        return configs

    def _legacy_recommend(
        self,
        evaluations: list[EvaluationResult],
        model_results: list[ModelResult],
        task_type: TaskType,
        experiment_id: str,
    ) -> Recommendation:
        """Fallback to legacy scoring method for backward compatibility."""
        # Adapt weights based on dataset/task context
        weights = self._adapt_weights(model_results, task_type)

        # Score each model
        scored = [
            self._score_result(mr, weights, model_results)
            for mr in model_results
            if mr.succeeded
        ]
        scored.sort(key=lambda s: s.composite_score, reverse=True)

        # Assign ranks
        for i, s in enumerate(scored):
            s.model_result.rank = i + 1

        best = scored[0]

        # Build all rankings list for the response
        all_rankings = [
            {
                "rank": i + 1,
                "config_name": s.model_result.config_name,
                "display_name": s.model_result.display_name,
                "composite_score": round(s.composite_score, 4),
                "primary_metric": s.model_result.primary_metric,
                "cv_score": s.model_result.cv_score,
                "is_overfitting": s.model_result.is_overfitting,
            }
            for i, s in enumerate(scored)
        ]

        explanation_text = self._build_explanation(best, scored, task_type)
        rationale = self._build_rationale_bullets(best, scored)

        logger.info(
            "legacy_recommendation_generated",
            experiment_id=experiment_id,
            best_model=best.model_result.config_name,
            score=best.composite_score,
            total_ranked=len(scored),
        )

        return Recommendation(
            experiment_id=experiment_id,
            model_result_id=best.model_result.id,
            composite_score=best.composite_score,
            score_breakdown=best.breakdown,
            rationale=rationale,
            explanation_text=explanation_text,
            all_rankings=all_rankings,
        )

    # ── Legacy Scoring (kept for fallback) ─────────────────────────────────────

    def _score_result(
        self,
        mr: ModelResult,
        weights: ScoringWeights,
        all_results: list[ModelResult],
    ) -> ScoredResult:
        breakdown = {}

        # 1. Performance: normalized primary metric
        all_primary = [r.primary_metric or 0 for r in all_results if r.succeeded]
        max_p = max(all_primary) if all_primary else 1.0
        perf_score = (mr.primary_metric or 0) / max_p if max_p > 0 else 0
        breakdown["performance"] = round(perf_score * weights.performance, 4)

        # 2. Generalization: cv_score normalized
        all_cv = [r.cv_score or 0 for r in all_results if r.succeeded]
        max_cv = max(all_cv) if all_cv else 1.0
        gen_score = (mr.cv_score or 0) / max_cv if max_cv > 0 else 0
        breakdown["generalization"] = round(gen_score * weights.generalization, 4)

        # 3. Training speed: inverse of training time, normalized
        all_train_times = [r.training_time_s for r in all_results if r.succeeded]
        max_t = max(all_train_times) if all_train_times else 1.0
        train_speed = 1.0 - (mr.training_time_s / max_t) if max_t > 0 else 1.0
        breakdown["train_speed"] = round(train_speed * weights.train_speed, 4)

        # 4. Prediction speed: inverse of prediction time, normalized
        all_pred_times = [r.prediction_time_s for r in all_results if r.succeeded]
        max_pt = max(all_pred_times) if all_pred_times else 1.0
        pred_speed = 1.0 - (mr.prediction_time_s / max_pt) if max_pt > 0 else 1.0
        breakdown["pred_speed"] = round(pred_speed * weights.pred_speed, 4)

        # 5. Interpretability: 1–5 scale normalized
        interp_score = (mr.interpretability_score - 1) / 4  # 0 to 1
        breakdown["interpretability"] = round(interp_score * weights.interpretability, 4)

        # 6. Overfitting penalty
        overfitting_penalty = -weights.overfitting_penalty if mr.is_overfitting else 0.0
        breakdown["overfitting_penalty"] = round(overfitting_penalty, 4)

        composite = sum(breakdown.values())
        return ScoredResult(
            model_result=mr,
            evaluation=None,  # type: ignore
            composite_score=round(composite, 4),
            breakdown=breakdown,
        )

    def _adapt_weights(
        self, model_results: list[ModelResult], task_type: TaskType
    ) -> ScoringWeights:
        """
        Slightly adapt weights based on context.
        Example: large datasets → speed matters more; small → interpretability matters more.
        """
        return self._weights  # Phase 1: use defaults

    # ── Explanation Generation ────────────────────────────────────────────────

    def _build_explanation(
        self, best: ScoredResult, all_scored: list[ScoredResult], task_type: TaskType
    ) -> str:
        mr = best.model_result
        primary_metric_name = self._primary_metric_name(task_type)
        primary_val = mr.primary_metric

        parts = [
            f"**{mr.display_name}** was selected as the best model for this experiment."
        ]

        if primary_val is not None:
            parts.append(
                f"It achieved the highest {primary_metric_name} of {primary_val:.4f}"
                + (
                    f" with strong cross-validation consistency (CV score: {mr.cv_score:.4f} ± {mr.cv_std:.4f})"
                    if mr.cv_score is not None
                    else ""
                )
                + "."
            )

        if not mr.is_overfitting:
            parts.append(
                "The model shows no signs of overfitting, indicating good generalization to unseen data."
            )
        else:
            parts.append(
                "Note: This model shows signs of overfitting. Consider adding regularization or "
                "reducing model complexity."
            )

        interp_labels = {1: "black box", 2: "low", 3: "moderate", 4: "high", 5: "fully interpretable"}
        interp_label = interp_labels.get(mr.interpretability_score, "unknown")
        parts.append(
            f"The model has {interp_label} interpretability (score {mr.interpretability_score}/5)."
        )

        if len(all_scored) > 1:
            runner_up = all_scored[1].model_result
            parts.append(
                f"The runner-up was {runner_up.display_name} "
                f"(composite score: {all_scored[1].composite_score:.4f} vs {best.composite_score:.4f})."
            )

        return " ".join(parts)

    def _build_rationale_bullets(
        self, best: ScoredResult, all_scored: list[ScoredResult]
    ) -> list[str]:
        mr = best.model_result
        bullets = []

        if mr.primary_metric is not None:
            bullets.append(f"Highest predictive performance ({mr.primary_metric:.4f})")

        if mr.cv_score is not None:
            bullets.append(f"Strong cross-validation score ({mr.cv_score:.4f} ± {mr.cv_std:.4f})")

        if not mr.is_overfitting:
            bullets.append("No overfitting detected — generalizes well to unseen data")

        if mr.training_time_s < 1.0:
            bullets.append(f"Fast training time ({mr.training_time_s:.2f}s)")

        if mr.interpretability_score >= 4:
            bullets.append(f"High interpretability score ({mr.interpretability_score}/5)")

        if mr.supports_feature_importance:
            bullets.append("Supports feature importance extraction")

        if mr.supports_shap:
            bullets.append("Compatible with SHAP explainability")

        return bullets

    @staticmethod
    def _primary_metric_name(task_type: TaskType) -> str:
        return {
            TaskType.BINARY_CLASSIFICATION: "F1 score",
            TaskType.MULTICLASS_CLASSIFICATION: "weighted F1 score",
            TaskType.REGRESSION: "R² score",
            TaskType.CLUSTERING: "Silhouette score",
        }.get(task_type, "primary metric")
