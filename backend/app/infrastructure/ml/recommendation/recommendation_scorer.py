"""
RecommendationScorer - Computes multi-dimensional scores for each model.

Every model receives independent scores across multiple dimensions:
- Predictive Performance Score
- Generalization Score
- Robustness Score
- Dataset Compatibility Score
- Interpretability Score
- Speed Score
- Resource Efficiency Score
- Overall Recommendation Score

All scores are normalized to 0-100.
"""

from dataclasses import dataclass

from app.core.logging import get_logger
from app.core.recommendation_config import RecommendationMode, ScoringWeights
from app.domain.entities.model_result import ModelResult
from app.infrastructure.ml.recommendation.dataset_compatibility_analyzer import (
    CompatibilityReport,
    CompatibilityScore,
)
from app.infrastructure.ml.recommendation.generalization_analyzer import (
    GeneralizationAnalysis,
    GeneralizationReport,
)

logger = get_logger(__name__)


@dataclass
class ModelScores:
    """Comprehensive scores for a single model."""
    model_id: str
    config_name: str

    # Individual dimension scores (0-100)
    predictive_performance: float
    generalization: float
    robustness: float
    dataset_compatibility: float
    interpretability: float
    speed: float
    resource_efficiency: float

    # Overall score (weighted combination)
    overall_score: float

    # Breakdown for transparency
    score_breakdown: dict[str, float]


@dataclass
class ScoringReport:
    """Aggregated scoring results across all models."""
    scores: dict[str, ModelScores]  # model_id -> scores
    best_overall: str | None  # model_id
    best_predictive: str | None  # model_id
    best_generalized: str | None  # model_id
    fastest: str | None  # model_id
    most_explainable: str | None  # model_id


class RecommendationScorer:
    """
    Computes multi-dimensional scores for model recommendation.
    
    This component combines analysis from other analyzers to produce
    comprehensive scores for each model across multiple dimensions.
    """

    def __init__(self, weights: ScoringWeights | None = None) -> None:
        self._weights = weights or ScoringWeights()

    def score(
        self,
        model_results: list[ModelResult],
        generalization_report: GeneralizationReport,
        compatibility_report: CompatibilityReport,
        primary_metric: str,
        mode: RecommendationMode = RecommendationMode.BEST_OVERALL,
    ) -> ScoringReport:
        """
        Compute scores for all models.
        
        Args:
            model_results: List of model results
            generalization_report: Generalization analysis results
            compatibility_report: Dataset compatibility analysis results
            primary_metric: The primary metric name for performance scoring
            mode: Recommendation mode for weighting
            
        Returns:
            ScoringReport with scores for each model
        """
        scores = {}
        weights = self._weights.get_weights(mode)

        for mr in model_results:
            if not mr.succeeded:
                continue

            gen_analysis = generalization_report.analyses.get(mr.id)
            compat_score = compatibility_report.scores.get(mr.id)

            model_scores = self._compute_model_scores(
                mr, gen_analysis, compat_score, primary_metric, weights
            )
            scores[mr.id] = model_scores

        # Find best models by category
        best_overall = self._find_best(scores, "overall_score")
        best_predictive = self._find_best(scores, "predictive_performance")
        best_generalized = self._find_best(scores, "generalization")
        fastest = self._find_best(scores, "speed")
        most_explainable = self._find_best(scores, "interpretability")

        logger.info(
            "scoring_completed",
            models_scored=len(scores),
            mode=mode.value,
            best_overall=best_overall,
        )

        return ScoringReport(
            scores=scores,
            best_overall=best_overall,
            best_predictive=best_predictive,
            best_generalized=best_generalized,
            fastest=fastest,
            most_explainable=most_explainable,
        )

    def _compute_model_scores(
        self,
        mr: ModelResult,
        gen_analysis: GeneralizationAnalysis | None,
        compat_score: CompatibilityScore | None,
        primary_metric: str,
        weights: dict[str, float],
    ) -> ModelScores:
        """Compute comprehensive scores for a single model."""
        breakdown = {}

        # 1. Predictive Performance Score
        perf_score = self._compute_performance_score(mr, primary_metric)
        breakdown["predictive_performance"] = perf_score

        # 2. Generalization Score
        gen_score = self._compute_generalization_score(gen_analysis)
        breakdown["generalization"] = gen_score

        # 3. Robustness Score
        robust_score = self._compute_robustness_score(mr, gen_analysis)
        breakdown["robustness"] = robust_score

        # 4. Dataset Compatibility Score
        compat = self._compute_compatibility_score(compat_score)
        breakdown["dataset_compatibility"] = compat

        # 5. Interpretability Score
        interp = self._compute_interpretability_score(mr)
        breakdown["interpretability"] = interp

        # 6. Speed Score
        speed = self._compute_speed_score(mr)
        breakdown["speed"] = speed

        # 7. Resource Efficiency Score
        resource = self._compute_resource_score(mr)
        breakdown["resource_efficiency"] = resource

        # 8. Overall Score (weighted combination)
        overall = (
            perf_score * weights.get("predictive_performance", 0.35)
            + gen_score * weights.get("generalization", 0.25)
            + robust_score * weights.get("robustness", 0.15)
            + compat * weights.get("dataset_compatibility", 0.10)
            + interp * weights.get("interpretability", 0.10)
            + speed * weights.get("speed", 0.05)
        )
        breakdown["overall"] = overall

        return ModelScores(
            model_id=mr.id,
            config_name=mr.config_name,
            predictive_performance=perf_score,
            generalization=gen_score,
            robustness=robust_score,
            dataset_compatibility=compat,
            interpretability=interp,
            speed=speed,
            resource_efficiency=resource,
            overall_score=overall,
            score_breakdown=breakdown,
        )

    def _compute_performance_score(self, mr: ModelResult, primary_metric: str) -> float:
        """Compute predictive performance score (0-100)."""
        # Use the primary metric value
        if primary_metric in mr.metrics:
            value = mr.metrics[primary_metric]
            # Normalize based on typical ranges
            # Most metrics are 0-1, but some (like RMSE) can be higher
            if primary_metric in ["rmse", "mae", "mse"]:
                # Lower is better for these metrics
                # This is a simple normalization - in practice, we'd use the dataset range
                return max(0, 100 - value * 10)
            else:
                # Higher is better
                return min(100, value * 100)
        return 50.0  # Default if metric not available

    def _compute_generalization_score(
        self, gen_analysis: GeneralizationAnalysis | None
    ) -> float:
        """Compute generalization score (0-100)."""
        if gen_analysis:
            return gen_analysis.normalized_score
        return 50.0  # Default if analysis not available

    def _compute_robustness_score(
        self, mr: ModelResult, gen_analysis: GeneralizationAnalysis | None
    ) -> float:
        """Compute robustness score (0-100)."""
        # Robustness considers:
        # - CV stability (low std)
        # - No severe overfitting
        # - Consistent performance

        score = 100.0

        # Penalty for overfitting
        if mr.is_overfitting:
            score -= 20.0

        # CV stability
        if gen_analysis and gen_analysis.cv_std is not None:
            # Lower std = higher stability
            if gen_analysis.cv_std > 0.1:
                score -= 15.0
            elif gen_analysis.cv_std > 0.05:
                score -= 5.0

        return max(0.0, min(100.0, score))

    def _compute_compatibility_score(
        self, compat_score: CompatibilityScore | None
    ) -> float:
        """Compute dataset compatibility score (0-100)."""
        if compat_score:
            return compat_score.overall_score
        return 50.0  # Default if analysis not available

    def _compute_interpretability_score(self, mr: ModelResult) -> float:
        """Compute interpretability score (0-100)."""
        # Interpretability is 1-5, normalize to 0-100
        return (mr.interpretability_score - 1) / 4 * 100

    def _compute_speed_score(self, mr: ModelResult) -> float:
        """Compute speed score (0-100) combining training and prediction speed."""
        # Normalize times - faster is better
        # Training time: <1s = 100, >10s = 0
        train_score = max(0, 100 - mr.training_time_s * 10)

        # Prediction time: <0.01s = 100, >0.1s = 0
        pred_score = max(0, 100 - mr.prediction_time_s * 1000)

        # Combine with more weight on prediction speed (inference)
        return train_score * 0.3 + pred_score * 0.7

    def _compute_resource_score(self, mr: ModelResult) -> float:
        """Compute resource efficiency score (0-100)."""
        # This is a simplified version - in practice, we'd measure actual memory usage
        # For now, use interpretability as a proxy (simpler models typically use less resources)
        return self._compute_interpretability_score(mr)

    def _find_best(self, scores: dict[str, ModelScores], field: str) -> str | None:
        """Find the model with the best score for a given field."""
        if not scores:
            return None
        return max(scores.items(), key=lambda x: getattr(x[1], field))[0]


