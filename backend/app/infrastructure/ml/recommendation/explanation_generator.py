"""
ExplanationGenerator - Generates natural language explanations for recommendations.

Every recommendation must explain WHY a model was selected, including:
- Primary reason for selection
- Strengths
- Weaknesses
- Recommended use cases
- Not recommended for
- Comparison with second-best model
"""

from dataclasses import dataclass

from app.core.logging import get_logger
from app.core.recommendation_config import RecommendationMode
from app.infrastructure.ml.recommendation.generalization_analyzer import GeneralizationAnalysis
from app.infrastructure.ml.recommendation.recommendation_strategy import RecommendationCandidate

logger = get_logger(__name__)


@dataclass
class ModelExplanation:
    """Natural language explanation for a model recommendation."""

    model_id: str
    config_name: str
    display_name: str
    mode: RecommendationMode
    primary_reason: str
    explanation_text: str
    strengths: list[str]
    weaknesses: list[str]
    recommended_for: list[str]
    not_recommended_for: list[str]
    comparison_with_runner_up: str | None


@dataclass
class ExplanationReport:
    """Aggregated explanations for all recommended models."""

    explanations: dict[str, ModelExplanation]  # model_id -> explanation


class ExplanationGenerator:
    """
    Generates human-readable explanations for model recommendations.

    The explanations are transparent and explainable, covering:
    - Why this model was selected
    - Its strengths and weaknesses
    - Suitable use cases
    - Comparison with alternatives
    """

    def __init__(self) -> None:
        pass

    def generate_explanations(
        self,
        candidates: list[RecommendationCandidate],
        generalization_analyses: dict[str, GeneralizationAnalysis],
    ) -> ExplanationReport:
        """
        Generate explanations for all recommended models.

        Args:
            candidates: List of recommendation candidates
            generalization_analyses: Generalization analysis for each model

        Returns:
            ExplanationReport with explanations for each model
        """
        explanations = {}

        for candidate in candidates:
            gen_analysis = generalization_analyses.get(candidate.model_id)

            explanation = self._generate_single_explanation(candidate, gen_analysis)
            explanations[candidate.model_id] = explanation

        logger.info(
            "explanations_generated",
            count=len(explanations),
        )

        return ExplanationReport(explanations=explanations)

    def _generate_single_explanation(
        self,
        candidate: RecommendationCandidate,
        gen_analysis: GeneralizationAnalysis | None,
    ) -> ModelExplanation:
        """Generate explanation for a single model."""
        mr = candidate.model_result
        mode = candidate.mode

        # Primary reason
        primary_reason = self._generate_primary_reason(candidate, gen_analysis)

        # Full explanation text
        explanation_text = self._generate_explanation_text(candidate, gen_analysis, primary_reason)

        # Strengths
        strengths = self._generate_strengths(candidate, gen_analysis)

        # Weaknesses
        weaknesses = self._generate_weaknesses(candidate, gen_analysis)

        # Recommended use cases
        recommended_for = self._generate_recommended_for(candidate)

        # Not recommended for
        not_recommended_for = self._generate_not_recommended_for(candidate)

        # Comparison with runner-up (if available)
        comparison = self._generate_comparison(candidate)

        return ModelExplanation(
            model_id=mr.id,
            config_name=mr.config_name,
            display_name=mr.display_name,
            mode=mode,
            primary_reason=primary_reason,
            explanation_text=explanation_text,
            strengths=strengths,
            weaknesses=weaknesses,
            recommended_for=recommended_for,
            not_recommended_for=not_recommended_for,
            comparison_with_runner_up=comparison,
        )

    def _generate_primary_reason(
        self,
        candidate: RecommendationCandidate,
        gen_analysis: GeneralizationAnalysis | None,
    ) -> str:
        """Generate the primary reason for selecting this model."""
        mr = candidate.model_result
        scores = candidate.scores
        mode = candidate.mode

        if mode == RecommendationMode.BEST_OVERALL:
            if scores.generalization >= 80:
                return "Excellent generalization with strong predictive performance"
            elif scores.predictive_performance >= 85:
                return "Highest predictive performance with good generalization"
            else:
                return "Best balance of performance, generalization, and efficiency"

        elif mode == RecommendationMode.BEST_PREDICTIVE:
            return f"Highest predictive performance ({scores.predictive_performance:.1f}/100)"

        elif mode == RecommendationMode.FASTEST:
            return f"Fastest inference speed ({mr.prediction_time_s * 1000:.2f}ms per prediction)"

        elif mode == RecommendationMode.MOST_EXPLAINABLE:
            return f"Highest interpretability score ({mr.interpretability_score}/5)"

        return "Selected based on overall performance metrics"

    def _generate_explanation_text(
        self,
        candidate: RecommendationCandidate,
        gen_analysis: GeneralizationAnalysis | None,
        primary_reason: str,
    ) -> str:
        """Generate the full explanation text."""
        mr = candidate.model_result
        scores = candidate.scores
        mode = candidate.mode

        parts = [
            f"**{mr.display_name}** was selected as the {mode.value.replace('_', ' ')} model.",
            f"{primary_reason}.",
        ]

        # Add performance details
        if mr.primary_metric is not None:
            parts.append(f"It achieved a primary metric score of {mr.primary_metric:.4f}")

        # Add generalization details
        if gen_analysis:
            if gen_analysis.level.value == "excellent":
                parts.append("with excellent generalization to unseen data")
            elif gen_analysis.level.value == "good":
                parts.append("with good generalization capability")
            elif gen_analysis.level.value == "moderate":
                parts.append("with moderate generalization (some overfitting detected)")
            else:
                parts.append("with high generalization gap (potential overfitting)")

        # Add mode-specific details
        if mode == RecommendationMode.BEST_OVERALL:
            if scores.dataset_compatibility >= 70:
                parts.append("and is well-suited for the dataset characteristics")
            if scores.speed >= 70:
                parts.append("with fast inference suitable for production deployment")

        elif mode == RecommendationMode.FASTEST:
            parts.append("making it ideal for real-time applications")

        elif mode == RecommendationMode.MOST_EXPLAINABLE:
            if mr.supports_feature_importance:
                parts.append("and supports feature importance extraction")
            if mr.supports_shap:
                parts.append("with SHAP compatibility for detailed explanations")

        return " ".join(parts) + "."

    def _generate_strengths(
        self,
        candidate: RecommendationCandidate,
        gen_analysis: GeneralizationAnalysis | None,
    ) -> list[str]:
        """Generate list of model strengths."""
        mr = candidate.model_result
        scores = candidate.scores
        strengths = []

        # Performance
        if scores.predictive_performance >= 80:
            strengths.append("Strong predictive performance")
        elif scores.predictive_performance >= 60:
            strengths.append("Acceptable predictive performance")

        # Generalization
        if gen_analysis:
            if gen_analysis.level.value == "excellent":
                strengths.append("Excellent generalization to unseen data")
            elif gen_analysis.level.value == "good":
                strengths.append("Good generalization capability")
            elif gen_analysis.level.value == "moderate":
                strengths.append("Moderate generalization")

        # Robustness
        if scores.robustness >= 70:
            strengths.append("Robust and stable performance")
        if not mr.is_overfitting:
            strengths.append("No significant overfitting")

        # Speed
        if scores.speed >= 80:
            strengths.append("Fast inference speed")
        if mr.training_time_s < 1.0:
            strengths.append("Fast training time")

        # Interpretability
        if scores.interpretability >= 80:
            strengths.append("High interpretability")
        if mr.interpretability_score >= 4:
            strengths.append(f"Interpretability score {mr.interpretability_score}/5")
        if mr.supports_feature_importance:
            strengths.append("Supports feature importance extraction")
        if mr.supports_shap:
            strengths.append("SHAP-compatible for detailed explanations")

        # Dataset compatibility
        if scores.dataset_compatibility >= 70:
            strengths.append("Well-suited for dataset characteristics")

        return strengths

    def _generate_weaknesses(
        self,
        candidate: RecommendationCandidate,
        gen_analysis: GeneralizationAnalysis | None,
    ) -> list[str]:
        """Generate list of model weaknesses."""
        mr = candidate.model_result
        scores = candidate.scores
        weaknesses = []

        # Performance
        if scores.predictive_performance < 60:
            weaknesses.append("Lower predictive performance")

        # Generalization
        if gen_analysis and gen_analysis.level.value == "high":
            weaknesses.append("High generalization gap (potential overfitting)")
        if mr.is_overfitting:
            weaknesses.append("Signs of overfitting detected")

        # Speed
        if scores.speed < 40:
            weaknesses.append("Slow inference speed")
        if mr.training_time_s > 10.0:
            weaknesses.append("Long training time")

        # Interpretability
        if scores.interpretability < 40:
            weaknesses.append("Low interpretability (black box)")
        if mr.interpretability_score <= 2:
            weaknesses.append("Difficult to interpret")

        # Resource usage
        if scores.resource_efficiency < 40:
            weaknesses.append("High resource requirements")

        return weaknesses

    def _generate_recommended_for(
        self,
        candidate: RecommendationCandidate,
    ) -> list[str]:
        """Generate recommended use cases."""
        scores = candidate.scores
        recommended = []

        # Based on speed
        if scores.speed >= 70:
            recommended.append("Real-time prediction applications")
            recommended.append("High-throughput inference")

        # Based on interpretability
        if scores.interpretability >= 70:
            recommended.append("Regulated industries requiring transparency")
            recommended.append("Applications where model explanation is critical")

        # Based on performance
        if scores.predictive_performance >= 80:
            recommended.append("Applications requiring high accuracy")
            recommended.append("Competitive ML scenarios")

        # Based on generalization
        if scores.generalization >= 70:
            recommended.append("Production deployment with unseen data")
            recommended.append("Dynamic environments")

        # Based on dataset compatibility
        if scores.dataset_compatibility >= 70:
            recommended.append("Datasets with similar characteristics")

        return recommended

    def _generate_not_recommended_for(
        self,
        candidate: RecommendationCandidate,
    ) -> list[str]:
        """Generate not recommended use cases."""
        scores = candidate.scores
        not_recommended = []

        # Based on speed
        if scores.speed < 40:
            not_recommended.append("Real-time prediction applications")
            not_recommended.append("Low-latency requirements")

        # Based on interpretability
        if scores.interpretability < 40:
            not_recommended.append("Regulated industries requiring transparency")
            not_recommended.append("Applications where model explanation is critical")

        # Based on performance
        if scores.predictive_performance < 60:
            not_recommended.append("Applications requiring high accuracy")
            not_recommended.append("Competitive ML scenarios")

        # Based on generalization
        if scores.generalization < 50:
            not_recommended.append("Production deployment with unseen data")

        # Based on resource usage
        if scores.resource_efficiency < 40:
            not_recommended.append("Resource-constrained environments")
            not_recommended.append("Edge deployment")

        return not_recommended

    def _generate_comparison(
        self,
        candidate: RecommendationCandidate,
    ) -> str | None:
        """Generate comparison with runner-up model."""
        # This would require access to the second-best model
        # For now, return None - this can be enhanced later
        return None
