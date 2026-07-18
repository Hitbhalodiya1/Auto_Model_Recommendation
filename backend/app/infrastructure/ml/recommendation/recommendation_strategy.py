"""
RecommendationStrategy - Implements different recommendation modes.

Generates FOUR recommendations instead of only one:
1. Best Overall - Optimized for production
2. Best Predictive Performance - Maximize predictive performance
3. Fastest Model - Prioritize inference latency
4. Most Explainable Model - Prioritize interpretability
"""

from dataclasses import dataclass

from app.core.logging import get_logger
from app.core.recommendation_config import RecommendationMode
from app.domain.entities.model_result import ModelResult
from app.infrastructure.ml.recommendation.recommendation_scorer import ModelScores, ScoringReport

logger = get_logger(__name__)


@dataclass
class RecommendationCandidate:
    """A candidate model for recommendation."""
    model_id: str
    model_result: ModelResult
    scores: ModelScores
    mode: RecommendationMode
    rationale: list[str]


@dataclass
class StrategyResult:
    """Result of applying a recommendation strategy."""
    mode: RecommendationMode
    candidates: list[RecommendationCandidate]
    selected: RecommendationCandidate | None


@dataclass
class MultiModeRecommendations:
    """Recommendations across all modes."""
    best_overall: StrategyResult
    best_predictive: StrategyResult
    fastest: StrategyResult
    most_explainable: StrategyResult


class RecommendationStrategy:
    """
    Implements different recommendation strategies based on use case.
    
    Each strategy prioritizes different aspects of model performance:
    - Best Overall: Balanced approach for production deployment
    - Best Predictive: Maximum accuracy regardless of complexity
    - Fastest: Minimum inference latency for real-time applications
    - Most Explainable: Maximum interpretability for regulated industries
    """

    def __init__(self) -> None:
        pass

    def generate_recommendations(
        self,
        model_results: list[ModelResult],
        scoring_report: ScoringReport,
    ) -> MultiModeRecommendations:
        """
        Generate recommendations for all modes.
        
        Args:
            model_results: List of model results
            scoring_report: Scoring report with all model scores
            
        Returns:
            MultiModeRecommendations with recommendations for each mode
        """
        # Build model lookup
        model_map = {mr.id: mr for mr in model_results}

        # Generate recommendations for each mode
        best_overall = self._recommend_best_overall(
            model_map, scoring_report
        )
        best_predictive = self._recommend_best_predictive(
            model_map, scoring_report
        )
        fastest = self._recommend_fastest(
            model_map, scoring_report
        )
        most_explainable = self._recommend_most_explainable(
            model_map, scoring_report
        )

        logger.info(
            "multi_mode_recommendations_generated",
            best_overall=best_overall.selected.model_id if best_overall.selected else None,
            best_predictive=best_predictive.selected.model_id if best_predictive.selected else None,
            fastest=fastest.selected.model_id if fastest.selected else None,
            most_explainable=most_explainable.selected.model_id if most_explainable.selected else None,
        )

        return MultiModeRecommendations(
            best_overall=best_overall,
            best_predictive=best_predictive,
            fastest=fastest,
            most_explainable=most_explainable,
        )

    def _recommend_best_overall(
        self,
        model_map: dict[str, ModelResult],
        scoring_report: ScoringReport,
    ) -> StrategyResult:
        """
        Recommend the best overall model for production.
        
        Priority:
        1. Generalization (most important)
        2. Predictive performance
        3. Robustness
        4. Dataset compatibility
        5. Speed
        6. Interpretability
        """
        candidates = []

        for model_id, scores in scoring_report.scores.items():
            mr = model_map.get(model_id)
            if not mr:
                continue

            rationale = self._build_overall_rationale(scores, mr)
            candidates.append(
                RecommendationCandidate(
                    model_id=model_id,
                    model_result=mr,
                    scores=scores,
                    mode=RecommendationMode.BEST_OVERALL,
                    rationale=rationale,
                )
            )

        # Sort by overall score
        candidates.sort(key=lambda c: c.scores.overall_score, reverse=True)

        selected = candidates[0] if candidates else None

        return StrategyResult(
            mode=RecommendationMode.BEST_OVERALL,
            candidates=candidates[:5],  # Top 5
            selected=selected,
        )

    def _recommend_best_predictive(
        self,
        model_map: dict[str, ModelResult],
        scoring_report: ScoringReport,
    ) -> StrategyResult:
        """
        Recommend the model with best predictive performance.
        
        Ignores interpretability and resource constraints.
        Maximizes predictive performance and generalization.
        """
        candidates = []

        for model_id, scores in scoring_report.scores.items():
            mr = model_map.get(model_id)
            if not mr:
                continue

            rationale = self._build_predictive_rationale(scores, mr)
            candidates.append(
                RecommendationCandidate(
                    model_id=model_id,
                    model_result=mr,
                    scores=scores,
                    mode=RecommendationMode.BEST_PREDICTIVE,
                    rationale=rationale,
                )
            )

        # Sort by predictive performance and generalization
        candidates.sort(
            key=lambda c: (
                c.scores.predictive_performance * 0.6
                + c.scores.generalization * 0.4
            ),
            reverse=True,
        )

        selected = candidates[0] if candidates else None

        return StrategyResult(
            mode=RecommendationMode.BEST_PREDICTIVE,
            candidates=candidates[:5],
            selected=selected,
        )

    def _recommend_fastest(
        self,
        model_map: dict[str, ModelResult],
        scoring_report: ScoringReport,
    ) -> StrategyResult:
        """
        Recommend the fastest model for inference.
        
        Prioritizes prediction speed while maintaining acceptable performance.
        """
        candidates = []

        for model_id, scores in scoring_report.scores.items():
            mr = model_map.get(model_id)
            if not mr:
                continue

            rationale = self._build_speed_rationale(scores, mr)
            candidates.append(
                RecommendationCandidate(
                    model_id=model_id,
                    model_result=mr,
                    scores=scores,
                    mode=RecommendationMode.FASTEST,
                    rationale=rationale,
                )
            )

        # Sort by speed score, but require minimum performance
        min_performance = 50.0  # Require at least 50% performance score
        viable = [c for c in candidates if c.scores.predictive_performance >= min_performance]

        if not viable:
            # If no viable models, use all candidates
            viable = candidates

        viable.sort(key=lambda c: c.scores.speed, reverse=True)

        selected = viable[0] if viable else None

        return StrategyResult(
            mode=RecommendationMode.FASTEST,
            candidates=viable[:5],
            selected=selected,
        )

    def _recommend_most_explainable(
        self,
        model_map: dict[str, ModelResult],
        scoring_report: ScoringReport,
    ) -> StrategyResult:
        """
        Recommend the most explainable model.
        
        Prioritizes interpretability while maintaining acceptable performance.
        """
        candidates = []

        for model_id, scores in scoring_report.scores.items():
            mr = model_map.get(model_id)
            if not mr:
                continue

            rationale = self._build_explainability_rationale(scores, mr)
            candidates.append(
                RecommendationCandidate(
                    model_id=model_id,
                    model_result=mr,
                    scores=scores,
                    mode=RecommendationMode.MOST_EXPLAINABLE,
                    rationale=rationale,
                )
            )

        # Sort by interpretability, but require minimum performance
        min_performance = 50.0
        viable = [c for c in candidates if c.scores.predictive_performance >= min_performance]

        if not viable:
            viable = candidates

        viable.sort(key=lambda c: c.scores.interpretability, reverse=True)

        selected = viable[0] if viable else None

        return StrategyResult(
            mode=RecommendationMode.MOST_EXPLAINABLE,
            candidates=viable[:5],
            selected=selected,
        )

    def _build_overall_rationale(self, scores: ModelScores, mr: ModelResult) -> list[str]:
        """Build rationale for best overall recommendation."""
        rationale = []

        if scores.generalization >= 80:
            rationale.append("Excellent generalization to unseen data")
        elif scores.generalization >= 60:
            rationale.append("Good generalization capability")

        if scores.predictive_performance >= 80:
            rationale.append("Strong predictive performance")
        elif scores.predictive_performance >= 60:
            rationale.append("Acceptable predictive performance")

        if scores.robustness >= 70:
            rationale.append("Robust model with stable performance")

        if scores.dataset_compatibility >= 70:
            rationale.append("Well-suited for dataset characteristics")

        if scores.speed >= 70:
            rationale.append("Fast inference suitable for production")

        if scores.interpretability >= 70:
            rationale.append("High interpretability for transparency")

        return rationale

    def _build_predictive_rationale(self, scores: ModelScores, mr: ModelResult) -> list[str]:
        """Build rationale for best predictive recommendation."""
        rationale = []

        if scores.predictive_performance >= 90:
            rationale.append("Exceptional predictive performance")
        elif scores.predictive_performance >= 75:
            rationale.append("Strong predictive performance")

        if scores.generalization >= 70:
            rationale.append("Good generalization capability")

        rationale.append("Optimized for maximum accuracy")

        return rationale

    def _build_speed_rationale(self, scores: ModelScores, mr: ModelResult) -> list[str]:
        """Build rationale for fastest recommendation."""
        rationale = []

        if scores.speed >= 90:
            rationale.append("Extremely fast inference")
        elif scores.speed >= 70:
            rationale.append("Fast prediction speed")

        if mr.prediction_time_s < 0.01:
            rationale.append(f"Sub-millisecond inference ({mr.prediction_time_s*1000:.2f}ms)")
        elif mr.prediction_time_s < 0.1:
            rationale.append(f"Low latency inference ({mr.prediction_time_s*1000:.1f}ms)")

        if scores.predictive_performance >= 60:
            rationale.append("Maintains acceptable performance")

        return rationale

    def _build_explainability_rationale(self, scores: ModelScores, mr: ModelResult) -> list[str]:
        """Build rationale for most explainable recommendation."""
        rationale = []

        if scores.interpretability >= 90:
            rationale.append("Fully interpretable model")
        elif scores.interpretability >= 70:
            rationale.append("High interpretability")

        if mr.interpretability_score >= 4:
            rationale.append(f"Interpretability score {mr.interpretability_score}/5")

        if mr.supports_feature_importance:
            rationale.append("Supports feature importance extraction")

        if mr.supports_shap:
            rationale.append("Compatible with SHAP explainability")

        if scores.predictive_performance >= 60:
            rationale.append("Maintains acceptable performance")

        return rationale
