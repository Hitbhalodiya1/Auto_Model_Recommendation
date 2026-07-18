"""
RecommendationFormatter - Formats recommendation results for API responses.

This component transforms internal recommendation data structures into
DTOs and response formats compatible with the existing API contracts.
"""

from dataclasses import dataclass
from typing import Any

from app.core.logging import get_logger
from app.domain.entities.model_result import ModelResult
from app.infrastructure.ml.recommendation.explanation_generator import ModelExplanation
from app.infrastructure.ml.recommendation.recommendation_strategy import (
    MultiModeRecommendations,
    RecommendationCandidate,
)

logger = get_logger(__name__)


@dataclass
class FormattedRecommendation:
    """Formatted recommendation for API response."""
    mode: str
    model_id: str
    config_name: str
    display_name: str
    overall_score: float
    score_breakdown: dict[str, float]
    rationale: list[str]
    explanation_text: str
    strengths: list[str]
    weaknesses: list[str]
    recommended_for: list[str]
    not_recommended_for: list[str]


@dataclass
class FormattedRanking:
    """Formatted ranking entry for all_rankings list."""
    rank: int
    config_name: str
    display_name: str
    composite_score: float
    primary_metric: float | None
    cv_score: float | None
    is_overfitting: bool


@dataclass
class FormattedRecommendations:
    """All formatted recommendations for API response."""
    best_overall: FormattedRecommendation | None
    best_predictive: FormattedRecommendation | None
    fastest: FormattedRecommendation | None
    most_explainable: FormattedRecommendation | None
    all_rankings: list[FormattedRanking]


class RecommendationFormatter:
    """
    Formats recommendation results for API responses.
    
    This component ensures backward compatibility with existing DTOs
    while supporting the new multi-mode recommendation structure.
    """

    def __init__(self) -> None:
        pass

    def format_recommendations(
        self,
        multi_mode_recommendations: MultiModeRecommendations,
        explanations: dict[str, ModelExplanation],
        model_results: list[ModelResult],
    ) -> FormattedRecommendations:
        """
        Format all recommendations for API response.
        
        Args:
            multi_mode_recommendations: Recommendations for all modes
            explanations: Explanations for each model
            model_results: List of all model results
            
        Returns:
            FormattedRecommendations with all formatted data
        """
        # Format each mode
        best_overall = self._format_single_recommendation(
            multi_mode_recommendations.best_overall,
            explanations,
        )
        best_predictive = self._format_single_recommendation(
            multi_mode_recommendations.best_predictive,
            explanations,
        )
        fastest = self._format_single_recommendation(
            multi_mode_recommendations.fastest,
            explanations,
        )
        most_explainable = self._format_single_recommendation(
            multi_mode_recommendations.most_explainable,
            explanations,
        )

        # Format all rankings (use best_overall as the primary ranking)
        all_rankings = self._format_all_rankings(
            multi_mode_recommendations.best_overall.candidates,
            model_results,
        )

        logger.info(
            "recommendations_formatted",
            best_overall=best_overall.config_name if best_overall else None,
            best_predictive=best_predictive.config_name if best_predictive else None,
            fastest=fastest.config_name if fastest else None,
            most_explainable=most_explainable.config_name if most_explainable else None,
        )

        return FormattedRecommendations(
            best_overall=best_overall,
            best_predictive=best_predictive,
            fastest=fastest,
            most_explainable=most_explainable,
            all_rankings=all_rankings,
        )

    def _format_single_recommendation(
        self,
        strategy_result: Any,
        explanations: dict[str, ModelExplanation],
    ) -> FormattedRecommendation | None:
        """Format a single recommendation from strategy result."""
        if not strategy_result or not strategy_result.selected:
            return None

        candidate = strategy_result.selected
        explanation = explanations.get(candidate.model_id)

        if not explanation:
            # Fallback if explanation not available
            return FormattedRecommendation(
                mode=candidate.mode.value,
                model_id=candidate.model_id,
                config_name=candidate.model_result.config_name,
                display_name=candidate.model_result.display_name,
                overall_score=candidate.scores.overall_score,
                score_breakdown=candidate.scores.score_breakdown,
                rationale=candidate.rationale,
                explanation_text="",
                strengths=[],
                weaknesses=[],
                recommended_for=[],
                not_recommended_for=[],
            )

        return FormattedRecommendation(
            mode=explanation.mode.value,
            model_id=explanation.model_id,
            config_name=explanation.config_name,
            display_name=explanation.display_name,
            overall_score=candidate.scores.overall_score,
            score_breakdown=candidate.scores.score_breakdown,
            rationale=explanation.strengths,  # Use strengths as rationale for backward compatibility
            explanation_text=explanation.explanation_text,
            strengths=explanation.strengths,
            weaknesses=explanation.weaknesses,
            recommended_for=explanation.recommended_for,
            not_recommended_for=explanation.not_recommended_for,
        )

    def _format_all_rankings(
        self,
        candidates: list[RecommendationCandidate],
        model_results: list[ModelResult],
    ) -> list[FormattedRanking]:
        """Format all rankings for the all_rankings list."""
        rankings = []

        for i, candidate in enumerate(candidates):
            mr = candidate.model_result

            ranking = FormattedRanking(
                rank=i + 1,
                config_name=mr.config_name,
                display_name=mr.display_name,
                composite_score=candidate.scores.overall_score,
                primary_metric=mr.primary_metric,
                cv_score=mr.cv_score,
                is_overfitting=mr.is_overfitting,
            )
            rankings.append(ranking)

        return rankings

    def format_for_backward_compatibility(
        self,
        formatted: FormattedRecommendations,
    ) -> dict[str, Any]:
        """
        Format recommendations for backward compatibility with existing API.
        
        Returns a dict compatible with the existing RecommendationDTO structure.
        """
        # Use best_overall as the primary recommendation for backward compatibility
        primary = formatted.best_overall

        if not primary:
            return {}

        return {
            "mode": primary.mode,
            "model_id": primary.model_id,
            "config_name": primary.config_name,
            "display_name": primary.display_name,
            "composite_score": primary.overall_score,
            "score_breakdown": primary.score_breakdown,
            "rationale": primary.rationale,
            "explanation_text": primary.explanation_text,
            "all_rankings": [
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
            ],
            # New fields for multi-mode support
            "best_predictive": {
                "model_id": formatted.best_predictive.model_id if formatted.best_predictive else None,
                "config_name": formatted.best_predictive.config_name if formatted.best_predictive else None,
                "display_name": formatted.best_predictive.display_name if formatted.best_predictive else None,
                "overall_score": formatted.best_predictive.overall_score if formatted.best_predictive else None,
            } if formatted.best_predictive else None,
            "fastest": {
                "model_id": formatted.fastest.model_id if formatted.fastest else None,
                "config_name": formatted.fastest.config_name if formatted.fastest else None,
                "display_name": formatted.fastest.display_name if formatted.fastest else None,
                "overall_score": formatted.fastest.overall_score if formatted.fastest else None,
            } if formatted.fastest else None,
            "most_explainable": {
                "model_id": formatted.most_explainable.model_id if formatted.most_explainable else None,
                "config_name": formatted.most_explainable.config_name if formatted.most_explainable else None,
                "display_name": formatted.most_explainable.display_name if formatted.most_explainable else None,
                "overall_score": formatted.most_explainable.overall_score if formatted.most_explainable else None,
            } if formatted.most_explainable else None,
        }
