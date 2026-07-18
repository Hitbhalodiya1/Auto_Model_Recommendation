"""
GeneralizationAnalyzer - Evaluates model generalization capability.

Generalization is the MOST IMPORTANT factor in model selection.
This component analyzes the gap between training and cross-validation scores
to assess how well a model will perform on unseen data.
"""

from dataclasses import dataclass

from app.core.logging import get_logger
from app.core.recommendation_config import (
    GeneralizationLevel,
    GeneralizationThresholds,
)
from app.domain.entities.model_result import ModelResult

logger = get_logger(__name__)


@dataclass
class GeneralizationAnalysis:
    """Result of generalization analysis for a single model."""

    model_id: str
    config_name: str
    training_score: float
    cv_score: float
    cv_std: float | None
    gap: float
    gap_pct: float
    level: GeneralizationLevel
    normalized_score: float  # 0-100, higher is better


@dataclass
class GeneralizationReport:
    """Aggregated generalization analysis across all models."""

    analyses: dict[str, GeneralizationAnalysis]  # model_id -> analysis
    best_generalized: str | None  # model_id
    worst_generalized: str | None  # model_id


class GeneralizationAnalyzer:
    """
    Analyzes model generalization using cross-validation scores.

    The generalization gap is computed as:
    gap = training_score - cv_score

    This gap is then classified:
    - Gap < 3%: Excellent
    - Gap 3-7%: Good
    - Gap 7-15%: Moderate
    - Gap > 15%: High (potential overfitting)
    """

    def __init__(self, thresholds: GeneralizationThresholds | None = None) -> None:
        self._thresholds = thresholds or GeneralizationThresholds()

    def analyze(self, model_results: list[ModelResult]) -> GeneralizationReport:
        """
        Analyze generalization for all models.

        Args:
            model_results: List of model results with training and CV scores

        Returns:
            GeneralizationReport with analysis for each model
        """
        analyses = {}

        for mr in model_results:
            if not mr.succeeded or mr.primary_metric is None or mr.cv_score is None:
                # Skip models without required scores
                continue

            analysis = self._analyze_single(mr)
            analyses[mr.id] = analysis

        # Find best and worst generalized models
        if analyses:
            sorted_by_score = sorted(
                analyses.values(),
                key=lambda a: a.normalized_score,
                reverse=True,
            )
            best = sorted_by_score[0].model_id
            worst = sorted_by_score[-1].model_id
        else:
            best = None
            worst = None

        logger.info(
            "generalization_analysis_completed",
            models_analyzed=len(analyses),
            best_generalized=best,
            worst_generalized=worst,
        )

        return GeneralizationReport(
            analyses=analyses,
            best_generalized=best,
            worst_generalized=worst,
        )

    def _analyze_single(self, mr: ModelResult) -> GeneralizationAnalysis:
        """Analyze generalization for a single model."""
        training_score = mr.primary_metric
        cv_score = mr.cv_score
        cv_std = mr.cv_std

        # Compute gap
        gap = training_score - cv_score
        gap_pct = abs(gap) / max(training_score, 0.001) * 100

        # Classify gap
        level = self._thresholds.classify(gap_pct)

        # Normalize to 0-100 score
        # Higher gap = lower score
        # Excellent (<3%) -> 90-100
        # Good (3-7%) -> 70-90
        # Moderate (7-15%) -> 50-70
        # High (>15%) -> 0-50
        normalized_score = self._normalize_gap_score(gap_pct)

        return GeneralizationAnalysis(
            model_id=mr.id,
            config_name=mr.config_name,
            training_score=training_score,
            cv_score=cv_score,
            cv_std=cv_std,
            gap=gap,
            gap_pct=gap_pct,
            level=level,
            normalized_score=normalized_score,
        )

    def _normalize_gap_score(self, gap_pct: float) -> float:
        """
        Normalize gap percentage to a 0-100 score.

        Smaller gaps get higher scores.
        """
        if gap_pct < self._thresholds.excellent_gap_pct:
            # Excellent: 90-100 based on how close to 0
            score = 100 - (gap_pct / self._thresholds.excellent_gap_pct) * 10
        elif gap_pct < self._thresholds.good_gap_pct:
            # Good: 70-90
            range_size = self._thresholds.good_gap_pct - self._thresholds.excellent_gap_pct
            position = (gap_pct - self._thresholds.excellent_gap_pct) / range_size
            score = 90 - position * 20
        elif gap_pct < self._thresholds.moderate_gap_pct:
            # Moderate: 50-70
            range_size = self._thresholds.moderate_gap_pct - self._thresholds.good_gap_pct
            position = (gap_pct - self._thresholds.good_gap_pct) / range_size
            score = 70 - position * 20
        else:
            # High: 0-50, decaying exponentially
            excess = gap_pct - self._thresholds.moderate_gap_pct
            score = max(0, 50 - excess * 2)

        return max(0.0, min(100.0, score))
