"""
CandidateFilter - Filters out invalid or severely underperforming models.

This component removes models that:
- Failed training or evaluation
- Have invalid metrics
- Have severe overfitting (configurable threshold)
"""

from dataclasses import dataclass

from app.core.logging import get_logger
from app.core.recommendation_config import FilteringThresholds
from app.domain.entities.model_result import ModelResult

logger = get_logger(__name__)


@dataclass
class FilteringResult:
    """Result of the filtering process."""

    candidates: list[ModelResult]
    filtered_out: list[ModelResult]
    reasons: dict[str, str]  # model_id -> reason for filtering


class CandidateFilter:
    """
    Filters model candidates based on quality thresholds.

    Only removes models with severe issues. Moderate overfitting is allowed
    as it may be acceptable depending on the use case.
    """

    def __init__(self, thresholds: FilteringThresholds | None = None) -> None:
        self._thresholds = thresholds or FilteringThresholds()

    def filter(self, model_results: list[ModelResult]) -> FilteringResult:
        """
        Filter model results to remove invalid candidates.

        Args:
            model_results: List of all model results from training

        Returns:
            FilteringResult with valid candidates and filtered models
        """
        candidates = []
        filtered_out = []
        reasons = {}

        for mr in model_results:
            reason = self._should_filter(mr)
            if reason:
                filtered_out.append(mr)
                reasons[mr.id] = reason
                logger.debug(
                    "model_filtered",
                    model_id=mr.id,
                    config_name=mr.config_name,
                    reason=reason,
                )
            else:
                candidates.append(mr)

        logger.info(
            "candidate_filtering_completed",
            total=len(model_results),
            candidates=len(candidates),
            filtered=len(filtered_out),
        )

        return FilteringResult(
            candidates=candidates,
            filtered_out=filtered_out,
            reasons=reasons,
        )

    def _should_filter(self, mr: ModelResult) -> str | None:
        """
        Determine if a model should be filtered out.

        Returns None if model should be kept, otherwise returns reason string.
        """
        # Check if training/evaluation failed
        if not self._thresholds.allow_failed_models and not mr.succeeded:
            return f"Training/evaluation failed: {mr.error_message}"

        # Check for invalid primary metric
        primary_metric = mr.primary_metric
        if primary_metric is None:
            return "No primary metric available"

        if not (
            self._thresholds.min_primary_metric
            <= primary_metric
            <= self._thresholds.max_primary_metric
        ):
            return f"Primary metric {primary_metric} out of valid range"

        # Check for severe overfitting
        if mr.is_overfitting:
            # Calculate generalization gap if we have both scores
            if mr.cv_score is not None and primary_metric is not None:
                gap_pct = abs(primary_metric - mr.cv_score) / max(primary_metric, 0.001) * 100
                if gap_pct > self._thresholds.severe_overfitting_gap_pct:
                    return f"Severe overfitting: {gap_pct:.1f}% generalization gap"

        return None
