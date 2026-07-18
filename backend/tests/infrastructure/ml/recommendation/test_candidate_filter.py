"""
Unit tests for CandidateFilter component.
"""

import pytest

from app.core.recommendation_config import FilteringThresholds
from app.domain.entities.model_result import ModelResult
from app.infrastructure.ml.recommendation.candidate_filter import (
    CandidateFilter,
    FilteringResult,
)


@pytest.fixture
def sample_model_results():
    """Create sample model results for testing."""
    return [
        ModelResult(
            id="model1",
            config_name="rf_100",
            display_name="Random Forest",
            algorithm_name="RandomForest",
            metrics={"accuracy": 0.85, "f1_score": 0.84},
            cv_score=0.82,
            cv_std=0.02,
            is_overfitting=False,
            training_time_s=5.0,
            prediction_time_s=0.01,
            interpretability_score=3,
        ),
        ModelResult(
            id="model2",
            config_name="lr_lbfgs",
            display_name="Logistic Regression",
            algorithm_name="LogisticRegression",
            metrics={"accuracy": 0.80, "f1_score": 0.79},
            cv_score=0.78,
            cv_std=0.03,
            is_overfitting=False,
            training_time_s=1.0,
            prediction_time_s=0.001,
            interpretability_score=4,
        ),
        ModelResult(
            id="model3",
            config_name="failed_model",
            display_name="Failed Model",
            algorithm_name="SVM",
            metrics={"accuracy": 0.50},
            cv_score=None,
            cv_std=None,
            is_overfitting=False,
            training_time_s=0.0,
            prediction_time_s=0.0,
            interpretability_score=2,
            error_message="Training failed due to convergence issues",
        ),
        ModelResult(
            id="model4",
            config_name="overfit_model",
            display_name="Overfit Model",
            algorithm_name="DecisionTree",
            metrics={"accuracy": 0.95, "f1_score": 0.94},
            cv_score=0.70,
            cv_std=0.10,
            is_overfitting=True,
            training_time_s=0.5,
            prediction_time_s=0.001,
            interpretability_score=5,
        ),
    ]


class TestCandidateFilter:
    """Test cases for CandidateFilter."""

    def test_filter_removes_failed_models(self, sample_model_results):
        """Test that failed models are filtered out."""
        filter = CandidateFilter()
        result = filter.filter(sample_model_results)

        assert len(result.candidates) == 2  # Should exclude failed model and overfit model
        assert len(result.filtered_out) == 2
        assert "model3" in result.reasons
        assert "Training failed" in result.reasons["model3"]

    def test_filter_keeps_successful_models(self, sample_model_results):
        """Test that successful models are kept."""
        filter = CandidateFilter()
        result = filter.filter(sample_model_results)

        candidate_ids = [c.id for c in result.candidates]
        assert "model1" in candidate_ids
        assert "model2" in candidate_ids

    def test_filter_removes_severe_overfitting(self, sample_model_results):
        """Test that models with severe overfitting are filtered."""
        thresholds = FilteringThresholds(severe_overfitting_gap_pct=15.0)
        filter = CandidateFilter(thresholds)
        result = filter.filter(sample_model_results)

        # model4 has 0.95 training vs 0.70 CV = 26% gap, should be filtered
        candidate_ids = [c.id for c in result.candidates]
        assert "model4" not in candidate_ids
        assert "model4" in result.reasons
        assert "Severe overfitting" in result.reasons["model4"]

    def test_filter_allows_moderate_overfitting(self, sample_model_results):
        """Test that models with moderate overfitting are kept."""
        thresholds = FilteringThresholds(severe_overfitting_gap_pct=30.0)
        filter = CandidateFilter(thresholds)
        result = filter.filter(sample_model_results)

        # model4 should be kept with higher threshold
        candidate_ids = [c.id for c in result.candidates]
        assert "model4" in candidate_ids

    def test_filter_empty_list(self):
        """Test filtering an empty list."""
        filter = CandidateFilter()
        result = filter.filter([])

        assert len(result.candidates) == 0
        assert len(result.filtered_out) == 0

    def test_filter_all_failed(self):
        """Test when all models failed."""
        failed_models = [
            ModelResult(
                id=f"model{i}",
                config_name=f"config{i}",
                display_name=f"Model {i}",
                algorithm_name="Test",
                metrics={},
                cv_score=None,
                cv_std=None,
                is_overfitting=False,
                training_time_s=0.0,
                prediction_time_s=0.0,
                interpretability_score=1,
                error_message="Failed",
            )
            for i in range(3)
        ]

        filter = CandidateFilter()
        result = filter.filter(failed_models)

        assert len(result.candidates) == 0
        assert len(result.filtered_out) == 3

    def test_filter_with_allow_failed(self, sample_model_results):
        """Test filtering with allow_failed=True."""
        thresholds = FilteringThresholds(allow_failed_models=True, severe_overfitting_gap_pct=30.0)
        filter = CandidateFilter(thresholds)
        result = filter.filter(sample_model_results)

        # Failed model should be kept, and overfit model should also be kept (gap < 30%)
        candidate_ids = [c.id for c in result.candidates]
        assert "model3" in candidate_ids
        assert "model4" in candidate_ids
        assert len(result.candidates) == 4  # All models kept
