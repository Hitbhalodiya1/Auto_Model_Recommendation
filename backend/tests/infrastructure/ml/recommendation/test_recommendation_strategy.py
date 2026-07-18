"""
Unit tests for RecommendationStrategy component.
"""

import pytest

from app.core.recommendation_config import RecommendationMode
from app.domain.entities.model_result import ModelResult
from app.infrastructure.ml.recommendation.recommendation_scorer import ModelScores, ScoringReport
from app.infrastructure.ml.recommendation.recommendation_strategy import (
    MultiModeRecommendations,
    RecommendationCandidate,
    RecommendationStrategy,
    StrategyResult,
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
            config_name="dt_gini",
            display_name="Decision Tree",
            algorithm_name="DecisionTree",
            metrics={"accuracy": 0.78, "f1_score": 0.77},
            cv_score=0.72,
            cv_std=0.05,
            is_overfitting=True,
            training_time_s=0.5,
            prediction_time_s=0.0005,
            interpretability_score=5,
        ),
    ]


@pytest.fixture
def sample_scoring_report():
    """Create sample scoring report."""
    return ScoringReport(
        scores={
            "model1": ModelScores(
                model_id="model1",
                config_name="rf_100",
                predictive_performance=85.0,
                generalization=82.0,
                robustness=80.0,
                dataset_compatibility=75.0,
                interpretability=60.0,
                speed=70.0,
                resource_efficiency=60.0,
                overall_score=78.0,
                score_breakdown={},
            ),
            "model2": ModelScores(
                model_id="model2",
                config_name="lr_lbfgs",
                predictive_performance=80.0,
                generalization=85.0,
                robustness=85.0,
                dataset_compatibility=70.0,
                interpretability=80.0,
                speed=90.0,
                resource_efficiency=80.0,
                overall_score=82.0,
                score_breakdown={},
            ),
            "model3": ModelScores(
                model_id="model3",
                config_name="dt_gini",
                predictive_performance=78.0,
                generalization=60.0,
                robustness=50.0,
                dataset_compatibility=65.0,
                interpretability=100.0,
                speed=95.0,
                resource_efficiency=90.0,
                overall_score=70.0,
                score_breakdown={},
            ),
        },
        best_overall="model2",
        best_predictive="model1",
        best_generalized="model2",
        fastest="model3",
        most_explainable="model3",
    )


class TestRecommendationStrategy:
    """Test cases for RecommendationStrategy."""

    def test_generate_recommendations(
        self, sample_model_results, sample_scoring_report
    ):
        """Test generation of recommendations for all modes."""
        strategy = RecommendationStrategy()
        result = strategy.generate_recommendations(
            sample_model_results, sample_scoring_report
        )

        assert isinstance(result, MultiModeRecommendations)
        assert result.best_overall is not None
        assert result.best_predictive is not None
        assert result.fastest is not None
        assert result.most_explainable is not None

    def test_best_overall_recommendation(
        self, sample_model_results, sample_scoring_report
    ):
        """Test best overall recommendation."""
        strategy = RecommendationStrategy()
        result = strategy.generate_recommendations(
            sample_model_results, sample_scoring_report
        )

        assert result.best_overall.mode == RecommendationMode.BEST_OVERALL
        assert result.best_overall.selected is not None
        assert len(result.best_overall.candidates) > 0

    def test_best_predictive_recommendation(
        self, sample_model_results, sample_scoring_report
    ):
        """Test best predictive recommendation."""
        strategy = RecommendationStrategy()
        result = strategy.generate_recommendations(
            sample_model_results, sample_scoring_report
        )

        assert result.best_predictive.mode == RecommendationMode.BEST_PREDICTIVE
        assert result.best_predictive.selected is not None

    def test_fastest_recommendation(
        self, sample_model_results, sample_scoring_report
    ):
        """Test fastest recommendation."""
        strategy = RecommendationStrategy()
        result = strategy.generate_recommendations(
            sample_model_results, sample_scoring_report
        )

        assert result.fastest.mode == RecommendationMode.FASTEST
        assert result.fastest.selected is not None

    def test_most_explainable_recommendation(
        self, sample_model_results, sample_scoring_report
    ):
        """Test most explainable recommendation."""
        strategy = RecommendationStrategy()
        result = strategy.generate_recommendations(
            sample_model_results, sample_scoring_report
        )

        assert result.most_explainable.mode == RecommendationMode.MOST_EXPLAINABLE
        assert result.most_explainable.selected is not None

    def test_rationale_generation(
        self, sample_model_results, sample_scoring_report
    ):
        """Test that rationale is generated for recommendations."""
        strategy = RecommendationStrategy()
        result = strategy.generate_recommendations(
            sample_model_results, sample_scoring_report
        )

        # Check that rationale is generated
        assert len(result.best_overall.selected.rationale) > 0
        assert len(result.best_predictive.selected.rationale) > 0
        assert len(result.fastest.selected.rationale) > 0
        assert len(result.most_explainable.selected.rationale) > 0

    def test_fastest_requires_minimum_performance(
        self, sample_model_results, sample_scoring_report
    ):
        """Test that fastest recommendation requires minimum performance."""
        # Create a model with very low performance but high speed
        low_perf_model = ModelResult(
            id="model4",
            config_name="fast_but_bad",
            display_name="Fast But Bad",
            algorithm_name="Test",
            metrics={"accuracy": 0.30, "f1_score": 0.25},
            cv_score=0.28,
            cv_std=0.05,
            is_overfitting=False,
            training_time_s=0.1,
            prediction_time_s=0.0001,
            interpretability_score=2,
        )
        sample_model_results.append(low_perf_model)

        sample_scoring_report.scores["model4"] = ModelScores(
            model_id="model4",
            config_name="fast_but_bad",
            predictive_performance=30.0,  # Very low
            generalization=40.0,
            robustness=50.0,
            dataset_compatibility=50.0,
            interpretability=50.0,
            speed=100.0,  # Very fast
            resource_efficiency=80.0,
            overall_score=50.0,
            score_breakdown={},
        )

        strategy = RecommendationStrategy()
        result = strategy.generate_recommendations(
            sample_model_results, sample_scoring_report
        )

        # Fastest should not select the low performance model
        assert result.fastest.selected.model_id != "model4"

    def test_most_explainable_requires_minimum_performance(
        self, sample_model_results, sample_scoring_report
    ):
        """Test that most explainable recommendation requires minimum performance."""
        # Create a model with very low performance but high interpretability
        low_perf_model = ModelResult(
            id="model5",
            config_name="explainable_but_bad",
            display_name="Explainable But Bad",
            algorithm_name="Test",
            metrics={"accuracy": 0.35, "f1_score": 0.30},
            cv_score=0.32,
            cv_std=0.05,
            is_overfitting=False,
            training_time_s=0.5,
            prediction_time_s=0.01,
            interpretability_score=5,
        )
        sample_model_results.append(low_perf_model)

        sample_scoring_report.scores["model5"] = ModelScores(
            model_id="model5",
            config_name="explainable_but_bad",
            predictive_performance=35.0,
            generalization=40.0,
            robustness=50.0,
            dataset_compatibility=50.0,
            interpretability=100.0,  # Very explainable
            speed=70.0,
            resource_efficiency=80.0,
            overall_score=55.0,
            score_breakdown={},
        )

        strategy = RecommendationStrategy()
        result = strategy.generate_recommendations(
            sample_model_results, sample_scoring_report
        )

        # Most explainable should not select the low performance model
        assert result.most_explainable.selected.model_id != "model5"

    def test_empty_model_list(self):
        """Test with empty model list."""
        strategy = RecommendationStrategy()
        result = strategy.generate_recommendations([], ScoringReport(scores={}, best_overall=None, best_predictive=None, best_generalized=None, fastest=None, most_explainable=None))

        assert result.best_overall.selected is None
        assert result.best_predictive.selected is None
        assert result.fastest.selected is None
        assert result.most_explainable.selected is None
