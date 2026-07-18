"""
Unit tests for RecommendationScorer component.
"""

import pytest

from app.core.recommendation_config import RecommendationMode
from app.domain.entities.model_result import ModelResult
from app.infrastructure.ml.recommendation.dataset_compatibility_analyzer import (
    CompatibilityReport,
    CompatibilityScore,
)
from app.infrastructure.ml.recommendation.generalization_analyzer import (
    GeneralizationAnalysis,
    GeneralizationLevel,
    GeneralizationReport,
)
from app.infrastructure.ml.recommendation.recommendation_scorer import (
    ModelScores,
    RecommendationScorer,
    ScoringReport,
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
    ]


@pytest.fixture
def sample_generalization_report():
    """Create sample generalization report."""
    return GeneralizationReport(
        analyses={
            "model1": GeneralizationAnalysis(
                model_id="model1",
                config_name="rf_100",
                training_score=0.85,
                cv_score=0.82,
                cv_std=0.02,
                gap=0.03,
                gap_pct=3.5,
                level=GeneralizationLevel.GOOD,
                normalized_score=85.0,
            ),
            "model2": GeneralizationAnalysis(
                model_id="model2",
                config_name="lr_lbfgs",
                training_score=0.80,
                cv_score=0.78,
                cv_std=0.03,
                gap=0.02,
                gap_pct=2.5,
                level=GeneralizationLevel.EXCELLENT,
                normalized_score=92.0,
            ),
        },
        best_generalized="model2",
        worst_generalized="model1",
    )


@pytest.fixture
def sample_compatibility_report():
    """Create sample compatibility report."""
    return CompatibilityReport(
        scores={
            "model1": CompatibilityScore(
                model_id="model1",
                config_name="rf_100",
                algorithm_family="RandomForest",
                overall_score=75.0,
                breakdown={"size": 15.0, "features": 12.0, "nonlinearity": 15.0},
            ),
            "model2": CompatibilityScore(
                model_id="model2",
                config_name="lr_lbfgs",
                algorithm_family="LogisticRegression",
                overall_score=70.0,
                breakdown={"size": 14.0, "features": 10.0, "nonlinearity": 12.0},
            ),
        },
        best_compatible="model1",
    )


class TestRecommendationScorer:
    """Test cases for RecommendationScorer."""

    def test_score_models(
        self,
        sample_model_results,
        sample_generalization_report,
        sample_compatibility_report,
    ):
        """Test scoring of models."""
        scorer = RecommendationScorer()
        result = scorer.score(
            sample_model_results,
            sample_generalization_report,
            sample_compatibility_report,
            "accuracy",
        )

        assert len(result.scores) == 2
        assert "model1" in result.scores
        assert "model2" in result.scores

    def test_score_breakdown(
        self,
        sample_model_results,
        sample_generalization_report,
        sample_compatibility_report,
    ):
        """Test that score breakdown includes all dimensions."""
        scorer = RecommendationScorer()
        result = scorer.score(
            sample_model_results,
            sample_generalization_report,
            sample_compatibility_report,
            "accuracy",
        )

        scores = result.scores["model1"]
        assert scores.predictive_performance >= 0
        assert scores.generalization >= 0
        assert scores.robustness >= 0
        assert scores.dataset_compatibility >= 0
        assert scores.interpretability >= 0
        assert scores.speed >= 0
        assert scores.resource_efficiency >= 0

    def test_overall_score_calculation(
        self,
        sample_model_results,
        sample_generalization_report,
        sample_compatibility_report,
    ):
        """Test that overall score is calculated correctly."""
        scorer = RecommendationScorer()
        result = scorer.score(
            sample_model_results,
            sample_generalization_report,
            sample_compatibility_report,
            "accuracy",
        )

        scores = result.scores["model1"]
        # Overall score should be weighted combination
        assert 0 <= scores.overall_score <= 100

    def test_identifies_best_overall(
        self,
        sample_model_results,
        sample_generalization_report,
        sample_compatibility_report,
    ):
        """Test that best overall model is identified."""
        scorer = RecommendationScorer()
        result = scorer.score(
            sample_model_results,
            sample_generalization_report,
            sample_compatibility_report,
            "accuracy",
        )

        assert result.best_overall is not None
        assert result.best_overall in result.scores

    def test_identifies_best_predictive(
        self,
        sample_model_results,
        sample_generalization_report,
        sample_compatibility_report,
    ):
        """Test that best predictive model is identified."""
        scorer = RecommendationScorer()
        result = scorer.score(
            sample_model_results,
            sample_generalization_report,
            sample_compatibility_report,
            "accuracy",
        )

        assert result.best_predictive is not None
        assert result.best_predictive in result.scores

    def test_identifies_fastest(
        self,
        sample_model_results,
        sample_generalization_report,
        sample_compatibility_report,
    ):
        """Test that fastest model is identified."""
        scorer = RecommendationScorer()
        result = scorer.score(
            sample_model_results,
            sample_generalization_report,
            sample_compatibility_report,
            "accuracy",
        )

        assert result.fastest is not None
        assert result.fastest in result.scores

    def test_identifies_most_explainable(
        self,
        sample_model_results,
        sample_generalization_report,
        sample_compatibility_report,
    ):
        """Test that most explainable model is identified."""
        scorer = RecommendationScorer()
        result = scorer.score(
            sample_model_results,
            sample_generalization_report,
            sample_compatibility_report,
            "accuracy",
        )

        assert result.most_explainable is not None
        assert result.most_explainable in result.scores

    def test_different_recommendation_modes(
        self,
        sample_model_results,
        sample_generalization_report,
        sample_compatibility_report,
    ):
        """Test scoring with different recommendation modes."""
        scorer = RecommendationScorer()

        result_best_overall = scorer.score(
            sample_model_results,
            sample_generalization_report,
            sample_compatibility_report,
            "accuracy",
            mode=RecommendationMode.BEST_OVERALL,
        )

        result_fastest = scorer.score(
            sample_model_results,
            sample_generalization_report,
            sample_compatibility_report,
            "accuracy",
            mode=RecommendationMode.FASTEST,
        )

        # Scores should differ based on mode
        assert (
            result_best_overall.scores["model1"].overall_score
            != result_fastest.scores["model1"].overall_score
        )

    def test_skips_failed_models(self, sample_model_results):
        """Test that failed models are skipped."""
        # Add a failed model
        failed_model = ModelResult(
            id="model3",
            config_name="failed",
            display_name="Failed",
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
        sample_model_results.append(failed_model)

        scorer = RecommendationScorer()
        result = scorer.score(
            sample_model_results,
            GeneralizationReport(analyses={}, best_generalized=None, worst_generalized=None),
            CompatibilityReport(scores={}, best_compatible=None),
            "accuracy",
        )

        # Failed model should not be scored
        assert "model3" not in result.scores

    def test_empty_model_list(self):
        """Test scoring with empty model list."""
        scorer = RecommendationScorer()
        result = scorer.score(
            [],
            GeneralizationReport(analyses={}, best_generalized=None, worst_generalized=None),
            CompatibilityReport(scores={}, best_compatible=None),
            "accuracy",
        )

        assert len(result.scores) == 0
        assert result.best_overall is None
