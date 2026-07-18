"""
Unit tests for GeneralizationAnalyzer component.
"""

import pytest

from app.core.recommendation_config import (
    GeneralizationLevel,
    GeneralizationThresholds,
)
from app.domain.entities.model_result import ModelResult
from app.infrastructure.ml.recommendation.generalization_analyzer import (
    GeneralizationAnalysis,
    GeneralizationAnalyzer,
    GeneralizationReport,
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
            metrics={"accuracy": 0.85},
            cv_score=0.84,
            cv_std=0.01,
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
            metrics={"accuracy": 0.80},
            cv_score=0.78,
            cv_std=0.02,
            is_overfitting=False,
            training_time_s=1.0,
            prediction_time_s=0.001,
            interpretability_score=4,
        ),
        ModelResult(
            id="model3",
            config_name="overfit_model",
            display_name="Overfit Model",
            algorithm_name="DecisionTree",
            metrics={"accuracy": 0.95},
            cv_score=0.70,
            cv_std=0.10,
            is_overfitting=True,
            training_time_s=0.5,
            prediction_time_s=0.001,
            interpretability_score=5,
        ),
    ]


class TestGeneralizationAnalyzer:
    """Test cases for GeneralizationAnalyzer."""

    def test_analyze_excellent_generalization(self, sample_model_results):
        """Test analysis of model with excellent generalization."""
        analyzer = GeneralizationAnalyzer()
        result = analyzer.analyze(sample_model_results)

        # model1 has 0.85 training vs 0.84 CV = 1.2% gap -> excellent
        analysis = result.analyses.get("model1")
        assert analysis is not None
        assert analysis.level == GeneralizationLevel.EXCELLENT
        assert analysis.normalized_score >= 90

    def test_analyze_good_generalization(self, sample_model_results):
        """Test analysis of model with good generalization."""
        analyzer = GeneralizationAnalyzer()
        result = analyzer.analyze(sample_model_results)

        # model2 has 0.80 training vs 0.78 CV = 2.5% gap -> excellent (threshold is 3%)
        analysis = result.analyses.get("model2")
        assert analysis is not None
        assert analysis.level == GeneralizationLevel.EXCELLENT
        assert analysis.normalized_score >= 90

    def test_analyze_high_generalization_gap(self, sample_model_results):
        """Test analysis of model with high generalization gap."""
        analyzer = GeneralizationAnalyzer()
        result = analyzer.analyze(sample_model_results)

        # model3 has 0.95 training vs 0.70 CV = 26% gap -> high
        analysis = result.analyses.get("model3")
        assert analysis is not None
        assert analysis.level == GeneralizationLevel.HIGH
        assert analysis.normalized_score < 50

    def test_gap_calculation(self, sample_model_results):
        """Test that gap is calculated correctly."""
        analyzer = GeneralizationAnalyzer()
        result = analyzer.analyze(sample_model_results)

        analysis = result.analyses.get("model1")
        assert analysis is not None
        assert analysis.gap == pytest.approx(0.01, abs=0.001)
        assert analysis.gap_pct == pytest.approx(1.2, abs=0.1)

    def test_identifies_best_generalized(self, sample_model_results):
        """Test that best generalized model is identified."""
        analyzer = GeneralizationAnalyzer()
        result = analyzer.analyze(sample_model_results)

        # model1 should be best generalized
        assert result.best_generalized == "model1"

    def test_identifies_worst_generalized(self, sample_model_results):
        """Test that worst generalized model is identified."""
        analyzer = GeneralizationAnalyzer()
        result = analyzer.analyze(sample_model_results)

        # model3 should be worst generalized
        assert result.worst_generalized == "model3"

    def test_skips_models_without_scores(self):
        """Test that models without required scores are skipped."""
        model_results = [
            ModelResult(
                id="model1",
                config_name="test",
                display_name="Test",
                algorithm_name="Test",
                metrics={},
                cv_score=None,
                cv_std=None,
                is_overfitting=False,
                training_time_s=0.0,
                prediction_time_s=0.0,
                interpretability_score=1,
            )
        ]

        analyzer = GeneralizationAnalyzer()
        result = analyzer.analyze(model_results)

        assert len(result.analyses) == 0

    def test_custom_thresholds(self, sample_model_results):
        """Test with custom generalization thresholds."""
        thresholds = GeneralizationThresholds(
            excellent_gap_pct=5.0,
            good_gap_pct=10.0,
            moderate_gap_pct=20.0,
        )
        analyzer = GeneralizationAnalyzer(thresholds)
        result = analyzer.analyze(sample_model_results)

        # With custom thresholds, model1 (1.2% gap) should still be excellent
        analysis = result.analyses.get("model1")
        assert analysis is not None
        assert analysis.level == GeneralizationLevel.EXCELLENT

    def test_empty_model_list(self):
        """Test analysis with empty model list."""
        analyzer = GeneralizationAnalyzer()
        result = analyzer.analyze([])

        assert len(result.analyses) == 0
        assert result.best_generalized is None
        assert result.worst_generalized is None
