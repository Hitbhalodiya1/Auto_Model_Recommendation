"""
Unit tests for DatasetCompatibilityAnalyzer component.
"""

import pytest

from app.domain.interfaces.registry.model_registry import ModelConfig
from app.domain.value_objects.task_type import TaskType
from app.infrastructure.ml.recommendation.dataset_compatibility_analyzer import (
    CompatibilityReport,
    DatasetCompatibilityAnalyzer,
)


@pytest.fixture
def sample_model_configs():
    """Create sample model configs for testing."""
    return {
        "model1": ModelConfig(
            name="rf_100",
            display_name="Random Forest",
            algorithm_family="RandomForest",
            params={},
            task_types=[TaskType.BINARY_CLASSIFICATION],
            handles_missing_values=False,
            handles_nonlinear_data=True,
            suitable_for_large_datasets=True,
            suitable_for_small_datasets=True,
            supports_high_dimensional_data=True,
            training_complexity="medium",
            prediction_complexity="medium",
            memory_complexity="medium",
            native_categorical_support=False,
        ),
        "model2": ModelConfig(
            name="lr_lbfgs",
            display_name="Logistic Regression",
            algorithm_family="LogisticRegression",
            params={},
            task_types=[TaskType.BINARY_CLASSIFICATION],
            handles_missing_values=False,
            handles_nonlinear_data=False,
            suitable_for_large_datasets=True,
            suitable_for_small_datasets=True,
            supports_high_dimensional_data=True,
            training_complexity="low",
            prediction_complexity="low",
            memory_complexity="low",
            native_categorical_support=False,
        ),
    }


@pytest.fixture
def sample_dataset_analysis():
    """Create sample dataset analysis for testing."""
    return {
        "row_count": 5000,
        "column_count": 20,
        "missing_value_pct": 0.05,
        "is_imbalanced": True,
        "column_profiles": [
            {
                "name": "feature1",
                "dtype": "float64",
                "feature_type": "numeric",
                "null_count": 10,
                "null_pct": 0.02,
                "unique_count": 100,
                "unique_pct": 0.02,
            },
            {
                "name": "feature2",
                "dtype": "object",
                "feature_type": "categorical",
                "null_count": 5,
                "null_pct": 0.01,
                "unique_count": 5,
                "unique_pct": 0.01,
            },
        ],
        "correlation_matrix": {},
        "outlier_counts": {"feature1": 50},
    }


class TestDatasetCompatibilityAnalyzer:
    """Test cases for DatasetCompatibilityAnalyzer."""

    def test_analyze_small_dataset(self, sample_model_configs):
        """Test compatibility analysis for small dataset."""
        analyzer = DatasetCompatibilityAnalyzer()
        dataset_analysis = {
            "row_count": 500,
            "column_count": 10,
            "missing_value_pct": 0.0,
            "is_imbalanced": False,
            "column_profiles": [],
            "correlation_matrix": {},
            "outlier_counts": {},
        }

        result = analyzer.analyze(sample_model_configs, dataset_analysis)

        assert len(result.scores) == 2
        assert "model1" in result.scores
        assert "model2" in result.scores

    def test_analyze_large_dataset(self, sample_model_configs):
        """Test compatibility analysis for large dataset."""
        analyzer = DatasetCompatibilityAnalyzer()
        dataset_analysis = {
            "row_count": 200000,
            "column_count": 50,
            "missing_value_pct": 0.1,
            "is_imbalanced": False,
            "column_profiles": [],
            "correlation_matrix": {},
            "outlier_counts": {},
        }

        result = analyzer.analyze(sample_model_configs, dataset_analysis)

        assert len(result.scores) == 2

    def test_analyze_high_dimensional(self, sample_model_configs):
        """Test compatibility analysis for high-dimensional dataset."""
        analyzer = DatasetCompatibilityAnalyzer()
        dataset_analysis = {
            "row_count": 10000,
            "column_count": 150,
            "missing_value_pct": 0.0,
            "is_imbalanced": False,
            "column_profiles": [],
            "correlation_matrix": {},
            "outlier_counts": {},
        }

        result = analyzer.analyze(sample_model_configs, dataset_analysis)

        # Models that support high dimensional data should score better
        assert result.scores["model1"].overall_score >= result.scores["model2"].overall_score

    def test_analyze_imbalanced_dataset(self, sample_model_configs):
        """Test compatibility analysis for imbalanced dataset."""
        analyzer = DatasetCompatibilityAnalyzer()
        dataset_analysis = {
            "row_count": 5000,
            "column_count": 20,
            "missing_value_pct": 0.0,
            "is_imbalanced": True,
            "column_profiles": [],
            "correlation_matrix": {},
            "outlier_counts": {},
        }

        result = analyzer.analyze(sample_model_configs, dataset_analysis)

        # RandomForest should handle imbalance better
        assert result.scores["model1"].overall_score >= 70

    def test_identifies_best_compatible(self, sample_model_configs, sample_dataset_analysis):
        """Test that best compatible model is identified."""
        analyzer = DatasetCompatibilityAnalyzer()
        result = analyzer.analyze(sample_model_configs, sample_dataset_analysis)

        assert result.best_compatible is not None
        # best_compatible should be the model config key, not the name
        assert result.best_compatible in sample_model_configs

    def test_score_breakdown(self, sample_model_configs, sample_dataset_analysis):
        """Test that score breakdown includes all components."""
        analyzer = DatasetCompatibilityAnalyzer()
        result = analyzer.analyze(sample_model_configs, sample_dataset_analysis)

        score = result.scores["model1"]
        assert "size" in score.breakdown
        assert "features" in score.breakdown
        assert "nonlinearity" in score.breakdown
        assert "dimensionality" in score.breakdown
        assert "imbalance" in score.breakdown
        assert "noise" in score.breakdown

    def test_empty_model_configs(self):
        """Test with empty model configs."""
        analyzer = DatasetCompatibilityAnalyzer()
        dataset_analysis = {
            "row_count": 1000,
            "column_count": 10,
            "missing_value_pct": 0.0,
            "is_imbalanced": False,
            "column_profiles": [],
            "correlation_matrix": {},
            "outlier_counts": {},
        }

        result = analyzer.analyze({}, dataset_analysis)

        assert len(result.scores) == 0
        assert result.best_compatible is None

    def test_custom_weights(self, sample_model_configs, sample_dataset_analysis):
        """Test with custom compatibility weights."""
        from app.core.recommendation_config import DatasetCompatibilityWeights

        weights = DatasetCompatibilityWeights(size_weight=0.5, feature_count_weight=0.1)
        analyzer = DatasetCompatibilityAnalyzer(weights)
        result = analyzer.analyze(sample_model_configs, sample_dataset_analysis)

        assert len(result.scores) == 2
