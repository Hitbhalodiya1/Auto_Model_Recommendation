"""
Unit tests for PrimaryMetricSelector component.
"""

import pytest

from app.domain.value_objects.task_type import TaskType
from app.infrastructure.ml.recommendation.primary_metric_selector import (
    MetricSelection,
    PrimaryMetricSelector,
)


class TestPrimaryMetricSelector:
    """Test cases for PrimaryMetricSelector."""

    def test_select_binary_classification_balanced(self):
        """Test metric selection for balanced binary classification."""
        selector = PrimaryMetricSelector()
        result = selector.select(
            task_type=TaskType.BINARY_CLASSIFICATION,
            is_imbalanced=False,
            available_metrics=["accuracy", "f1_score", "roc_auc"],
        )

        assert result.primary_metric == "accuracy"
        assert "balanced" in result.rationale.lower()

    def test_select_binary_classification_imbalanced(self):
        """Test metric selection for imbalanced binary classification."""
        selector = PrimaryMetricSelector()
        result = selector.select(
            task_type=TaskType.BINARY_CLASSIFICATION,
            is_imbalanced=True,
            available_metrics=["accuracy", "f1_score", "roc_auc"],
        )

        # Should prioritize F1 for imbalanced datasets
        assert result.primary_metric in ["f1_score", "f1_weighted"]
        assert "imbalanced" in result.rationale.lower()

    def test_select_regression(self):
        """Test metric selection for regression."""
        selector = PrimaryMetricSelector()
        result = selector.select(
            task_type=TaskType.REGRESSION,
            is_imbalanced=False,
            available_metrics=["r2_score", "rmse", "mae"],
        )

        assert result.primary_metric == "r2_score"
        assert "regression" in result.rationale.lower()

    def test_select_clustering(self):
        """Test metric selection for clustering."""
        selector = PrimaryMetricSelector()
        result = selector.select(
            task_type=TaskType.CLUSTERING,
            is_imbalanced=False,
            available_metrics=["silhouette_score", "calinski_harabasz"],
        )

        assert result.primary_metric == "silhouette_score"
        assert "clustering" in result.rationale.lower()

    def test_select_with_limited_metrics(self):
        """Test metric selection when only some metrics are available."""
        selector = PrimaryMetricSelector()
        result = selector.select(
            task_type=TaskType.BINARY_CLASSIFICATION,
            is_imbalanced=False,
            available_metrics=["f1_score"],  # Only F1 available
        )

        assert result.primary_metric == "f1_score"

    def test_select_with_no_available_metrics(self):
        """Test metric selection when no metrics are available."""
        selector = PrimaryMetricSelector()
        result = selector.select(
            task_type=TaskType.BINARY_CLASSIFICATION,
            is_imbalanced=False,
            available_metrics=[],
        )

        # Should default to accuracy
        assert result.primary_metric == "accuracy"
        assert "accuracy" in result.rationale.lower()

    def test_select_multiclass_classification(self):
        """Test metric selection for multiclass classification."""
        selector = PrimaryMetricSelector()
        result = selector.select(
            task_type=TaskType.MULTICLASS_CLASSIFICATION,
            is_imbalanced=True,
            available_metrics=["accuracy", "f1_weighted", "f1_score"],
        )

        # Should use weighted F1 for imbalanced multiclass
        assert result.primary_metric in ["f1_weighted", "f1_score"]

    def test_secondary_metrics_included(self):
        """Test that secondary metrics are included in selection."""
        selector = PrimaryMetricSelector()
        result = selector.select(
            task_type=TaskType.REGRESSION,
            is_imbalanced=False,
            available_metrics=["r2_score", "rmse", "mae", "mse"],
        )

        assert len(result.secondary_metrics) > 0
        assert "rmse" in result.secondary_metrics or "mae" in result.secondary_metrics
