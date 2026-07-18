"""
PrimaryMetricSelector - Automatically determines the primary evaluation metric.

For classification:
- Balanced datasets: Accuracy
- Imbalanced datasets: F1 Score, Balanced Accuracy, ROC-AUC (prioritized)

For regression:
- Primary: R²
- Secondary: RMSE, MAE

For clustering:
- Primary: Silhouette Score
"""

from dataclasses import dataclass
from enum import Enum

from app.core.logging import get_logger
from app.domain.value_objects.task_type import TaskType

logger = get_logger(__name__)


class MetricPriority(Enum):
    """Priority levels for metrics."""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"


@dataclass
class MetricDefinition:
    """Definition of an evaluation metric."""

    name: str
    display_name: str
    priority: MetricPriority
    higher_is_better: bool = True
    applicable_tasks: list[TaskType] | None = None


@dataclass
class MetricSelection:
    """Result of metric selection."""

    primary_metric: str
    secondary_metrics: list[str]
    rationale: str


class PrimaryMetricSelector:
    """
    Selects the primary evaluation metric based on task type and dataset characteristics.

    The selection is automatic and considers:
    - Task type (classification, regression, clustering)
    - Class balance (for classification)
    - Dataset size (can influence metric choice)
    """

    # Metric definitions by task type
    CLASSIFICATION_METRICS = [
        MetricDefinition("f1_weighted", "Weighted F1 Score", MetricPriority.PRIMARY),
        MetricDefinition("f1_score", "F1 Score", MetricPriority.PRIMARY),
        MetricDefinition("balanced_accuracy", "Balanced Accuracy", MetricPriority.PRIMARY),
        MetricDefinition("roc_auc", "ROC-AUC", MetricPriority.PRIMARY),
        MetricDefinition("accuracy", "Accuracy", MetricPriority.SECONDARY),
        MetricDefinition("precision", "Precision", MetricPriority.TERTIARY),
        MetricDefinition("recall", "Recall", MetricPriority.TERTIARY),
    ]

    REGRESSION_METRICS = [
        MetricDefinition("r2_score", "R² Score", MetricPriority.PRIMARY),
        MetricDefinition("rmse", "RMSE", MetricPriority.SECONDARY, higher_is_better=False),
        MetricDefinition("mae", "MAE", MetricPriority.TERTIARY, higher_is_better=False),
        MetricDefinition("mse", "MSE", MetricPriority.TERTIARY, higher_is_better=False),
    ]

    CLUSTERING_METRICS = [
        MetricDefinition("silhouette_score", "Silhouette Score", MetricPriority.PRIMARY),
        MetricDefinition("calinski_harabasz", "Calinski-Harabasz Index", MetricPriority.SECONDARY),
        MetricDefinition(
            "davies_bouldin",
            "Davies-Bouldin Index",
            MetricPriority.TERTIARY,
            higher_is_better=False,
        ),
    ]

    def __init__(self) -> None:
        self._metric_map = {
            TaskType.BINARY_CLASSIFICATION: self.CLASSIFICATION_METRICS,
            TaskType.MULTICLASS_CLASSIFICATION: self.CLASSIFICATION_METRICS,
            TaskType.REGRESSION: self.REGRESSION_METRICS,
            TaskType.CLUSTERING: self.CLUSTERING_METRICS,
        }

    def select(
        self,
        task_type: TaskType,
        is_imbalanced: bool = False,
        available_metrics: list[str] | None = None,
    ) -> MetricSelection:
        """
        Select the primary metric for the given task and dataset characteristics.

        Args:
            task_type: The type of ML task
            is_imbalanced: Whether the dataset has class imbalance (for classification)
            available_metrics: List of metrics actually available in the results

        Returns:
            MetricSelection with primary metric and rationale
        """
        metrics = self._metric_map.get(task_type, self.CLASSIFICATION_METRICS)

        if available_metrics:
            # Filter to only available metrics
            metrics = [m for m in metrics if m.name in available_metrics]

        if not metrics:
            logger.warning("no_available_metrics", task_type=task_type)
            return MetricSelection(
                primary_metric="accuracy",
                secondary_metrics=[],
                rationale="No metrics available, defaulting to accuracy",
            )

        # For classification, adjust based on imbalance
        if task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]:
            return self._select_classification_metric(metrics, is_imbalanced)
        elif task_type == TaskType.REGRESSION:
            return self._select_regression_metric(metrics)
        elif task_type == TaskType.CLUSTERING:
            return self._select_clustering_metric(metrics)
        else:
            # Default to first available
            primary = metrics[0].name
            secondary = [m.name for m in metrics[1:4] if m.priority != MetricPriority.PRIMARY]
            return MetricSelection(
                primary_metric=primary,
                secondary_metrics=secondary,
                rationale=f"Default selection for {task_type.value}",
            )

    def _select_classification_metric(
        self, metrics: list[MetricDefinition], is_imbalanced: bool
    ) -> MetricSelection:
        """Select metric for classification tasks."""
        if is_imbalanced:
            # Prioritize F1, Balanced Accuracy, ROC-AUC for imbalanced datasets
            priority_metrics = [m for m in metrics if m.priority == MetricPriority.PRIMARY]
            if priority_metrics:
                primary = priority_metrics[0].name
                secondary = [m.name for m in priority_metrics[1:3]]
                rationale = (
                    f"Dataset is imbalanced. Using {primary} as primary metric "
                    f"to better handle class imbalance."
                )
            else:
                primary = metrics[0].name
                secondary = [m.name for m in metrics[1:3]]
                rationale = "Using available primary metric for imbalanced dataset"
        else:
            # For balanced datasets, prefer accuracy if available
            accuracy = next((m for m in metrics if m.name == "accuracy"), None)
            if accuracy:
                primary = accuracy.name
                secondary = [m.name for m in metrics if m.name != "accuracy"][:3]
                rationale = "Dataset is balanced. Using accuracy as primary metric."
            else:
                # Fall back to first primary metric
                primary = metrics[0].name
                secondary = [m.name for m in metrics[1:4]]
                rationale = "Using best available metric for balanced dataset"

        return MetricSelection(
            primary_metric=primary,
            secondary_metrics=secondary,
            rationale=rationale,
        )

    def _select_regression_metric(self, metrics: list[MetricDefinition]) -> MetricSelection:
        """Select metric for regression tasks."""
        primary = metrics[0].name  # R² should be first
        secondary = [m.name for m in metrics[1:3]]
        rationale = "Using R² as primary metric for regression task."

        return MetricSelection(
            primary_metric=primary,
            secondary_metrics=secondary,
            rationale=rationale,
        )

    def _select_clustering_metric(self, metrics: list[MetricDefinition]) -> MetricSelection:
        """Select metric for clustering tasks."""
        primary = metrics[0].name  # Silhouette should be first
        secondary = [m.name for m in metrics[1:3]]
        rationale = "Using Silhouette Score as primary metric for clustering task."

        return MetricSelection(
            primary_metric=primary,
            secondary_metrics=secondary,
            rationale=rationale,
        )
