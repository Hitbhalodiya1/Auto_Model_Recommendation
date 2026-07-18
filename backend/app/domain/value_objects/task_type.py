"""
TaskType value object — represents the detected ML task.
"""

from enum import StrEnum


class TaskType(StrEnum):
    """The machine learning task detected from dataset characteristics."""

    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"

    @property
    def is_classification(self) -> bool:
        return self in (
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
        )

    @property
    def is_supervised(self) -> bool:
        return self != TaskType.CLUSTERING

    @property
    def display_name(self) -> str:
        return {
            TaskType.BINARY_CLASSIFICATION: "Binary Classification",
            TaskType.MULTICLASS_CLASSIFICATION: "Multi-Class Classification",
            TaskType.REGRESSION: "Regression",
            TaskType.CLUSTERING: "Clustering",
        }[self]
