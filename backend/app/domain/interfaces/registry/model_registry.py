"""
Model Registry interface — the domain contract for the plugin system.
Infrastructure provides the concrete implementation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.domain.value_objects.task_type import TaskType


@dataclass
class ModelConfig:
    """
    Describes a single algorithm configuration (a specific variant/set of hyperparameters).
    One plugin can expose many ModelConfigs.
    """

    name: str                            # unique identifier e.g. "rf_gini_100"
    display_name: str                    # human-readable e.g. "Random Forest (Gini, 100 trees)"
    algorithm_family: str                # e.g. "RandomForest"
    params: dict[str, Any] = field(default_factory=dict)
    task_types: list[TaskType] = field(default_factory=list)
    requires_scaling: bool = False
    supports_feature_importance: bool = False
    supports_shap: bool = False
    interpretability_score: int = 1      # 1 = black box, 5 = fully interpretable

    # Enhanced metadata for intelligent recommendation
    handles_missing_values: bool = False
    handles_nonlinear_data: bool = True
    suitable_for_large_datasets: bool = True
    suitable_for_small_datasets: bool = True
    supports_high_dimensional_data: bool = True
    training_complexity: str = "medium"  # low, medium, high
    prediction_complexity: str = "medium"  # low, medium, high
    memory_complexity: str = "medium"  # low, medium, high
    native_categorical_support: bool = False

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModelConfig):
            return NotImplemented
        return self.name == other.name


class IBaseModel(ABC):
    """
    Every model plugin must implement this interface.
    A plugin declares its configurations and knows how to build estimators.
    """

    @property
    @abstractmethod
    def configs(self) -> list[ModelConfig]:
        """Return all ModelConfig variants this plugin exposes."""
        ...

    @abstractmethod
    def build(self, config: ModelConfig) -> Any:
        """
        Instantiate and return a scikit-learn compatible estimator for the given config.
        The estimator must implement fit(), predict(), and score().
        """
        ...


class IModelRegistry(ABC):
    """Contract for the model plugin registry."""

    @abstractmethod
    def register(self, plugin: IBaseModel) -> None:
        """Register a model plugin. Called at application startup."""
        ...

    @abstractmethod
    def get_models_for_task(self, task: TaskType) -> list[ModelConfig]:
        """Return all registered ModelConfigs compatible with the given task."""
        ...

    @abstractmethod
    def build_estimator(self, config: ModelConfig) -> Any:
        """Build and return the estimator for a given ModelConfig."""
        ...

    @abstractmethod
    def get_config_by_name(self, name: str) -> ModelConfig | None:
        """Look up a ModelConfig by its unique name."""
        ...
