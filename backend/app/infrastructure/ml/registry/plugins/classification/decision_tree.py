"""Decision Tree classification plugin."""

from sklearn.tree import DecisionTreeClassifier

from app.domain.interfaces.registry.model_registry import IBaseModel, ModelConfig
from app.domain.value_objects.task_type import TaskType

_TASKS = [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]


class DecisionTreePlugin(IBaseModel):
    @property
    def configs(self) -> list[ModelConfig]:
        return [
            ModelConfig(
                name="dt_gini",
                display_name="Decision Tree (Gini)",
                algorithm_family="DecisionTree",
                params={"criterion": "gini", "random_state": 42},
                task_types=_TASKS,
                requires_scaling=False,
                supports_feature_importance=True,
                supports_shap=True,
                interpretability_score=5,
            ),
            ModelConfig(
                name="dt_entropy",
                display_name="Decision Tree (Entropy)",
                algorithm_family="DecisionTree",
                params={"criterion": "entropy", "random_state": 42},
                task_types=_TASKS,
                requires_scaling=False,
                supports_feature_importance=True,
                supports_shap=True,
                interpretability_score=5,
            ),
            ModelConfig(
                name="dt_gini_depth5",
                display_name="Decision Tree (Gini, max_depth=5)",
                algorithm_family="DecisionTree",
                params={"criterion": "gini", "max_depth": 5, "random_state": 42},
                task_types=_TASKS,
                requires_scaling=False,
                supports_feature_importance=True,
                supports_shap=True,
                interpretability_score=5,
            ),
        ]

    def build(self, config: ModelConfig) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(**config.params)
