"""AdaBoost classification plugin."""

from sklearn.ensemble import AdaBoostClassifier

from app.domain.interfaces.registry.model_registry import IBaseModel, ModelConfig
from app.domain.value_objects.task_type import TaskType

_TASKS = [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]


class AdaBoostPlugin(IBaseModel):
    @property
    def configs(self) -> list[ModelConfig]:
        return [
            ModelConfig(
                name="ada_50",
                display_name="AdaBoost (50 estimators)",
                algorithm_family="AdaBoost",
                params={"n_estimators": 50, "learning_rate": 1.0, "random_state": 42},
                task_types=_TASKS,
                requires_scaling=False,
                supports_feature_importance=True,
                supports_shap=True,
                interpretability_score=2,
            ),
            ModelConfig(
                name="ada_100",
                display_name="AdaBoost (100 estimators)",
                algorithm_family="AdaBoost",
                params={"n_estimators": 100, "learning_rate": 1.0, "random_state": 42},
                task_types=_TASKS,
                requires_scaling=False,
                supports_feature_importance=True,
                supports_shap=True,
                interpretability_score=2,
            ),
            ModelConfig(
                name="ada_lr05",
                display_name="AdaBoost (100 estimators, lr=0.5)",
                algorithm_family="AdaBoost",
                params={"n_estimators": 100, "learning_rate": 0.5, "random_state": 42},
                task_types=_TASKS,
                requires_scaling=False,
                supports_feature_importance=True,
                supports_shap=True,
                interpretability_score=2,
            ),
        ]

    def build(self, config: ModelConfig) -> AdaBoostClassifier:
        return AdaBoostClassifier(**config.params)
