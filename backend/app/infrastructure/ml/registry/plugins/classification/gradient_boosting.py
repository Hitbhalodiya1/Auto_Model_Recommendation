"""Gradient Boosting classification plugin."""

from sklearn.ensemble import GradientBoostingClassifier

from app.domain.interfaces.registry.model_registry import IBaseModel, ModelConfig
from app.domain.value_objects.task_type import TaskType

_TASKS = [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]


class GradientBoostingPlugin(IBaseModel):
    @property
    def configs(self) -> list[ModelConfig]:
        return [
            ModelConfig(
                name="gb_default",
                display_name="Gradient Boosting (lr=0.1, 100 estimators)",
                algorithm_family="GradientBoosting",
                params={"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": 42},
                task_types=_TASKS,
                requires_scaling=False,
                supports_feature_importance=True,
                supports_shap=True,
                interpretability_score=2,
            ),
            ModelConfig(
                name="gb_lr005",
                display_name="Gradient Boosting (lr=0.05, 200 estimators)",
                algorithm_family="GradientBoosting",
                params={"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3, "random_state": 42},
                task_types=_TASKS,
                requires_scaling=False,
                supports_feature_importance=True,
                supports_shap=True,
                interpretability_score=2,
            ),
            ModelConfig(
                name="gb_deep",
                display_name="Gradient Boosting (lr=0.1, depth=5)",
                algorithm_family="GradientBoosting",
                params={"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5, "random_state": 42},
                task_types=_TASKS,
                requires_scaling=False,
                supports_feature_importance=True,
                supports_shap=True,
                interpretability_score=2,
            ),
        ]

    def build(self, config: ModelConfig) -> GradientBoostingClassifier:
        return GradientBoostingClassifier(**config.params)
