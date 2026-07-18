"""Random Forest classification plugin."""

from sklearn.ensemble import RandomForestClassifier

from app.domain.interfaces.registry.model_registry import IBaseModel, ModelConfig
from app.domain.value_objects.task_type import TaskType

_TASKS = [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]


class RandomForestPlugin(IBaseModel):
    @property
    def configs(self) -> list[ModelConfig]:
        return [
            ModelConfig(
                name="rf_gini_100",
                display_name="Random Forest (Gini, 100 trees)",
                algorithm_family="RandomForest",
                params={"n_estimators": 100, "criterion": "gini", "random_state": 42, "n_jobs": -1},
                task_types=_TASKS,
                requires_scaling=False,
                supports_feature_importance=True,
                supports_shap=True,
                interpretability_score=3,
            ),
            ModelConfig(
                name="rf_entropy_100",
                display_name="Random Forest (Entropy, 100 trees)",
                algorithm_family="RandomForest",
                params={
                    "n_estimators": 100,
                    "criterion": "entropy",
                    "random_state": 42,
                    "n_jobs": -1,
                },
                task_types=_TASKS,
                requires_scaling=False,
                supports_feature_importance=True,
                supports_shap=True,
                interpretability_score=3,
            ),
            ModelConfig(
                name="rf_gini_200",
                display_name="Random Forest (Gini, 200 trees)",
                algorithm_family="RandomForest",
                params={"n_estimators": 200, "criterion": "gini", "random_state": 42, "n_jobs": -1},
                task_types=_TASKS,
                requires_scaling=False,
                supports_feature_importance=True,
                supports_shap=True,
                interpretability_score=3,
            ),
        ]

    def build(self, config: ModelConfig) -> RandomForestClassifier:
        return RandomForestClassifier(**config.params)
