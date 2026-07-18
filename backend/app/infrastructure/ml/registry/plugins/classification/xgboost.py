"""XGBoost classification plugin."""

from xgboost import XGBClassifier

from app.domain.interfaces.registry.model_registry import IBaseModel, ModelConfig
from app.domain.value_objects.task_type import TaskType

_TASKS = [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]


class XGBoostPlugin(IBaseModel):
    @property
    def configs(self) -> list[ModelConfig]:
        return [
            ModelConfig(
                name="xgb_default",
                display_name="XGBoost (default)",
                algorithm_family="XGBoost",
                params={
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 6,
                    "random_state": 42,
                    "eval_metric": "logloss",
                    "verbosity": 0,
                },
                task_types=_TASKS,
                requires_scaling=False,
                supports_feature_importance=True,
                supports_shap=True,
                interpretability_score=2,
            ),
            ModelConfig(
                name="xgb_deep",
                display_name="XGBoost (depth=8, 200 estimators)",
                algorithm_family="XGBoost",
                params={
                    "n_estimators": 200,
                    "learning_rate": 0.05,
                    "max_depth": 8,
                    "random_state": 42,
                    "eval_metric": "logloss",
                    "verbosity": 0,
                },
                task_types=_TASKS,
                requires_scaling=False,
                supports_feature_importance=True,
                supports_shap=True,
                interpretability_score=2,
            ),
            ModelConfig(
                name="xgb_regularized",
                display_name="XGBoost (regularized, alpha=1)",
                algorithm_family="XGBoost",
                params={
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 6,
                    "reg_alpha": 1.0,
                    "reg_lambda": 1.0,
                    "random_state": 42,
                    "eval_metric": "logloss",
                    "verbosity": 0,
                },
                task_types=_TASKS,
                requires_scaling=False,
                supports_feature_importance=True,
                supports_shap=True,
                interpretability_score=2,
            ),
        ]

    def build(self, config: ModelConfig) -> XGBClassifier:
        return XGBClassifier(**config.params)
