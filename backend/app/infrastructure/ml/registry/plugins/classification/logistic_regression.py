"""Logistic Regression classification plugin."""

from sklearn.linear_model import LogisticRegression

from app.domain.interfaces.registry.model_registry import IBaseModel, ModelConfig
from app.domain.value_objects.task_type import TaskType

_TASKS = [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]


class LogisticRegressionPlugin(IBaseModel):
    @property
    def configs(self) -> list[ModelConfig]:
        return [
            ModelConfig(
                name="lr_lbfgs_l2",
                display_name="Logistic Regression (lbfgs, L2)",
                algorithm_family="LogisticRegression",
                params={
                    "solver": "lbfgs",
                    "penalty": "l2",
                    "C": 1.0,
                    "max_iter": 1000,
                    "random_state": 42,
                },
                task_types=_TASKS,
                requires_scaling=True,
                supports_feature_importance=True,
                supports_shap=True,
                interpretability_score=5,
            ),
            ModelConfig(
                name="lr_saga_l1",
                display_name="Logistic Regression (saga, L1)",
                algorithm_family="LogisticRegression",
                params={
                    "solver": "saga",
                    "penalty": "l1",
                    "C": 1.0,
                    "max_iter": 1000,
                    "random_state": 42,
                },
                task_types=_TASKS,
                requires_scaling=True,
                supports_feature_importance=True,
                supports_shap=True,
                interpretability_score=5,
            ),
            ModelConfig(
                name="lr_saga_elasticnet",
                display_name="Logistic Regression (saga, ElasticNet)",
                algorithm_family="LogisticRegression",
                params={
                    "solver": "saga",
                    "penalty": "elasticnet",
                    "l1_ratio": 0.5,
                    "C": 1.0,
                    "max_iter": 1000,
                    "random_state": 42,
                },
                task_types=_TASKS,
                requires_scaling=True,
                supports_feature_importance=True,
                supports_shap=True,
                interpretability_score=5,
            ),
        ]

    def build(self, config: ModelConfig) -> LogisticRegression:
        return LogisticRegression(**config.params)
