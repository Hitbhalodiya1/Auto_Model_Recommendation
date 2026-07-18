"""Linear, Ridge, Lasso, ElasticNet regression plugins."""

from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

from app.domain.interfaces.registry.model_registry import IBaseModel, ModelConfig
from app.domain.value_objects.task_type import TaskType

_TASKS = [TaskType.REGRESSION]


class LinearRegressionPlugin(IBaseModel):
    @property
    def configs(self) -> list[ModelConfig]:
        return [
            ModelConfig(
                name="linreg",
                display_name="Linear Regression (OLS)",
                algorithm_family="LinearRegression",
                params={},
                task_types=_TASKS,
                requires_scaling=True,
                supports_feature_importance=True,
                supports_shap=True,
                interpretability_score=5,
            ),
        ]

    def build(self, config: ModelConfig) -> LinearRegression:
        return LinearRegression(**config.params)


class RidgePlugin(IBaseModel):
    @property
    def configs(self) -> list[ModelConfig]:
        return [
            ModelConfig(
                name="ridge_1",
                display_name="Ridge Regression (alpha=1.0)",
                algorithm_family="Ridge",
                params={"alpha": 1.0, "random_state": 42},
                task_types=_TASKS,
                requires_scaling=True,
                supports_feature_importance=True,
                supports_shap=True,
                interpretability_score=5,
            ),
            ModelConfig(
                name="ridge_10",
                display_name="Ridge Regression (alpha=10.0)",
                algorithm_family="Ridge",
                params={"alpha": 10.0, "random_state": 42},
                task_types=_TASKS,
                requires_scaling=True,
                supports_feature_importance=True,
                supports_shap=True,
                interpretability_score=5,
            ),
        ]

    def build(self, config: ModelConfig) -> Ridge:
        return Ridge(**config.params)


class LassoPlugin(IBaseModel):
    @property
    def configs(self) -> list[ModelConfig]:
        return [
            ModelConfig(
                name="lasso_1",
                display_name="Lasso Regression (alpha=1.0)",
                algorithm_family="Lasso",
                params={"alpha": 1.0, "random_state": 42, "max_iter": 2000},
                task_types=_TASKS,
                requires_scaling=True,
                supports_feature_importance=True,
                supports_shap=True,
                interpretability_score=5,
            ),
            ModelConfig(
                name="lasso_01",
                display_name="Lasso Regression (alpha=0.1)",
                algorithm_family="Lasso",
                params={"alpha": 0.1, "random_state": 42, "max_iter": 2000},
                task_types=_TASKS,
                requires_scaling=True,
                supports_feature_importance=True,
                supports_shap=True,
                interpretability_score=5,
            ),
        ]

    def build(self, config: ModelConfig) -> Lasso:
        return Lasso(**config.params)


class ElasticNetPlugin(IBaseModel):
    @property
    def configs(self) -> list[ModelConfig]:
        return [
            ModelConfig(
                name="enet_5050",
                display_name="ElasticNet (l1_ratio=0.5)",
                algorithm_family="ElasticNet",
                params={"alpha": 1.0, "l1_ratio": 0.5, "random_state": 42, "max_iter": 2000},
                task_types=_TASKS,
                requires_scaling=True,
                supports_feature_importance=True,
                supports_shap=True,
                interpretability_score=5,
            ),
            ModelConfig(
                name="enet_l1heavy",
                display_name="ElasticNet (l1_ratio=0.8)",
                algorithm_family="ElasticNet",
                params={"alpha": 1.0, "l1_ratio": 0.8, "random_state": 42, "max_iter": 2000},
                task_types=_TASKS,
                requires_scaling=True,
                supports_feature_importance=True,
                supports_shap=True,
                interpretability_score=5,
            ),
        ]

    def build(self, config: ModelConfig) -> ElasticNet:
        return ElasticNet(**config.params)
