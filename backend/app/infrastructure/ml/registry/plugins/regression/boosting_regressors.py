"""GradientBoosting, AdaBoost, XGBoost, MLP regression plugins."""

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from app.domain.interfaces.registry.model_registry import IBaseModel, ModelConfig
from app.domain.value_objects.task_type import TaskType

_TASKS = [TaskType.REGRESSION]


class GradientBoostingRegressorPlugin(IBaseModel):
    @property
    def configs(self) -> list[ModelConfig]:
        return [
            ModelConfig(
                name="gbr_default",
                display_name="Gradient Boosting Regressor (default)",
                algorithm_family="GradientBoostingRegressor",
                params={
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 3,
                    "random_state": 42,
                },
                task_types=_TASKS,
                requires_scaling=False,
                supports_feature_importance=True,
                supports_shap=True,
                interpretability_score=2,
            ),
            ModelConfig(
                name="gbr_lr005",
                display_name="Gradient Boosting Regressor (lr=0.05)",
                algorithm_family="GradientBoostingRegressor",
                params={
                    "n_estimators": 200,
                    "learning_rate": 0.05,
                    "max_depth": 3,
                    "random_state": 42,
                },
                task_types=_TASKS,
                requires_scaling=False,
                supports_feature_importance=True,
                supports_shap=True,
                interpretability_score=2,
            ),
        ]

    def build(self, config: ModelConfig) -> GradientBoostingRegressor:
        return GradientBoostingRegressor(**config.params)


class AdaBoostRegressorPlugin(IBaseModel):
    @property
    def configs(self) -> list[ModelConfig]:
        return [
            ModelConfig(
                name="adar_50",
                display_name="AdaBoost Regressor (50 estimators)",
                algorithm_family="AdaBoostRegressor",
                params={"n_estimators": 50, "learning_rate": 1.0, "random_state": 42},
                task_types=_TASKS,
                requires_scaling=False,
                supports_feature_importance=True,
                supports_shap=True,
                interpretability_score=2,
            ),
            ModelConfig(
                name="adar_100",
                display_name="AdaBoost Regressor (100 estimators)",
                algorithm_family="AdaBoostRegressor",
                params={"n_estimators": 100, "learning_rate": 1.0, "random_state": 42},
                task_types=_TASKS,
                requires_scaling=False,
                supports_feature_importance=True,
                supports_shap=True,
                interpretability_score=2,
            ),
        ]

    def build(self, config: ModelConfig) -> AdaBoostRegressor:
        return AdaBoostRegressor(**config.params)


class XGBoostRegressorPlugin(IBaseModel):
    @property
    def configs(self) -> list[ModelConfig]:
        return [
            ModelConfig(
                name="xgbr_default",
                display_name="XGBoost Regressor (default)",
                algorithm_family="XGBoostRegressor",
                params={
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 6,
                    "random_state": 42,
                    "verbosity": 0,
                },
                task_types=_TASKS,
                requires_scaling=False,
                supports_feature_importance=True,
                supports_shap=True,
                interpretability_score=2,
            ),
            ModelConfig(
                name="xgbr_regularized",
                display_name="XGBoost Regressor (regularized)",
                algorithm_family="XGBoostRegressor",
                params={
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 6,
                    "reg_alpha": 1.0,
                    "reg_lambda": 1.0,
                    "random_state": 42,
                    "verbosity": 0,
                },
                task_types=_TASKS,
                requires_scaling=False,
                supports_feature_importance=True,
                supports_shap=True,
                interpretability_score=2,
            ),
        ]

    def build(self, config: ModelConfig) -> XGBRegressor:
        return XGBRegressor(**config.params)


class MLPRegressorPlugin(IBaseModel):
    @property
    def configs(self) -> list[ModelConfig]:
        return [
            ModelConfig(
                name="mlpr_relu_adam",
                display_name="MLP Regressor (relu, adam, [100])",
                algorithm_family="MLPRegressor",
                params={
                    "hidden_layer_sizes": (100,),
                    "activation": "relu",
                    "solver": "adam",
                    "max_iter": 500,
                    "random_state": 42,
                },
                task_types=_TASKS,
                requires_scaling=True,
                supports_feature_importance=False,
                supports_shap=True,
                interpretability_score=1,
            ),
            ModelConfig(
                name="mlpr_relu_adam_deep",
                display_name="MLP Regressor (relu, adam, [100, 50])",
                algorithm_family="MLPRegressor",
                params={
                    "hidden_layer_sizes": (100, 50),
                    "activation": "relu",
                    "solver": "adam",
                    "max_iter": 500,
                    "random_state": 42,
                },
                task_types=_TASKS,
                requires_scaling=True,
                supports_feature_importance=False,
                supports_shap=True,
                interpretability_score=1,
            ),
        ]

    def build(self, config: ModelConfig) -> MLPRegressor:
        return MLPRegressor(**config.params)
