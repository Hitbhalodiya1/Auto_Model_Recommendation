"""Decision Tree, Random Forest, KNN, SVR regression plugins."""

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from app.domain.interfaces.registry.model_registry import IBaseModel, ModelConfig
from app.domain.value_objects.task_type import TaskType

_TASKS = [TaskType.REGRESSION]


class DecisionTreeRegressorPlugin(IBaseModel):
    @property
    def configs(self) -> list[ModelConfig]:
        return [
            ModelConfig(
                name="dtr_mse",
                display_name="Decision Tree Regressor (MSE)",
                algorithm_family="DecisionTreeRegressor",
                params={"criterion": "squared_error", "random_state": 42},
                task_types=_TASKS,
                requires_scaling=False,
                supports_feature_importance=True,
                supports_shap=True,
                interpretability_score=5,
            ),
            ModelConfig(
                name="dtr_mae",
                display_name="Decision Tree Regressor (MAE)",
                algorithm_family="DecisionTreeRegressor",
                params={"criterion": "absolute_error", "random_state": 42},
                task_types=_TASKS,
                requires_scaling=False,
                supports_feature_importance=True,
                supports_shap=True,
                interpretability_score=5,
            ),
        ]

    def build(self, config: ModelConfig) -> DecisionTreeRegressor:
        return DecisionTreeRegressor(**config.params)


class RandomForestRegressorPlugin(IBaseModel):
    @property
    def configs(self) -> list[ModelConfig]:
        return [
            ModelConfig(
                name="rfr_100",
                display_name="Random Forest Regressor (100 trees)",
                algorithm_family="RandomForestRegressor",
                params={"n_estimators": 100, "random_state": 42, "n_jobs": -1},
                task_types=_TASKS,
                requires_scaling=False,
                supports_feature_importance=True,
                supports_shap=True,
                interpretability_score=3,
            ),
            ModelConfig(
                name="rfr_200",
                display_name="Random Forest Regressor (200 trees)",
                algorithm_family="RandomForestRegressor",
                params={"n_estimators": 200, "random_state": 42, "n_jobs": -1},
                task_types=_TASKS,
                requires_scaling=False,
                supports_feature_importance=True,
                supports_shap=True,
                interpretability_score=3,
            ),
        ]

    def build(self, config: ModelConfig) -> RandomForestRegressor:
        return RandomForestRegressor(**config.params)


class KNNRegressorPlugin(IBaseModel):
    @property
    def configs(self) -> list[ModelConfig]:
        configs = []
        for k in (3, 5, 7):
            for weights in ("uniform", "distance"):
                configs.append(ModelConfig(
                    name=f"knnr_k{k}_{weights}",
                    display_name=f"KNN Regressor (k={k}, {weights})",
                    algorithm_family="KNNRegressor",
                    params={"n_neighbors": k, "weights": weights},
                    task_types=_TASKS,
                    requires_scaling=True,
                    supports_feature_importance=False,
                    supports_shap=True,
                    interpretability_score=3,
                ))
        return configs

    def build(self, config: ModelConfig) -> KNeighborsRegressor:
        return KNeighborsRegressor(**config.params)


class SVRPlugin(IBaseModel):
    @property
    def configs(self) -> list[ModelConfig]:
        return [
            ModelConfig(
                name="svr_rbf",
                display_name="Support Vector Regressor (RBF)",
                algorithm_family="SVR",
                params={"kernel": "rbf", "C": 1.0, "gamma": "scale"},
                task_types=_TASKS,
                requires_scaling=True,
                supports_feature_importance=False,
                supports_shap=True,
                interpretability_score=1,
            ),
            ModelConfig(
                name="svr_linear",
                display_name="Support Vector Regressor (Linear)",
                algorithm_family="SVR",
                params={"kernel": "linear", "C": 1.0},
                task_types=_TASKS,
                requires_scaling=True,
                supports_feature_importance=True,
                supports_shap=True,
                interpretability_score=2,
            ),
        ]

    def build(self, config: ModelConfig) -> SVR:
        return SVR(**config.params)
