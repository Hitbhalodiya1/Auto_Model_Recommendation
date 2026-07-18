"""KNN classification plugin."""

from sklearn.neighbors import KNeighborsClassifier

from app.domain.interfaces.registry.model_registry import IBaseModel, ModelConfig
from app.domain.value_objects.task_type import TaskType

_TASKS = [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]


class KNNPlugin(IBaseModel):
    @property
    def configs(self) -> list[ModelConfig]:
        configs = []
        for k in (3, 5, 7):
            for weights in ("uniform", "distance"):
                for metric in ("euclidean", "manhattan"):
                    configs.append(ModelConfig(
                        name=f"knn_k{k}_{weights}_{metric}",
                        display_name=f"KNN (k={k}, {weights}, {metric})",
                        algorithm_family="KNN",
                        params={"n_neighbors": k, "weights": weights, "metric": metric},
                        task_types=_TASKS,
                        requires_scaling=True,
                        supports_feature_importance=False,
                        supports_shap=True,
                        interpretability_score=3,
                    ))
        return configs

    def build(self, config: ModelConfig) -> KNeighborsClassifier:
        return KNeighborsClassifier(**config.params)
