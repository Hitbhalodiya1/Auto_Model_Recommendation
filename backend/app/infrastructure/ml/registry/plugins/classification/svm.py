"""SVM classification plugin."""

from sklearn.svm import SVC

from app.domain.interfaces.registry.model_registry import IBaseModel, ModelConfig
from app.domain.value_objects.task_type import TaskType

_TASKS = [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]


class SVMPlugin(IBaseModel):
    @property
    def configs(self) -> list[ModelConfig]:
        return [
            ModelConfig(
                name="svm_linear",
                display_name="SVM (Linear kernel)",
                algorithm_family="SVM",
                params={"kernel": "linear", "C": 1.0, "probability": True, "random_state": 42},
                task_types=_TASKS,
                requires_scaling=True,
                supports_feature_importance=True,
                supports_shap=True,
                interpretability_score=2,
            ),
            ModelConfig(
                name="svm_rbf",
                display_name="SVM (RBF kernel)",
                algorithm_family="SVM",
                params={"kernel": "rbf", "C": 1.0, "gamma": "scale", "probability": True, "random_state": 42},
                task_types=_TASKS,
                requires_scaling=True,
                supports_feature_importance=False,
                supports_shap=True,
                interpretability_score=1,
            ),
            ModelConfig(
                name="svm_poly",
                display_name="SVM (Polynomial kernel)",
                algorithm_family="SVM",
                params={"kernel": "poly", "degree": 3, "C": 1.0, "probability": True, "random_state": 42},
                task_types=_TASKS,
                requires_scaling=True,
                supports_feature_importance=False,
                supports_shap=True,
                interpretability_score=1,
            ),
            ModelConfig(
                name="svm_sigmoid",
                display_name="SVM (Sigmoid kernel)",
                algorithm_family="SVM",
                params={"kernel": "sigmoid", "C": 1.0, "probability": True, "random_state": 42},
                task_types=_TASKS,
                requires_scaling=True,
                supports_feature_importance=False,
                supports_shap=True,
                interpretability_score=1,
            ),
        ]

    def build(self, config: ModelConfig) -> SVC:
        return SVC(**config.params)
