"""Linear Discriminant Analysis classification plugin."""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from app.domain.interfaces.registry.model_registry import IBaseModel, ModelConfig
from app.domain.value_objects.task_type import TaskType

_TASKS = [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]


class LDAPlugin(IBaseModel):
    @property
    def configs(self) -> list[ModelConfig]:
        return [
            ModelConfig(
                name="lda_svd",
                display_name="Linear Discriminant Analysis (SVD)",
                algorithm_family="LDA",
                params={"solver": "svd"},
                task_types=_TASKS,
                requires_scaling=False,
                supports_feature_importance=True,
                supports_shap=True,
                interpretability_score=4,
            ),
            ModelConfig(
                name="lda_lsqr",
                display_name="Linear Discriminant Analysis (LSQR)",
                algorithm_family="LDA",
                params={"solver": "lsqr", "shrinkage": "auto"},
                task_types=_TASKS,
                requires_scaling=False,
                supports_feature_importance=True,
                supports_shap=True,
                interpretability_score=4,
            ),
        ]

    def build(self, config: ModelConfig) -> LinearDiscriminantAnalysis:
        return LinearDiscriminantAnalysis(**config.params)
