"""Naive Bayes (Gaussian and Bernoulli) classification plugin."""

from sklearn.naive_bayes import BernoulliNB, GaussianNB

from app.domain.interfaces.registry.model_registry import IBaseModel, ModelConfig
from app.domain.value_objects.task_type import TaskType

_TASKS = [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]


class NaiveBayesPlugin(IBaseModel):
    @property
    def configs(self) -> list[ModelConfig]:
        return [
            ModelConfig(
                name="gnb",
                display_name="Gaussian Naive Bayes",
                algorithm_family="NaiveBayes",
                params={},
                task_types=_TASKS,
                requires_scaling=False,
                supports_feature_importance=False,
                supports_shap=True,
                interpretability_score=4,
            ),
            ModelConfig(
                name="bnb",
                display_name="Bernoulli Naive Bayes",
                algorithm_family="NaiveBayes",
                params={"alpha": 1.0},
                task_types=_TASKS,
                requires_scaling=False,
                supports_feature_importance=False,
                supports_shap=True,
                interpretability_score=4,
            ),
        ]

    def build(self, config: ModelConfig):
        if config.name == "gnb":
            return GaussianNB(**config.params)
        return BernoulliNB(**config.params)
