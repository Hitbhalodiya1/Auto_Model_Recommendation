"""Multi-Layer Perceptron classification plugin."""

from sklearn.neural_network import MLPClassifier

from app.domain.interfaces.registry.model_registry import IBaseModel, ModelConfig
from app.domain.value_objects.task_type import TaskType

_TASKS = [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]


class MLPClassifierPlugin(IBaseModel):
    @property
    def configs(self) -> list[ModelConfig]:
        return [
            ModelConfig(
                name="mlp_relu_adam",
                display_name="MLP (relu, adam, [100])",
                algorithm_family="MLP",
                params={"hidden_layer_sizes": (100,), "activation": "relu", "solver": "adam", "max_iter": 500, "random_state": 42},
                task_types=_TASKS,
                requires_scaling=True,
                supports_feature_importance=False,
                supports_shap=True,
                interpretability_score=1,
            ),
            ModelConfig(
                name="mlp_tanh_adam",
                display_name="MLP (tanh, adam, [100])",
                algorithm_family="MLP",
                params={"hidden_layer_sizes": (100,), "activation": "tanh", "solver": "adam", "max_iter": 500, "random_state": 42},
                task_types=_TASKS,
                requires_scaling=True,
                supports_feature_importance=False,
                supports_shap=True,
                interpretability_score=1,
            ),
            ModelConfig(
                name="mlp_relu_adam_deep",
                display_name="MLP (relu, adam, [100, 50])",
                algorithm_family="MLP",
                params={"hidden_layer_sizes": (100, 50), "activation": "relu", "solver": "adam", "max_iter": 500, "random_state": 42},
                task_types=_TASKS,
                requires_scaling=True,
                supports_feature_importance=False,
                supports_shap=True,
                interpretability_score=1,
            ),
            ModelConfig(
                name="mlp_relu_sgd",
                display_name="MLP (relu, sgd, [100])",
                algorithm_family="MLP",
                params={"hidden_layer_sizes": (100,), "activation": "relu", "solver": "sgd", "max_iter": 500, "random_state": 42},
                task_types=_TASKS,
                requires_scaling=True,
                supports_feature_importance=False,
                supports_shap=True,
                interpretability_score=1,
            ),
        ]

    def build(self, config: ModelConfig) -> MLPClassifier:
        return MLPClassifier(**config.params)
