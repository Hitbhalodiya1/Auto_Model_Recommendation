import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression

from app.domain.entities.model_result import ModelResult
from app.domain.interfaces.registry.model_registry import ModelConfig
from app.domain.value_objects.task_type import TaskType
from app.infrastructure.ml.engines.explainability_engine import ExplainabilityEngine
from app.infrastructure.ml.engines.training_engine import TrainingEngine


class EmptyRegistry:
    def get_models_for_task(self, task):
        return []

    def build_estimator(self, config):
        raise RuntimeError("no models")


class FailingRegistry:
    def __init__(self, config):
        self._config = config

    def get_models_for_task(self, task):
        return [self._config]

    def build_estimator(self, config):
        raise RuntimeError("build failed")


class DummyRegistry:
    def __init__(self, config):
        self._config = config

    def get_models_for_task(self, task):
        return [self._config]

    def build_estimator(self, config):
        return DummyClassifier(strategy="most_frequent")


def test_training_engine_no_models_returns_empty():
    engine = TrainingEngine(EmptyRegistry())
    X_train = np.ones((10, 1))
    y_train = np.ones(10)
    X_test = np.ones((5, 1))
    y_test = np.ones(5)

    results = engine.train_all(
        X_train,
        y_train,
        X_test,
        y_test,
        TaskType.BINARY_CLASSIFICATION,
        experiment_id="exp-1",
    )

    assert results == []


def test_training_engine_handles_failed_build():
    config = ModelConfig(
        name="bad_model",
        display_name="Bad Model",
        algorithm_family="Bad",
        params={},
        task_types=[TaskType.BINARY_CLASSIFICATION],
    )
    engine = TrainingEngine(FailingRegistry(config))
    X_train = np.ones((10, 1))
    y_train = np.ones(10)
    X_test = np.ones((5, 1))
    y_test = np.ones(5)

    results = engine.train_all(
        X_train,
        y_train,
        X_test,
        y_test,
        TaskType.BINARY_CLASSIFICATION,
        experiment_id="exp-1",
    )

    assert len(results) == 1
    assert not results[0].succeeded
    assert results[0].error is not None


def test_training_engine_trains_dummy_estimator():
    config = ModelConfig(
        name="dummy_model",
        display_name="Dummy Model",
        algorithm_family="Dummy",
        params={},
        task_types=[TaskType.BINARY_CLASSIFICATION],
    )
    engine = TrainingEngine(DummyRegistry(config))
    X_train = np.vstack([np.zeros((10, 1)), np.ones((10, 1))])
    y_train = np.array([0] * 10 + [1] * 10)
    X_test = np.vstack([np.zeros((5, 1)), np.ones((5, 1))])
    y_test = np.array([0] * 5 + [1] * 5)

    results = engine.train_all(
        X_train,
        y_train,
        X_test,
        y_test,
        TaskType.BINARY_CLASSIFICATION,
        experiment_id="exp-2",
    )

    assert len(results) == 1
    assert results[0].succeeded
    assert results[0].predictions is not None
    assert results[0].train_score is not None


def test_explainability_engine_returns_feature_importance_and_shap():
    engine = ExplainabilityEngine()
    X_train = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y_train = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    X_test = np.array([[1.5], [2.5], [3.5]])

    estimator = LinearRegression()
    estimator.fit(X_train, y_train)

    model_result = ModelResult(
        experiment_id="exp-1",
        algorithm_name="LinearRegression",
        config_name="linreg",
        display_name="Linear Regression",
        configuration={},
        metrics={},
        cv_score=1.0,
        cv_std=0.0,
        is_overfitting=False,
        training_time_s=0.1,
        prediction_time_s=0.01,
        interpretability_score=5,
        supports_feature_importance=True,
        supports_shap=True,
    )

    result = engine.explain(
        estimator=estimator,
        X_train=X_train,
        X_test=X_test,
        feature_names=["x"],
        model_result=model_result,
        max_shap_samples=3,
    )

    assert len(result.feature_importances) == 1
    assert result.feature_importances[0].feature == "x"
    assert result.method_used in {"coefficients", "feature_importances", "permutation_importance"}
    assert result.shap_values is not None
    assert result.top_features == ["x"]
