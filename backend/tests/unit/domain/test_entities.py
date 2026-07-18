"""
Unit tests for domain entities, value objects, and exceptions.
Pure Python — no I/O or infrastructure dependencies.
"""

import pytest

from app.domain.entities.dataset import Dataset, DatasetStatus
from app.domain.entities.experiment import Experiment, ExperimentStatus
from app.domain.exceptions.domain_exceptions import (
    DatasetNotFoundError,
    ExperimentNotFoundError,
    FileTooLargeError,
    InvalidFileTypeError,
)
from app.domain.value_objects.feature_type import FeatureType
from app.domain.value_objects.metric import Metric, MetricSet
from app.domain.value_objects.task_type import TaskType

# ── TaskType ──────────────────────────────────────────────────────────────────

class TestTaskType:
    def test_binary_is_classification(self):
        assert TaskType.BINARY_CLASSIFICATION.is_classification is True

    def test_multiclass_is_classification(self):
        assert TaskType.MULTICLASS_CLASSIFICATION.is_classification is True

    def test_regression_not_classification(self):
        assert TaskType.REGRESSION.is_classification is False

    def test_regression_is_supervised(self):
        assert TaskType.REGRESSION.is_supervised is True

    def test_clustering_not_supervised(self):
        assert TaskType.CLUSTERING.is_supervised is False

    def test_binary_display_name_contains_binary(self):
        assert "Binary" in TaskType.BINARY_CLASSIFICATION.display_name

    def test_regression_display_name_contains_regression(self):
        assert "Regression" in TaskType.REGRESSION.display_name

    def test_string_value(self):
        assert TaskType.REGRESSION == "regression"


# ── FeatureType ───────────────────────────────────────────────────────────────

class TestFeatureType:
    def test_numeric_continuous_is_numeric(self):
        assert FeatureType.NUMERIC_CONTINUOUS.is_numeric is True

    def test_numeric_discrete_is_numeric(self):
        assert FeatureType.NUMERIC_DISCRETE.is_numeric is True

    def test_categorical_is_categorical(self):
        assert FeatureType.CATEGORICAL_NOMINAL.is_categorical is True

    def test_boolean_not_categorical(self):
        assert FeatureType.BOOLEAN.is_categorical is False

    def test_categorical_requires_encoding(self):
        assert FeatureType.CATEGORICAL_NOMINAL.requires_encoding is True

    def test_boolean_requires_encoding(self):
        assert FeatureType.BOOLEAN.requires_encoding is True

    def test_numeric_no_encoding_needed(self):
        assert FeatureType.NUMERIC_CONTINUOUS.requires_encoding is False


# ── Metric ────────────────────────────────────────────────────────────────────

class TestMetric:
    def test_create_metric(self):
        m = Metric(name="accuracy", value=0.95)
        assert m.name == "accuracy"
        assert m.value == 0.95

    def test_metric_is_immutable(self):
        m = Metric(name="f1", value=0.9)
        with pytest.raises(AttributeError):
            m.value = 0.8  # type: ignore

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            Metric(name="", value=0.5)

    def test_format_output(self):
        m = Metric(name="accuracy", value=0.9512)
        assert "0.9512" in m.format()

    def test_format_custom_decimals(self):
        m = Metric(name="accuracy", value=0.951234)
        assert "0.95" in m.format(2)


class TestMetricSet:
    def test_primary_score(self):
        ms = MetricSet(
            metrics=(Metric("f1_score", 0.9), Metric("accuracy", 0.92)),
            primary_metric_name="f1_score",
        )
        assert ms.primary_score == 0.9

    def test_to_dict(self):
        ms = MetricSet(
            metrics=(Metric("f1_score", 0.9), Metric("accuracy", 0.92)),
            primary_metric_name="f1_score",
        )
        d = ms.to_dict()
        assert d["f1_score"] == 0.9
        assert d["accuracy"] == 0.92

    def test_get_existing_metric(self):
        ms = MetricSet(
            metrics=(Metric("accuracy", 0.9),),
            primary_metric_name="accuracy",
        )
        assert ms.get("accuracy") == 0.9

    def test_get_missing_metric_returns_none(self):
        ms = MetricSet(
            metrics=(Metric("accuracy", 0.9),),
            primary_metric_name="accuracy",
        )
        assert ms.get("f1_score") is None

    def test_missing_primary_raises(self):
        ms = MetricSet(
            metrics=(Metric("accuracy", 0.9),),
            primary_metric_name="f1_score",
        )
        with pytest.raises(KeyError):
            _ = ms.primary_score


# ── Dataset Entity ────────────────────────────────────────────────────────────

class TestDataset:
    def test_default_status_is_uploaded(self):
        d = Dataset()
        assert d.status == DatasetStatus.UPLOADED

    def test_id_auto_generated(self):
        d1 = Dataset()
        d2 = Dataset()
        assert d1.id != d2.id

    def test_mark_analyzing(self):
        d = Dataset()
        d.mark_analyzing()
        assert d.status == DatasetStatus.ANALYZING

    def test_mark_analyzed(self, sample_analysis):
        d = Dataset()
        d.mark_analyzed(sample_analysis)
        assert d.status == DatasetStatus.ANALYZED
        assert d.analysis is not None

    def test_is_ready_after_analyzed(self, sample_analysis):
        d = Dataset()
        d.mark_analyzed(sample_analysis)
        assert d.is_ready is True

    def test_not_ready_when_uploaded(self):
        assert Dataset().is_ready is False

    def test_not_ready_when_analyzing(self):
        d = Dataset()
        d.mark_analyzing()
        assert d.is_ready is False

    def test_mark_error(self):
        d = Dataset()
        d.mark_error()
        assert d.status == DatasetStatus.ERROR


# ── Experiment Entity ─────────────────────────────────────────────────────────

class TestExperiment:
    def test_default_status_created(self):
        e = Experiment(name="test")
        assert e.status == ExperimentStatus.CREATED

    def test_id_auto_generated(self):
        e1 = Experiment(name="a")
        e2 = Experiment(name="b")
        assert e1.id != e2.id

    def test_transition_to_training(self):
        e = Experiment(name="test")
        e.transition_to(ExperimentStatus.TRAINING)
        assert e.status == ExperimentStatus.TRAINING

    def test_is_running_during_training(self):
        e = Experiment(name="test")
        e.transition_to(ExperimentStatus.TRAINING)
        assert e.is_running is True

    def test_is_running_during_evaluating(self):
        e = Experiment(name="test")
        e.transition_to(ExperimentStatus.EVALUATING)
        assert e.is_running is True

    def test_not_running_when_complete(self):
        e = Experiment(name="test")
        e.transition_to(ExperimentStatus.COMPLETE)
        assert e.is_running is False

    def test_is_complete(self):
        e = Experiment(name="test")
        e.transition_to(ExperimentStatus.COMPLETE)
        assert e.is_complete is True

    def test_is_not_complete_when_created(self):
        assert Experiment(name="test").is_complete is False


# ── Domain Exceptions ─────────────────────────────────────────────────────────

class TestDomainExceptions:
    def test_dataset_not_found_contains_id(self):
        exc = DatasetNotFoundError("abc-123")
        assert "abc-123" in exc.message
        assert exc.details["dataset_id"] == "abc-123"

    def test_experiment_not_found_contains_id(self):
        exc = ExperimentNotFoundError("exp-456")
        assert "exp-456" in exc.message

    def test_invalid_file_type_message(self):
        exc = InvalidFileTypeError("report.pdf", frozenset({".csv", ".xlsx"}))
        assert "report.pdf" in exc.message

    def test_file_too_large_message(self):
        exc = FileTooLargeError(600 * 1024 * 1024, 500 * 1024 * 1024)
        assert "600" in exc.message or "500" in exc.message

    def test_exception_hierarchy(self):
        from app.domain.exceptions.domain_exceptions import AutoRecError, NotFoundError
        assert isinstance(DatasetNotFoundError("x"), NotFoundError)
        assert isinstance(DatasetNotFoundError("x"), AutoRecError)
