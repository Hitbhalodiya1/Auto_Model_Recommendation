"""
Unit tests for domain entities and value objects.
No I/O, no infrastructure dependencies.
"""


import pytest

from app.domain.entities.dataset import Dataset, DatasetStatus
from app.domain.entities.experiment import (
    Experiment,
    ExperimentStatus,
)
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

    def test_regression_not_classification(self):
        assert TaskType.REGRESSION.is_classification is False

    def test_clustering_not_supervised(self):
        assert TaskType.CLUSTERING.is_supervised is False

    def test_regression_is_supervised(self):
        assert TaskType.REGRESSION.is_supervised is True

    def test_display_names(self):
        assert "Binary" in TaskType.BINARY_CLASSIFICATION.display_name
        assert "Regression" in TaskType.REGRESSION.display_name


# ── FeatureType ───────────────────────────────────────────────────────────────

class TestFeatureType:
    def test_numeric_continuous_is_numeric(self):
        assert FeatureType.NUMERIC_CONTINUOUS.is_numeric is True

    def test_categorical_not_numeric(self):
        assert FeatureType.CATEGORICAL_NOMINAL.is_numeric is False

    def test_categorical_requires_encoding(self):
        assert FeatureType.CATEGORICAL_NOMINAL.requires_encoding is True

    def test_numeric_no_encoding(self):
        assert FeatureType.NUMERIC_CONTINUOUS.requires_encoding is False


# ── Metric ────────────────────────────────────────────────────────────────────

class TestMetric:
    def test_metric_creation(self):
        m = Metric(name="accuracy", value=0.95)
        assert m.name == "accuracy"
        assert m.value == 0.95

    def test_metric_immutable(self):
        m = Metric(name="f1", value=0.9)
        with pytest.raises(AttributeError):
            m.value = 0.8  # type: ignore

    def test_metric_empty_name_raises(self):
        with pytest.raises(ValueError):
            Metric(name="", value=0.5)

    def test_metric_format(self):
        m = Metric(name="accuracy", value=0.9512)
        assert "0.9512" in m.format()

    def test_metric_set_primary_score(self):
        ms = MetricSet(
            metrics=(Metric("f1_score", 0.9), Metric("accuracy", 0.92)),
            primary_metric_name="f1_score",
        )
        assert ms.primary_score == 0.9

    def test_metric_set_missing_primary_raises(self):
        ms = MetricSet(
            metrics=(Metric("accuracy", 0.9),),
            primary_metric_name="f1_score",
        )
        with pytest.raises(KeyError):
            _ = ms.primary_score


# ── Dataset Entity ────────────────────────────────────────────────────────────

class TestDataset:
    def test_default_status_is_uploaded(self):
        d = Dataset(filename="test.csv", original_name="test.csv")
        assert d.status == DatasetStatus.UPLOADED

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
        d = Dataset()
        assert d.is_ready is False

    def test_mark_error(self):
        d = Dataset()
        d.mark_error()
        assert d.status == DatasetStatus.ERROR


# ── Experiment Entity ─────────────────────────────────────────────────────────

class TestExperiment:
    def test_default_status_is_created(self):
        e = Experiment(name="test")
        assert e.status == ExperimentStatus.CREATED

    def test_transition_status(self):
        e = Experiment(name="test")
        e.transition_to(ExperimentStatus.TRAINING)
        assert e.status == ExperimentStatus.TRAINING

    def test_is_running_during_training(self):
        e = Experiment(name="test")
        e.transition_to(ExperimentStatus.TRAINING)
        assert e.is_running is True

    def test_is_not_running_when_complete(self):
        e = Experiment(name="test")
        e.transition_to(ExperimentStatus.COMPLETE)
        assert e.is_running is False

    def test_is_complete(self):
        e = Experiment(name="test")
        e.transition_to(ExperimentStatus.COMPLETE)
        assert e.is_complete is True


# ── Domain Exceptions ─────────────────────────────────────────────────────────

class TestDomainExceptions:
    def test_dataset_not_found_message(self):
        exc = DatasetNotFoundError("abc-123")
        assert "abc-123" in str(exc)
        assert exc.details["dataset_id"] == "abc-123"

    def test_invalid_file_type(self):
        exc = InvalidFileTypeError("file.pdf", frozenset({".csv", ".xlsx"}))
        assert "file.pdf" in exc.message

    def test_file_too_large(self):
        exc = FileTooLargeError(600 * 1024 * 1024, 500 * 1024 * 1024)
        assert "600.0MB" in exc.message or "600" in exc.message

    def test_experiment_not_found(self):
        exc = ExperimentNotFoundError("exp-1")
        assert "exp-1" in str(exc)
