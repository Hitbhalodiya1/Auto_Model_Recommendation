"""
Experiment domain entity.
Represents a complete ML workflow run against a dataset.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class ExperimentStatus(StrEnum):
    CREATED = "created"
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    EVALUATING = "evaluating"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class PreprocessingStep:
    """A single step in the preprocessing pipeline."""

    name: str  # e.g. "handle_missing_values"
    display_name: str  # e.g. "Handle Missing Values"
    strategy: str  # e.g. "median_imputation"
    params: dict = field(default_factory=dict)
    explanation: str = ""  # human-readable rationale
    affects_columns: list[str] = field(default_factory=list)


@dataclass
class PreprocessingPipeline:
    """The preprocessing pipeline applied to a dataset."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    experiment_id: str = ""
    steps: list[PreprocessingStep] = field(default_factory=list)
    pipeline_path: str | None = None  # path to serialized .pkl
    executed_at: datetime | None = None
    is_executed: bool = False

    def mark_executed(self, pipeline_path: str) -> None:
        self.pipeline_path = pipeline_path
        self.executed_at = datetime.utcnow()
        self.is_executed = True


@dataclass
class Experiment:
    """
    Core experiment entity.
    Tracks the full lifecycle from dataset → recommendation.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    dataset_id: str = ""
    analysis_id: str | None = None
    status: ExperimentStatus = ExperimentStatus.CREATED
    task_type: str | None = None
    target_column: str | None = None
    preprocessing_pipeline: PreprocessingPipeline | None = None
    config: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def transition_to(self, status: ExperimentStatus) -> None:
        """Transition experiment to a new status, updating timestamp."""
        self.status = status
        self.updated_at = datetime.utcnow()

    @property
    def is_complete(self) -> bool:
        return self.status == ExperimentStatus.COMPLETE

    @property
    def is_running(self) -> bool:
        return self.status in (
            ExperimentStatus.PREPROCESSING,
            ExperimentStatus.TRAINING,
            ExperimentStatus.EVALUATING,
        )
