"""
Dataset domain entity.
Represents an uploaded dataset and its lifecycle status.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class DatasetStatus(StrEnum):
    UPLOADED = "uploaded"
    ANALYZING = "analyzing"
    ANALYZED = "analyzed"
    ERROR = "error"


@dataclass
class ColumnProfile:
    """Profile of a single dataset column."""

    name: str
    dtype: str
    feature_type: str
    null_count: int
    null_pct: float
    unique_count: int
    unique_pct: float
    sample_values: list[Any] = field(default_factory=list)
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    median: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None


@dataclass
class DatasetAnalysis:
    """Results of automated dataset analysis."""

    dataset_id: str
    task_type: str
    suggested_target_column: str | None
    quality_score: float
    row_count: int
    column_count: int
    duplicate_row_count: int
    missing_value_pct: float
    column_profiles: list[ColumnProfile]
    class_distribution: dict[str, int] | None = None
    is_imbalanced: bool = False
    correlation_matrix: dict[str, dict[str, float]] | None = None
    outlier_counts: dict[str, int] | None = None
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    analyzed_at: datetime = field(default_factory=datetime.utcnow)
    # Progress reporting: percentage 0-100 and optional steps tracking
    progress: int = 0
    steps_total: int | None = None
    steps_completed: int | None = None


@dataclass
class Dataset:
    """
    Core dataset entity.
    Tracks file metadata and analysis lifecycle.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    filename: str = ""
    original_name: str = ""
    file_path: str = ""
    file_size: int = 0
    row_count: int | None = None
    column_count: int | None = None
    status: DatasetStatus = DatasetStatus.UPLOADED
    analysis: DatasetAnalysis | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def mark_analyzing(self) -> None:
        self.status = DatasetStatus.ANALYZING
        self.updated_at = datetime.utcnow()

    def mark_analyzed(self, analysis: DatasetAnalysis) -> None:
        self.analysis = analysis
        self.status = DatasetStatus.ANALYZED
        self.updated_at = datetime.utcnow()

    def mark_error(self) -> None:
        self.status = DatasetStatus.ERROR
        self.updated_at = datetime.utcnow()

    @property
    def is_ready(self) -> bool:
        return self.status == DatasetStatus.ANALYZED
