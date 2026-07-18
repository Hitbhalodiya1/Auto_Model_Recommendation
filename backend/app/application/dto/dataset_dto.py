"""
Data Transfer Objects (DTOs) for the AutoRec application layer.
These are Pydantic models used for inter-layer communication and API schemas.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ── Shared ────────────────────────────────────────────────────────────────────


class APIResponse(BaseModel):
    """Standardized API response envelope."""

    success: bool = True
    data: Any = None
    message: str = ""
    errors: dict = Field(default_factory=dict)
    meta: dict = Field(default_factory=dict)


# ── Dataset DTOs ──────────────────────────────────────────────────────────────


class DatasetUploadResponse(BaseModel):
    id: str
    filename: str
    original_name: str
    file_size: int
    status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ColumnProfileDTO(BaseModel):
    name: str
    dtype: str
    feature_type: str
    null_count: int
    null_pct: float
    unique_count: int
    unique_pct: float
    sample_values: list[Any] = []
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    median: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None


class DatasetAnalysisDTO(BaseModel):
    dataset_id: str
    task_type: str
    suggested_target_column: str | None
    quality_score: float
    row_count: int
    column_count: int
    duplicate_row_count: int
    missing_value_pct: float
    column_profiles: list[ColumnProfileDTO]
    class_distribution: dict[str, int] | None = None
    is_imbalanced: bool = False
    correlation_matrix: dict[str, dict[str, float]] | None = None
    outlier_counts: dict[str, int] | None = None
    warnings: list[str] = []
    recommendations: list[str] = []
    analyzed_at: datetime
    progress: int = 0
    steps_total: int | None = None
    steps_completed: int | None = None


class DatasetDTO(BaseModel):
    id: str
    filename: str
    original_name: str
    file_size: int
    row_count: int | None
    column_count: int | None
    status: str
    analysis: DatasetAnalysisDTO | None = None
    created_at: datetime
    updated_at: datetime


class DatasetPreviewDTO(BaseModel):
    columns: list[str]
    rows: list[dict[str, Any]]
    total_rows: int


# ── Experiment DTOs ───────────────────────────────────────────────────────────


class CreateExperimentRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str = ""
    dataset_id: str
    target_column: str
    task_type: str | None = None  # auto-detected if None


class ExperimentDTO(BaseModel):
    id: str
    name: str
    description: str
    dataset_id: str
    status: str
    task_type: str | None
    target_column: str | None
    created_at: datetime
    updated_at: datetime


# ── Preprocessing DTOs ────────────────────────────────────────────────────────


class PreprocessingStepDTO(BaseModel):
    name: str
    display_name: str
    strategy: str
    params: dict = {}
    explanation: str = ""
    affects_columns: list[str] = []


class PreprocessingRecommendationDTO(BaseModel):
    experiment_id: str
    steps: list[PreprocessingStepDTO]
    step_count: int
    recommended_scaler: str


class PreprocessingStatusDTO(BaseModel):
    experiment_id: str
    is_executed: bool
    executed_at: datetime | None = None
    pipeline_path: str | None = None


# ── Training DTOs ─────────────────────────────────────────────────────────────


class TrainingStatusDTO(BaseModel):
    experiment_id: str
    status: str
    total_models: int | None = None
    completed_models: int | None = None
    message: str = ""


# ── Evaluation / Model Result DTOs ────────────────────────────────────────────


class ModelResultDTO(BaseModel):
    id: str
    experiment_id: str
    algorithm_name: str
    config_name: str
    display_name: str
    configuration: dict
    metrics: dict[str, float]
    cv_score: float | None
    cv_std: float | None
    is_overfitting: bool
    training_time_s: float
    prediction_time_s: float
    is_recommended: bool
    rank: int | None
    requires_scaling: bool
    supports_feature_importance: bool
    supports_shap: bool
    interpretability_score: int
    error_message: str | None = None
    created_at: datetime


# ── Recommendation DTOs ───────────────────────────────────────────────────────


class RecommendationDTO(BaseModel):
    id: str
    experiment_id: str
    model_result_id: str
    composite_score: float
    score_breakdown: dict[str, float]
    rationale: list[str]
    explanation_text: str
    all_rankings: list[dict[str, Any]]
    recommended_model: ModelResultDTO | None = None
    created_at: datetime


# ── Explainability DTOs ───────────────────────────────────────────────────────


class FeatureImportanceDTO(BaseModel):
    feature: str
    importance: float
    rank: int


class ExplainabilityDTO(BaseModel):
    model_result_id: str
    feature_importances: list[FeatureImportanceDTO]
    shap_values: list[list[float]] | None = None
    shap_base_value: float | None = None
    top_features: list[str]
    method_used: str
    error: str | None = None
