"""
SQLAlchemy ORM models for Experiment, ModelResult, Recommendation, and Report tables.
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.database.session import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class ExperimentModel(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    dataset_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    analysis_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="created")
    task_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    target_column: Mapped[str | None] = mapped_column(String(255), nullable=True)
    preprocessing_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    config_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class ModelResultModel(Base):
    __tablename__ = "model_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    experiment_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    algorithm_name: Mapped[str] = mapped_column(String(100), nullable=False)
    config_name: Mapped[str] = mapped_column(String(100), nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    configuration_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    metrics_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    cv_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    cv_std: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_overfitting: Mapped[bool] = mapped_column(Boolean, default=False)
    training_time_s: Mapped[float] = mapped_column(Float, default=0.0)
    prediction_time_s: Mapped[float] = mapped_column(Float, default=0.0)
    model_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    is_recommended: Mapped[bool] = mapped_column(Boolean, default=False)
    rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    requires_scaling: Mapped[bool] = mapped_column(Boolean, default=False)
    supports_feature_importance: Mapped[bool] = mapped_column(Boolean, default=False)
    supports_shap: Mapped[bool] = mapped_column(Boolean, default=False)
    interpretability_score: Mapped[int] = mapped_column(Integer, default=1)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class RecommendationModel(Base):
    __tablename__ = "recommendations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    experiment_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    model_result_id: Mapped[str] = mapped_column(String(36), nullable=False)
    composite_score: Mapped[float] = mapped_column(Float, default=0.0)
    score_breakdown_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    rationale_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    explanation_text: Mapped[str] = mapped_column(Text, default="")
    all_rankings_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ReportModel(Base):
    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    experiment_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    format: Mapped[str] = mapped_column(String(10), nullable=False)
    file_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
