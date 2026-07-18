"""
ModelResult domain entity.
Represents the outcome of training and evaluating a single model configuration.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ModelResult:
    """
    Stores all outputs for a trained + evaluated model configuration.
    Immutable after creation except for rank assignment.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    experiment_id: str = ""
    algorithm_name: str = ""  # e.g. "RandomForest"
    config_name: str = ""  # e.g. "rf_gini_100"
    display_name: str = ""  # e.g. "Random Forest (Gini, 100 trees)"
    configuration: dict = field(default_factory=dict)  # hyperparameters
    metrics: dict[str, float] = field(default_factory=dict)
    cv_score: float | None = None
    cv_std: float | None = None
    is_overfitting: bool = False
    training_time_s: float = 0.0
    prediction_time_s: float = 0.0
    model_path: str | None = None  # path to serialized .pkl
    is_recommended: bool = False
    rank: int | None = None
    requires_scaling: bool = False
    supports_feature_importance: bool = False
    supports_shap: bool = False
    interpretability_score: int = 1
    error_message: str | None = None  # populated if training failed
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def succeeded(self) -> bool:
        return self.error_message is None

    @property
    def primary_metric(self) -> float | None:
        """Return the most important metric value for ranking."""
        # Priority order for classification: f1, accuracy
        # Priority order for regression: r2
        for key in ("f1_score", "f1_weighted", "roc_auc", "accuracy", "r2_score"):
            if key in self.metrics:
                return self.metrics[key]
        return None


@dataclass
class Recommendation:
    """The recommended model for an experiment with its rationale."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    experiment_id: str = ""
    model_result_id: str = ""
    composite_score: float = 0.0
    score_breakdown: dict[str, float] = field(default_factory=dict)
    rationale: list[str] = field(default_factory=list)  # bullet points
    explanation_text: str = ""  # plain-language paragraph
    all_rankings: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
