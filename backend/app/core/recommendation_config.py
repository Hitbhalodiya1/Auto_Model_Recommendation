"""
Recommendation Engine Configuration.

Defines thresholds, weights, and parameters for the recommendation pipeline.
These values can be customized without changing source code.
"""

from dataclasses import dataclass, field
from enum import Enum


class GeneralizationLevel(Enum):
    """Classification of generalization gap."""

    EXCELLENT = "excellent"
    GOOD = "good"
    MODERATE = "moderate"
    HIGH = "high"


class RecommendationMode(Enum):
    """Different recommendation strategies."""

    BEST_OVERALL = "best_overall"
    BEST_PREDICTIVE = "best_predictive"
    FASTEST = "fastest"
    MOST_EXPLAINABLE = "most_explainable"


@dataclass
class GeneralizationThresholds:
    """Thresholds for classifying generalization gaps."""

    excellent_gap_pct: float = 3.0
    good_gap_pct: float = 7.0
    moderate_gap_pct: float = 15.0

    def classify(self, gap_pct: float) -> GeneralizationLevel:
        """Classify a generalization gap percentage."""
        if gap_pct < self.excellent_gap_pct:
            return GeneralizationLevel.EXCELLENT
        elif gap_pct < self.good_gap_pct:
            return GeneralizationLevel.GOOD
        elif gap_pct < self.moderate_gap_pct:
            return GeneralizationLevel.MODERATE
        else:
            return GeneralizationLevel.HIGH


@dataclass
class ScoringWeights:
    """Weights for different scoring components (0-1, should sum to 1.0)."""

    # Primary metric weights by recommendation mode
    best_overall: dict[str, float] = field(
        default_factory=lambda: {
            "predictive_performance": 0.35,
            "generalization": 0.25,
            "robustness": 0.15,
            "dataset_compatibility": 0.10,
            "interpretability": 0.10,
            "speed": 0.05,
        }
    )

    best_predictive: dict[str, float] = field(
        default_factory=lambda: {
            "predictive_performance": 0.50,
            "generalization": 0.30,
            "robustness": 0.15,
            "dataset_compatibility": 0.05,
            "interpretability": 0.0,
            "speed": 0.0,
        }
    )

    fastest: dict[str, float] = field(
        default_factory=lambda: {
            "predictive_performance": 0.20,
            "generalization": 0.20,
            "robustness": 0.10,
            "dataset_compatibility": 0.10,
            "interpretability": 0.05,
            "speed": 0.35,
        }
    )

    most_explainable: dict[str, float] = field(
        default_factory=lambda: {
            "predictive_performance": 0.25,
            "generalization": 0.20,
            "robustness": 0.15,
            "dataset_compatibility": 0.10,
            "interpretability": 0.30,
            "speed": 0.0,
        }
    )

    def get_weights(self, mode: RecommendationMode) -> dict[str, float]:
        """Get weights for a specific recommendation mode."""
        weights_map = {
            RecommendationMode.BEST_OVERALL: self.best_overall,
            RecommendationMode.BEST_PREDICTIVE: self.best_predictive,
            RecommendationMode.FASTEST: self.fastest,
            RecommendationMode.MOST_EXPLAINABLE: self.most_explainable,
        }
        return weights_map.get(mode, self.best_overall)


@dataclass
class FilteringThresholds:
    """Thresholds for filtering candidate models."""

    # Remove models with severe overfitting
    severe_overfitting_gap_pct: float = 20.0

    # Remove models with invalid metrics
    min_primary_metric: float = 0.0
    max_primary_metric: float = 1.0

    # Remove models that failed training/evaluation
    allow_failed_models: bool = False


@dataclass
class DatasetCompatibilityWeights:
    """Weights for dataset compatibility scoring."""

    size_weight: float = 0.20
    feature_count_weight: float = 0.15
    nonlinearity_weight: float = 0.20
    dimensionality_weight: float = 0.15
    imbalance_weight: float = 0.15
    noise_weight: float = 0.15


@dataclass
class RecommendationConfig:
    """Main configuration for the recommendation engine."""

    generalization: GeneralizationThresholds = field(default_factory=GeneralizationThresholds)
    scoring: ScoringWeights = field(default_factory=ScoringWeights)
    filtering: FilteringThresholds = field(default_factory=FilteringThresholds)
    compatibility: DatasetCompatibilityWeights = field(default_factory=DatasetCompatibilityWeights)

    # Generate recommendations for all modes
    generate_all_modes: bool = True

    # Number of recommendations to return per mode
    recommendations_per_mode: int = 1


# Default configuration instance
default_config = RecommendationConfig()
