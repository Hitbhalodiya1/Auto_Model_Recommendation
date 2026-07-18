"""
DatasetCompatibilityAnalyzer - Analyzes dataset characteristics and computes
algorithm compatibility scores.

This component uses information from the analysis engine to determine how well
each algorithm suits the dataset characteristics. The compatibility score is
a BONUS factor and never overrides significantly better predictive performance.
"""

from dataclasses import dataclass
from typing import Any

from app.core.logging import get_logger
from app.core.recommendation_config import DatasetCompatibilityWeights
from app.domain.interfaces.registry.model_registry import ModelConfig

logger = get_logger(__name__)


@dataclass
class DatasetCharacteristics:
    """Key characteristics of the dataset for compatibility analysis."""
    row_count: int
    column_count: int
    missing_value_pct: float
    is_imbalanced: bool
    has_categorical: bool
    categorical_ratio: float  # ratio of categorical features
    has_high_correlation: bool
    outlier_ratio: float
    feature_types: dict[str, int]  # feature_type -> count


@dataclass
class CompatibilityScore:
    """Compatibility score for a single model."""
    model_id: str
    config_name: str
    algorithm_family: str
    overall_score: float  # 0-100
    breakdown: dict[str, float]  # component -> score


@dataclass
class CompatibilityReport:
    """Aggregated compatibility analysis across all models."""
    scores: dict[str, CompatibilityScore]  # model_id -> score
    best_compatible: str | None  # model_id


class DatasetCompatibilityAnalyzer:
    """
    Analyzes dataset compatibility for different algorithms.
    
    The compatibility score considers:
    - Dataset size (small vs large datasets favor different algorithms)
    - Feature count (dimensionality)
    - Nonlinearity (tree models vs linear models)
    - Class imbalance (some algorithms handle this better)
    - Noise (robustness to noise)
    - Categorical features (native support vs encoding required)
    """

    # Dataset size thresholds
    SMALL_DATASET_THRESHOLD = 1000
    LARGE_DATASET_THRESHOLD = 100000

    # Dimensionality thresholds
    HIGH_DIMENSIONAL_THRESHOLD = 100

    def __init__(self, weights: DatasetCompatibilityWeights | None = None) -> None:
        self._weights = weights or DatasetCompatibilityWeights()

    def analyze(
        self,
        model_configs: dict[str, ModelConfig],
        dataset_analysis: dict[str, Any],
    ) -> CompatibilityReport:
        """
        Analyze compatibility for all models.
        
        Args:
            model_configs: Mapping of model_id to ModelConfig
            dataset_analysis: Dataset analysis results from analysis engine
            
        Returns:
            CompatibilityReport with scores for each model
        """
        characteristics = self._extract_characteristics(dataset_analysis)
        scores = {}

        for model_id, config in model_configs.items():
            score = self._compute_compatibility(model_id, config, characteristics)
            scores[model_id] = score

        # Find best compatible model
        if scores:
            best = max(scores.values(), key=lambda s: s.overall_score)
            best_id = best.model_id
        else:
            best_id = None

        logger.info(
            "compatibility_analysis_completed",
            models_analyzed=len(scores),
            best_compatible=best_id,
            dataset_size=characteristics.row_count,
            feature_count=characteristics.column_count,
        )

        return CompatibilityReport(
            scores=scores,
            best_compatible=best_id,
        )

    def _extract_characteristics(self, analysis: dict[str, Any]) -> DatasetCharacteristics:
        """Extract key characteristics from dataset analysis."""
        row_count = analysis.get("row_count", 0)
        column_count = analysis.get("column_count", 0)
        missing_value_pct = analysis.get("missing_value_pct", 0.0)
        is_imbalanced = analysis.get("is_imbalanced", False)

        # Analyze feature types
        column_profiles = analysis.get("column_profiles", [])
        categorical_count = 0
        numeric_count = 0
        for col in column_profiles:
            if col.get("feature_type") == "categorical":
                categorical_count += 1
            elif col.get("feature_type") == "numeric":
                numeric_count += 1

        categorical_ratio = categorical_count / max(column_count, 1)
        has_categorical = categorical_count > 0

        # Check for high correlation
        correlation_matrix = analysis.get("correlation_matrix", {})
        has_high_correlation = self._detect_high_correlation(correlation_matrix)

        # Estimate outlier ratio
        outlier_counts = analysis.get("outlier_counts", {})
        total_outliers = sum(outlier_counts.values())
        outlier_ratio = total_outliers / max(row_count * column_count, 1)

        feature_types = {
            "categorical": categorical_count,
            "numeric": numeric_count,
        }

        return DatasetCharacteristics(
            row_count=row_count,
            column_count=column_count,
            missing_value_pct=missing_value_pct,
            is_imbalanced=is_imbalanced,
            has_categorical=has_categorical,
            categorical_ratio=categorical_ratio,
            has_high_correlation=has_high_correlation,
            outlier_ratio=outlier_ratio,
            feature_types=feature_types,
        )

    def _detect_high_correlation(self, correlation_matrix: dict[str, dict[str, float]]) -> bool:
        """Detect if dataset has high feature correlation."""
        high_corr_count = 0
        total_pairs = 0

        for col1, correlations in correlation_matrix.items():
            for col2, corr in correlations.items():
                if col1 != col2 and abs(corr) > 0.8:
                    high_corr_count += 1
                total_pairs += 1

        if total_pairs == 0:
            return False

        return (high_corr_count / total_pairs) > 0.1

    def _compute_compatibility(
        self, model_id: str, config: ModelConfig, characteristics: DatasetCharacteristics
    ) -> CompatibilityScore:
        """Compute compatibility score for a single model."""
        breakdown = {}

        # 1. Size compatibility
        size_score = self._compute_size_compatibility(config, characteristics)
        breakdown["size"] = size_score * self._weights.size_weight

        # 2. Feature count compatibility
        feature_score = self._compute_feature_compatibility(config, characteristics)
        breakdown["features"] = feature_score * self._weights.feature_count_weight

        # 3. Nonlinearity compatibility
        nonlinearity_score = self._compute_nonlinearity_compatibility(config, characteristics)
        breakdown["nonlinearity"] = nonlinearity_score * self._weights.nonlinearity_weight

        # 4. Dimensionality compatibility
        dimensionality_score = self._compute_dimensionality_compatibility(config, characteristics)
        breakdown["dimensionality"] = dimensionality_score * self._weights.dimensionality_weight

        # 5. Imbalance compatibility
        imbalance_score = self._compute_imbalance_compatibility(config, characteristics)
        breakdown["imbalance"] = imbalance_score * self._weights.imbalance_weight

        # 6. Noise compatibility
        noise_score = self._compute_noise_compatibility(config, characteristics)
        breakdown["noise"] = noise_score * self._weights.noise_weight

        overall = sum(breakdown.values())

        return CompatibilityScore(
            model_id=model_id,
            config_name=config.name,
            algorithm_family=config.algorithm_family,
            overall_score=overall,
            breakdown=breakdown,
        )

    def _compute_size_compatibility(
        self, config: ModelConfig, characteristics: DatasetCharacteristics
    ) -> float:
        """Compute size compatibility score (0-100)."""
        row_count = characteristics.row_count

        if row_count < self.SMALL_DATASET_THRESHOLD:
            # Small dataset: prefer simpler models
            if config.suitable_for_small_datasets:
                if config.training_complexity == "low":
                    return 90.0
                elif config.training_complexity == "medium":
                    return 70.0
                else:
                    return 50.0
            else:
                return 30.0
        elif row_count > self.LARGE_DATASET_THRESHOLD:
            # Large dataset: prefer scalable models
            if config.suitable_for_large_datasets:
                if config.training_complexity == "high":
                    return 90.0
                elif config.training_complexity == "medium":
                    return 80.0
                else:
                    return 70.0
            else:
                return 40.0
        else:
            # Medium dataset: most models work well
            return 80.0

    def _compute_feature_compatibility(
        self, config: ModelConfig, characteristics: DatasetCharacteristics
    ) -> float:
        """Compute feature count compatibility score (0-100)."""
        column_count = characteristics.column_count

        if column_count > self.HIGH_DIMENSIONAL_THRESHOLD:
            # High dimensional: prefer models that handle this well
            if config.supports_high_dimensional_data:
                return 90.0
            else:
                return 50.0
        elif column_count < 10:
            # Low dimensional: most models work well
            return 85.0
        else:
            # Medium dimensional
            return 80.0

    def _compute_nonlinearity_compatibility(
        self, config: ModelConfig, characteristics: DatasetCharacteristics
    ) -> float:
        """Compute nonlinearity compatibility score (0-100)."""
        # Use correlation as a proxy for linearity
        if characteristics.has_high_correlation:
            # High correlation suggests linear relationships
            if config.algorithm_family in ["Linear", "LogisticRegression", "SVM"]:
                return 85.0
            else:
                return 70.0
        else:
            # Low correlation suggests nonlinearity
            if config.handles_nonlinear_data:
                return 90.0
            else:
                return 50.0

    def _compute_dimensionality_compatibility(
        self, config: ModelConfig, characteristics: DatasetCharacteristics
    ) -> float:
        """Compute dimensionality compatibility score (0-100)."""
        column_count = characteristics.column_count
        categorical_ratio = characteristics.categorical_ratio

        if column_count > self.HIGH_DIMENSIONAL_THRESHOLD:
            # High dimensional
            if config.supports_high_dimensional_data:
                return 90.0
            else:
                return 40.0
        elif categorical_ratio > 0.5:
            # Many categorical features
            if config.native_categorical_support:
                return 85.0
            else:
                return 60.0
        else:
            return 80.0

    def _compute_imbalance_compatibility(
        self, config: ModelConfig, characteristics: DatasetCharacteristics
    ) -> float:
        """Compute class imbalance compatibility score (0-100)."""
        if not characteristics.is_imbalanced:
            # Balanced dataset: all models work well
            return 85.0

        # Imbalanced dataset: some algorithms handle this better
        # Tree-based models and ensemble methods typically handle imbalance better
        if config.algorithm_family in [
            "RandomForest",
            "GradientBoosting",
            "XGBoost",
            "LightGBM",
            "DecisionTree",
        ]:
            return 90.0
        elif config.algorithm_family in ["LogisticRegression", "SVM"]:
            # Can handle imbalance with class weights
            return 75.0
        else:
            return 60.0

    def _compute_noise_compatibility(
        self, config: ModelConfig, characteristics: DatasetCharacteristics
    ) -> float:
        """Compute noise robustness compatibility score (0-100)."""
        outlier_ratio = characteristics.outlier_ratio

        if outlier_ratio > 0.1:
            # High noise: prefer robust models
            if config.algorithm_family in [
                "RandomForest",
                "GradientBoosting",
                "XGBoost",
                "SVM",
            ]:
                return 85.0
            else:
                return 60.0
        else:
            # Low noise: most models work well
            return 80.0
