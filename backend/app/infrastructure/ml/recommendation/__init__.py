"""
Recommendation module - Modular recommendation components for AutoRec.

This module contains the refactored recommendation engine components:
- CandidateFilter: Filters invalid or severely underperforming models
- PrimaryMetricSelector: Automatically determines primary evaluation metric
- GeneralizationAnalyzer: Evaluates model generalization capability
- DatasetCompatibilityAnalyzer: Analyzes dataset-algorithm compatibility
- RecommendationScorer: Computes multi-dimensional model scores
- RecommendationStrategy: Implements different recommendation modes
- ExplanationGenerator: Generates natural language explanations
- RecommendationFormatter: Formats results for API responses
- RecommendationResultBuilder: Orchestrates the recommendation pipeline
"""

from app.infrastructure.ml.recommendation.candidate_filter import (
    CandidateFilter,
    FilteringResult,
)
from app.infrastructure.ml.recommendation.dataset_compatibility_analyzer import (
    CompatibilityReport,
    CompatibilityScore,
    DatasetCompatibilityAnalyzer,
)
from app.infrastructure.ml.recommendation.explanation_generator import (
    ExplanationGenerator,
    ExplanationReport,
    ModelExplanation,
)
from app.infrastructure.ml.recommendation.generalization_analyzer import (
    GeneralizationAnalysis,
    GeneralizationAnalyzer,
    GeneralizationReport,
)
from app.infrastructure.ml.recommendation.primary_metric_selector import (
    MetricDefinition,
    MetricPriority,
    MetricSelection,
    PrimaryMetricSelector,
)
from app.infrastructure.ml.recommendation.recommendation_formatter import (
    FormattedRanking,
    FormattedRecommendation,
    FormattedRecommendations,
    RecommendationFormatter,
)
from app.infrastructure.ml.recommendation.recommendation_result_builder import (
    RecommendationPipelineResult,
    RecommendationResultBuilder,
)
from app.infrastructure.ml.recommendation.recommendation_scorer import (
    ModelScores,
    RecommendationScorer,
    ScoringReport,
)
from app.infrastructure.ml.recommendation.recommendation_strategy import (
    MultiModeRecommendations,
    RecommendationCandidate,
    RecommendationStrategy,
    StrategyResult,
)

__all__ = [
    # CandidateFilter
    "CandidateFilter",
    "FilteringResult",
    # PrimaryMetricSelector
    "PrimaryMetricSelector",
    "MetricSelection",
    "MetricDefinition",
    "MetricPriority",
    # GeneralizationAnalyzer
    "GeneralizationAnalyzer",
    "GeneralizationReport",
    "GeneralizationAnalysis",
    # DatasetCompatibilityAnalyzer
    "DatasetCompatibilityAnalyzer",
    "CompatibilityReport",
    "CompatibilityScore",
    # RecommendationScorer
    "RecommendationScorer",
    "ScoringReport",
    "ModelScores",
    # RecommendationStrategy
    "RecommendationStrategy",
    "MultiModeRecommendations",
    "RecommendationCandidate",
    "StrategyResult",
    # ExplanationGenerator
    "ExplanationGenerator",
    "ExplanationReport",
    "ModelExplanation",
    # RecommendationFormatter
    "RecommendationFormatter",
    "FormattedRecommendations",
    "FormattedRecommendation",
    "FormattedRanking",
    # RecommendationResultBuilder
    "RecommendationResultBuilder",
    "RecommendationPipelineResult",
]
