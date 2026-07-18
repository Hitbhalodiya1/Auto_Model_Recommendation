"""
RecommendationResultBuilder - Orchestrates all recommendation components
to build the final recommendation result.

This component coordinates the entire recommendation pipeline:
1. Candidate filtering
2. Primary metric selection
3. Generalization analysis
4. Dataset compatibility analysis
5. Multi-dimensional scoring
6. Multi-mode strategy application
7. Explanation generation
8. Result formatting
"""

from dataclasses import dataclass
from typing import Any

from app.core.logging import get_logger
from app.core.recommendation_config import RecommendationConfig
from app.domain.entities.model_result import ModelResult
from app.domain.value_objects.task_type import TaskType
from app.infrastructure.ml.recommendation.candidate_filter import (
    CandidateFilter,
    FilteringResult,
)
from app.infrastructure.ml.recommendation.dataset_compatibility_analyzer import (
    CompatibilityReport,
    DatasetCompatibilityAnalyzer,
)
from app.infrastructure.ml.recommendation.explanation_generator import (
    ExplanationGenerator,
    ExplanationReport,
)
from app.infrastructure.ml.recommendation.generalization_analyzer import (
    GeneralizationAnalyzer,
    GeneralizationReport,
)
from app.infrastructure.ml.recommendation.primary_metric_selector import (
    MetricSelection,
    PrimaryMetricSelector,
)
from app.infrastructure.ml.recommendation.recommendation_formatter import (
    FormattedRecommendations,
    RecommendationFormatter,
)
from app.infrastructure.ml.recommendation.recommendation_scorer import (
    RecommendationScorer,
    ScoringReport,
)
from app.infrastructure.ml.recommendation.recommendation_strategy import (
    MultiModeRecommendations,
    RecommendationStrategy,
)

logger = get_logger(__name__)


@dataclass
class RecommendationPipelineResult:
    """Complete result of the recommendation pipeline."""
    filtering_result: FilteringResult
    metric_selection: MetricSelection
    generalization_report: GeneralizationReport
    compatibility_report: CompatibilityReport
    scoring_report: ScoringReport
    multi_mode_recommendations: MultiModeRecommendations
    explanation_report: ExplanationReport
    formatted_recommendations: FormattedRecommendations


class RecommendationResultBuilder:
    """
    Orchestrates the entire recommendation pipeline.
    
    This component coordinates all the individual recommendation components
    to produce a comprehensive recommendation result.
    """

    def __init__(self, config: RecommendationConfig | None = None) -> None:
        self._config = config or RecommendationConfig()

        # Initialize components
        self._candidate_filter = CandidateFilter(self._config.filtering)
        self._metric_selector = PrimaryMetricSelector()
        self._generalization_analyzer = GeneralizationAnalyzer(
            self._config.generalization
        )
        self._compatibility_analyzer = DatasetCompatibilityAnalyzer(
            self._config.compatibility
        )
        self._scorer = RecommendationScorer(self._config.scoring)
        self._strategy = RecommendationStrategy()
        self._explanation_generator = ExplanationGenerator()
        self._formatter = RecommendationFormatter()

    def build_recommendation(
        self,
        model_results: list[ModelResult],
        model_configs: dict[str, Any],
        dataset_analysis: dict[str, Any] | None,
        task_type: TaskType,
        experiment_id: str,
    ) -> RecommendationPipelineResult:
        """
        Build the complete recommendation result.
        
        Args:
            model_results: List of model results from training
            model_configs: Mapping of model_id to ModelConfig
            dataset_analysis: Dataset analysis results (optional)
            task_type: The ML task type
            experiment_id: Experiment ID for logging
            
        Returns:
            RecommendationPipelineResult with all pipeline outputs
        """
        logger.info(
            "recommendation_pipeline_started",
            experiment_id=experiment_id,
            total_models=len(model_results),
            task_type=task_type.value,
        )

        # Stage 1: Candidate Filtering
        filtering_result = self._candidate_filter.filter(model_results)
        candidates = filtering_result.candidates

        if not candidates:
            logger.warning("no_valid_candidates", experiment_id=experiment_id)
            # Return empty result
            return self._build_empty_result(model_results, experiment_id)

        # Stage 2: Primary Metric Selection
        is_imbalanced = dataset_analysis.get("is_imbalanced", False) if dataset_analysis else False
        available_metrics = self._get_available_metrics(candidates)
        metric_selection = self._metric_selector.select(
            task_type, is_imbalanced, available_metrics
        )

        logger.info(
            "primary_metric_selected",
            metric=metric_selection.primary_metric,
            rationale=metric_selection.rationale,
        )

        # Stage 3: Generalization Analysis
        generalization_report = self._generalization_analyzer.analyze(candidates)

        # Stage 4: Dataset Compatibility Analysis
        if dataset_analysis:
            compatibility_report = self._compatibility_analyzer.analyze(
                model_configs, dataset_analysis
            )
        else:
            # Create empty compatibility report if no dataset analysis
            compatibility_report = self._build_empty_compatibility_report(
                model_configs
            )

        # Stage 5: Multi-dimensional Scoring
        scoring_report = self._scorer.score(
            candidates,
            generalization_report,
            compatibility_report,
            metric_selection.primary_metric,
        )

        # Stage 6: Multi-mode Strategy Application
        multi_mode_recommendations = self._strategy.generate_recommendations(
            candidates, scoring_report
        )

        # Stage 7: Explanation Generation
        # Collect all candidates from all modes
        all_candidates = []
        for strategy_result in [
            multi_mode_recommendations.best_overall,
            multi_mode_recommendations.best_predictive,
            multi_mode_recommendations.fastest,
            multi_mode_recommendations.most_explainable,
        ]:
            all_candidates.extend(strategy_result.candidates)

        explanation_report = self._explanation_generator.generate_explanations(
            all_candidates, generalization_report.analyses
        )

        # Stage 8: Result Formatting
        formatted_recommendations = self._formatter.format_recommendations(
            multi_mode_recommendations,
            explanation_report.explanations,
            model_results,
        )

        logger.info(
            "recommendation_pipeline_completed",
            experiment_id=experiment_id,
            best_overall=formatted_recommendations.best_overall.config_name if formatted_recommendations.best_overall else None,
        )

        return RecommendationPipelineResult(
            filtering_result=filtering_result,
            metric_selection=metric_selection,
            generalization_report=generalization_report,
            compatibility_report=compatibility_report,
            scoring_report=scoring_report,
            multi_mode_recommendations=multi_mode_recommendations,
            explanation_report=explanation_report,
            formatted_recommendations=formatted_recommendations,
        )

    def _get_available_metrics(self, model_results: list[ModelResult]) -> list[str]:
        """Get list of available metrics from model results."""
        metrics = set()
        for mr in model_results:
            metrics.update(mr.metrics.keys())
        return list(metrics)

    def _build_empty_compatibility_report(
        self, model_configs: dict[str, Any]
    ) -> CompatibilityReport:
        """Build empty compatibility report when dataset analysis is not available."""
        from app.infrastructure.ml.recommendation.dataset_compatibility_analyzer import (
            CompatibilityScore,
        )

        scores = {}
        for model_id, config in model_configs.items():
            scores[model_id] = CompatibilityScore(
                model_id=model_id,
                config_name=config.name,
                algorithm_family=config.algorithm_family,
                overall_score=50.0,  # Neutral score
                breakdown={},
            )

        return CompatibilityReport(scores=scores, best_compatible=None)

    def _build_empty_result(
        self, model_results: list[ModelResult], experiment_id: str
    ) -> RecommendationPipelineResult:
        """Build empty result when no valid candidates exist."""
        from app.infrastructure.ml.recommendation.dataset_compatibility_analyzer import (
            CompatibilityReport,
        )
        from app.infrastructure.ml.recommendation.explanation_generator import (
            ExplanationReport,
        )
        from app.infrastructure.ml.recommendation.generalization_analyzer import (
            GeneralizationReport,
        )
        from app.infrastructure.ml.recommendation.primary_metric_selector import (
            MetricSelection,
        )
        from app.infrastructure.ml.recommendation.recommendation_formatter import (
            FormattedRecommendations,
        )
        from app.infrastructure.ml.recommendation.recommendation_scorer import (
            ScoringReport,
        )
        from app.infrastructure.ml.recommendation.recommendation_strategy import (
            MultiModeRecommendations,
            StrategyResult,
        )

        return RecommendationPipelineResult(
            filtering_result=FilteringResult(
                candidates=[],
                filtered_out=model_results,
                reasons={mr.id: "No valid candidates" for mr in model_results},
            ),
            metric_selection=MetricSelection(
                primary_metric="accuracy",
                secondary_metrics=[],
                rationale="No valid candidates, using default",
            ),
            generalization_report=GeneralizationReport(analyses={}, best_generalized=None, worst_generalized=None),
            compatibility_report=CompatibilityReport(scores={}, best_compatible=None),
            scoring_report=ScoringReport(
                scores={},
                best_overall=None,
                best_predictive=None,
                best_generalized=None,
                fastest=None,
                most_explainable=None,
            ),
            multi_mode_recommendations=MultiModeRecommendations(
                best_overall=StrategyResult(
                    mode=None, candidates=[], selected=None
                ),
                best_predictive=StrategyResult(
                    mode=None, candidates=[], selected=None
                ),
                fastest=StrategyResult(mode=None, candidates=[], selected=None),
                most_explainable=StrategyResult(
                    mode=None, candidates=[], selected=None
                ),
            ),
            explanation_report=ExplanationReport(explanations={}),
            formatted_recommendations=FormattedRecommendations(
                best_overall=None,
                best_predictive=None,
                fastest=None,
                most_explainable=None,
                all_rankings=[],
            ),
        )
