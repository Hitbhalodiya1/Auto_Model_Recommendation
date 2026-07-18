"""
Unit tests for refactored RecommendationEngine backward compatibility.
"""

import pytest

from app.core.recommendation_config import RecommendationConfig
from app.domain.entities.model_result import ModelResult, Recommendation
from app.domain.value_objects.task_type import TaskType
from app.infrastructure.ml.engines.evaluation_engine import EvaluationResult
from app.infrastructure.ml.engines.recommendation_engine import RecommendationEngine


@pytest.fixture
def sample_model_results():
    """Create sample model results for testing."""
    return [
        ModelResult(
            id="model1",
            config_name="rf_100",
            display_name="Random Forest",
            algorithm_name="RandomForest",
            metrics={"accuracy": 0.85, "f1_score": 0.84},
            cv_score=0.82,
            cv_std=0.02,
            is_overfitting=False,
            training_time_s=5.0,
            prediction_time_s=0.01,
            interpretability_score=3,
        ),
        ModelResult(
            id="model2",
            config_name="lr_lbfgs",
            display_name="Logistic Regression",
            algorithm_name="LogisticRegression",
            metrics={"accuracy": 0.80, "f1_score": 0.79},
            cv_score=0.78,
            cv_std=0.03,
            is_overfitting=False,
            training_time_s=1.0,
            prediction_time_s=0.001,
            interpretability_score=4,
        ),
    ]


@pytest.fixture
def sample_evaluations():
    """Create sample evaluation results."""
    from app.infrastructure.ml.engines.training_engine import TrainingResult
    from app.domain.interfaces.registry.model_registry import ModelConfig
    import numpy as np

    return [
        EvaluationResult(
            training_result=TrainingResult(
                config=ModelConfig(
                    name="rf_100",
                    display_name="Random Forest",
                    algorithm_family="RandomForest",
                    params={},
                    task_types=[],
                ),
                estimator=None,
                predictions=np.array([1, 0, 1]),
                train_score=0.85,
                training_time_s=5.0,
            ),
            metrics={"accuracy": 0.85, "f1_score": 0.84},
            cv_mean=0.82,
            cv_std=0.02,
        ),
        EvaluationResult(
            training_result=TrainingResult(
                config=ModelConfig(
                    name="lr_lbfgs",
                    display_name="Logistic Regression",
                    algorithm_family="LogisticRegression",
                    params={},
                    task_types=[],
                ),
                estimator=None,
                predictions=np.array([1, 0, 1]),
                train_score=0.80,
                training_time_s=1.0,
            ),
            metrics={"accuracy": 0.80, "f1_score": 0.79},
            cv_mean=0.78,
            cv_std=0.03,
        ),
    ]


class TestRecommendationEngineBackwardCompatibility:
    """Test backward compatibility of refactored RecommendationEngine."""

    def test_recommend_returns_recommendation_entity(
        self, sample_evaluations, sample_model_results
    ):
        """Test that recommend() returns a Recommendation entity."""
        engine = RecommendationEngine()
        result = engine.recommend(
            sample_evaluations,
            sample_model_results,
            TaskType.BINARY_CLASSIFICATION,
            "test_experiment",
        )

        assert isinstance(result, Recommendation)
        assert result.experiment_id == "test_experiment"
        assert result.model_result_id is not None

    def test_recommend_with_minimal_params(
        self, sample_evaluations, sample_model_results
    ):
        """Test recommend() with minimal parameters (backward compatibility)."""
        engine = RecommendationEngine()
        result = engine.recommend(
            sample_evaluations,
            sample_model_results,
            TaskType.BINARY_CLASSIFICATION,
            "test_experiment",
        )

        # Should work without dataset_analysis and model_configs
        assert result is not None
        assert result.model_result_id is not None

    def test_recommend_with_optional_params(
        self, sample_evaluations, sample_model_results
    ):
        """Test recommend() with optional new parameters."""
        engine = RecommendationEngine()
        dataset_analysis = {
            "row_count": 1000,
            "column_count": 10,
            "missing_value_pct": 0.0,
            "is_imbalanced": False,
            "column_profiles": [],
            "correlation_matrix": {},
            "outlier_counts": {},
        }

        result = engine.recommend(
            sample_evaluations,
            sample_model_results,
            TaskType.BINARY_CLASSIFICATION,
            "test_experiment",
            dataset_analysis=dataset_analysis,
        )

        assert result is not None
        assert result.model_result_id is not None

    def test_recommend_has_composite_score(
        self, sample_evaluations, sample_model_results
    ):
        """Test that recommendation has composite score."""
        engine = RecommendationEngine()
        result = engine.recommend(
            sample_evaluations,
            sample_model_results,
            TaskType.BINARY_CLASSIFICATION,
            "test_experiment",
        )

        assert result.composite_score is not None
        assert 0 <= result.composite_score <= 100

    def test_recommend_has_score_breakdown(
        self, sample_evaluations, sample_model_results
    ):
        """Test that recommendation has score breakdown."""
        engine = RecommendationEngine()
        result = engine.recommend(
            sample_evaluations,
            sample_model_results,
            TaskType.BINARY_CLASSIFICATION,
            "test_experiment",
        )

        assert result.score_breakdown is not None
        assert len(result.score_breakdown) > 0

    def test_recommend_has_rationale(
        self, sample_evaluations, sample_model_results
    ):
        """Test that recommendation has rationale."""
        engine = RecommendationEngine()
        result = engine.recommend(
            sample_evaluations,
            sample_model_results,
            TaskType.BINARY_CLASSIFICATION,
            "test_experiment",
        )

        assert result.rationale is not None
        assert len(result.rationale) > 0

    def test_recommend_has_explanation_text(
        self, sample_evaluations, sample_model_results
    ):
        """Test that recommendation has explanation text."""
        engine = RecommendationEngine()
        result = engine.recommend(
            sample_evaluations,
            sample_model_results,
            TaskType.BINARY_CLASSIFICATION,
            "test_experiment",
        )

        assert result.explanation_text is not None
        assert len(result.explanation_text) > 0

    def test_recommend_has_all_rankings(
        self, sample_evaluations, sample_model_results
    ):
        """Test that recommendation has all rankings."""
        engine = RecommendationEngine()
        result = engine.recommend(
            sample_evaluations,
            sample_model_results,
            TaskType.BINARY_CLASSIFICATION,
            "test_experiment",
        )

        assert result.all_rankings is not None
        assert len(result.all_rankings) > 0

    def test_all_rankings_have_required_fields(
        self, sample_evaluations, sample_model_results
    ):
        """Test that all rankings have required fields."""
        engine = RecommendationEngine()
        result = engine.recommend(
            sample_evaluations,
            sample_model_results,
            TaskType.BINARY_CLASSIFICATION,
            "test_experiment",
        )

        for ranking in result.all_rankings:
            assert "rank" in ranking
            assert "config_name" in ranking
            assert "display_name" in ranking
            assert "composite_score" in ranking
            assert "primary_metric" in ranking
            assert "cv_score" in ranking
            assert "is_overfitting" in ranking

    def test_raises_error_on_empty_model_list(self):
        """Test that recommend() raises error on empty model list."""
        engine = RecommendationEngine()
        with pytest.raises(ValueError, match="No model results"):
            engine.recommend([], [], TaskType.BINARY_CLASSIFICATION, "test_experiment")

    def test_custom_config(self, sample_evaluations, sample_model_results):
        """Test with custom recommendation config."""
        config = RecommendationConfig()
        engine = RecommendationEngine(config=config)
        result = engine.recommend(
            sample_evaluations,
            sample_model_results,
            TaskType.BINARY_CLASSIFICATION,
            "test_experiment",
        )

        assert result is not None

    def test_custom_weights(self, sample_evaluations, sample_model_results):
        """Test with custom scoring weights."""
        from app.infrastructure.ml.engines.recommendation_engine import ScoringWeights

        weights = ScoringWeights(performance=0.5, generalization=0.3)
        engine = RecommendationEngine(weights=weights)
        result = engine.recommend(
            sample_evaluations,
            sample_model_results,
            TaskType.BINARY_CLASSIFICATION,
            "test_experiment",
        )

        assert result is not None

    def test_ranks_are_assigned(
        self, sample_evaluations, sample_model_results
    ):
        """Test that ranks are assigned to models."""
        engine = RecommendationEngine()
        result = engine.recommend(
            sample_evaluations,
            sample_model_results,
            TaskType.BINARY_CLASSIFICATION,
            "test_experiment",
        )

        # Check that ranks are sequential starting from 1
        ranks = [r["rank"] for r in result.all_rankings]
        assert ranks == sorted(ranks)
        assert ranks[0] == 1
        assert ranks[-1] == len(result.all_rankings)

    def test_best_model_has_highest_score(
        self, sample_evaluations, sample_model_results
    ):
        """Test that the recommended model has the highest score."""
        engine = RecommendationEngine()
        result = engine.recommend(
            sample_evaluations,
            sample_model_results,
            TaskType.BINARY_CLASSIFICATION,
            "test_experiment",
        )

        # The recommended model should have the highest composite score
        best_ranking = result.all_rankings[0]
        assert best_ranking["composite_score"] >= max(
            r["composite_score"] for r in result.all_rankings
        )
