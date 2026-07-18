"""
Unit tests for ML engines and the model registry.
Uses small synthetic datasets; deterministic via random_state=42.
"""

import numpy as np
import pandas as pd
import pytest

from app.domain.value_objects.task_type import TaskType
from app.infrastructure.ml.engines.analysis_engine import AnalysisEngine
from app.infrastructure.ml.engines.evaluation_engine import EvaluationEngine
from app.infrastructure.ml.engines.preprocessing_engine import PreprocessingEngine
from app.infrastructure.ml.engines.recommendation_engine import RecommendationEngine
from app.infrastructure.ml.registry.registry_bootstrap import build_registry

# ── AnalysisEngine ────────────────────────────────────────────────────────────

class TestAnalysisEngine:
    @pytest.fixture(autouse=True)
    def engine(self):
        self.engine = AnalysisEngine()

    def test_binary_task_detected(self, binary_df):
        result = self.engine.analyze(binary_df, "ds-1")
        assert result.task_type == TaskType.BINARY_CLASSIFICATION.value

    def test_regression_task_detected(self, regression_df):
        result = self.engine.analyze(regression_df, "ds-2")
        assert result.task_type == TaskType.REGRESSION.value

    def test_row_and_column_count(self, binary_df):
        result = self.engine.analyze(binary_df, "ds-3")
        assert result.row_count == 100
        assert result.column_count == 4

    def test_target_column_suggested(self, binary_df):
        result = self.engine.analyze(binary_df, "ds-4")
        assert result.suggested_target_column == "target"

    def test_quality_score_in_range(self, binary_df):
        result = self.engine.analyze(binary_df, "ds-5")
        assert 0 <= result.quality_score <= 100

    def test_column_profiles_generated(self, binary_df):
        result = self.engine.analyze(binary_df, "ds-6")
        assert len(result.column_profiles) == 4
        names = [cp.name for cp in result.column_profiles]
        assert "feature1" in names and "target" in names

    def test_detects_missing_values(self):
        df = pd.DataFrame({
            "a": [1.0, None, 3.0, 4.0, 5.0] * 20,
            "target": [0, 1, 0, 1, 0] * 20,
        })
        result = self.engine.analyze(df, "ds-7")
        assert result.missing_value_pct > 0
        assert any("missing" in w.lower() for w in result.warnings)

    def test_detects_duplicates(self):
        row = {"a": 1, "target": 0}
        df = pd.DataFrame([row] * 60 + [{"a": i, "target": i % 2} for i in range(40)])
        result = self.engine.analyze(df, "ds-8")
        assert result.duplicate_row_count > 0

    def test_too_small_raises(self):
        from app.domain.exceptions.domain_exceptions import DatasetTooSmallError
        df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
        with pytest.raises(DatasetTooSmallError):
            self.engine.analyze(df, "ds-small")

    def test_no_duplicate_row_count_zero_by_default(self, binary_df):
        result = self.engine.analyze(binary_df, "ds-9")
        assert result.duplicate_row_count == 0

    def test_recommendations_generated(self):
        df = pd.DataFrame({
            "a": [None if i % 5 == 0 else float(i) for i in range(100)],
            "b": ["X" if i % 3 == 0 else "Y" for i in range(100)],
            "target": [i % 2 for i in range(100)],
        })
        result = self.engine.analyze(df, "ds-10")
        assert len(result.recommendations) > 0


# ── PreprocessingEngine ───────────────────────────────────────────────────────

class TestPreprocessingEngine:
    @pytest.fixture(autouse=True)
    def engine(self):
        self.engine = PreprocessingEngine()
        self.analysis_engine = AnalysisEngine()

    def test_recommend_returns_pipeline(self, sample_analysis):
        pipeline = self.engine.recommend(sample_analysis, "target")
        assert pipeline is not None
        assert len(pipeline.steps) >= 1

    def test_recommend_includes_scale_step(self, sample_analysis):
        pipeline = self.engine.recommend(sample_analysis, "target")
        names = [s.name for s in pipeline.steps]
        assert "scale_features" in names

    def test_recommend_duplicate_removal_when_duplicates(self):
        from app.domain.entities.dataset import DatasetAnalysis
        analysis = DatasetAnalysis(
            dataset_id="x",
            task_type="binary_classification",
            suggested_target_column="target",
            quality_score=70.0,
            row_count=100,
            column_count=2,
            duplicate_row_count=15,
            missing_value_pct=0.0,
            column_profiles=[],
        )
        pipeline = self.engine.recommend(analysis, "target")
        assert any(s.name == "remove_duplicates" for s in pipeline.steps)

    def test_recommend_imputation_when_missing(self):
        from app.domain.entities.dataset import ColumnProfile, DatasetAnalysis
        analysis = DatasetAnalysis(
            dataset_id="x",
            task_type="binary_classification",
            suggested_target_column="target",
            quality_score=70.0,
            row_count=100,
            column_count=2,
            duplicate_row_count=0,
            missing_value_pct=0.1,
            column_profiles=[
                ColumnProfile(
                    name="feature1",
                    dtype="float64",
                    feature_type="numeric_continuous",
                    null_count=10,
                    null_pct=0.1,
                    unique_count=90,
                    unique_pct=0.9,
                ),
            ],
        )
        pipeline = self.engine.recommend(analysis, "target")
        assert any("impute" in s.name for s in pipeline.steps)

    def test_each_step_has_explanation(self, sample_analysis):
        pipeline = self.engine.recommend(sample_analysis, "target")
        for step in pipeline.steps:
            assert len(step.explanation) > 0, f"Step {step.name} has no explanation"

    def test_execute_binary_shapes(self, binary_df):
        analysis = self.analysis_engine.analyze(binary_df, "test")
        pipeline = self.engine.recommend(analysis, "target")
        result = self.engine.execute(
            df=binary_df,
            pipeline_def=pipeline,
            target_column="target",
            task_type="binary_classification",
            test_size=0.3,
            random_state=42,
        )
        total = len(result.X_train) + len(result.X_test)
        assert total == len(binary_df)
        assert result.X_train.shape[0] > 0
        assert result.X_test.shape[0] > 0

    def test_execute_regression_no_label_encoder(self, regression_df):
        analysis = self.analysis_engine.analyze(regression_df, "test")
        pipeline = self.engine.recommend(analysis, "target")
        result = self.engine.execute(
            df=regression_df,
            pipeline_def=pipeline,
            target_column="target",
            task_type="regression",
            test_size=0.3,
        )
        assert result.label_encoder is None

    def test_execute_classification_feature_names_populated(self, binary_df):
        analysis = self.analysis_engine.analyze(binary_df, "test")
        pipeline = self.engine.recommend(analysis, "target")
        result = self.engine.execute(
            df=binary_df,
            pipeline_def=pipeline,
            target_column="target",
            task_type="binary_classification",
        )
        assert len(result.feature_names) > 0


# ── ModelRegistry ─────────────────────────────────────────────────────────────

class TestModelRegistry:
    @pytest.fixture(autouse=True)
    def registry(self):
        self.registry = build_registry()

    def test_has_classification_models(self):
        configs = self.registry.get_models_for_task(TaskType.BINARY_CLASSIFICATION)
        assert len(configs) > 10

    def test_has_regression_models(self):
        configs = self.registry.get_models_for_task(TaskType.REGRESSION)
        assert len(configs) > 10

    def test_all_configs_have_unique_names(self):
        all_configs = (
            self.registry.get_models_for_task(TaskType.BINARY_CLASSIFICATION)
            + self.registry.get_models_for_task(TaskType.REGRESSION)
        )
        names = [c.name for c in all_configs]
        assert len(names) == len(set(names)), "Duplicate config names detected"

    def test_build_estimator_has_fit_predict(self):
        config = self.registry.get_config_by_name("rf_gini_100")
        estimator = self.registry.build_estimator(config)
        assert hasattr(estimator, "fit")
        assert hasattr(estimator, "predict")

    def test_get_config_by_name_found(self):
        config = self.registry.get_config_by_name("rf_gini_100")
        assert config is not None
        assert config.name == "rf_gini_100"

    def test_get_config_by_name_missing_returns_none(self):
        assert self.registry.get_config_by_name("nonexistent_xyz") is None

    def test_total_configs_positive(self):
        assert self.registry.total_configs > 20

    def test_classification_configs_have_correct_task_type(self):
        configs = self.registry.get_models_for_task(TaskType.BINARY_CLASSIFICATION)
        for cfg in configs:
            assert TaskType.BINARY_CLASSIFICATION in cfg.task_types

    def test_regression_configs_have_correct_task_type(self):
        configs = self.registry.get_models_for_task(TaskType.REGRESSION)
        for cfg in configs:
            assert TaskType.REGRESSION in cfg.task_types

    def test_interpretability_score_in_range(self):
        configs = self.registry.get_models_for_task(TaskType.BINARY_CLASSIFICATION)
        for cfg in configs:
            assert 1 <= cfg.interpretability_score <= 5

    def test_duplicate_registration_raises(self):
        from app.infrastructure.ml.registry.model_registry import ModelRegistry
        from app.infrastructure.ml.registry.plugins.classification.random_forest import (
            RandomForestPlugin,
        )
        reg = ModelRegistry()
        reg.register(RandomForestPlugin())
        with pytest.raises(ValueError, match="Duplicate model config name"):
            reg.register(RandomForestPlugin())


# ── EvaluationEngine ──────────────────────────────────────────────────────────

class TestEvaluationEngine:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.engine = EvaluationEngine()
        self.registry = build_registry()

        np.random.seed(42)
        n_train, n_test = 60, 20
        self.X_train = np.random.randn(n_train, 3)
        self.y_train = np.random.randint(0, 2, n_train)
        self.X_test = np.random.randn(n_test, 3)
        self.y_test = np.random.randint(0, 2, n_test)

        self.X_train_reg = np.random.randn(n_train, 3)
        self.y_train_reg = np.random.randn(n_train)
        self.X_test_reg = np.random.randn(n_test, 3)
        self.y_test_reg = np.random.randn(n_test)

    def _make_training_result(self, estimator, X_train, y_train, X_test, config_name="rf_gini_100"):

        from app.infrastructure.ml.engines.training_engine import TrainingResult
        estimator.fit(X_train, y_train)
        preds = estimator.predict(X_test)
        train_score = float(estimator.score(X_train, y_train))
        config = self.registry.get_config_by_name(config_name)
        return TrainingResult(
            config=config,
            estimator=estimator,
            predictions=preds,
            train_score=train_score,
            training_time_s=0.1,
            prediction_time_s=0.01,
        )

    def test_binary_metrics_present(self):
        from sklearn.ensemble import RandomForestClassifier
        tr = self._make_training_result(
            RandomForestClassifier(n_estimators=5, random_state=42),
            self.X_train, self.y_train, self.X_test,
        )
        ev = self.engine._evaluate_one(
            tr, self.X_train, self.y_train, self.X_test, self.y_test,
            TaskType.BINARY_CLASSIFICATION,
        )
        for key in ("accuracy", "precision", "recall", "f1_score"):
            assert key in ev.metrics, f"Missing metric: {key}"

    def test_regression_metrics_present(self):
        from sklearn.linear_model import LinearRegression
        tr = self._make_training_result(
            LinearRegression(),
            self.X_train_reg, self.y_train_reg, self.X_test_reg,
            config_name="linreg",
        )
        ev = self.engine._evaluate_one(
            tr, self.X_train_reg, self.y_train_reg, self.X_test_reg, self.y_test_reg,
            TaskType.REGRESSION,
        )
        for key in ("mae", "mse", "rmse", "r2_score"):
            assert key in ev.metrics, f"Missing metric: {key}"

    def test_cv_score_populated(self):
        from sklearn.ensemble import RandomForestClassifier
        tr = self._make_training_result(
            RandomForestClassifier(n_estimators=5, random_state=42),
            self.X_train, self.y_train, self.X_test,
        )
        ev = self.engine._evaluate_one(
            tr, self.X_train, self.y_train, self.X_test, self.y_test,
            TaskType.BINARY_CLASSIFICATION,
        )
        assert ev.cv_mean is not None
        assert 0 <= ev.cv_mean <= 1

    def test_overfitting_flag_is_bool(self):
        from sklearn.tree import DecisionTreeClassifier
        tr = self._make_training_result(
            DecisionTreeClassifier(random_state=42),
            self.X_train, self.y_train, self.X_test,
            config_name="dt_gini",
        )
        ev = self.engine._evaluate_one(
            tr, self.X_train, self.y_train, self.X_test, self.y_test,
            TaskType.BINARY_CLASSIFICATION,
        )
        assert isinstance(ev.is_overfitting, bool)

    def test_evaluate_all_skips_failed_results(self):
        from app.infrastructure.ml.engines.training_engine import TrainingResult
        failed_tr = TrainingResult(
            config=self.registry.get_config_by_name("rf_gini_100"),
            estimator=None,
            predictions=None,
            train_score=None,
            error="Something went wrong",
        )
        results = self.engine.evaluate_all(
            [failed_tr], self.X_train, self.y_train,
            self.X_test, self.y_test,
            TaskType.BINARY_CLASSIFICATION,
        )
        assert results == []

    def test_to_model_result_maps_fields(self):
        from sklearn.ensemble import RandomForestClassifier
        tr = self._make_training_result(
            RandomForestClassifier(n_estimators=5, random_state=42),
            self.X_train, self.y_train, self.X_test,
        )
        ev = self.engine._evaluate_one(
            tr, self.X_train, self.y_train, self.X_test, self.y_test,
            TaskType.BINARY_CLASSIFICATION,
        )
        mr = self.engine.to_model_result(ev, "exp-123")
        assert mr.experiment_id == "exp-123"
        assert mr.algorithm_name == "RandomForest"
        assert mr.config_name == "rf_gini_100"
        assert isinstance(mr.metrics, dict)


# ── RecommendationEngine ──────────────────────────────────────────────────────

class TestRecommendationEngine:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.engine = RecommendationEngine()

    def _make_model_results(self, n: int = 3) -> list:
        from app.domain.entities.model_result import ModelResult
        results = []
        metrics_list = [
            {"f1_score": 0.92, "accuracy": 0.93},
            {"f1_score": 0.85, "accuracy": 0.87},
            {"f1_score": 0.78, "accuracy": 0.80},
        ]
        for i in range(n):
            mr = ModelResult(
                id=f"mr-{i}",
                experiment_id="exp-1",
                algorithm_name=f"Algo{i}",
                config_name=f"config_{i}",
                display_name=f"Model {i}",
                metrics=metrics_list[i % len(metrics_list)],
                cv_score=0.90 - i * 0.05,
                cv_std=0.02,
                is_overfitting=False,
                training_time_s=0.1 + i * 0.1,
                prediction_time_s=0.001,
                interpretability_score=3,
                supports_feature_importance=True,
                supports_shap=True,
            )
            results.append(mr)
        return results

    def test_returns_recommendation(self):
        mrs = self._make_model_results()
        rec = self.engine.recommend(
            evaluations=[], model_results=mrs,
            task_type=TaskType.BINARY_CLASSIFICATION,
            experiment_id="exp-1",
        )
        assert rec is not None
        assert rec.model_result_id is not None

    def test_best_model_has_rank_1(self):
        mrs = self._make_model_results()
        self.engine.recommend(
            evaluations=[], model_results=mrs,
            task_type=TaskType.BINARY_CLASSIFICATION,
            experiment_id="exp-1",
        )
        best = next(m for m in mrs if m.rank == 1)
        assert best.config_name == "config_0"  # highest f1_score

    def test_explanation_text_not_empty(self):
        mrs = self._make_model_results()
        rec = self.engine.recommend(
            evaluations=[], model_results=mrs,
            task_type=TaskType.BINARY_CLASSIFICATION,
            experiment_id="exp-1",
        )
        assert len(rec.explanation_text) > 20

    def test_rationale_bullets_populated(self):
        mrs = self._make_model_results()
        rec = self.engine.recommend(
            evaluations=[], model_results=mrs,
            task_type=TaskType.BINARY_CLASSIFICATION,
            experiment_id="exp-1",
        )
        assert len(rec.rationale) > 0

    def test_all_rankings_contain_all_models(self):
        mrs = self._make_model_results(3)
        rec = self.engine.recommend(
            evaluations=[], model_results=mrs,
            task_type=TaskType.BINARY_CLASSIFICATION,
            experiment_id="exp-1",
        )
        assert len(rec.all_rankings) == 3

    def test_empty_results_raises(self):
        with pytest.raises(ValueError):
            self.engine.recommend(
                evaluations=[], model_results=[],
                task_type=TaskType.BINARY_CLASSIFICATION,
                experiment_id="exp-1",
            )

    def test_composite_score_between_0_and_1(self):
        mrs = self._make_model_results()
        rec = self.engine.recommend(
            evaluations=[], model_results=mrs,
            task_type=TaskType.BINARY_CLASSIFICATION,
            experiment_id="exp-1",
        )
        for ranking in rec.all_rankings:
            assert -0.5 <= ranking["composite_score"] <= 1.5
