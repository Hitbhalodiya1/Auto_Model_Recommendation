"""
Experiment, preprocessing, training, evaluation, and recommendation use cases.
"""

import pickle

from app.application.dto.dataset_dto import (
    CreateExperimentRequest,
    ExperimentDTO,
    ExplainabilityDTO,
    FeatureImportanceDTO,
    ModelResultDTO,
    PreprocessingRecommendationDTO,
    PreprocessingStatusDTO,
    PreprocessingStepDTO,
    RecommendationDTO,
    TrainingStatusDTO,
)
from app.application.use_cases.dataset.dataset_use_cases import _read_dataframe
from app.core.logging import get_logger
from app.domain.entities.experiment import Experiment, ExperimentStatus
from app.domain.entities.model_result import ModelResult
from app.domain.exceptions.domain_exceptions import (
    AnalysisNotCompleteError,
    DatasetNotFoundError,
    ExperimentNotFoundError,
    PreprocessingNotCompleteError,
)
from app.domain.interfaces.repositories.dataset_repository import IDatasetRepository
from app.domain.interfaces.repositories.experiment_repository import IExperimentRepository
from app.domain.interfaces.services.storage_service import IStorageService
from app.domain.value_objects.task_type import TaskType
from app.infrastructure.ml.engines.evaluation_engine import EvaluationEngine
from app.infrastructure.ml.engines.explainability_engine import ExplainabilityEngine
from app.infrastructure.ml.engines.preprocessing_engine import PreprocessingEngine
from app.infrastructure.ml.engines.recommendation_engine import RecommendationEngine
from app.infrastructure.ml.engines.training_engine import TrainingEngine

logger = get_logger(__name__)


def _map_experiment(e: Experiment) -> ExperimentDTO:
    return ExperimentDTO(
        id=e.id,
        name=e.name,
        description=e.description,
        dataset_id=e.dataset_id,
        status=e.status.value,
        task_type=e.task_type,
        target_column=e.target_column,
        created_at=e.created_at,
        updated_at=e.updated_at,
    )


def _map_model_result(mr: ModelResult) -> ModelResultDTO:
    return ModelResultDTO(
        id=mr.id,
        experiment_id=mr.experiment_id,
        algorithm_name=mr.algorithm_name,
        config_name=mr.config_name,
        display_name=mr.display_name,
        configuration=mr.configuration,
        metrics=mr.metrics,
        cv_score=mr.cv_score,
        cv_std=mr.cv_std,
        is_overfitting=mr.is_overfitting,
        training_time_s=mr.training_time_s,
        prediction_time_s=mr.prediction_time_s,
        is_recommended=mr.is_recommended,
        rank=mr.rank,
        requires_scaling=mr.requires_scaling,
        supports_feature_importance=mr.supports_feature_importance,
        supports_shap=mr.supports_shap,
        interpretability_score=mr.interpretability_score,
        error_message=mr.error_message,
        created_at=mr.created_at,
    )


# ── Experiment CRUD ───────────────────────────────────────────────────────────

class CreateExperimentUseCase:
    def __init__(
        self,
        experiment_repo: IExperimentRepository,
        dataset_repo: IDatasetRepository,
    ) -> None:
        self._exp_repo = experiment_repo
        self._ds_repo = dataset_repo

    async def execute(self, req: CreateExperimentRequest) -> ExperimentDTO:
        dataset = await self._ds_repo.get_by_id(req.dataset_id)
        if not dataset:
            raise DatasetNotFoundError(req.dataset_id)
        if not dataset.is_ready:
            raise AnalysisNotCompleteError(req.dataset_id)

        # Determine task type: from request or from analysis
        analysis = await self._ds_repo.get_analysis(req.dataset_id)
        task_type = req.task_type or (analysis.task_type if analysis else None)

        experiment = Experiment(
            name=req.name,
            description=req.description,
            dataset_id=req.dataset_id,
            task_type=task_type,
            target_column=req.target_column,
        )
        experiment = await self._exp_repo.save(experiment)
        logger.info("experiment_created", experiment_id=experiment.id, task_type=task_type)
        return _map_experiment(experiment)


class GetExperimentUseCase:
    def __init__(self, experiment_repo: IExperimentRepository) -> None:
        self._repo = experiment_repo

    async def execute(self, experiment_id: str) -> ExperimentDTO:
        exp = await self._repo.get_by_id(experiment_id)
        if not exp:
            raise ExperimentNotFoundError(experiment_id)
        return _map_experiment(exp)


class ListExperimentsUseCase:
    def __init__(self, experiment_repo: IExperimentRepository) -> None:
        self._repo = experiment_repo

    async def execute(self, limit: int = 20, offset: int = 0) -> list[ExperimentDTO]:
        exps = await self._repo.list_all(limit=limit, offset=offset)
        return [_map_experiment(e) for e in exps]


class DeleteExperimentUseCase:
    def __init__(self, experiment_repo: IExperimentRepository) -> None:
        self._repo = experiment_repo

    async def execute(self, experiment_id: str) -> None:
        exp = await self._repo.get_by_id(experiment_id)
        if not exp:
            raise ExperimentNotFoundError(experiment_id)
        await self._repo.delete(experiment_id)


# ── Preprocessing ─────────────────────────────────────────────────────────────

class RecommendPreprocessingUseCase:
    def __init__(
        self,
        experiment_repo: IExperimentRepository,
        dataset_repo: IDatasetRepository,
        preprocessing_engine: PreprocessingEngine,
    ) -> None:
        self._exp_repo = experiment_repo
        self._ds_repo = dataset_repo
        self._engine = preprocessing_engine

    async def execute(self, experiment_id: str) -> PreprocessingRecommendationDTO:
        exp = await self._exp_repo.get_by_id(experiment_id)
        if not exp:
            raise ExperimentNotFoundError(experiment_id)

        analysis = await self._ds_repo.get_analysis(exp.dataset_id)
        if not analysis:
            raise AnalysisNotCompleteError(exp.dataset_id)

        pipeline = self._engine.recommend(analysis, exp.target_column)

        # Save recommended pipeline to experiment
        pipeline.experiment_id = experiment_id
        exp.preprocessing_pipeline = pipeline
        await self._exp_repo.save(exp)

        scaler_step = next((s for s in pipeline.steps if s.name == "scale_features"), None)
        return PreprocessingRecommendationDTO(
            experiment_id=experiment_id,
            steps=[
                PreprocessingStepDTO(
                    name=s.name,
                    display_name=s.display_name,
                    strategy=s.strategy,
                    params=s.params,
                    explanation=s.explanation,
                    affects_columns=s.affects_columns,
                )
                for s in pipeline.steps
            ],
            step_count=len(pipeline.steps),
            recommended_scaler=scaler_step.strategy if scaler_step else "standard",
        )


class ExecutePreprocessingUseCase:
    def __init__(
        self,
        experiment_repo: IExperimentRepository,
        dataset_repo: IDatasetRepository,
        storage: IStorageService,
        preprocessing_engine: PreprocessingEngine,
    ) -> None:
        self._exp_repo = experiment_repo
        self._ds_repo = dataset_repo
        self._storage = storage
        self._engine = preprocessing_engine

    async def execute(self, experiment_id: str) -> PreprocessingStatusDTO:
        exp = await self._exp_repo.get_by_id(experiment_id)
        if not exp:
            raise ExperimentNotFoundError(experiment_id)
        if not exp.preprocessing_pipeline:
            raise PreprocessingNotCompleteError(experiment_id)

        dataset = await self._ds_repo.get_by_id(exp.dataset_id)
        file_bytes = await self._storage.read_file(dataset.file_path)
        df = _read_dataframe(file_bytes, dataset.original_name)

        from app.core.config import get_settings
        settings = get_settings()

        result = self._engine.execute(
            df=df,
            pipeline_def=exp.preprocessing_pipeline,
            target_column=exp.target_column,
            task_type=exp.task_type,
            test_size=settings.TEST_SIZE,
            random_state=settings.RANDOM_STATE,
        )

        # Serialize and save preprocessed data
        artifact = pickle.dumps({
            "X_train": result.X_train,
            "X_test": result.X_test,
            "y_train": result.y_train,
            "y_test": result.y_test,
            "feature_names": result.feature_names,
            "label_encoder": result.label_encoder,
        })
        artifact_path = f"experiments/{experiment_id}/preprocessed_data.pkl"
        await self._storage.save_artifact(artifact, artifact_path)

        exp.preprocessing_pipeline.mark_executed(artifact_path)
        await self._exp_repo.save(exp)

        return PreprocessingStatusDTO(
            experiment_id=experiment_id,
            is_executed=True,
            executed_at=exp.preprocessing_pipeline.executed_at,
            pipeline_path=artifact_path,
        )


# ── Training ──────────────────────────────────────────────────────────────────

class RunTrainingUseCase:
    """
    Loads preprocessed data, runs TrainingEngine + EvaluationEngine,
    then RecommendationEngine, persisting all results.
    """

    def __init__(
        self,
        experiment_repo: IExperimentRepository,
        storage: IStorageService,
        training_engine: TrainingEngine,
        evaluation_engine: EvaluationEngine,
        recommendation_engine: RecommendationEngine,
    ) -> None:
        self._exp_repo = experiment_repo
        self._storage = storage
        self._training_engine = training_engine
        self._evaluation_engine = evaluation_engine
        self._recommendation_engine = recommendation_engine

    async def execute(self, experiment_id: str) -> TrainingStatusDTO:
        exp = await self._exp_repo.get_by_id(experiment_id)
        if not exp:
            raise ExperimentNotFoundError(experiment_id)
        if not exp.preprocessing_pipeline or not exp.preprocessing_pipeline.is_executed:
            raise PreprocessingNotCompleteError(experiment_id)

        # Load preprocessed data
        data_bytes = await self._storage.read_file(
            exp.preprocessing_pipeline.pipeline_path
        )
        data = pickle.loads(data_bytes)
        X_train = data["X_train"]
        X_test = data["X_test"]
        y_train = data["y_train"]
        y_test = data["y_test"]
        feature_names = data["feature_names"]

        task_type = TaskType(exp.task_type)

        exp.transition_to(ExperimentStatus.TRAINING)
        await self._exp_repo.save(exp)

        # Train all compatible models
        training_results = self._training_engine.train_all(
            X_train, y_train, X_test, y_test, task_type, experiment_id
        )

        exp.transition_to(ExperimentStatus.EVALUATING)
        await self._exp_repo.save(exp)

        # Evaluate
        evaluations = self._evaluation_engine.evaluate_all(
            training_results, X_train, y_train, X_test, y_test, task_type
        )

        # Convert to ModelResult entities and save
        model_results = []
        for ev in evaluations:
            mr = self._evaluation_engine.to_model_result(ev, experiment_id)
            mr = await self._exp_repo.save_model_result(mr)
            model_results.append(mr)

            # Save serialized model artifact
            if ev.training_result.estimator:
                try:
                    model_bytes = pickle.dumps(ev.training_result.estimator)
                    artifact_path = f"experiments/{experiment_id}/models/{mr.config_name}.pkl"
                    await self._storage.save_artifact(model_bytes, artifact_path)
                    mr.model_path = artifact_path
                    await self._exp_repo.save_model_result(mr)
                except Exception as exc:
                    logger.warning("model_serialization_failed", config=mr.config_name, error=str(exc))

        # Generate recommendation
        recommendation = self._recommendation_engine.recommend(
            evaluations, model_results, task_type, experiment_id
        )
        await self._exp_repo.save_recommendation(recommendation)

        # Mark recommended model
        for mr in model_results:
            if mr.id == recommendation.model_result_id:
                mr.is_recommended = True
                await self._exp_repo.save_model_result(mr)

        exp.transition_to(ExperimentStatus.COMPLETE)
        await self._exp_repo.save(exp)

        logger.info(
            "training_pipeline_complete",
            experiment_id=experiment_id,
            models_trained=len(model_results),
            recommended=recommendation.model_result_id,
        )

        return TrainingStatusDTO(
            experiment_id=experiment_id,
            status=ExperimentStatus.COMPLETE.value,
            total_models=len(model_results),
            completed_models=len([r for r in model_results if r.succeeded]),
            message=f"Training complete. {len(model_results)} models evaluated.",
        )


# ── Evaluation Results ────────────────────────────────────────────────────────

class GetEvaluationResultsUseCase:
    def __init__(self, experiment_repo: IExperimentRepository) -> None:
        self._repo = experiment_repo

    async def execute(self, experiment_id: str) -> list[ModelResultDTO]:
        exp = await self._repo.get_by_id(experiment_id)
        if not exp:
            raise ExperimentNotFoundError(experiment_id)
        results = await self._repo.get_model_results(experiment_id)
        return [_map_model_result(mr) for mr in results]


# ── Recommendation ────────────────────────────────────────────────────────────

class GetRecommendationUseCase:
    def __init__(self, experiment_repo: IExperimentRepository) -> None:
        self._repo = experiment_repo

    async def execute(self, experiment_id: str) -> RecommendationDTO:
        exp = await self._repo.get_by_id(experiment_id)
        if not exp:
            raise ExperimentNotFoundError(experiment_id)

        rec = await self._repo.get_recommendation(experiment_id)
        if not rec:
            raise ExperimentNotFoundError(f"No recommendation found for experiment {experiment_id}")

        best_mr = await self._repo.get_model_result_by_id(rec.model_result_id)

        return RecommendationDTO(
            id=rec.id,
            experiment_id=rec.experiment_id,
            model_result_id=rec.model_result_id,
            composite_score=rec.composite_score,
            score_breakdown=rec.score_breakdown,
            rationale=rec.rationale,
            explanation_text=rec.explanation_text,
            all_rankings=rec.all_rankings,
            recommended_model=_map_model_result(best_mr) if best_mr else None,
            created_at=rec.created_at,
        )


# ── Explainability ────────────────────────────────────────────────────────────

class ExplainModelUseCase:
    def __init__(
        self,
        experiment_repo: IExperimentRepository,
        storage: IStorageService,
        explainability_engine: ExplainabilityEngine,
    ) -> None:
        self._exp_repo = experiment_repo
        self._storage = storage
        self._engine = explainability_engine

    async def execute(self, experiment_id: str, model_id: str) -> ExplainabilityDTO:
        exp = await self._exp_repo.get_by_id(experiment_id)
        if not exp:
            raise ExperimentNotFoundError(experiment_id)

        mr = await self._exp_repo.get_model_result_by_id(model_id)
        if not mr:
            from app.domain.exceptions.domain_exceptions import ModelResultNotFoundError
            raise ModelResultNotFoundError(model_id)

        # Load preprocessed data and serialized model
        data_bytes = await self._storage.read_file(exp.preprocessing_pipeline.pipeline_path)
        data = pickle.loads(data_bytes)
        X_train, X_test = data["X_train"], data["X_test"]
        feature_names = data["feature_names"]

        model_bytes = await self._storage.read_file(mr.model_path)
        estimator = pickle.loads(model_bytes)

        explanation = self._engine.explain(
            estimator=estimator,
            X_train=X_train,
            X_test=X_test,
            feature_names=feature_names,
            model_result=mr,
        )

        return ExplainabilityDTO(
            model_result_id=explanation.model_result_id,
            feature_importances=[
                FeatureImportanceDTO(
                    feature=fi.feature,
                    importance=fi.importance,
                    rank=fi.rank,
                )
                for fi in explanation.feature_importances
            ],
            shap_values=explanation.shap_values,
            shap_base_value=explanation.shap_base_value,
            top_features=explanation.top_features,
            method_used=explanation.method_used,
            error=explanation.error,
        )
