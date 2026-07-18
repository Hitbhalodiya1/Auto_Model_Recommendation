"""
Concrete SQLAlchemy implementation of IExperimentRepository.
"""

import json

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.entities.experiment import (
    Experiment,
    ExperimentStatus,
    PreprocessingPipeline,
    PreprocessingStep,
)
from app.domain.entities.model_result import ModelResult, Recommendation
from app.domain.interfaces.repositories.experiment_repository import IExperimentRepository
from app.infrastructure.database.models.experiment_model import (
    ExperimentModel,
    ModelResultModel,
    RecommendationModel,
)


class ExperimentRepository(IExperimentRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ── Experiment ────────────────────────────────────────────────────────────

    async def save(self, experiment: Experiment) -> Experiment:
        preprocessing_json = None
        if experiment.preprocessing_pipeline:
            pipeline = experiment.preprocessing_pipeline
            preprocessing_json = json.dumps(
                {
                    "id": pipeline.id,
                    "steps": [vars(s) for s in pipeline.steps],
                    "pipeline_path": pipeline.pipeline_path,
                    "is_executed": pipeline.is_executed,
                    "executed_at": pipeline.executed_at.isoformat()
                    if pipeline.executed_at
                    else None,
                }
            )

        existing = await self._session.get(ExperimentModel, experiment.id)
        if existing:
            existing.name = experiment.name
            existing.description = experiment.description
            existing.status = experiment.status.value
            existing.task_type = experiment.task_type
            existing.target_column = experiment.target_column
            existing.preprocessing_json = preprocessing_json
            existing.config_json = json.dumps(experiment.config)
            existing.updated_at = experiment.updated_at
        else:
            model = ExperimentModel(
                id=experiment.id,
                name=experiment.name,
                description=experiment.description,
                dataset_id=experiment.dataset_id,
                analysis_id=experiment.analysis_id,
                status=experiment.status.value,
                task_type=experiment.task_type,
                target_column=experiment.target_column,
                preprocessing_json=preprocessing_json,
                config_json=json.dumps(experiment.config),
                created_at=experiment.created_at,
                updated_at=experiment.updated_at,
            )
            self._session.add(model)
        await self._session.flush()
        return experiment

    async def get_by_id(self, experiment_id: str) -> Experiment | None:
        model = await self._session.get(ExperimentModel, experiment_id)
        return self._to_entity(model) if model else None

    async def list_all(self, limit: int = 20, offset: int = 0) -> list[Experiment]:
        stmt = (
            select(ExperimentModel)
            .order_by(ExperimentModel.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(stmt)
        return [self._to_entity(m) for m in result.scalars().all()]

    async def delete(self, experiment_id: str) -> bool:
        model = await self._session.get(ExperimentModel, experiment_id)
        if not model:
            return False
        await self._session.delete(model)
        await self._session.flush()
        return True

    # ── Model Results ─────────────────────────────────────────────────────────

    async def save_model_result(self, result: ModelResult) -> ModelResult:
        existing = await self._session.get(ModelResultModel, result.id)
        if existing:
            existing.metrics_json = json.dumps(result.metrics)
            existing.cv_score = result.cv_score
            existing.cv_std = result.cv_std
            existing.is_overfitting = result.is_overfitting
            existing.training_time_s = result.training_time_s
            existing.prediction_time_s = result.prediction_time_s
            existing.model_path = result.model_path
            existing.is_recommended = result.is_recommended
            existing.rank = result.rank
            existing.error_message = result.error_message
        else:
            model = ModelResultModel(
                id=result.id,
                experiment_id=result.experiment_id,
                algorithm_name=result.algorithm_name,
                config_name=result.config_name,
                display_name=result.display_name,
                configuration_json=json.dumps(result.configuration),
                metrics_json=json.dumps(result.metrics),
                cv_score=result.cv_score,
                cv_std=result.cv_std,
                is_overfitting=result.is_overfitting,
                training_time_s=result.training_time_s,
                prediction_time_s=result.prediction_time_s,
                model_path=result.model_path,
                is_recommended=result.is_recommended,
                rank=result.rank,
                requires_scaling=result.requires_scaling,
                supports_feature_importance=result.supports_feature_importance,
                supports_shap=result.supports_shap,
                interpretability_score=result.interpretability_score,
                error_message=result.error_message,
                created_at=result.created_at,
            )
            self._session.add(model)
        await self._session.flush()
        return result

    async def get_model_results(self, experiment_id: str) -> list[ModelResult]:
        stmt = (
            select(ModelResultModel)
            .where(ModelResultModel.experiment_id == experiment_id)
            .order_by(ModelResultModel.rank.asc().nullslast())
        )
        result = await self._session.execute(stmt)
        return [self._result_to_entity(m) for m in result.scalars().all()]

    async def get_model_result_by_id(self, model_id: str) -> ModelResult | None:
        model = await self._session.get(ModelResultModel, model_id)
        return self._result_to_entity(model) if model else None

    # ── Recommendations ───────────────────────────────────────────────────────

    async def save_recommendation(self, recommendation: Recommendation) -> Recommendation:
        stmt = select(RecommendationModel).where(
            RecommendationModel.experiment_id == recommendation.experiment_id
        )
        result = await self._session.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            existing.model_result_id = recommendation.model_result_id
            existing.composite_score = recommendation.composite_score
            existing.score_breakdown_json = json.dumps(recommendation.score_breakdown)
            existing.rationale_json = json.dumps(recommendation.rationale)
            existing.explanation_text = recommendation.explanation_text
            existing.all_rankings_json = json.dumps(recommendation.all_rankings)
        else:
            model = RecommendationModel(
                id=recommendation.id,
                experiment_id=recommendation.experiment_id,
                model_result_id=recommendation.model_result_id,
                composite_score=recommendation.composite_score,
                score_breakdown_json=json.dumps(recommendation.score_breakdown),
                rationale_json=json.dumps(recommendation.rationale),
                explanation_text=recommendation.explanation_text,
                all_rankings_json=json.dumps(recommendation.all_rankings),
                created_at=recommendation.created_at,
            )
            self._session.add(model)
        await self._session.flush()
        return recommendation

    async def get_recommendation(self, experiment_id: str) -> Recommendation | None:
        stmt = select(RecommendationModel).where(RecommendationModel.experiment_id == experiment_id)
        result = await self._session.execute(stmt)
        model = result.scalar_one_or_none()
        return self._rec_to_entity(model) if model else None

    # ── Mappers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to_entity(model: ExperimentModel) -> Experiment:
        pipeline = None
        if model.preprocessing_json:
            data = json.loads(model.preprocessing_json)
            steps = [PreprocessingStep(**s) for s in data.get("steps", [])]
            pipeline = PreprocessingPipeline(
                id=data.get("id", ""),
                experiment_id=model.id,
                steps=steps,
                pipeline_path=data.get("pipeline_path"),
                is_executed=data.get("is_executed", False),
            )

        return Experiment(
            id=model.id,
            name=model.name,
            description=model.description or "",
            dataset_id=model.dataset_id,
            analysis_id=model.analysis_id,
            status=ExperimentStatus(model.status),
            task_type=model.task_type,
            target_column=model.target_column,
            preprocessing_pipeline=pipeline,
            config=json.loads(model.config_json) if model.config_json else {},
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    @staticmethod
    def _result_to_entity(model: ModelResultModel) -> ModelResult:
        return ModelResult(
            id=model.id,
            experiment_id=model.experiment_id,
            algorithm_name=model.algorithm_name,
            config_name=model.config_name,
            display_name=model.display_name,
            configuration=json.loads(model.configuration_json) if model.configuration_json else {},
            metrics=json.loads(model.metrics_json) if model.metrics_json else {},
            cv_score=model.cv_score,
            cv_std=model.cv_std,
            is_overfitting=model.is_overfitting,
            training_time_s=model.training_time_s,
            prediction_time_s=model.prediction_time_s,
            model_path=model.model_path,
            is_recommended=model.is_recommended,
            rank=model.rank,
            requires_scaling=model.requires_scaling,
            supports_feature_importance=model.supports_feature_importance,
            supports_shap=model.supports_shap,
            interpretability_score=model.interpretability_score,
            error_message=model.error_message,
            created_at=model.created_at,
        )

    @staticmethod
    def _rec_to_entity(model: RecommendationModel) -> Recommendation:
        return Recommendation(
            id=model.id,
            experiment_id=model.experiment_id,
            model_result_id=model.model_result_id,
            composite_score=model.composite_score,
            score_breakdown=json.loads(model.score_breakdown_json)
            if model.score_breakdown_json
            else {},
            rationale=json.loads(model.rationale_json) if model.rationale_json else [],
            explanation_text=model.explanation_text or "",
            all_rankings=json.loads(model.all_rankings_json) if model.all_rankings_json else [],
            created_at=model.created_at,
        )
