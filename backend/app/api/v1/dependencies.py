"""
Dependency injection for FastAPI routes.
All use cases and services are wired here and injected via Depends().
"""

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.application.use_cases.dataset.dataset_use_cases import (
    AnalyzeDatasetUseCase,
    UploadDatasetUseCase,
)
from app.application.use_cases.training.training_use_cases import (
    CreateExperimentUseCase,
    ExecutePreprocessingUseCase,
    ExplainModelUseCase,
    RecommendPreprocessingUseCase,
    RunTrainingUseCase,
)
from app.core.config import Settings, get_settings
from app.infrastructure.database.repositories.dataset_repository import DatasetRepository
from app.infrastructure.database.repositories.experiment_repository import ExperimentRepository
from app.infrastructure.database.session import get_db_session, get_session_factory
from app.infrastructure.ml.engines.analysis_engine import AnalysisEngine
from app.infrastructure.ml.engines.evaluation_engine import EvaluationEngine
from app.infrastructure.ml.engines.explainability_engine import ExplainabilityEngine
from app.infrastructure.ml.engines.preprocessing_engine import PreprocessingEngine
from app.infrastructure.ml.engines.recommendation_engine import RecommendationEngine
from app.infrastructure.ml.engines.training_engine import TrainingEngine
from app.infrastructure.ml.registry.model_registry import ModelRegistry
from app.infrastructure.services.analysis_background_service import AnalysisBackgroundService
from app.infrastructure.services.training_background_service import TrainingBackgroundService
from app.infrastructure.storage.local_storage import LocalStorageService

# ── Singleton services (built once) ──────────────────────────────────────────

_registry: ModelRegistry | None = None
_storage: LocalStorageService | None = None
_analysis_bg_service: AnalysisBackgroundService | None = None


def get_model_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        from app.infrastructure.ml.registry.registry_bootstrap import build_registry

        _registry = build_registry()
    return _registry


def get_storage_service() -> LocalStorageService:
    global _storage
    if _storage is None:
        _storage = LocalStorageService()
    return _storage


def get_analysis_background_service() -> AnalysisBackgroundService:
    """
    Return the singleton AnalysisBackgroundService.

    This service holds the session *factory* (not a session), so it is safe
    to reuse across requests.  Each call to ``svc.run(dataset_id)`` opens its
    own fresh AsyncSession that is completely independent of any HTTP request.
    """
    global _analysis_bg_service
    if _analysis_bg_service is None:
        _analysis_bg_service = AnalysisBackgroundService(
            session_factory=get_session_factory(),
            storage=get_storage_service(),
            analysis_engine=AnalysisEngine(),
        )
    return _analysis_bg_service


_training_bg_service: TrainingBackgroundService | None = None


# ── Per-request dependencies ──────────────────────────────────────────────────


def get_dataset_repo(
    db: AsyncSession = Depends(get_db_session),
) -> DatasetRepository:
    return DatasetRepository(db)


def get_experiment_repo(
    db: AsyncSession = Depends(get_db_session),
) -> ExperimentRepository:
    return ExperimentRepository(db)


def get_analysis_engine() -> AnalysisEngine:
    return AnalysisEngine()


def get_preprocessing_engine() -> PreprocessingEngine:
    return PreprocessingEngine()


def get_training_engine(
    registry: ModelRegistry = Depends(get_model_registry),
) -> TrainingEngine:
    return TrainingEngine(registry)


def get_evaluation_engine() -> EvaluationEngine:
    return EvaluationEngine()


def get_recommendation_engine() -> RecommendationEngine:
    return RecommendationEngine()


def get_explainability_engine() -> ExplainabilityEngine:
    return ExplainabilityEngine()


# ── Background services ────────────────────────────────────────────────────────

_training_bg_service: TrainingBackgroundService | None = None


def get_training_background_service(
    storage: LocalStorageService = Depends(get_storage_service),
    training_engine: TrainingEngine = Depends(get_training_engine),
    eval_engine: EvaluationEngine = Depends(get_evaluation_engine),
    rec_engine: RecommendationEngine = Depends(get_recommendation_engine),
) -> TrainingBackgroundService:
    """
    Return the singleton TrainingBackgroundService.

    This service holds the ML engines and storage service, so it is safe
    to reuse across requests. Each call to ``svc.run(experiment_id)`` opens its
    own fresh AsyncSession that is completely independent of any HTTP request.
    """
    global _training_bg_service
    if _training_bg_service is None:
        _training_bg_service = TrainingBackgroundService(
            storage=storage,
            training_engine=training_engine,
            evaluation_engine=eval_engine,
            recommendation_engine=rec_engine,
        )
    return _training_bg_service


# ── Use case factories ────────────────────────────────────────────────────────


def get_upload_use_case(
    repo: DatasetRepository = Depends(get_dataset_repo),
    storage: LocalStorageService = Depends(get_storage_service),
    settings: Settings = Depends(get_settings),
) -> UploadDatasetUseCase:
    return UploadDatasetUseCase(repo, storage, settings.max_upload_bytes)


def get_analyze_use_case(
    repo: DatasetRepository = Depends(get_dataset_repo),
    storage: LocalStorageService = Depends(get_storage_service),
    engine: AnalysisEngine = Depends(get_analysis_engine),
) -> AnalyzeDatasetUseCase:
    return AnalyzeDatasetUseCase(repo, storage, engine)


def get_create_experiment_use_case(
    exp_repo: ExperimentRepository = Depends(get_experiment_repo),
    ds_repo: DatasetRepository = Depends(get_dataset_repo),
) -> CreateExperimentUseCase:
    return CreateExperimentUseCase(exp_repo, ds_repo)


def get_recommend_preprocessing_use_case(
    exp_repo: ExperimentRepository = Depends(get_experiment_repo),
    ds_repo: DatasetRepository = Depends(get_dataset_repo),
    engine: PreprocessingEngine = Depends(get_preprocessing_engine),
) -> RecommendPreprocessingUseCase:
    return RecommendPreprocessingUseCase(exp_repo, ds_repo, engine)


def get_execute_preprocessing_use_case(
    exp_repo: ExperimentRepository = Depends(get_experiment_repo),
    ds_repo: DatasetRepository = Depends(get_dataset_repo),
    storage: LocalStorageService = Depends(get_storage_service),
    engine: PreprocessingEngine = Depends(get_preprocessing_engine),
) -> ExecutePreprocessingUseCase:
    return ExecutePreprocessingUseCase(exp_repo, ds_repo, storage, engine)


def get_run_training_use_case(
    exp_repo: ExperimentRepository = Depends(get_experiment_repo),
    storage: LocalStorageService = Depends(get_storage_service),
    training_engine: TrainingEngine = Depends(get_training_engine),
    eval_engine: EvaluationEngine = Depends(get_evaluation_engine),
    rec_engine: RecommendationEngine = Depends(get_recommendation_engine),
) -> RunTrainingUseCase:
    return RunTrainingUseCase(exp_repo, storage, training_engine, eval_engine, rec_engine)


def get_explain_use_case(
    exp_repo: ExperimentRepository = Depends(get_experiment_repo),
    storage: LocalStorageService = Depends(get_storage_service),
    engine: ExplainabilityEngine = Depends(get_explainability_engine),
) -> ExplainModelUseCase:
    return ExplainModelUseCase(exp_repo, storage, engine)
