"""
Background service for running training pipeline with proper session management.
Similar to analysis_background_service, this ensures the training use case
runs with a fresh database session independent of the HTTP request lifecycle.
"""


from app.application.use_cases.training.training_use_cases import RunTrainingUseCase
from app.core.logging import get_logger
from app.domain.interfaces.services.storage_service import IStorageService
from app.infrastructure.database.repositories.experiment_repository import ExperimentRepository
from app.infrastructure.database.session import get_session_factory
from app.infrastructure.ml.engines.evaluation_engine import EvaluationEngine
from app.infrastructure.ml.engines.recommendation_engine import RecommendationEngine
from app.infrastructure.ml.engines.training_engine import TrainingEngine

logger = get_logger(__name__)


class TrainingBackgroundService:
    """
    Service to run training as a background task with proper session management.
    Creates a fresh database session for each training job to avoid session issues.
    """

    def __init__(
        self,
        storage: IStorageService,
        training_engine: TrainingEngine,
        evaluation_engine: EvaluationEngine,
        recommendation_engine: RecommendationEngine,
    ) -> None:
        self._storage = storage
        self._training_engine = training_engine
        self._evaluation_engine = evaluation_engine
        self._recommendation_engine = recommendation_engine
        logger.info("TrainingBackgroundService initialized")

    async def run(self, experiment_id: str) -> None:
        """
        Run training for the given experiment as a background task.
        Creates a fresh database session for this operation.
        """
        logger.info(
            "training_background_started",
            extra={"experiment_id": experiment_id},
        )

        session_factory = get_session_factory()
        async with session_factory() as session:
            try:
                # Create repository with fresh session
                repo = ExperimentRepository(session)

                # Create use case with fresh dependencies
                use_case = RunTrainingUseCase(
                    repo,
                    self._storage,
                    self._training_engine,
                    self._evaluation_engine,
                    self._recommendation_engine,
                )

                # Execute training
                await use_case.execute(experiment_id)

                await session.commit()
                logger.info(
                    "training_background_completed",
                    extra={"experiment_id": experiment_id},
                )
            except Exception as e:
                await session.rollback()
                logger.error(
                    "training_background_failed",
                    extra={"experiment_id": experiment_id, "error": str(e)},
                )
                raise
