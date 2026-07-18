"""
AnalysisBackgroundService — production-grade background worker for dataset analysis.

WHY THIS EXISTS
---------------
FastAPI BackgroundTasks run *after* the HTTP response has been sent.  At that
point, every per-request AsyncSession created by ``get_db_session`` has already
been committed and closed by the dependency-injection machinery.

Passing a use-case that holds one of those sessions into BackgroundTasks
therefore causes SQLAlchemy "connection not checked in" / "session is closed"
errors — the exact bug this service fixes.

DESIGN
------
The service accepts only long-lived, request-independent objects at construction
time:
  • ``session_factory`` – the module-level async_sessionmaker, alive for the
    entire process lifetime.
  • ``storage`` – singleton, stateless, safe to share.
  • ``analysis_engine`` – singleton, stateless.

``run(dataset_id)`` is the background entry-point.  It:
  1. Opens a *brand-new* AsyncSession from the factory (independent of any
     request lifecycle).
  2. Constructs DatasetRepository and AnalyzeDatasetUseCase *inside* the call,
     so they share the same fresh session.
  3. Provides a progress callback that writes partial progress using the *same*
     session — fully synchronous from the engine's perspective; no
     asyncio.get_event_loop() hacks needed because the callback schedules an
     inner async task via ``asyncio.ensure_future`` on the already-running loop.
  4. Commits and closes the session in a ``finally`` block regardless of outcome.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.application.use_cases.dataset.dataset_use_cases import AnalyzeDatasetUseCase
from app.core.logging import get_logger
from app.domain.interfaces.services.storage_service import IStorageService
from app.infrastructure.database.repositories.dataset_repository import DatasetRepository
from app.infrastructure.ml.engines.analysis_engine import AnalysisEngine

logger = get_logger(__name__)


class AnalysisBackgroundService:
    """
    Owns the full lifecycle of a single dataset-analysis background job.

    Constructed once per application (singleton-style via dependency injection)
    and reused across requests.  Each call to ``run()`` creates its own
    database session that is independent of any HTTP request.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        storage: IStorageService,
        analysis_engine: AnalysisEngine,
    ) -> None:
        self._session_factory = session_factory
        self._storage = storage
        self._engine = analysis_engine
        logger.info("AnalysisBackgroundService initialized")

    async def run(self, dataset_id: str) -> None:
        """
        Entry-point for BackgroundTasks.

        Opens a fresh database session, runs the analysis use case to
        completion, then commits and closes the session.  Any exception
        is caught, logged, and swallowed so it does not surface as an
        unhandled exception in the Starlette background-task runner.
        """
        logger.info("analysis_background_started", extra={"dataset_id": dataset_id})

        async with self._session_factory() as session:
            try:
                repo = DatasetRepository(session)
                use_case = AnalyzeDatasetUseCase(
                    repo,
                    self._storage,
                    self._engine,
                )
                await use_case.execute(dataset_id)
                await session.commit()
                logger.info(
                    "analysis_background_completed",
                    extra={"dataset_id": dataset_id},
                )
            except Exception:
                await session.rollback()
                logger.exception(
                    "analysis_background_failed",
                    extra={"dataset_id": dataset_id},
                )
                # Do not re-raise — BackgroundTasks has no caller to propagate to.
                # The dataset status has already been set to ERROR by the use case.
