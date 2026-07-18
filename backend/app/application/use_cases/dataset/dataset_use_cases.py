"""
Dataset use cases: upload, analyze, and preview datasets.
"""

import io
from pathlib import Path

import pandas as pd

from app.application.dto.dataset_dto import (
    DatasetAnalysisDTO,
    DatasetDTO,
    DatasetPreviewDTO,
    DatasetUploadResponse,
)
from app.core.constants import ALLOWED_EXTENSIONS, PREVIEW_ROWS
from app.core.logging import get_logger
from app.domain.entities.dataset import ColumnProfile, Dataset, DatasetAnalysis
from app.domain.exceptions.domain_exceptions import (
    DatasetNotFoundError,
    DuplicateDatasetError,
    FileTooLargeError,
    InvalidFileTypeError,
)
from app.domain.interfaces.repositories.dataset_repository import IDatasetRepository
from app.domain.interfaces.services.storage_service import IStorageService
from app.infrastructure.ml.engines.analysis_engine import AnalysisEngine

logger = get_logger(__name__)


def _map_column_profile(cp: ColumnProfile):
    from app.application.dto.dataset_dto import ColumnProfileDTO
    return ColumnProfileDTO(**vars(cp))


def _map_analysis(a: DatasetAnalysis) -> DatasetAnalysisDTO:
    return DatasetAnalysisDTO(
        dataset_id=a.dataset_id,
        task_type=a.task_type,
        suggested_target_column=a.suggested_target_column,
        quality_score=a.quality_score,
        row_count=a.row_count,
        column_count=a.column_count,
        duplicate_row_count=a.duplicate_row_count,
        missing_value_pct=a.missing_value_pct,
        column_profiles=[_map_column_profile(cp) for cp in a.column_profiles],
        class_distribution=a.class_distribution,
        is_imbalanced=a.is_imbalanced,
        correlation_matrix=a.correlation_matrix,
        outlier_counts=a.outlier_counts,
        warnings=a.warnings,
        recommendations=a.recommendations,
        analyzed_at=a.analyzed_at,
        progress=getattr(a, "progress", 0),
        steps_total=getattr(a, "steps_total", None),
        steps_completed=getattr(a, "steps_completed", None),
    )


def _map_dataset(d: Dataset, analysis: DatasetAnalysis | None = None) -> DatasetDTO:
    return DatasetDTO(
        id=d.id,
        filename=d.filename,
        original_name=d.original_name,
        file_size=d.file_size,
        row_count=d.row_count,
        column_count=d.column_count,
        status=d.status.value,
        analysis=_map_analysis(analysis) if analysis else None,
        created_at=d.created_at,
        updated_at=d.updated_at,
    )


class UploadDatasetUseCase:
    """Handle file upload, validation, storage, and initial DB record creation."""

    def __init__(
        self,
        dataset_repo: IDatasetRepository,
        storage: IStorageService,
        max_upload_bytes: int,
    ) -> None:
        self._repo = dataset_repo
        self._storage = storage
        self._max_bytes = max_upload_bytes

    async def execute(
        self, file_bytes: bytes, filename: str
    ) -> DatasetUploadResponse:
        # Validate extension
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise InvalidFileTypeError(filename, ALLOWED_EXTENSIONS)

        # Validate size
        if len(file_bytes) > self._max_bytes:
            raise FileTooLargeError(len(file_bytes), self._max_bytes)

        # Check for duplicate filename
        existing_datasets = await self._repo.list_all(limit=1000, offset=0)
        filename_only = Path(filename).name
        if any(d.filename == filename_only or d.original_name == filename for d in existing_datasets):
            raise DuplicateDatasetError(filename)

        # Store file
        file_path = await self._storage.save_upload(file_bytes, filename)

        # Read to get row/column count
        df = _read_dataframe(file_bytes, filename)

        # Persist dataset record
        dataset = Dataset(
            filename=Path(filename).name,
            original_name=filename,
            file_path=file_path,
            file_size=len(file_bytes),
            row_count=len(df),
            column_count=len(df.columns),
        )
        dataset = await self._repo.save(dataset)

        logger.info(
            "dataset_uploaded",
            dataset_id=dataset.id,
            filename=filename,
            rows=dataset.row_count,
            columns=dataset.column_count,
        )
        return DatasetUploadResponse(
            id=dataset.id,
            filename=dataset.filename,
            original_name=dataset.original_name,
            file_size=dataset.file_size,
            status=dataset.status.value,
            created_at=dataset.created_at,
        )


class AnalyzeDatasetUseCase:
    """Run the AnalysisEngine on a stored dataset and persist results."""

    def __init__(
        self,
        dataset_repo: IDatasetRepository,
        storage: IStorageService,
        analysis_engine: AnalysisEngine,
    ) -> None:
        self._repo = dataset_repo
        self._storage = storage
        self._engine = analysis_engine

    async def execute(self, dataset_id: str) -> DatasetAnalysisDTO:
        dataset = await self._repo.get_by_id(dataset_id)
        if not dataset:
            raise DatasetNotFoundError(dataset_id)

        # Load file and parse
        logger.info("analysis_started", dataset_id=dataset_id)
        file_bytes = await self._storage.read_file(dataset.file_path)
        df = _read_dataframe(file_bytes, dataset.original_name)

        # Update status to analyzing
        dataset.mark_analyzing()
        await self._repo.save(dataset)

        try:
            analysis = self._engine.analyze(df, dataset_id)
            await self._repo.save_analysis(analysis)

            # Update dataset with row/column counts and analyzed status
            dataset.mark_analyzed(analysis)
            dataset.row_count = analysis.row_count
            dataset.column_count = analysis.column_count
            await self._repo.save(dataset)

            logger.info("analysis_persisted", dataset_id=dataset_id)
            return _map_analysis(analysis)

        except Exception as exc:
            logger.error("analysis_failed", dataset_id=dataset_id, error=str(exc), exc_info=True)
            dataset.mark_error()
            await self._repo.save(dataset)
            raise


class GetDatasetUseCase:
    def __init__(self, dataset_repo: IDatasetRepository) -> None:
        self._repo = dataset_repo

    async def execute(self, dataset_id: str) -> DatasetDTO:
        dataset = await self._repo.get_by_id(dataset_id)
        if not dataset:
            raise DatasetNotFoundError(dataset_id)
        analysis = await self._repo.get_analysis(dataset_id)
        return _map_dataset(dataset, analysis)


class PreviewDatasetUseCase:
    def __init__(
        self,
        dataset_repo: IDatasetRepository,
        storage: IStorageService,
    ) -> None:
        self._repo = dataset_repo
        self._storage = storage

    async def execute(self, dataset_id: str, n_rows: int = PREVIEW_ROWS) -> DatasetPreviewDTO:
        dataset = await self._repo.get_by_id(dataset_id)
        if not dataset:
            raise DatasetNotFoundError(dataset_id)

        file_bytes = await self._storage.read_file(dataset.file_path)
        df = _read_dataframe(file_bytes, dataset.original_name)
        preview_df = df.head(n_rows)

        return DatasetPreviewDTO(
            columns=list(preview_df.columns),
            rows=preview_df.fillna("").astype(str).to_dict(orient="records"),
            total_rows=len(df),
        )


class ListDatasetsUseCase:
    def __init__(self, dataset_repo: IDatasetRepository) -> None:
        self._repo = dataset_repo

    async def execute(self, limit: int = 20, offset: int = 0) -> list[DatasetDTO]:
        datasets = await self._repo.list_all(limit=limit, offset=offset)
        return [_map_dataset(d) for d in datasets]


class DeleteDatasetUseCase:
    def __init__(
        self,
        dataset_repo: IDatasetRepository,
        storage: IStorageService,
    ) -> None:
        self._repo = dataset_repo
        self._storage = storage

    async def execute(self, dataset_id: str) -> None:
        dataset = await self._repo.get_by_id(dataset_id)
        if not dataset:
            raise DatasetNotFoundError(dataset_id)
        await self._storage.delete_file(dataset.file_path)
        await self._repo.delete(dataset_id)
        logger.info("dataset_deleted", dataset_id=dataset_id)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_dataframe(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parse CSV or Excel bytes into a DataFrame."""
    ext = Path(filename).suffix.lower()
    buf = io.BytesIO(file_bytes)
    if ext == ".csv":
        return pd.read_csv(buf)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(buf)
    raise InvalidFileTypeError(filename, ALLOWED_EXTENSIONS)
