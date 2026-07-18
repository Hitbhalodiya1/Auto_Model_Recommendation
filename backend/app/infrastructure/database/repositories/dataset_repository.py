"""
Concrete SQLAlchemy implementation of IDatasetRepository.
"""

import json

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.entities.dataset import ColumnProfile, Dataset, DatasetAnalysis, DatasetStatus
from app.domain.interfaces.repositories.dataset_repository import IDatasetRepository
from app.infrastructure.database.models.dataset_model import DatasetAnalysisModel, DatasetModel


class DatasetRepository(IDatasetRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ── Dataset CRUD ─────────────────────────────────────────────────────────

    async def save(self, dataset: Dataset) -> Dataset:
        existing = await self._session.get(DatasetModel, dataset.id)
        if existing:
            existing.filename = dataset.filename
            existing.original_name = dataset.original_name
            existing.file_path = dataset.file_path
            existing.file_size = dataset.file_size
            existing.row_count = dataset.row_count
            existing.column_count = dataset.column_count
            existing.status = dataset.status.value
            existing.updated_at = dataset.updated_at
        else:
            model = DatasetModel(
                id=dataset.id,
                filename=dataset.filename,
                original_name=dataset.original_name,
                file_path=dataset.file_path,
                file_size=dataset.file_size,
                row_count=dataset.row_count,
                column_count=dataset.column_count,
                status=dataset.status.value,
                created_at=dataset.created_at,
                updated_at=dataset.updated_at,
            )
            self._session.add(model)
        await self._session.flush()
        return dataset

    async def get_by_id(self, dataset_id: str) -> Dataset | None:
        model = await self._session.get(DatasetModel, dataset_id)
        if not model:
            return None
        return self._to_entity(model)

    async def list_all(self, limit: int = 20, offset: int = 0) -> list[Dataset]:
        stmt = (
            select(DatasetModel)
            .order_by(DatasetModel.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(stmt)
        return [self._to_entity(m) for m in result.scalars().all()]

    async def delete(self, dataset_id: str) -> bool:
        model = await self._session.get(DatasetModel, dataset_id)
        if not model:
            return False
        await self._session.delete(model)
        await self._session.flush()
        return True

    # ── Analysis ──────────────────────────────────────────────────────────────

    async def save_analysis(self, analysis: DatasetAnalysis) -> DatasetAnalysis:
        existing = await self._get_analysis_model(analysis.dataset_id)
        analysis_json = json.dumps(
            {
                "column_profiles": [vars(cp) for cp in analysis.column_profiles],
                "class_distribution": analysis.class_distribution,
                "correlation_matrix": analysis.correlation_matrix,
                "outlier_counts": analysis.outlier_counts,
                "warnings": analysis.warnings,
                "recommendations": analysis.recommendations,
                "progress": getattr(analysis, "progress", 0),
                "steps_total": getattr(analysis, "steps_total", None),
                "steps_completed": getattr(analysis, "steps_completed", None),
            }
        )

        if existing:
            existing.task_type = analysis.task_type
            existing.suggested_target_column = analysis.suggested_target_column
            existing.quality_score = analysis.quality_score
            existing.row_count = analysis.row_count
            existing.column_count = analysis.column_count
            existing.duplicate_row_count = analysis.duplicate_row_count
            existing.missing_value_pct = analysis.missing_value_pct
            existing.is_imbalanced = analysis.is_imbalanced
            existing.analysis_json = analysis_json
        else:
            model = DatasetAnalysisModel(
                dataset_id=analysis.dataset_id,
                task_type=analysis.task_type,
                suggested_target_column=analysis.suggested_target_column,
                quality_score=analysis.quality_score,
                row_count=analysis.row_count,
                column_count=analysis.column_count,
                duplicate_row_count=analysis.duplicate_row_count,
                missing_value_pct=analysis.missing_value_pct,
                is_imbalanced=analysis.is_imbalanced,
                analysis_json=analysis_json,
                analyzed_at=analysis.analyzed_at,
            )
            self._session.add(model)

        await self._session.flush()
        return analysis

    async def get_analysis(self, dataset_id: str) -> DatasetAnalysis | None:
        model = await self._get_analysis_model(dataset_id)
        if not model:
            return None
        return self._analysis_to_entity(model)

    async def _get_analysis_model(self, dataset_id: str) -> DatasetAnalysisModel | None:
        stmt = select(DatasetAnalysisModel).where(DatasetAnalysisModel.dataset_id == dataset_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    # ── Mappers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to_entity(model: DatasetModel) -> Dataset:
        return Dataset(
            id=model.id,
            filename=model.filename,
            original_name=model.original_name,
            file_path=model.file_path,
            file_size=model.file_size,
            row_count=model.row_count,
            column_count=model.column_count,
            status=DatasetStatus(model.status),
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    @staticmethod
    def _analysis_to_entity(model: DatasetAnalysisModel) -> DatasetAnalysis:
        extra: dict = {}
        if model.analysis_json:
            extra = json.loads(model.analysis_json)

        column_profiles = [ColumnProfile(**cp) for cp in extra.get("column_profiles", [])]

        progress = extra.get("progress", 0)
        steps_total = extra.get("steps_total")
        steps_completed = extra.get("steps_completed")

        return DatasetAnalysis(
            dataset_id=model.dataset_id,
            task_type=model.task_type or "",
            suggested_target_column=model.suggested_target_column,
            quality_score=model.quality_score or 0.0,
            row_count=model.row_count,
            column_count=model.column_count,
            duplicate_row_count=model.duplicate_row_count,
            missing_value_pct=model.missing_value_pct,
            column_profiles=column_profiles,
            class_distribution=extra.get("class_distribution"),
            is_imbalanced=model.is_imbalanced,
            correlation_matrix=extra.get("correlation_matrix"),
            outlier_counts=extra.get("outlier_counts"),
            warnings=extra.get("warnings", []),
            recommendations=extra.get("recommendations", []),
            analyzed_at=model.analyzed_at,
            progress=progress,
            steps_total=steps_total,
            steps_completed=steps_completed,
        )
