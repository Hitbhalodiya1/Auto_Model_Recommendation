"""
Dataset API routes — upload, analyze, preview, list, delete.
"""

from fastapi import APIRouter, BackgroundTasks, Depends, File, Query, UploadFile, status

from app.api.v1.dependencies import (
    get_analysis_background_service,
    get_dataset_repo,
    get_storage_service,
    get_upload_use_case,
)
from app.application.use_cases.dataset.dataset_use_cases import (
    DeleteDatasetUseCase,
    GetDatasetUseCase,
    ListDatasetsUseCase,
    PreviewDatasetUseCase,
    UploadDatasetUseCase,
)
from app.core.constants import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE, PREVIEW_ROWS
from app.infrastructure.services.analysis_background_service import AnalysisBackgroundService

router = APIRouter(prefix="/datasets", tags=["Datasets"])


def _ok(data, message: str = "") -> dict:
    return {"success": True, "data": data, "message": message, "errors": {}}


# ── Upload ────────────────────────────────────────────────────────────────────

@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    use_case: UploadDatasetUseCase = Depends(get_upload_use_case),
    analysis_bg_service: AnalysisBackgroundService = Depends(get_analysis_background_service),
):
    """
    Upload a CSV or Excel dataset.
    Analysis is automatically triggered as a background task.
    """
    file_bytes = await file.read()
    result = await use_case.execute(file_bytes, file.filename or "upload")

    # Trigger analysis in background immediately after upload
    background_tasks.add_task(analysis_bg_service.run, result.id)

    return _ok(result.model_dump(), "Dataset uploaded. Analysis started in background.")


# ── List ──────────────────────────────────────────────────────────────────────

@router.get("")
async def list_datasets(
    limit: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    offset: int = Query(0, ge=0),
    repo=Depends(get_dataset_repo),
):
    use_case = ListDatasetsUseCase(repo)
    datasets = await use_case.execute(limit=limit, offset=offset)
    return _ok([d.model_dump() for d in datasets])


# ── Get ───────────────────────────────────────────────────────────────────────

@router.get("/{dataset_id}")
async def get_dataset(
    dataset_id: str,
    repo=Depends(get_dataset_repo),
):
    use_case = GetDatasetUseCase(repo)
    dataset = await use_case.execute(dataset_id)
    return _ok(dataset.model_dump())


# ── Preview ───────────────────────────────────────────────────────────────────

@router.get("/{dataset_id}/preview")
async def preview_dataset(
    dataset_id: str,
    n_rows: int = Query(PREVIEW_ROWS, ge=1, le=500),
    repo=Depends(get_dataset_repo),
    storage=Depends(get_storage_service),
):
    use_case = PreviewDatasetUseCase(repo, storage)
    preview = await use_case.execute(dataset_id, n_rows=n_rows)
    return _ok(preview.model_dump())


# ── Analyze (manual trigger) ──────────────────────────────────────────────────

@router.post("/{dataset_id}/analyze", status_code=status.HTTP_202_ACCEPTED)
async def analyze_dataset(
    dataset_id: str,
    background_tasks: BackgroundTasks,
    analysis_bg_service: AnalysisBackgroundService = Depends(get_analysis_background_service),
    repo=Depends(get_dataset_repo),
):
    """Manually re-trigger dataset analysis (runs as background task)."""
    from app.domain.exceptions.domain_exceptions import DatasetNotFoundError

    if not await repo.get_by_id(dataset_id):
        raise DatasetNotFoundError(dataset_id)

    background_tasks.add_task(analysis_bg_service.run, dataset_id)
    return _ok({"dataset_id": dataset_id}, "Analysis started.")


# ── Analysis Results ──────────────────────────────────────────────────────────

@router.get("/{dataset_id}/analysis")
async def get_analysis(
    dataset_id: str,
    repo=Depends(get_dataset_repo),
):
    """Get the analysis results for a dataset."""
    from app.domain.exceptions.domain_exceptions import (
        AnalysisNotCompleteError,
        DatasetNotFoundError,
    )
    dataset = await repo.get_by_id(dataset_id)
    if not dataset:
        raise DatasetNotFoundError(dataset_id)
    analysis = await repo.get_analysis(dataset_id)
    if not analysis:
        raise AnalysisNotCompleteError(dataset_id)
    from app.application.use_cases.dataset.dataset_use_cases import _map_analysis
    return _ok(_map_analysis(analysis).model_dump())


# ── Delete ────────────────────────────────────────────────────────────────────

@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: str,
    repo=Depends(get_dataset_repo),
    storage=Depends(get_storage_service),
):
    use_case = DeleteDatasetUseCase(repo, storage)
    await use_case.execute(dataset_id)
