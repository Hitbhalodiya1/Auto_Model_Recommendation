"""
Preprocessing routes — recommend and execute preprocessing pipelines.
"""

from fastapi import APIRouter, Depends, status

from app.api.v1.dependencies import (
    get_execute_preprocessing_use_case,
    get_experiment_repo,
    get_recommend_preprocessing_use_case,
)
from app.application.use_cases.training.training_use_cases import (
    ExecutePreprocessingUseCase,
    RecommendPreprocessingUseCase,
)

router = APIRouter(tags=["Preprocessing"])


def _ok(data, message: str = "") -> dict:
    return {"success": True, "data": data, "message": message, "errors": {}}


@router.post("/experiments/{experiment_id}/preprocessing/recommend")
async def recommend_preprocessing(
    experiment_id: str,
    use_case: RecommendPreprocessingUseCase = Depends(get_recommend_preprocessing_use_case),
):
    """Generate intelligent preprocessing recommendations for an experiment."""
    result = await use_case.execute(experiment_id)
    return _ok(result.model_dump(), f"Generated {result.step_count} preprocessing steps.")


@router.post("/experiments/{experiment_id}/preprocessing/execute", status_code=status.HTTP_202_ACCEPTED)
async def execute_preprocessing(
    experiment_id: str,
    use_case: ExecutePreprocessingUseCase = Depends(get_execute_preprocessing_use_case),
):
    """Execute the recommended preprocessing pipeline and prepare train/test splits."""
    result = await use_case.execute(experiment_id)
    return _ok(result.model_dump(), "Preprocessing executed successfully.")


@router.get("/experiments/{experiment_id}/preprocessing/status")
async def get_preprocessing_status(
    experiment_id: str,
    repo=Depends(get_experiment_repo),
):
    """Get the current preprocessing status for an experiment."""
    from app.domain.exceptions.domain_exceptions import ExperimentNotFoundError
    exp = await repo.get_by_id(experiment_id)
    if not exp:
        raise ExperimentNotFoundError(experiment_id)

    pipeline = exp.preprocessing_pipeline
    return _ok({
        "experiment_id": experiment_id,
        "is_executed": pipeline.is_executed if pipeline else False,
        "executed_at": pipeline.executed_at.isoformat() if pipeline and pipeline.executed_at else None,
        "pipeline_path": pipeline.pipeline_path if pipeline else None,
    })
