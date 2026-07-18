"""
Experiment API routes — CRUD for experiments.
"""

from fastapi import APIRouter, Depends, Query, status

from app.api.v1.dependencies import (
    get_create_experiment_use_case,
    get_experiment_repo,
)
from app.application.dto.dataset_dto import CreateExperimentRequest
from app.application.use_cases.training.training_use_cases import (
    CreateExperimentUseCase,
    DeleteExperimentUseCase,
    GetExperimentUseCase,
    ListExperimentsUseCase,
)
from app.core.constants import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE

router = APIRouter(prefix="/experiments", tags=["Experiments"])


def _ok(data, message: str = "") -> dict:
    return {"success": True, "data": data, "message": message, "errors": {}}


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_experiment(
    req: CreateExperimentRequest,
    use_case: CreateExperimentUseCase = Depends(get_create_experiment_use_case),
):
    """Create a new experiment linked to an analyzed dataset."""
    exp = await use_case.execute(req)
    return _ok(exp.model_dump(), "Experiment created.")


@router.get("")
async def list_experiments(
    limit: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    offset: int = Query(0, ge=0),
    repo=Depends(get_experiment_repo),
):
    use_case = ListExperimentsUseCase(repo)
    experiments = await use_case.execute(limit=limit, offset=offset)
    return _ok([e.model_dump() for e in experiments])


@router.get("/{experiment_id}")
async def get_experiment(
    experiment_id: str,
    repo=Depends(get_experiment_repo),
):
    use_case = GetExperimentUseCase(repo)
    exp = await use_case.execute(experiment_id)
    return _ok(exp.model_dump())


@router.delete("/{experiment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_experiment(
    experiment_id: str,
    repo=Depends(get_experiment_repo),
):
    use_case = DeleteExperimentUseCase(repo)
    await use_case.execute(experiment_id)
