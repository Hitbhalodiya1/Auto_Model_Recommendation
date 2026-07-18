"""
Training routes — start training, poll status, retrieve results.
"""

from fastapi import APIRouter, BackgroundTasks, Depends, status

from app.api.v1.dependencies import (
    get_experiment_repo,
    get_explain_use_case,
    get_training_background_service,
)
from app.application.use_cases.training.training_use_cases import (
    ExplainModelUseCase,
    GetEvaluationResultsUseCase,
    GetRecommendationUseCase,
)
from app.infrastructure.services.training_background_service import TrainingBackgroundService

router = APIRouter(tags=["Training & Evaluation"])


def _ok(data, message: str = "") -> dict:
    return {"success": True, "data": data, "message": message, "errors": {}}


# ── Training ──────────────────────────────────────────────────────────────────


@router.post("/experiments/{experiment_id}/training/start", status_code=status.HTTP_202_ACCEPTED)
async def start_training(
    experiment_id: str,
    background_tasks: BackgroundTasks,
    training_bg_service: TrainingBackgroundService = Depends(get_training_background_service),
    repo=Depends(get_experiment_repo),
):
    """
    Start the full training pipeline (train + evaluate + recommend).
    Runs as a background task. Poll /training/status for completion.
    """
    from app.domain.exceptions.domain_exceptions import ExperimentNotFoundError

    if not await repo.get_by_id(experiment_id):
        raise ExperimentNotFoundError(experiment_id)

    # Use the background service which handles the session properly
    background_tasks.add_task(training_bg_service.run, experiment_id)
    return _ok(
        {"experiment_id": experiment_id, "status": "training"},
        "Training started in background.",
    )


@router.get("/experiments/{experiment_id}/training/status")
async def get_training_status(
    experiment_id: str,
    repo=Depends(get_experiment_repo),
):
    """Poll the training status of an experiment."""
    from app.domain.exceptions.domain_exceptions import ExperimentNotFoundError

    exp = await repo.get_by_id(experiment_id)
    if not exp:
        raise ExperimentNotFoundError(experiment_id)
    return _ok({"experiment_id": experiment_id, "status": exp.status.value})


@router.get("/experiments/{experiment_id}/training/results")
async def get_training_results(
    experiment_id: str,
    repo=Depends(get_experiment_repo),
):
    """Get all model training and evaluation results for an experiment."""
    use_case = GetEvaluationResultsUseCase(repo)
    results = await use_case.execute(experiment_id)
    return _ok([r.model_dump() for r in results])


# ── Evaluation ────────────────────────────────────────────────────────────────


@router.get("/experiments/{experiment_id}/evaluation")
async def get_evaluation_summary(
    experiment_id: str,
    repo=Depends(get_experiment_repo),
):
    """Get evaluation summary (same as training results, sorted by rank)."""
    use_case = GetEvaluationResultsUseCase(repo)
    results = await use_case.execute(experiment_id)
    return _ok([r.model_dump() for r in results])


@router.get("/experiments/{experiment_id}/evaluation/{model_id}")
async def get_model_evaluation(
    experiment_id: str,
    model_id: str,
    repo=Depends(get_experiment_repo),
):
    """Get detailed evaluation metrics for a specific model."""
    from app.domain.exceptions.domain_exceptions import ModelResultNotFoundError

    mr = await repo.get_model_result_by_id(model_id)
    if not mr:
        raise ModelResultNotFoundError(model_id)
    from app.application.use_cases.training.training_use_cases import _map_model_result

    return _ok(_map_model_result(mr).model_dump())


# ── Recommendation ────────────────────────────────────────────────────────────


@router.get("/experiments/{experiment_id}/recommendation")
async def get_recommendation(
    experiment_id: str,
    repo=Depends(get_experiment_repo),
):
    """Get the best model recommendation with explanation for an experiment."""
    use_case = GetRecommendationUseCase(repo)
    rec = await use_case.execute(experiment_id)
    return _ok(rec.model_dump())


# ── Explainability ────────────────────────────────────────────────────────────


@router.post("/experiments/{experiment_id}/explain/{model_id}", status_code=status.HTTP_200_OK)
async def explain_model(
    experiment_id: str,
    model_id: str,
    use_case: ExplainModelUseCase = Depends(get_explain_use_case),
):
    """
    Compute feature importance and SHAP values for a specific model.
    """
    result = await use_case.execute(experiment_id, model_id)
    return _ok(result.model_dump())
