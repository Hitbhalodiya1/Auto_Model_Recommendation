import pytest

from app.domain.entities.dataset import (
    DatasetStatus,
)
from app.domain.entities.experiment import Experiment, ExperimentStatus
from app.domain.entities.model_result import ModelResult, Recommendation
from app.infrastructure.database.repositories.dataset_repository import DatasetRepository
from app.infrastructure.database.repositories.experiment_repository import ExperimentRepository


@pytest.mark.asyncio
async def test_dataset_repository_save_get_list_delete(db_session, sample_dataset, sample_analysis):
    repo = DatasetRepository(db_session)

    saved = await repo.save(sample_dataset)
    assert saved.id == sample_dataset.id
    assert saved.status == DatasetStatus.ANALYZED

    retrieved = await repo.get_by_id(sample_dataset.id)
    assert retrieved is not None
    assert retrieved.filename == sample_dataset.filename

    analysis = await repo.save_analysis(sample_analysis)
    assert analysis.dataset_id == sample_analysis.dataset_id

    retrieved_analysis = await repo.get_analysis(sample_dataset.id)
    assert retrieved_analysis is not None
    assert retrieved_analysis.task_type == sample_analysis.task_type

    all_datasets = await repo.list_all(limit=10, offset=0)
    assert any(ds.id == sample_dataset.id for ds in all_datasets)

    deleted = await repo.delete(sample_dataset.id)
    assert deleted is True
    assert await repo.get_by_id(sample_dataset.id) is None


@pytest.mark.asyncio
async def test_experiment_repository_save_and_model_results(db_session):
    exp_repo = ExperimentRepository(db_session)

    experiment = Experiment(
        name="Test Experiment",
        dataset_id="ds-test-001",
        target_column="target",
    )
    saved_exp = await exp_repo.save(experiment)
    assert saved_exp.id == experiment.id
    assert saved_exp.status == ExperimentStatus.CREATED

    retrieved_exp = await exp_repo.get_by_id(saved_exp.id)
    assert retrieved_exp is not None
    assert retrieved_exp.dataset_id == experiment.dataset_id

    model_result = ModelResult(
        experiment_id=saved_exp.id,
        algorithm_name="RandomForest",
        config_name="rf_gini_100",
        display_name="Random Forest",
        configuration={"n_estimators": 10},
        metrics={"f1_score": 0.9},
        cv_score=0.9,
        cv_std=0.05,
        is_overfitting=False,
        training_time_s=0.1,
        prediction_time_s=0.01,
        interpretability_score=3,
        supports_feature_importance=True,
        supports_shap=True,
    )
    saved_mr = await exp_repo.save_model_result(model_result)
    assert saved_mr.id == model_result.id

    model_results = await exp_repo.get_model_results(saved_exp.id)
    assert len(model_results) == 1
    assert model_results[0].config_name == "rf_gini_100"

    recommendation = Recommendation(
        experiment_id=saved_exp.id,
        model_result_id=saved_mr.id,
        composite_score=0.92,
        score_breakdown={"f1_score": 0.92},
        rationale=["Best overall performance"],
        explanation_text="Recommended model based on F1 and validation performance.",
        all_rankings=[{"model_result_id": saved_mr.id, "composite_score": 0.92}],
    )
    saved_rec = await exp_repo.save_recommendation(recommendation)
    assert saved_rec.model_result_id == recommendation.model_result_id

    fetched_rec = await exp_repo.get_recommendation(saved_exp.id)
    assert fetched_rec is not None
    assert fetched_rec.model_result_id == saved_rec.model_result_id
