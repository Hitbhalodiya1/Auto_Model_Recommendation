"""
Extended API integration tests for AutoRec backend routes.
"""

import io

import pytest


@pytest.mark.asyncio
async def test_preview_dataset_not_found(test_client):
    response = await test_client.get("/api/v1/datasets/nonexistent/preview")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_analyze_dataset_not_found(test_client):
    response = await test_client.post("/api/v1/datasets/nonexistent/analyze")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_start_training_experiment_not_found(test_client):
    response = await test_client.post("/api/v1/experiments/nonexistent/training/start")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_dataset_analysis_not_found(test_client):
    response = await test_client.get("/api/v1/datasets/nonexistent/analysis")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_experiment_not_found(test_client):
    response = await test_client.delete("/api/v1/experiments/nonexistent")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_training_status_not_found(test_client):
    response = await test_client.get("/api/v1/experiments/nonexistent/training/status")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_evaluation_summary_not_found(test_client):
    response = await test_client.get("/api/v1/experiments/nonexistent/evaluation")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_model_evaluation_not_found(test_client):
    response = await test_client.get("/api/v1/experiments/nonexistent/evaluation/nonexistent-model")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_recommendation_not_found(test_client):
    response = await test_client.get("/api/v1/experiments/nonexistent/recommendation")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_explain_model_not_found(test_client):
    response = await test_client.post("/api/v1/experiments/nonexistent/explain/nonexistent-model")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_upload_then_preview_dataset(test_client):
    import pandas as pd

    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [10, 20, 30, 40, 50],
        "label": [0, 1, 0, 1, 0],
    })
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    csv_bytes = buf.getvalue().encode()

    upload_resp = await test_client.post(
        "/api/v1/datasets/upload",
        files={"file": ("preview.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert upload_resp.status_code == 201
    dataset_id = upload_resp.json()["data"]["id"]

    preview_resp = await test_client.get(f"/api/v1/datasets/{dataset_id}/preview?n_rows=3")
    assert preview_resp.status_code == 200
    payload = preview_resp.json()
    assert payload["success"] is True
    assert payload["data"]["total_rows"] == 5
    assert len(payload["data"]["rows"]) == 3


# ── Preprocessing Routes ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_recommend_preprocessing_not_found(test_client):
    response = await test_client.post("/api/v1/experiments/nonexistent/preprocessing/recommend")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_execute_preprocessing_not_found(test_client):
    response = await test_client.post("/api/v1/experiments/nonexistent/preprocessing/execute")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_preprocessing_status_not_found(test_client):
    response = await test_client.get("/api/v1/experiments/nonexistent/preprocessing/status")
    assert response.status_code == 404


# ── Training Routes ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_training_results_not_found(test_client):
    response = await test_client.get("/api/v1/experiments/nonexistent/training/results")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_recommendation_not_found(test_client):
    response = await test_client.get("/api/v1/experiments/nonexistent/recommendation")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_explain_model_not_found(test_client):
    response = await test_client.post("/api/v1/experiments/nonexistent/explain/nonexistent-model")
    assert response.status_code == 404


# ── Report Generation ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_generate_report_not_found(test_client):
    response = await test_client.post("/api/v1/experiments/nonexistent/reports/generate")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_report_not_found(test_client):
    response = await test_client.get("/api/v1/experiments/nonexistent/reports/report-123")
    assert response.status_code == 404


# ── File Upload Edge Cases ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_upload_no_file(test_client):
    response = await test_client.post("/api/v1/datasets/upload")
    assert response.status_code == 422
