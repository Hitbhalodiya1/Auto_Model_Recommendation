"""
API integration tests using HTTPX AsyncClient with TestClient + in-memory DB.
"""

import io

import pandas as pd
import pytest

# ── Health ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_endpoint(test_client):
    response = await test_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["app"] == "AutoRec"


# ── Datasets ──────────────────────────────────────────────────────────────────

def _make_csv_bytes(n_rows: int = 50) -> bytes:
    import io

    import numpy as np
    df = pd.DataFrame({
        "feature1": np.random.randn(n_rows),
        "feature2": np.random.randn(n_rows),
        "label": [0, 1] * (n_rows // 2),
    })
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


@pytest.mark.asyncio
async def test_upload_dataset(test_client):
    csv_bytes = _make_csv_bytes()
    response = await test_client.post(
        "/api/v1/datasets/upload",
        files={"file": ("test.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["success"] is True
    assert data["data"]["filename"] == "test.csv"
    assert data["data"]["status"] == "uploaded"


@pytest.mark.asyncio
async def test_upload_invalid_extension(test_client):
    response = await test_client.post(
        "/api/v1/datasets/upload",
        files={"file": ("malware.exe", io.BytesIO(b"bad"), "application/octet-stream")},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_list_datasets_empty(test_client):
    response = await test_client.get("/api/v1/datasets")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert isinstance(data["data"], list)


@pytest.mark.asyncio
async def test_get_dataset_not_found(test_client):
    response = await test_client.get("/api/v1/datasets/nonexistent-id")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_dataset_not_found(test_client):
    response = await test_client.delete("/api/v1/datasets/nonexistent-id")
    assert response.status_code == 404


# ── Experiments ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_experiments_empty(test_client):
    response = await test_client.get("/api/v1/experiments")
    assert response.status_code == 200
    assert response.json()["data"] == []


@pytest.mark.asyncio
async def test_get_experiment_not_found(test_client):
    response = await test_client.get("/api/v1/experiments/nonexistent")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_create_experiment_dataset_not_found(test_client):
    response = await test_client.post(
        "/api/v1/experiments",
        json={
            "name": "Test Exp",
            "dataset_id": "nonexistent-dataset",
            "target_column": "label",
        },
    )
    assert response.status_code == 404


# ── Upload → List → Get roundtrip ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_upload_then_list(test_client):
    csv_bytes = _make_csv_bytes()
    # Upload
    up_resp = await test_client.post(
        "/api/v1/datasets/upload",
        files={"file": ("roundtrip.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert up_resp.status_code == 201
    dataset_id = up_resp.json()["data"]["id"]

    # List should include it
    list_resp = await test_client.get("/api/v1/datasets")
    assert list_resp.status_code == 200
    ids = [d["id"] for d in list_resp.json()["data"]]
    assert dataset_id in ids

    # Get by ID
    get_resp = await test_client.get(f"/api/v1/datasets/{dataset_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["data"]["id"] == dataset_id
