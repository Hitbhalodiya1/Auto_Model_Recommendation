"""
Shared pytest fixtures for AutoRec backend tests.
"""

import asyncio
import io
from collections.abc import AsyncGenerator

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.domain.entities.dataset import ColumnProfile, Dataset, DatasetAnalysis, DatasetStatus
from app.domain.entities.experiment import Experiment, ExperimentStatus
from app.domain.value_objects.task_type import TaskType
from app.infrastructure.database.session import Base

# ── Event loop ────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ── Event loop ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ── In-memory SQLite database ─────────────────────────────────────────────────

@pytest_asyncio.fixture
async def db_engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_engine) -> AsyncGenerator[AsyncSession, None]:
    factory = async_sessionmaker(
        bind=db_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with factory() as session:
        yield session
        await session.rollback()


# ── Sample DataFrames ─────────────────────────────────────────────────────────

@pytest.fixture
def binary_df() -> pd.DataFrame:
    """100-row binary classification dataset."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "feature1": np.random.randn(n),
        "feature2": np.random.randn(n),
        "cat_col": np.random.choice(["A", "B", "C"], n),
        "target": np.random.choice([0, 1], n),
    })


@pytest.fixture
def regression_df() -> pd.DataFrame:
    """100-row regression dataset."""
    np.random.seed(42)
    n = 100
    x = np.random.randn(n)
    return pd.DataFrame({
        "x1": x,
        "x2": np.random.randn(n),
        "target": 3 * x + np.random.randn(n) * 0.5,
    })


@pytest.fixture
def csv_bytes(binary_df) -> bytes:
    buf = io.StringIO()
    binary_df.to_csv(buf, index=False)
    return buf.getvalue().encode()


# ── Domain object fixtures ────────────────────────────────────────────────────

@pytest.fixture
def sample_dataset() -> Dataset:
    return Dataset(
        id="ds-test-001",
        filename="test.csv",
        original_name="test.csv",
        file_path="uploads/test.csv",
        file_size=2048,
        row_count=100,
        column_count=4,
        status=DatasetStatus.ANALYZED,
    )


@pytest.fixture
def sample_analysis(sample_dataset) -> DatasetAnalysis:
    return DatasetAnalysis(
        dataset_id=sample_dataset.id,
        task_type=TaskType.BINARY_CLASSIFICATION.value,
        suggested_target_column="target",
        quality_score=85.0,
        row_count=100,
        column_count=4,
        duplicate_row_count=0,
        missing_value_pct=0.0,
        column_profiles=[
            ColumnProfile(
                name="feature1",
                dtype="float64",
                feature_type="numeric_continuous",
                null_count=0, null_pct=0.0,
                unique_count=100, unique_pct=1.0,
                mean=0.0, std=1.0, min=-3.0, max=3.0,
                median=0.0, skewness=0.05,
            ),
            ColumnProfile(
                name="feature2",
                dtype="float64",
                feature_type="numeric_continuous",
                null_count=0, null_pct=0.0,
                unique_count=100, unique_pct=1.0,
            ),
            ColumnProfile(
                name="cat_col",
                dtype="object",
                feature_type="categorical_nominal",
                null_count=0, null_pct=0.0,
                unique_count=3, unique_pct=0.03,
            ),
            ColumnProfile(
                name="target",
                dtype="int64",
                feature_type="boolean",
                null_count=0, null_pct=0.0,
                unique_count=2, unique_pct=0.02,
            ),
        ],
        warnings=[],
        recommendations=[],
    )


@pytest.fixture
def sample_experiment(sample_dataset) -> Experiment:
    return Experiment(
        id="exp-test-001",
        name="Test Experiment",
        dataset_id=sample_dataset.id,
        task_type=TaskType.BINARY_CLASSIFICATION.value,
        target_column="target",
        status=ExperimentStatus.CREATED,
    )


# ── FastAPI async test client ─────────────────────────────────────────────────

@pytest_asyncio.fixture
async def test_client(db_session) -> AsyncGenerator:
    from httpx import ASGITransport, AsyncClient

    from app.infrastructure.database.session import get_db_session
    from app.main import app

    async def _override_db():
        yield db_session

    app.dependency_overrides[get_db_session] = _override_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()
