import os

import pytest

from app.domain.exceptions.domain_exceptions import StorageError
from app.infrastructure.database.session import get_engine, get_session_factory, init_database
from app.infrastructure.storage.local_storage import LocalStorageService


@pytest.mark.asyncio
async def test_local_storage_save_read_delete_upload(tmp_path):
    storage = LocalStorageService(base_dir=str(tmp_path))
    contents = b"hello world"
    rel_path = await storage.save_upload(contents, "file.txt")

    assert os.path.normpath(rel_path) == os.path.normpath("uploads/file.txt")
    assert await storage.file_exists(rel_path)
    assert await storage.read_file(rel_path) == contents

    await storage.delete_file(rel_path)
    assert not await storage.file_exists(rel_path)


@pytest.mark.asyncio
async def test_local_storage_save_artifact_and_absolute_path(tmp_path):
    storage = LocalStorageService(base_dir=str(tmp_path))
    artifact_path = await storage.save_artifact(b"artifact", "experiments/exp-1/model.pkl")

    assert artifact_path == "experiments/exp-1/model.pkl"
    assert await storage.file_exists(artifact_path)
    assert str(tmp_path) in storage.get_absolute_path(artifact_path)


@pytest.mark.asyncio
async def test_local_storage_read_missing_file_raises(tmp_path):
    storage = LocalStorageService(base_dir=str(tmp_path))
    with pytest.raises(StorageError):
        await storage.read_file("missing.txt")


@pytest.mark.asyncio
async def test_local_storage_path_traversal_blocked(tmp_path):
    storage = LocalStorageService(base_dir=str(tmp_path))
    with pytest.raises(StorageError):
        await storage.read_file("../outside.txt")


@pytest.mark.asyncio
async def test_database_init_database_and_session_factory(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    monkeypatch.setenv("DATABASE_ECHO", "False")

    # Reinitialize settings after environment change
    from app.core.config import get_settings

    get_settings.cache_clear()
    init_database()

    engine = get_engine()
    assert engine is not None

    factory = get_session_factory()
    assert factory is not None

    async with factory() as session:
        assert session.bind is engine
        assert session.in_transaction() is False
