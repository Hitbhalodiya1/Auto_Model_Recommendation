"""
Local filesystem storage service implementation.
All files are stored under UPLOAD_DIR, organized by experiment.
"""

from pathlib import Path

import aiofiles

from app.core.config import get_settings
from app.core.logging import get_logger
from app.domain.exceptions.domain_exceptions import StorageError
from app.domain.interfaces.services.storage_service import IStorageService

logger = get_logger(__name__)


class LocalStorageService(IStorageService):
    """
    Stores files on the local filesystem.
    Base directory is configured via UPLOAD_DIR setting.
    """

    def __init__(self, base_dir: str | None = None) -> None:
        settings = get_settings()
        self._base_dir = Path(base_dir or settings.UPLOAD_DIR).resolve()
        self._base_dir.mkdir(parents=True, exist_ok=True)
        logger.info("local_storage_initialized", base_dir=str(self._base_dir))

    async def save_upload(
        self,
        file_bytes: bytes,
        filename: str,
        experiment_id: str | None = None,
    ) -> str:
        """
        Save uploaded file bytes.
        Files go to: {base_dir}/{filename} (for regular uploads)
        or           {base_dir}/experiments/{experiment_id}/{filename}
        Returns relative path from base_dir.
        """
        try:
            # Sanitize filename to prevent path traversal
            safe_name = Path(filename).name
            if experiment_id:
                dest_dir = self._base_dir / "experiments" / experiment_id
            else:
                dest_dir = self._base_dir

            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / safe_name

            async with aiofiles.open(dest_path, "wb") as f:
                await f.write(file_bytes)

            rel_path = str(dest_path.relative_to(self._base_dir))
            logger.info("file_saved", path=rel_path, size=len(file_bytes))
            return rel_path

        except OSError as e:
            raise StorageError(f"Failed to save file '{filename}': {e}") from e

    async def read_file(self, path: str) -> bytes:
        """Read and return file bytes by relative or absolute path."""
        try:
            abs_path = self._resolve(path)
            async with aiofiles.open(abs_path, "rb") as f:
                return await f.read()
        except FileNotFoundError as e:
            raise StorageError(f"File not found: {path}") from e
        except OSError as e:
            raise StorageError(f"Failed to read file '{path}': {e}") from e

    async def delete_file(self, path: str) -> None:
        """Delete a file. Silently succeeds if not found."""
        try:
            abs_path = self._resolve(path)
            if abs_path.exists():
                abs_path.unlink()
                logger.info("file_deleted", path=path)
        except OSError as e:
            raise StorageError(f"Failed to delete file '{path}': {e}") from e

    async def file_exists(self, path: str) -> bool:
        abs_path = self._resolve(path)
        return abs_path.exists()

    async def save_artifact(self, data: bytes, relative_path: str) -> str:
        """
        Save a model artifact at a relative path under base_dir.
        Creates parent directories as needed.
        """
        try:
            abs_path = self._base_dir / relative_path
            abs_path.parent.mkdir(parents=True, exist_ok=True)

            async with aiofiles.open(abs_path, "wb") as f:
                await f.write(data)

            logger.info("artifact_saved", path=relative_path, size=len(data))
            return relative_path
        except OSError as e:
            raise StorageError(f"Failed to save artifact '{relative_path}': {e}") from e

    def get_absolute_path(self, path: str) -> str:
        return str(self._resolve(path))

    def _resolve(self, path: str) -> Path:
        """
        Resolve a path to absolute, preventing directory traversal.
        Accepts both absolute paths and paths relative to base_dir.
        """
        p = Path(path)
        if p.is_absolute():
            resolved = p.resolve()
        else:
            resolved = (self._base_dir / p).resolve()

        # Security: ensure resolved path stays within base_dir
        try:
            resolved.relative_to(self._base_dir)
        except ValueError as e:
            raise StorageError(f"Path traversal attempt blocked: {path}") from e

        return resolved
