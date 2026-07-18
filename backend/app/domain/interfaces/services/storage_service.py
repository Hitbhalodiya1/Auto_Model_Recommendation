"""
Storage service interface.
Abstracts file storage so local and S3 backends are interchangeable.
"""

from abc import ABC, abstractmethod


class IStorageService(ABC):
    """Contract for file storage operations."""

    @abstractmethod
    async def save_upload(self, file_bytes: bytes, filename: str, experiment_id: str | None = None) -> str:
        """
        Persist uploaded file bytes.
        Returns the storage path/key that can be used to retrieve the file.
        """
        ...

    @abstractmethod
    async def read_file(self, path: str) -> bytes:
        """Read and return the raw bytes of a stored file."""
        ...

    @abstractmethod
    async def delete_file(self, path: str) -> None:
        """Delete a file from storage. Silently succeeds if not found."""
        ...

    @abstractmethod
    async def file_exists(self, path: str) -> bool:
        """Return True if the file exists in storage."""
        ...

    @abstractmethod
    async def save_artifact(self, data: bytes, relative_path: str) -> str:
        """
        Save a model artifact (pickle, report, etc.) to experiment storage.
        Returns the full storage path.
        """
        ...

    @abstractmethod
    def get_absolute_path(self, path: str) -> str:
        """Resolve a storage path to an absolute filesystem path (local only)."""
        ...
