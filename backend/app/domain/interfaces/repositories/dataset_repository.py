"""
Repository interfaces for dataset and experiment persistence.
These are abstract contracts — infrastructure implements them.
"""

from abc import ABC, abstractmethod

from app.domain.entities.dataset import Dataset, DatasetAnalysis


class IDatasetRepository(ABC):
    """Persistence contract for Dataset entities."""

    @abstractmethod
    async def save(self, dataset: Dataset) -> Dataset:
        """Persist a new or updated dataset. Returns the saved entity."""
        ...

    @abstractmethod
    async def get_by_id(self, dataset_id: str) -> Dataset | None:
        """Return a dataset by ID, or None if not found."""
        ...

    @abstractmethod
    async def list_all(self, limit: int = 20, offset: int = 0) -> list[Dataset]:
        """Return paginated list of all datasets, newest first."""
        ...

    @abstractmethod
    async def delete(self, dataset_id: str) -> bool:
        """Delete a dataset by ID. Returns True if deleted, False if not found."""
        ...

    @abstractmethod
    async def save_analysis(self, analysis: DatasetAnalysis) -> DatasetAnalysis:
        """Persist dataset analysis results."""
        ...

    @abstractmethod
    async def get_analysis(self, dataset_id: str) -> DatasetAnalysis | None:
        """Return analysis for a dataset, or None."""
        ...
