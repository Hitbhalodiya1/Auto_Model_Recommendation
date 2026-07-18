"""
Repository interface for Experiment entities.
"""

from abc import ABC, abstractmethod

from app.domain.entities.experiment import Experiment
from app.domain.entities.model_result import ModelResult, Recommendation


class IExperimentRepository(ABC):
    """Persistence contract for Experiment entities."""

    @abstractmethod
    async def save(self, experiment: Experiment) -> Experiment: ...

    @abstractmethod
    async def get_by_id(self, experiment_id: str) -> Experiment | None: ...

    @abstractmethod
    async def list_all(self, limit: int = 20, offset: int = 0) -> list[Experiment]: ...

    @abstractmethod
    async def delete(self, experiment_id: str) -> bool: ...

    @abstractmethod
    async def save_model_result(self, result: ModelResult) -> ModelResult: ...

    @abstractmethod
    async def get_model_results(self, experiment_id: str) -> list[ModelResult]: ...

    @abstractmethod
    async def get_model_result_by_id(self, model_id: str) -> ModelResult | None: ...

    @abstractmethod
    async def save_recommendation(self, recommendation: Recommendation) -> Recommendation: ...

    @abstractmethod
    async def get_recommendation(self, experiment_id: str) -> Recommendation | None: ...
