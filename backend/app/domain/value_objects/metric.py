"""
Metric value object — an immutable named metric with a numeric value.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Metric:
    """An immutable metric with a name, value, and optional description."""

    name: str
    value: float
    description: str = ""
    higher_is_better: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Metric name cannot be empty.")

    def format(self, decimals: int = 4) -> str:
        return f"{self.name}: {self.value:.{decimals}f}"


@dataclass(frozen=True)
class MetricSet:
    """
    A collection of metrics for a single model evaluation.
    Carries the primary metric used for ranking.
    """

    metrics: tuple[Metric, ...]
    primary_metric_name: str

    @property
    def primary_score(self) -> float:
        for m in self.metrics:
            if m.name == self.primary_metric_name:
                return m.value
        raise KeyError(f"Primary metric '{self.primary_metric_name}' not found in metric set.")

    def to_dict(self) -> dict[str, Any]:
        return {m.name: m.value for m in self.metrics}

    def get(self, name: str) -> float | None:
        for m in self.metrics:
            if m.name == name:
                return m.value
        return None
