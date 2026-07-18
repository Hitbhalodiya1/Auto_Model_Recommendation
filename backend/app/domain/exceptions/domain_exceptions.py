"""
Domain exception hierarchy for AutoRec.

All application errors derive from AutoRecError.
Exceptions are raised in the domain/application layer and
translated to HTTP responses in the API middleware.
"""


class AutoRecError(Exception):
    """Root exception for all AutoRec errors."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


# ── Not Found ─────────────────────────────────────────────────────────────────


class NotFoundError(AutoRecError):
    """Raised when a requested resource does not exist."""


class DatasetNotFoundError(NotFoundError):
    def __init__(self, dataset_id: str) -> None:
        super().__init__(
            f"Dataset '{dataset_id}' not found.",
            {"dataset_id": dataset_id},
        )


class ExperimentNotFoundError(NotFoundError):
    def __init__(self, experiment_id: str) -> None:
        super().__init__(
            f"Experiment '{experiment_id}' not found.",
            {"experiment_id": experiment_id},
        )


class ModelResultNotFoundError(NotFoundError):
    def __init__(self, model_id: str) -> None:
        super().__init__(
            f"Model result '{model_id}' not found.",
            {"model_id": model_id},
        )


# ── Validation ────────────────────────────────────────────────────────────────


class ValidationError(AutoRecError):
    """Raised when input data fails business rule validation."""


class InvalidFileTypeError(ValidationError):
    def __init__(self, filename: str, allowed: frozenset[str]) -> None:
        super().__init__(
            f"File '{filename}' has an unsupported type. Allowed: {', '.join(sorted(allowed))}",
            {"filename": filename, "allowed_extensions": list(allowed)},
        )


class FileTooLargeError(ValidationError):
    def __init__(self, size_bytes: int, max_bytes: int) -> None:
        super().__init__(
            f"File size {size_bytes / 1024 / 1024:.1f}MB exceeds "
            f"the {max_bytes / 1024 / 1024:.0f}MB limit.",
            {"size_bytes": size_bytes, "max_bytes": max_bytes},
        )


class DuplicateDatasetError(ValidationError):
    def __init__(self, filename: str) -> None:
        super().__init__(
            f"A dataset with the name '{filename}' already exists. "
            "Please delete the existing dataset or upload with a different name.",
            {"filename": filename},
        )


class DatasetTooSmallError(ValidationError):
    def __init__(self, row_count: int, min_rows: int) -> None:
        super().__init__(
            f"Dataset has only {row_count} rows. Minimum required: {min_rows}.",
            {"row_count": row_count, "min_rows": min_rows},
        )


class InvalidTargetColumnError(ValidationError):
    def __init__(self, column: str, available: list[str]) -> None:
        super().__init__(
            f"Target column '{column}' not found in dataset.",
            {"column": column, "available_columns": available},
        )


# ── State Errors ──────────────────────────────────────────────────────────────


class InvalidStateError(AutoRecError):
    """Raised when an operation is attempted in an invalid state."""


class ExperimentAlreadyRunningError(InvalidStateError):
    def __init__(self, experiment_id: str) -> None:
        super().__init__(
            f"Experiment '{experiment_id}' is already running. "
            "Wait for it to complete or cancel it first.",
            {"experiment_id": experiment_id},
        )


class AnalysisNotCompleteError(InvalidStateError):
    def __init__(self, dataset_id: str) -> None:
        super().__init__(
            f"Dataset '{dataset_id}' analysis is not yet complete. "
            "Please wait for analysis to finish.",
            {"dataset_id": dataset_id},
        )


class PreprocessingNotCompleteError(InvalidStateError):
    def __init__(self, experiment_id: str) -> None:
        super().__init__(
            f"Preprocessing for experiment '{experiment_id}' has not been executed.",
            {"experiment_id": experiment_id},
        )


# ── ML Errors ─────────────────────────────────────────────────────────────────


class MLError(AutoRecError):
    """Base class for ML pipeline errors."""


class TrainingError(MLError):
    def __init__(self, algorithm: str, reason: str) -> None:
        super().__init__(
            f"Training failed for algorithm '{algorithm}': {reason}",
            {"algorithm": algorithm, "reason": reason},
        )


class InsufficientDataError(MLError):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class UnsupportedTaskTypeError(MLError):
    def __init__(self, task_type: str) -> None:
        super().__init__(
            f"Task type '{task_type}' is not supported.",
            {"task_type": task_type},
        )


# ── Infrastructure Errors ─────────────────────────────────────────────────────


class StorageError(AutoRecError):
    """Raised when a file storage operation fails."""


class ReportGenerationError(AutoRecError):
    """Raised when report generation fails."""
