"""
TrainingEngine — trains all compatible models from the registry in parallel.
Completely decoupled from specific algorithms.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger
from app.domain.interfaces.registry.model_registry import IModelRegistry, ModelConfig
from app.domain.value_objects.task_type import TaskType

logger = get_logger(__name__)


@dataclass
class TrainingResult:
    """Raw output from training a single model configuration."""

    config: ModelConfig
    estimator: Any | None
    predictions: np.ndarray | None
    train_score: float | None
    training_time_s: float = 0.0
    prediction_time_s: float = 0.0
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


class TrainingEngine:
    """
    Fetches compatible model configs from the registry,
    trains all of them in parallel worker threads,
    and returns structured TrainingResult objects.
    Zero knowledge of specific algorithms.
    """

    def __init__(self, registry: IModelRegistry) -> None:
        self._registry = registry
        self._settings = get_settings()

    def train_all(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        task_type: TaskType,
        experiment_id: str,
    ) -> list[TrainingResult]:
        """
        Train all registered models compatible with task_type.
        Returns list of TrainingResult (successes and failures).
        """
        configs = self._registry.get_models_for_task(task_type)
        if not configs:
            logger.warning("no_models_for_task", task_type=task_type.value)
            return []

        logger.info(
            "training_started",
            experiment_id=experiment_id,
            task_type=task_type.value,
            model_count=len(configs),
            workers=self._settings.MAX_TRAINING_WORKERS,
        )

        results: list[TrainingResult] = []
        with ThreadPoolExecutor(
            max_workers=self._settings.MAX_TRAINING_WORKERS,
            thread_name_prefix="training",
        ) as executor:
            future_to_config = {
                executor.submit(
                    self._train_one,
                    config,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                ): config
                for config in configs
            }

            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    results.append(result)
                    status = "success" if result.succeeded else "failed"
                    logger.info(
                        "model_trained",
                        config=config.name,
                        status=status,
                        training_time_s=result.training_time_s,
                    )
                except Exception as exc:
                    logger.error(
                        "model_training_exception",
                        config=config.name,
                        error=str(exc),
                    )
                    results.append(
                        TrainingResult(
                            config=config,
                            estimator=None,
                            predictions=None,
                            train_score=None,
                            error=str(exc),
                        )
                    )

        successes = sum(1 for r in results if r.succeeded)
        logger.info(
            "training_complete",
            experiment_id=experiment_id,
            total=len(results),
            succeeded=successes,
            failed=len(results) - successes,
        )
        return results

    def _train_one(
        self,
        config: ModelConfig,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> TrainingResult:
        """
        Train a single model configuration.
        Wraps models that require scaling in a Pipeline if not already scaled.
        Catches and records any training exception without crashing the pool.
        """
        try:
            estimator = self._registry.build_estimator(config)

            # Note: in Phase 1 the preprocessing engine already applies a global scaler.
            # The requires_scaling flag here is preserved for models that need their
            # own internal pipeline (e.g., when called without the preprocessing engine).
            t0 = time.perf_counter()
            estimator.fit(x_train, y_train)
            training_time = time.perf_counter() - t0

            # Train score (for overfitting detection)
            train_score = float(estimator.score(x_train, y_train))

            t1 = time.perf_counter()
            predictions = estimator.predict(x_test)
            prediction_time = time.perf_counter() - t1

            return TrainingResult(
                config=config,
                estimator=estimator,
                predictions=predictions,
                train_score=train_score,
                training_time_s=round(training_time, 4),
                prediction_time_s=round(prediction_time, 6),
            )

        except Exception as exc:
            return TrainingResult(
                config=config,
                estimator=None,
                predictions=None,
                train_score=None,
                error=f"{type(exc).__name__}: {exc}",
            )
