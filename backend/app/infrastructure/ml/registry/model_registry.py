"""
Concrete ModelRegistry implementation.
Holds registered plugins and provides task-filtered model configs.
"""

from app.core.logging import get_logger
from app.domain.interfaces.registry.model_registry import IBaseModel, IModelRegistry, ModelConfig
from app.domain.value_objects.task_type import TaskType

logger = get_logger(__name__)


class ModelRegistry(IModelRegistry):
    """
    Plugin-based model registry.
    New algorithms are added by calling register() with a plugin — zero
    changes required to the training engine.
    """

    def __init__(self) -> None:
        self._plugins: list[IBaseModel] = []
        self._config_map: dict[str, tuple[IBaseModel, ModelConfig]] = {}

    def register(self, plugin: IBaseModel) -> None:
        """Register a model plugin and index all its configs by name."""
        self._plugins.append(plugin)
        for config in plugin.configs:
            if config.name in self._config_map:
                raise ValueError(
                    f"Duplicate model config name '{config.name}'. "
                    "Each config must have a unique name."
                )
            self._config_map[config.name] = (plugin, config)

        logger.info(
            "plugin_registered",
            plugin=type(plugin).__name__,
            config_count=len(plugin.configs),
        )

    def get_models_for_task(self, task: TaskType) -> list[ModelConfig]:
        """Return all ModelConfigs compatible with the given task type."""
        return [config for _, config in self._config_map.values() if task in config.task_types]

    def build_estimator(self, config: ModelConfig):
        """Build and return a scikit-learn compatible estimator."""
        entry = self._config_map.get(config.name)
        if entry is None:
            raise ValueError(f"No plugin found for config '{config.name}'.")
        plugin, _ = entry
        return plugin.build(config)

    def get_config_by_name(self, name: str) -> ModelConfig | None:
        """Look up a ModelConfig by its unique name."""
        entry = self._config_map.get(name)
        return entry[1] if entry else None

    @property
    def total_configs(self) -> int:
        return len(self._config_map)

    def summary(self) -> dict:
        """Return a summary dict for logging/health checks."""
        from collections import defaultdict

        by_task: dict = defaultdict(list)
        for _, config in self._config_map.values():
            for task in config.task_types:
                by_task[task.value].append(config.name)
        return dict(by_task)
