"""
Registry bootstrap module.
Creates and configures the ModelRegistry with all Phase 1 plugins.
Called once at application startup.
"""

from app.core.logging import get_logger
from app.infrastructure.ml.registry.model_registry import ModelRegistry

# Classification plugins
from app.infrastructure.ml.registry.plugins.classification.adaboost import AdaBoostPlugin
from app.infrastructure.ml.registry.plugins.classification.decision_tree import DecisionTreePlugin
from app.infrastructure.ml.registry.plugins.classification.gradient_boosting import (
    GradientBoostingPlugin,
)
from app.infrastructure.ml.registry.plugins.classification.knn import KNNPlugin
from app.infrastructure.ml.registry.plugins.classification.lda import LDAPlugin
from app.infrastructure.ml.registry.plugins.classification.logistic_regression import (
    LogisticRegressionPlugin,
)
from app.infrastructure.ml.registry.plugins.classification.mlp import MLPClassifierPlugin
from app.infrastructure.ml.registry.plugins.classification.naive_bayes import NaiveBayesPlugin
from app.infrastructure.ml.registry.plugins.classification.random_forest import RandomForestPlugin
from app.infrastructure.ml.registry.plugins.classification.svm import SVMPlugin
from app.infrastructure.ml.registry.plugins.classification.xgboost import XGBoostPlugin

# Regression plugins
from app.infrastructure.ml.registry.plugins.regression.boosting_regressors import (
    AdaBoostRegressorPlugin,
    GradientBoostingRegressorPlugin,
    MLPRegressorPlugin,
    XGBoostRegressorPlugin,
)
from app.infrastructure.ml.registry.plugins.regression.linear_regression import (
    ElasticNetPlugin,
    LassoPlugin,
    LinearRegressionPlugin,
    RidgePlugin,
)
from app.infrastructure.ml.registry.plugins.regression.tree_ensemble_regressors import (
    DecisionTreeRegressorPlugin,
    KNNRegressorPlugin,
    RandomForestRegressorPlugin,
    SVRPlugin,
)

logger = get_logger(__name__)


def build_registry() -> ModelRegistry:
    """
    Instantiate and populate the ModelRegistry with all Phase 1 algorithm plugins.
    Returns the fully configured registry.
    """
    registry = ModelRegistry()

    # ── Classification ───────────────────────────────────────────────────────
    registry.register(LogisticRegressionPlugin())
    registry.register(DecisionTreePlugin())
    registry.register(RandomForestPlugin())
    registry.register(KNNPlugin())
    registry.register(SVMPlugin())
    registry.register(NaiveBayesPlugin())
    registry.register(LDAPlugin())
    registry.register(GradientBoostingPlugin())
    registry.register(AdaBoostPlugin())
    registry.register(XGBoostPlugin())
    registry.register(MLPClassifierPlugin())

    # ── Regression ───────────────────────────────────────────────────────────
    registry.register(LinearRegressionPlugin())
    registry.register(RidgePlugin())
    registry.register(LassoPlugin())
    registry.register(ElasticNetPlugin())
    registry.register(DecisionTreeRegressorPlugin())
    registry.register(RandomForestRegressorPlugin())
    registry.register(KNNRegressorPlugin())
    registry.register(SVRPlugin())
    registry.register(GradientBoostingRegressorPlugin())
    registry.register(AdaBoostRegressorPlugin())
    registry.register(XGBoostRegressorPlugin())
    registry.register(MLPRegressorPlugin())

    summary = registry.summary()
    logger.info(
        "registry_built",
        total_configs=registry.total_configs,
        tasks=list(summary.keys()),
    )
    return registry
