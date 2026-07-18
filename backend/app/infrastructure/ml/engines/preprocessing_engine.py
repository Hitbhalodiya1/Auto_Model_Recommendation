"""
PreprocessingEngine — recommends and executes preprocessing pipelines.
Each recommendation step carries a human-readable explanation.
"""

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from app.core.logging import get_logger
from app.domain.entities.dataset import DatasetAnalysis
from app.domain.entities.experiment import PreprocessingPipeline, PreprocessingStep
from app.domain.value_objects.feature_type import FeatureType
from app.domain.value_objects.task_type import TaskType

logger = get_logger(__name__)

SCALER_MAP = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
    "maxabs": MaxAbsScaler,
    "normalizer": Normalizer,
}


class PreprocessingResult:
    """Outcome of executing a preprocessing pipeline."""

    def __init__(
        self,
        x_train: np.ndarray,
        x_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: list[str],
        pipeline: Any,  # fitted sklearn Pipeline
        label_encoder: LabelEncoder | None,
    ) -> None:
        self.X_train = x_train
        self.X_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        self.pipeline = pipeline
        self.label_encoder = label_encoder


class PreprocessingEngine:
    """
    Recommends a preprocessing pipeline based on dataset analysis,
    then executes it to produce train/test splits.
    """

    def recommend(
        self,
        analysis: DatasetAnalysis,
        target_column: str,
    ) -> PreprocessingPipeline:
        """
        Generate an ordered list of preprocessing steps with explanations.
        Does NOT execute the pipeline — use execute() for that.
        """
        steps: list[PreprocessingStep] = []
        {p.name: p for p in analysis.column_profiles}

        # Step 1: Remove duplicates
        if analysis.duplicate_row_count > 0:
            steps.append(
                PreprocessingStep(
                    name="remove_duplicates",
                    display_name="Remove Duplicate Rows",
                    strategy="drop_duplicates",
                    explanation=(
                        f"The dataset contains {analysis.duplicate_row_count} duplicate rows. "
                        "Removing them prevents data leakage and biased evaluation metrics."
                    ),
                )
            )

        # Step 2: Drop high-missing columns (>80% missing)
        high_missing = [
            p.name for p in analysis.column_profiles if p.name != target_column and p.null_pct > 0.8
        ]
        if high_missing:
            steps.append(
                PreprocessingStep(
                    name="drop_high_missing_columns",
                    display_name="Drop Columns with >80% Missing Values",
                    strategy="drop_columns",
                    params={"columns": high_missing},
                    explanation=(
                        f"Columns {high_missing} have over 80% missing values. "
                        "Imputing such columns introduces more noise than signal."
                    ),
                    affects_columns=high_missing,
                )
            )

        # Step 3: Handle missing values
        numeric_missing = [
            p.name
            for p in analysis.column_profiles
            if p.name != target_column
            and p.null_count > 0
            and p.feature_type
            in (FeatureType.NUMERIC_CONTINUOUS.value, FeatureType.NUMERIC_DISCRETE.value)
            and p.name not in high_missing
        ]
        categorical_missing = [
            p.name
            for p in analysis.column_profiles
            if p.name != target_column
            and p.null_count > 0
            and p.feature_type
            in (FeatureType.CATEGORICAL_NOMINAL.value, FeatureType.CATEGORICAL_ORDINAL.value)
            and p.name not in high_missing
        ]

        # Choose median vs mean based on skewness
        outlier_cols = list((analysis.outlier_counts or {}).keys())
        skewed_cols = [
            p.name for p in analysis.column_profiles if p.skewness and abs(p.skewness) > 1.0
        ]

        if numeric_missing:
            use_robust = bool(outlier_cols or skewed_cols)
            strategy = "median" if use_robust else "mean"
            steps.append(
                PreprocessingStep(
                    name="impute_numeric",
                    display_name=f"Impute Numeric Missing Values ({strategy})",
                    strategy=f"{strategy}_imputation",
                    params={"strategy": strategy, "columns": numeric_missing},
                    explanation=(
                        "Median imputation chosen for numeric columns "
                        if use_robust
                        else "Mean imputation chosen for numeric columns "
                    )
                    + (
                        "because outliers or skewness were detected. "
                        if use_robust
                        else "because data appears normally distributed. "
                    )
                    + "Median is robust to extreme values.",
                    affects_columns=numeric_missing,
                )
            )

        if categorical_missing:
            steps.append(
                PreprocessingStep(
                    name="impute_categorical",
                    display_name="Impute Categorical Missing Values (most frequent)",
                    strategy="most_frequent_imputation",
                    params={"strategy": "most_frequent", "columns": categorical_missing},
                    explanation=(
                        "Most-frequent imputation fills categorical gaps with the mode, "
                        "preserving the existing distribution without introducing new categories."
                    ),
                    affects_columns=categorical_missing,
                )
            )

        # Step 4: Encode categorical columns
        categorical_cols = [
            p.name
            for p in analysis.column_profiles
            if p.name != target_column
            and p.feature_type
            in (
                FeatureType.CATEGORICAL_NOMINAL.value,
                FeatureType.CATEGORICAL_ORDINAL.value,
                FeatureType.BOOLEAN.value,
            )
            and p.name not in high_missing
        ]
        if categorical_cols:
            steps.append(
                PreprocessingStep(
                    name="encode_categorical",
                    display_name="Encode Categorical Features",
                    strategy="ordinal_encoding",
                    params={"columns": categorical_cols},
                    explanation=(
                        "Ordinal encoding converts categorical text values to integers. "
                        "This is compatible with all scikit-learn estimators and avoids "
                        "the dimensionality explosion of one-hot encoding."
                    ),
                    affects_columns=categorical_cols,
                )
            )

        # Step 5: Handle class imbalance (classification only)
        if analysis.is_imbalanced and analysis.task_type in (
            TaskType.BINARY_CLASSIFICATION.value,
            TaskType.MULTICLASS_CLASSIFICATION.value,
        ):
            steps.append(
                PreprocessingStep(
                    name="handle_class_imbalance",
                    display_name="Handle Class Imbalance (SMOTE)",
                    strategy="smote",
                    explanation=(
                        "The target class distribution is imbalanced. "
                        "SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic "
                        "samples for minority classes, improving recall and overall model fairness."
                    ),
                    affects_columns=[target_column],
                )
            )

        # Step 6: Recommend scaler
        scaler_name, scaler_explanation = self._recommend_scaler(analysis, outlier_cols)
        steps.append(
            PreprocessingStep(
                name="scale_features",
                display_name=f"Scale Numeric Features ({scaler_name})",
                strategy=scaler_name,
                explanation=scaler_explanation,
            )
        )

        pipeline = PreprocessingPipeline(steps=steps)
        logger.info(
            "preprocessing_recommended",
            step_count=len(steps),
            scaler=scaler_name,
        )
        return pipeline

    def _recommend_scaler(
        self, analysis: DatasetAnalysis, outlier_cols: list[str]
    ) -> tuple[str, str]:
        if outlier_cols:
            return (
                "robust",
                "RobustScaler is recommended because outliers were detected. "
                "It scales using the IQR, making it resistant to extreme values.",
            )
        skewed = [p.name for p in analysis.column_profiles if p.skewness and abs(p.skewness) > 1.0]
        if skewed:
            return (
                "robust",
                "RobustScaler is recommended because several features are skewed, "
                "which may indicate non-Gaussian distributions.",
            )
        return (
            "standard",
            "StandardScaler is recommended (zero mean, unit variance). "
            "It works well when data is approximately normally distributed "
            "and is required by distance-based and linear algorithms.",
        )

    def execute(
        self,
        df: pd.DataFrame,
        pipeline_def: PreprocessingPipeline,
        target_column: str,
        task_type: str,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> PreprocessingResult:
        """
        Apply the recommended preprocessing pipeline to the DataFrame.
        Returns train/test splits ready for the training engine.
        """
        from sklearn.impute import SimpleImputer
        from sklearn.model_selection import train_test_split

        df = df.copy()
        scaler_name = "standard"

        for step in pipeline_def.steps:
            if step.name == "remove_duplicates":
                df = df.drop_duplicates()
                logger.info("removed_duplicates", remaining_rows=len(df))

            elif step.name == "drop_high_missing_columns":
                cols = step.params.get("columns", [])
                df = df.drop(columns=[c for c in cols if c in df.columns], errors="ignore")

            elif step.name == "impute_numeric":
                cols = [c for c in step.params.get("columns", []) if c in df.columns]
                strategy = step.params.get("strategy", "median")
                if cols:
                    imputer = SimpleImputer(strategy=strategy)
                    df[cols] = imputer.fit_transform(df[cols])

            elif step.name == "impute_categorical":
                cols = [c for c in step.params.get("columns", []) if c in df.columns]
                if cols:
                    imputer = SimpleImputer(strategy="most_frequent")
                    df[cols] = imputer.fit_transform(df[cols])

            elif step.name == "encode_categorical":
                cols = [c for c in step.params.get("columns", []) if c in df.columns]
                if cols:
                    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                    df[cols] = enc.fit_transform(df[cols].astype(str))

            elif step.name == "scale_features":
                scaler_name = step.strategy

        # Separate features and target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found after preprocessing.")

        feature_cols = [c for c in df.columns if c != target_column]
        x_features = df[feature_cols].select_dtypes(include=[np.number]).values
        feature_names = [
            c for c in feature_cols if c in df.select_dtypes(include=[np.number]).columns
        ]
        y = df[target_column].values

        # Encode target for classification
        label_encoder = None
        if task_type in (
            TaskType.BINARY_CLASSIFICATION.value,
            TaskType.MULTICLASS_CLASSIFICATION.value,
        ):
            if not pd.api.types.is_numeric_dtype(df[target_column]):
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)

        # Train/test split (stratify for classification)
        stratify = (
            y
            if task_type
            in (
                TaskType.BINARY_CLASSIFICATION.value,
                TaskType.MULTICLASS_CLASSIFICATION.value,
            )
            else None
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x_train, x_test, y_train, y_test = train_test_split(
                x_features,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify,
            )

        # Apply scaler
        scaler_class = SCALER_MAP.get(scaler_name, StandardScaler)
        scaler = scaler_class()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        # Handle class imbalance
        smote_step = next(
            (s for s in pipeline_def.steps if s.name == "handle_class_imbalance"),
            None,
        )
        if smote_step:
            try:
                from collections import Counter

                from imblearn.over_sampling import SMOTE

                # Check if minority class has enough samples for SMOTE
                class_counts = Counter(y_train)
                min_class_count = min(class_counts.values())
                smote_neighbors = 6  # SMOTE default

                if min_class_count <= smote_neighbors:
                    logger.warning(
                        "smote_skipped",
                        reason=f"Minority class has only {min_class_count} samples, "
                        f"SMOTE requires at least {smote_neighbors + 1} samples",
                    )
                else:
                    sm = SMOTE(random_state=random_state)
                    x_train, y_train = sm.fit_resample(x_train, y_train)
                    logger.info("smote_applied", new_train_size=len(x_train))
            except ImportError:
                logger.warning("smote_skipped", reason="imbalanced-learn not installed")

        logger.info(
            "preprocessing_executed",
            X_train_shape=str(x_train.shape),
            X_test_shape=str(x_test.shape),
            features=len(feature_names),
        )

        return PreprocessingResult(
            X_train=x_train,
            X_test=x_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=feature_names,
            pipeline=scaler,  # store fitted scaler for artifact saving
            label_encoder=label_encoder,
        )
