"""
AnalysisEngine — performs automated statistical analysis on a dataset.
Detects task type, feature types, quality issues, and generates recommendations.
"""

import warnings

import numpy as np
import pandas as pd

from app.core.constants import (
    CLASS_IMBALANCE_THRESHOLD,
    HIGH_CARDINALITY_THRESHOLD,
    MIN_ROWS_FOR_TRAINING,
    MISSING_VALUE_CRITICAL_THRESHOLD,
    OUTLIER_IQR_MULTIPLIER,
    SKEWNESS_THRESHOLD,
)
from app.core.logging import get_logger
from app.domain.entities.dataset import ColumnProfile, DatasetAnalysis
from app.domain.exceptions.domain_exceptions import DatasetTooSmallError
from app.domain.value_objects.feature_type import FeatureType
from app.domain.value_objects.task_type import TaskType

logger = get_logger(__name__)


class AnalysisEngine:
    """
    Performs full automated analysis of a pandas DataFrame.
    Returns a DatasetAnalysis entity with all computed metrics.
    """

    def analyze(self, df: pd.DataFrame, dataset_id: str, progress_callback=None) -> DatasetAnalysis:
        """
        Run the full analysis pipeline on a DataFrame.
        Returns a DatasetAnalysis with all computed fields populated.
        """
        if len(df) < MIN_ROWS_FOR_TRAINING:
            raise DatasetTooSmallError(len(df), MIN_ROWS_FOR_TRAINING)

        logger.info("analysis_started", dataset_id=dataset_id, shape=str(df.shape))

        # Steps breakdown (used for progress reporting)
        steps = [
            "profile_columns",
            "detect_task",
            "compute_stats",
            "class_imbalance",
            "correlation",
            "outliers",
            "quality_score",
            "warnings",
            "recommendations",
        ]
        total_steps = len(steps)

        # Step 1: Profile every column
        column_profiles = []
        for i, col in enumerate(df.columns, start=1):
            column_profiles.append(self._profile_column(df, col))
        if progress_callback:
            # after profiling columns, mark progress (1 of total_steps)
            progress_callback(1, total_steps)

        # Step 2: Detect task type and target column
        task_type, suggested_target = self._detect_task_and_target(df, column_profiles)
        if progress_callback:
            progress_callback(2, total_steps)

        # Step 3: Compute dataset-level statistics
        duplicate_count = int(df.duplicated().sum())
        total_missing = int(df.isnull().sum().sum())
        total_cells = df.shape[0] * df.shape[1]
        missing_pct = round(total_missing / total_cells, 4) if total_cells > 0 else 0.0
        if progress_callback:
            progress_callback(3, total_steps)

        # Step 4: Class imbalance (only for classification targets)
        class_distribution = None
        is_imbalanced = False
        if suggested_target and task_type in (
            TaskType.BINARY_CLASSIFICATION.value,
            TaskType.MULTICLASS_CLASSIFICATION.value,
        ):
            class_distribution, is_imbalanced = self._check_class_imbalance(
                df[suggested_target]
            )
        if progress_callback:
            progress_callback(4, total_steps)

        # Step 5: Correlation matrix (numeric columns only)
        correlation_matrix = self._compute_correlation(df)
        if progress_callback:
            progress_callback(5, total_steps)

        # Step 6: Outlier detection
        outlier_counts = self._detect_outliers(df)
        if progress_callback:
            progress_callback(6, total_steps)

        # Step 7: Quality score
        quality_score = self._compute_quality_score(
            df, column_profiles, duplicate_count, missing_pct, is_imbalanced
        )
        if progress_callback:
            progress_callback(7, total_steps)

        # Step 8: Generate warnings and recommendations
        warnings_list = self._generate_warnings(
            df, column_profiles, duplicate_count, missing_pct, is_imbalanced, outlier_counts
        )
        if progress_callback:
            progress_callback(8, total_steps)
        recommendations = self._generate_recommendations(
            column_profiles, duplicate_count, missing_pct, is_imbalanced, task_type
        )
        if progress_callback:
            progress_callback(9, total_steps)

        analysis = DatasetAnalysis(
            dataset_id=dataset_id,
            task_type=task_type,
            suggested_target_column=suggested_target,
            quality_score=quality_score,
            row_count=len(df),
            column_count=len(df.columns),
            duplicate_row_count=duplicate_count,
            missing_value_pct=missing_pct,
            column_profiles=column_profiles,
            class_distribution=class_distribution,
            is_imbalanced=is_imbalanced,
            correlation_matrix=correlation_matrix,
            outlier_counts=outlier_counts,
            warnings=warnings_list,
            recommendations=recommendations,
        )

        logger.info(
            "analysis_complete",
            dataset_id=dataset_id,
            task_type=task_type,
            quality_score=quality_score,
            columns=len(df.columns),
            rows=len(df),
        )
        return analysis

    # ── Column Profiling ──────────────────────────────────────────────────────

    def _profile_column(self, df: pd.DataFrame, col: str) -> ColumnProfile:
        series = df[col]
        null_count = int(series.isnull().sum())
        null_pct = round(null_count / len(series), 4)
        unique_count = int(series.nunique())
        unique_pct = round(unique_count / len(series), 4)
        feature_type = self._infer_feature_type(series)

        profile = ColumnProfile(
            name=col,
            dtype=str(series.dtype),
            feature_type=feature_type.value,
            null_count=null_count,
            null_pct=null_pct,
            unique_count=unique_count,
            unique_pct=unique_pct,
            sample_values=series.dropna().head(5).tolist(),
        )

        # Numeric statistics
        if feature_type.is_numeric:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clean = series.dropna()
                if len(clean) > 0:
                    profile.mean = float(clean.mean())
                    profile.std = float(clean.std())
                    profile.min = float(clean.min())
                    profile.max = float(clean.max())
                    profile.median = float(clean.median())
                    profile.skewness = float(clean.skew())
                    profile.kurtosis = float(clean.kurt())

        return profile

    def _infer_feature_type(self, series: pd.Series) -> FeatureType:
        """Infer the semantic type of a column from its values."""
        if series.dtype == "bool" or series.nunique() == 2:
            return FeatureType.BOOLEAN

        if pd.api.types.is_datetime64_any_dtype(series):
            return FeatureType.DATETIME

        if pd.api.types.is_numeric_dtype(series):
            unique_ratio = series.nunique() / max(len(series), 1)
            if unique_ratio < 0.05 and series.nunique() <= 20:
                return FeatureType.NUMERIC_DISCRETE
            return FeatureType.NUMERIC_CONTINUOUS

        if pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
            if series.nunique() > HIGH_CARDINALITY_THRESHOLD:
                # Could be text or high-cardinality categorical
                avg_len = series.dropna().astype(str).str.len().mean()
                if avg_len > 50:
                    return FeatureType.TEXT
                return FeatureType.CATEGORICAL_NOMINAL
            return FeatureType.CATEGORICAL_NOMINAL

        return FeatureType.UNKNOWN

    # ── Task Type Detection ───────────────────────────────────────────────────

    def _detect_task_and_target(
        self, df: pd.DataFrame, profiles: list[ColumnProfile]
    ) -> tuple[str, str | None]:
        """
        Heuristically detect the ML task type and the most likely target column.
        Priority: last column → column named 'target'/'label'/'class'/'y' → best candidate
        """
        # Candidate target column names (common conventions)
        priority_names = {"target", "label", "class", "output", "y", "result", "outcome"}

        candidates: list[tuple[str, int]] = []  # (col_name, score)

        for profile in profiles:
            score = 0
            col_lower = profile.name.lower()

            if col_lower in priority_names:
                score += 10

            # Last column is often the target
            if profile.name == df.columns[-1]:
                score += 5

            # Prefer binary/low-cardinality columns for classification
            if profile.unique_count == 2:
                score += 4
            elif 3 <= profile.unique_count <= 20:
                score += 2

            # Avoid high-cardinality, datetime, text
            if profile.feature_type in (FeatureType.TEXT.value, FeatureType.DATETIME.value):
                score -= 10
            if profile.unique_count > 50:
                score -= 3

            candidates.append((profile.name, score))

        if not candidates:
            return TaskType.REGRESSION.value, None

        best_col = max(candidates, key=lambda x: x[1])[0]
        series = df[best_col]

        # Determine task type from target column
        n_unique = series.nunique()
        if n_unique == 2:
            return TaskType.BINARY_CLASSIFICATION.value, best_col
        elif 3 <= n_unique <= 20:
            return TaskType.MULTICLASS_CLASSIFICATION.value, best_col
        elif pd.api.types.is_numeric_dtype(series):
            return TaskType.REGRESSION.value, best_col
        else:
            return TaskType.MULTICLASS_CLASSIFICATION.value, best_col

    # ── Class Imbalance ───────────────────────────────────────────────────────

    def _check_class_imbalance(
        self, target: pd.Series
    ) -> tuple[dict[str, int], bool]:
        counts = target.value_counts()
        class_distribution = {str(k): int(v) for k, v in counts.items()}
        total = counts.sum()
        minority_pct = counts.min() / total if total > 0 else 1.0
        is_imbalanced = minority_pct < CLASS_IMBALANCE_THRESHOLD
        return class_distribution, is_imbalanced

    # ── Correlation ───────────────────────────────────────────────────────────

    def _compute_correlation(self, df: pd.DataFrame) -> dict[str, dict[str, float]] | None:
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            return None
        corr = numeric_df.corr().round(4)
        return corr.to_dict()

    # ── Outlier Detection ─────────────────────────────────────────────────────

    def _detect_outliers(self, df: pd.DataFrame) -> dict[str, int]:
        result = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 4:
                continue
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - OUTLIER_IQR_MULTIPLIER * iqr
            upper = q3 + OUTLIER_IQR_MULTIPLIER * iqr
            outlier_count = int(((series < lower) | (series > upper)).sum())
            if outlier_count > 0:
                result[col] = outlier_count
        return result

    # ── Quality Score ─────────────────────────────────────────────────────────

    def _compute_quality_score(
        self,
        df: pd.DataFrame,
        profiles: list[ColumnProfile],
        duplicate_count: int,
        missing_pct: float,
        is_imbalanced: bool,
    ) -> float:
        """
        Compute a 0–100 dataset quality score.
        Deductions: missing values, duplicates, class imbalance, constant columns.
        """
        score = 100.0

        # Missing values: up to -30
        score -= min(30, missing_pct * 100 * 1.5)

        # Duplicates: up to -15
        dup_pct = duplicate_count / len(df) if len(df) > 0 else 0
        score -= min(15, dup_pct * 100)

        # Class imbalance: -10
        if is_imbalanced:
            score -= 10

        # Constant / near-constant columns: -3 each, up to -15
        constant_penalty = sum(
            1 for p in profiles if p.unique_count <= 1
        )
        score -= min(15, constant_penalty * 3)

        # High missing columns: -2 each, up to -10
        high_missing = sum(
            1 for p in profiles if p.null_pct > MISSING_VALUE_CRITICAL_THRESHOLD
        )
        score -= min(10, high_missing * 2)

        return round(max(0.0, min(100.0, score)), 1)

    # ── Warnings and Recommendations ─────────────────────────────────────────

    def _generate_warnings(
        self,
        df: pd.DataFrame,
        profiles: list[ColumnProfile],
        duplicate_count: int,
        missing_pct: float,
        is_imbalanced: bool,
        outlier_counts: dict[str, int],
    ) -> list[str]:
        warns = []

        if duplicate_count > 0:
            pct = round(duplicate_count / len(df) * 100, 1)
            warns.append(f"{duplicate_count} duplicate rows detected ({pct}% of data).")

        if missing_pct > 0:
            warns.append(f"Dataset has {missing_pct * 100:.1f}% missing values overall.")

        critical_cols = [p.name for p in profiles if p.null_pct > MISSING_VALUE_CRITICAL_THRESHOLD]
        if critical_cols:
            warns.append(f"Columns with >50% missing values: {', '.join(critical_cols)}.")

        if is_imbalanced:
            warns.append("Target class distribution is imbalanced. Consider resampling strategies.")

        skewed = [p.name for p in profiles if p.skewness and abs(p.skewness) > SKEWNESS_THRESHOLD]
        if skewed:
            warns.append(f"{len(skewed)} highly skewed numeric column(s): {', '.join(skewed[:5])}.")

        if outlier_counts:
            warns.append(
                f"Outliers detected in {len(outlier_counts)} column(s): "
                f"{', '.join(list(outlier_counts.keys())[:5])}."
            )

        constant_cols = [p.name for p in profiles if p.unique_count <= 1]
        if constant_cols:
            warns.append(f"Constant columns (no variance): {', '.join(constant_cols)}.")

        return warns

    def _generate_recommendations(
        self,
        profiles: list[ColumnProfile],
        duplicate_count: int,
        missing_pct: float,
        is_imbalanced: bool,
        task_type: str,
    ) -> list[str]:
        recs = []

        if duplicate_count > 0:
            recs.append("Remove duplicate rows before training to prevent data leakage.")

        has_missing = any(p.null_count > 0 for p in profiles)
        if has_missing:
            recs.append(
                "Apply imputation for missing values. "
                "Use median imputation for numeric columns and most-frequent for categorical."
            )

        has_categorical = any(
            p.feature_type in (FeatureType.CATEGORICAL_NOMINAL.value, FeatureType.CATEGORICAL_ORDINAL.value)
            for p in profiles
        )
        if has_categorical:
            recs.append(
                "Encode categorical columns. "
                "Use one-hot encoding for nominal features and ordinal encoding for ordinal."
            )

        skewed = [p.name for p in profiles if p.skewness and abs(p.skewness) > SKEWNESS_THRESHOLD]
        if skewed:
            recs.append(
                f"Apply log or Box-Cox transformation to reduce skewness in: {', '.join(skewed[:5])}."
            )

        if is_imbalanced:
            recs.append(
                "Apply SMOTE or class weights to handle class imbalance. "
                "This improves recall for the minority class."
            )

        has_numeric = any(p.feature_type in (
            FeatureType.NUMERIC_CONTINUOUS.value, FeatureType.NUMERIC_DISCRETE.value
        ) for p in profiles)
        if has_numeric:
            recs.append(
                "Scale numeric features. "
                "StandardScaler works well for most algorithms; "
                "RobustScaler is better when outliers are present."
            )

        return recs
