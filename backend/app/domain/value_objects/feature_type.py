"""
FeatureType value object — describes the type of a dataset column.
"""

from enum import StrEnum


class FeatureType(StrEnum):
    """The inferred type of a dataset column."""

    NUMERIC_CONTINUOUS = "numeric_continuous"
    NUMERIC_DISCRETE = "numeric_discrete"
    CATEGORICAL_NOMINAL = "categorical_nominal"
    CATEGORICAL_ORDINAL = "categorical_ordinal"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"
    UNKNOWN = "unknown"

    @property
    def is_numeric(self) -> bool:
        return self in (FeatureType.NUMERIC_CONTINUOUS, FeatureType.NUMERIC_DISCRETE)

    @property
    def is_categorical(self) -> bool:
        return self in (FeatureType.CATEGORICAL_NOMINAL, FeatureType.CATEGORICAL_ORDINAL)

    @property
    def requires_encoding(self) -> bool:
        return self.is_categorical or self == FeatureType.BOOLEAN
