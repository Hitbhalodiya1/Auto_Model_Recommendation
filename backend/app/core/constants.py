"""
Application-wide constants. Never hardcode these values elsewhere.
"""

from typing import Final

# ── File Upload ───────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS: Final[frozenset[str]] = frozenset({".csv", ".xlsx", ".xls"})
ALLOWED_MIME_TYPES: Final[frozenset[str]] = frozenset(
    {
        "text/csv",
        "application/csv",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "text/plain",  # some systems send CSV as text/plain
    }
)

# ── Dataset Analysis ──────────────────────────────────────────────────────────
PREVIEW_ROWS: Final[int] = 100
MIN_ROWS_FOR_TRAINING: Final[int] = 20
HIGH_CARDINALITY_THRESHOLD: Final[int] = 50          # unique values → treat as categorical
MISSING_VALUE_CRITICAL_THRESHOLD: Final[float] = 0.5 # >50% missing → critical warning
OUTLIER_IQR_MULTIPLIER: Final[float] = 1.5
SKEWNESS_THRESHOLD: Final[float] = 1.0
CORRELATION_STRONG_THRESHOLD: Final[float] = 0.85
CLASS_IMBALANCE_THRESHOLD: Final[float] = 0.15       # minority class < 15% → imbalanced

# ── ML Engine ─────────────────────────────────────────────────────────────────
SCALER_NONE: Final[str] = "none"
SCALER_STANDARD: Final[str] = "standard"
SCALER_MINMAX: Final[str] = "minmax"
SCALER_ROBUST: Final[str] = "robust"
SCALER_MAXABS: Final[str] = "maxabs"
SCALER_NORMALIZER: Final[str] = "normalizer"

SHAP_MAX_BACKGROUND_SAMPLES: Final[int] = 100
FEATURE_IMPORTANCE_TOP_N: Final[int] = 20

# ── Recommendation Scoring Weights ────────────────────────────────────────────
DEFAULT_WEIGHT_PERFORMANCE: Final[float] = 0.35
DEFAULT_WEIGHT_GENERALIZATION: Final[float] = 0.20
DEFAULT_WEIGHT_TRAIN_SPEED: Final[float] = 0.10
DEFAULT_WEIGHT_PRED_SPEED: Final[float] = 0.10
DEFAULT_WEIGHT_INTERPRETABILITY: Final[float] = 0.15
DEFAULT_WEIGHT_OVERFITTING: Final[float] = 0.10

# ── API ───────────────────────────────────────────────────────────────────────
REQUEST_ID_HEADER: Final[str] = "X-Request-ID"
DEFAULT_PAGE_SIZE: Final[int] = 20
MAX_PAGE_SIZE: Final[int] = 100

# ── Report Formats ────────────────────────────────────────────────────────────
REPORT_FORMAT_PDF: Final[str] = "pdf"
REPORT_FORMAT_CSV: Final[str] = "csv"
REPORT_FORMAT_JSON: Final[str] = "json"
