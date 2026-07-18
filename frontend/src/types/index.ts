/**
 * Shared TypeScript type definitions for AutoRec frontend.
 * These mirror the backend DTOs.
 */

// ── API Response Envelope ─────────────────────────────────────────────────────

export interface APIResponse<T> {
  success: boolean;
  data: T;
  message: string;
  errors: Record<string, unknown>;
  meta?: {
    request_id: string;
    version: string;
  };
}

// ── Dataset Types ─────────────────────────────────────────────────────────────

export type DatasetStatus = "uploaded" | "analyzing" | "analyzed" | "error";

export interface ColumnProfile {
  name: string;
  dtype: string;
  feature_type: string;
  null_count: number;
  null_pct: number;
  unique_count: number;
  unique_pct: number;
  sample_values: unknown[];
  mean?: number;
  std?: number;
  min?: number;
  max?: number;
  median?: number;
  skewness?: number;
  kurtosis?: number;
}

export interface DatasetAnalysis {
  dataset_id: string;
  task_type: string;
  suggested_target_column: string | null;
  quality_score: number;
  row_count: number;
  column_count: number;
  duplicate_row_count: number;
  missing_value_pct: number;
  column_profiles: ColumnProfile[];
  class_distribution: Record<string, number> | null;
  is_imbalanced: boolean;
  correlation_matrix: Record<string, Record<string, number>> | null;
  outlier_counts: Record<string, number> | null;
  warnings: string[];
  recommendations: string[];
  analyzed_at: string;
  progress?: number;
  steps_total?: number | null;
  steps_completed?: number | null;
}

export interface Dataset {
  id: string;
  filename: string;
  original_name: string;
  file_size: number;
  row_count: number | null;
  column_count: number | null;
  status: DatasetStatus;
  analysis: DatasetAnalysis | null;
  created_at: string;
  updated_at: string;
}

export interface DatasetPreview {
  columns: string[];
  rows: Record<string, string>[];
  total_rows: number;
}

// ── Experiment Types ──────────────────────────────────────────────────────────

export type ExperimentStatus =
  | "created"
  | "preprocessing"
  | "training"
  | "evaluating"
  | "complete"
  | "error";

export interface Experiment {
  id: string;
  name: string;
  description: string;
  dataset_id: string;
  status: ExperimentStatus;
  task_type: string | null;
  target_column: string | null;
  created_at: string;
  updated_at: string;
}

export interface CreateExperimentRequest {
  name: string;
  description?: string;
  dataset_id: string;
  target_column: string;
  task_type?: string;
}

// ── Preprocessing Types ───────────────────────────────────────────────────────

export interface PreprocessingStep {
  name: string;
  display_name: string;
  strategy: string;
  params: Record<string, unknown>;
  explanation: string;
  affects_columns: string[];
}

export interface PreprocessingRecommendation {
  experiment_id: string;
  steps: PreprocessingStep[];
  step_count: number;
  recommended_scaler: string;
}

// ── Training / Model Result Types ─────────────────────────────────────────────

export interface ModelResult {
  id: string;
  experiment_id: string;
  algorithm_name: string;
  config_name: string;
  display_name: string;
  configuration: Record<string, unknown>;
  metrics: Record<string, number>;
  cv_score: number | null;
  cv_std: number | null;
  is_overfitting: boolean;
  training_time_s: number;
  prediction_time_s: number;
  is_recommended: boolean;
  rank: number | null;
  requires_scaling: boolean;
  supports_feature_importance: boolean;
  supports_shap: boolean;
  interpretability_score: number;
  error_message: string | null;
  created_at: string;
}

// ── Recommendation Types ──────────────────────────────────────────────────────

export interface RankingEntry {
  rank: number;
  config_name: string;
  display_name: string;
  composite_score: number;
  primary_metric: number | null;
  cv_score: number | null;
  is_overfitting: boolean;
}

export interface Recommendation {
  id: string;
  experiment_id: string;
  model_result_id: string;
  composite_score: number;
  score_breakdown: Record<string, number>;
  rationale: string[];
  explanation_text: string;
  all_rankings: RankingEntry[];
  recommended_model: ModelResult | null;
  created_at: string;
}

// ── Explainability Types ──────────────────────────────────────────────────────

export interface FeatureImportance {
  feature: string;
  importance: number;
  rank: number;
}

export interface Explainability {
  model_result_id: string;
  feature_importances: FeatureImportance[];
  shap_values: number[][] | null;
  shap_base_value: number | null;
  top_features: string[];
  method_used: string;
  error: string | null;
}
