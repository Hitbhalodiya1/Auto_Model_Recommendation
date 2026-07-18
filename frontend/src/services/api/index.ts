/**
 * API service functions for all backend endpoints.
 */

import { apiClient, apiDelete, apiGet, apiPost } from "./client";
import type {
  CreateExperimentRequest,
  Dataset,
  DatasetAnalysis,
  DatasetPreview,
  Experiment,
  Explainability,
  ModelResult,
  PreprocessingRecommendation,
  Recommendation,
} from "@/types";

// ── Datasets ──────────────────────────────────────────────────────────────────

export const datasetApi = {
  upload: async (file: File): Promise<Dataset> => {
    const formData = new FormData();
    formData.append("file", file);
    const resp = await apiClient.post("/datasets/upload", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return resp.data.data;
  },

  list: (limit = 20, offset = 0) =>
    apiGet<Dataset[]>(`/datasets?limit=${limit}&offset=${offset}`),

  get: (datasetId: string) => apiGet<Dataset>(`/datasets/${datasetId}`),

  preview: (datasetId: string, nRows = 100) =>
    apiGet<DatasetPreview>(`/datasets/${datasetId}/preview?n_rows=${nRows}`),

  getAnalysis: (datasetId: string) =>
    apiGet<DatasetAnalysis>(`/datasets/${datasetId}/analysis`),

  triggerAnalysis: (datasetId: string) =>
    apiPost<{ dataset_id: string }>(`/datasets/${datasetId}/analyze`),

  delete: (datasetId: string) => apiDelete(`/datasets/${datasetId}`),
};

// ── Experiments ───────────────────────────────────────────────────────────────

export const experimentApi = {
  create: (req: CreateExperimentRequest) =>
    apiPost<Experiment>("/experiments", req),

  list: (limit = 20, offset = 0) =>
    apiGet<Experiment[]>(`/experiments?limit=${limit}&offset=${offset}`),

  get: (experimentId: string) =>
    apiGet<Experiment>(`/experiments/${experimentId}`),

  delete: (experimentId: string) => apiDelete(`/experiments/${experimentId}`),
};

// ── Preprocessing ─────────────────────────────────────────────────────────────

export const preprocessingApi = {
  recommend: (experimentId: string) =>
    apiPost<PreprocessingRecommendation>(
      `/experiments/${experimentId}/preprocessing/recommend`
    ),

  execute: (experimentId: string) =>
    apiPost<{ is_executed: boolean; executed_at: string | null }>(
      `/experiments/${experimentId}/preprocessing/execute`
    ),

  status: (experimentId: string) =>
    apiGet<{ is_executed: boolean; executed_at: string | null }>(
      `/experiments/${experimentId}/preprocessing/status`
    ),
};

// ── Training ──────────────────────────────────────────────────────────────────

export const trainingApi = {
  start: (experimentId: string) =>
    apiPost<{ experiment_id: string; status: string }>(
      `/experiments/${experimentId}/training/start`
    ),

  status: (experimentId: string) =>
    apiGet<{ experiment_id: string; status: string }>(
      `/experiments/${experimentId}/training/status`
    ),

  results: (experimentId: string) =>
    apiGet<ModelResult[]>(`/experiments/${experimentId}/training/results`),

  recommendation: (experimentId: string) =>
    apiGet<Recommendation>(`/experiments/${experimentId}/recommendation`),

  explain: (experimentId: string, modelId: string) =>
    apiPost<Explainability>(`/experiments/${experimentId}/explain/${modelId}`),
};
