/**
 * Axios API client configured for the AutoRec backend.
 * All API calls go through this instance.
 */

import axios, { type AxiosInstance, type AxiosRequestConfig } from "axios";
import type { APIResponse } from "@/types";

const BASE_URL = import.meta.env.VITE_API_URL ?? "/api/v1";

function createApiClient(): AxiosInstance {
  const client = axios.create({
    baseURL: BASE_URL,
    headers: {
      "Content-Type": "application/json",
    },
    timeout: 30_000,
  });

  // Request interceptor: inject correlation header
  client.interceptors.request.use((config) => {
    config.headers["X-Client"] = "autorec-frontend";
    return config;
  });

  // Response interceptor: unwrap data or throw structured errors
  client.interceptors.response.use(
    (response) => response,
    (error) => {
      if (error.response) {
        const data = error.response.data;
        const status = error.response.status;

        // 409 Conflict for analysis means "not ready yet" - return null instead of error
        if (status === 409 && error.config?.url?.includes('/analysis')) {
          return Promise.resolve({ data: { data: null } });
        }

        const message =
          data?.message ?? `Request failed with status ${status}`;
        const apiError = new Error(message) as Error & {
          status: number;
          details: Record<string, unknown>;
        };
        apiError.status = status;
        apiError.details = data?.errors ?? {};
        return Promise.reject(apiError);
      }
      return Promise.reject(error);
    }
  );

  return client;
}

export const apiClient = createApiClient();

/** Unwrap the APIResponse envelope and return the data field. */
export async function apiGet<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
  const resp = await apiClient.get<APIResponse<T>>(url, config);
  return resp.data.data;
}

export async function apiPost<T>(
  url: string,
  data?: unknown,
  config?: AxiosRequestConfig
): Promise<T> {
  const resp = await apiClient.post<APIResponse<T>>(url, data, config);
  return resp.data.data;
}

export async function apiDelete(url: string): Promise<void> {
  await apiClient.delete(url);
}
