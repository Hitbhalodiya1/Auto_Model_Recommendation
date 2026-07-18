/**
 * ResultsPage — model comparison table and recommendation display.
 */

import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Award, Loader2, TrendingUp } from "lucide-react";
import { clsx } from "clsx";
import { trainingApi } from "@/services/api";
import type { ModelResult } from "@/types";

function MetricBadge({ value }: { value: number }) {
  const pct = Math.min(Math.max(value, 0), 1) * 100;
  const color = pct >= 80 ? "text-success-700" : pct >= 60 ? "text-warning-700" : "text-danger-700";
  return <span className={clsx("font-mono font-medium", color)}>{value.toFixed(4)}</span>;
}

export function ResultsPage() {
  const { experimentId } = useParams<{ experimentId: string }>();

  const { data: results, isLoading: loadingResults } = useQuery({
    queryKey: ["training-results", experimentId],
    queryFn: () => trainingApi.results(experimentId!),
    enabled: !!experimentId,
  });

  const { data: recommendation } = useQuery({
    queryKey: ["recommendation", experimentId],
    queryFn: () => trainingApi.recommendation(experimentId!),
    enabled: !!experimentId,
  });

  if (loadingResults) {
    return <div className="flex justify-center py-20"><Loader2 className="animate-spin text-gray-400" size={32} /></div>;
  }

  const primaryKey = results?.[0]
    ? Object.keys(results[0].metrics).find((k) =>
        ["f1_score", "f1_weighted", "r2_score", "accuracy"].includes(k)
      ) ?? "accuracy"
    : "accuracy";

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Results</h1>
        <p className="text-gray-500 mt-1">Model comparison and recommendation.</p>
      </div>

      {/* Recommendation Card */}
      {recommendation && (
        <div className="card p-6 border-primary-200 bg-gradient-to-r from-primary-50 to-white">
          <div className="flex items-start gap-4">
            <div className="w-10 h-10 rounded-lg bg-primary-100 flex items-center justify-center shrink-0">
              <Award className="text-primary-600" size={20} />
            </div>
            <div className="flex-1">
              <h2 className="font-semibold text-gray-900">
                Recommended: {recommendation.recommended_model?.display_name}
              </h2>
              <p className="text-sm text-gray-700 mt-2 leading-relaxed">
                {recommendation.explanation_text}
              </p>
              {recommendation.rationale.length > 0 && (
                <ul className="mt-3 space-y-1">
                  {recommendation.rationale.map((r, i) => (
                    <li key={i} className="text-sm text-gray-600 flex gap-2">
                      <span className="text-primary-500">✓</span> {r}
                    </li>
                  ))}
                </ul>
              )}
            </div>
            <div className="text-right shrink-0">
              <div className="text-2xl font-bold text-primary-700">
                {(recommendation.composite_score * 100).toFixed(1)}
              </div>
              <div className="text-xs text-gray-500">Composite Score</div>
            </div>
          </div>
        </div>
      )}

      {/* Model Rankings Table */}
      <div className="card overflow-hidden">
        <div className="px-4 py-3 border-b border-gray-200 flex items-center gap-2">
          <TrendingUp size={16} className="text-gray-500" />
          <h3 className="font-medium text-gray-900">
            All Models ({results?.length ?? 0})
          </h3>
        </div>
        <table className="w-full text-sm">
          <thead className="bg-gray-50 border-b border-gray-200">
            <tr>
              {["Rank", "Model", primaryKey, "CV Score", "Train Time", "Overfit", "Interpretability"].map((h) => (
                <th key={h} className="px-4 py-3 text-left font-medium text-gray-600">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {results?.map((r: ModelResult) => (
              <tr
                key={r.id}
                className={clsx(
                  "hover:bg-gray-50",
                  r.is_recommended && "bg-primary-50 hover:bg-primary-100"
                )}
              >
                <td className="px-4 py-3">
                  {r.is_recommended ? (
                    <span className="inline-flex items-center gap-1 text-primary-700 font-semibold">
                      <Award size={14} /> {r.rank}
                    </span>
                  ) : (
                    <span className="text-gray-500">{r.rank ?? "—"}</span>
                  )}
                </td>
                <td className="px-4 py-3 font-medium text-gray-900">{r.display_name}</td>
                <td className="px-4 py-3">
                  {r.metrics[primaryKey] !== undefined
                    ? <MetricBadge value={r.metrics[primaryKey]} />
                    : <span className="text-gray-400">—</span>}
                </td>
                <td className="px-4 py-3">
                  {r.cv_score !== null
                    ? <MetricBadge value={r.cv_score} />
                    : <span className="text-gray-400">—</span>}
                </td>
                <td className="px-4 py-3 text-gray-600">{r.training_time_s.toFixed(2)}s</td>
                <td className="px-4 py-3">
                  {r.is_overfitting
                    ? <span className="badge-danger badge">Yes</span>
                    : <span className="badge-success badge">No</span>}
                </td>
                <td className="px-4 py-3">
                  <div className="flex gap-0.5">
                    {Array.from({ length: 5 }).map((_, i) => (
                      <div
                        key={i}
                        className={clsx(
                          "w-3 h-3 rounded-sm",
                          i < r.interpretability_score ? "bg-primary-500" : "bg-gray-200"
                        )}
                      />
                    ))}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
