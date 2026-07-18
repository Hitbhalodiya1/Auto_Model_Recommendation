/**
 * PreprocessingPage — show recommendations and execute preprocessing pipeline.
 */

import { useNavigate, useParams } from "react-router-dom";
import { useMutation, useQuery } from "@tanstack/react-query";
import { CheckCircle, Loader2 } from "lucide-react";
import { preprocessingApi } from "@/services/api";

export function PreprocessingPage() {
  const { experimentId } = useParams<{ experimentId: string }>();
  const navigate = useNavigate();

  const { data: recommendation, isPending: loading } = useQuery({
    queryKey: ["preprocessing-recommendation", experimentId],
    queryFn: () => preprocessingApi.recommend(experimentId!),
    enabled: !!experimentId,
  });

  const execute = useMutation({
    mutationFn: () => preprocessingApi.execute(experimentId!),
    onSuccess: () => navigate(`/experiments/${experimentId}/training`),
  });

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Preprocessing Pipeline</h1>
        <p className="text-gray-500 mt-1">Review recommended steps before executing.</p>
      </div>

      {loading ? (
        <div className="flex justify-center py-20"><Loader2 className="animate-spin text-gray-400" size={32} /></div>
      ) : recommendation ? (
        <>
          <div className="space-y-3">
            {recommendation.steps.map((step, i) => (
              <div key={step.name} className="card p-4 flex gap-4">
                <div className="w-7 h-7 rounded-full bg-primary-100 text-primary-700 text-xs font-bold flex items-center justify-center shrink-0">
                  {i + 1}
                </div>
                <div className="flex-1">
                  <div className="font-medium text-gray-900">{step.display_name}</div>
                  <div className="text-sm text-gray-600 mt-1">{step.explanation}</div>
                </div>
              </div>
            ))}
          </div>

          <div className="card p-4 flex items-center gap-3 bg-primary-50 border-primary-200">
            <CheckCircle size={18} className="text-primary-600 shrink-0" />
            <span className="text-sm text-primary-800">
              Recommended scaler: <strong>{recommendation.recommended_scaler}</strong>
            </span>
          </div>

          <button
            className="btn-primary w-full"
            disabled={execute.isPending}
            onClick={() => execute.mutate()}
          >
            {execute.isPending
              ? <span className="flex items-center justify-center gap-2"><Loader2 className="animate-spin" size={16} /> Executing…</span>
              : `Execute Pipeline (${recommendation.step_count} steps) →`
            }
          </button>
        </>
      ) : null}
    </div>
  );
}
