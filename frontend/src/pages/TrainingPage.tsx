/**
 * TrainingPage — start training and poll status.
 */

import { useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Brain, CheckCircle, Loader2 } from "lucide-react";
import { trainingApi } from "@/services/api";

export function TrainingPage() {
  const { experimentId } = useParams<{ experimentId: string }>();
  const navigate = useNavigate();

  const start = useMutation({
    mutationFn: () => trainingApi.start(experimentId!),
  });

  const { data: status } = useQuery({
    queryKey: ["training-status", experimentId],
    queryFn: () => trainingApi.status(experimentId!),
    enabled: !!experimentId && start.isSuccess,
    refetchInterval: (query) => {
      const s = query.state.data?.status;
      return s === "complete" || s === "error" ? false : 2000;
    },
  });

  // Auto-navigate when complete
  useEffect(() => {
    if (status?.status === "complete") {
      navigate(`/experiments/${experimentId}/results`);
    }
  }, [status?.status, experimentId, navigate]);

  const isRunning = status?.status === "training" || status?.status === "evaluating";

  return (
    <div className="max-w-xl mx-auto text-center py-16 space-y-6">
      <div className="w-20 h-20 rounded-full bg-primary-50 flex items-center justify-center mx-auto">
        {isRunning
          ? <Loader2 className="animate-spin text-primary-600" size={36} />
          : <Brain className="text-primary-600" size={36} />
        }
      </div>

      <div>
        <h1 className="text-2xl font-bold text-gray-900">Model Training</h1>
        <p className="text-gray-500 mt-2">
          {start.isIdle
            ? "Ready to train all compatible models on your preprocessed dataset."
            : isRunning
            ? `Training in progress… Status: ${status?.status}`
            : status?.status === "complete"
            ? "Training complete! Redirecting to results…"
            : "Waiting to start…"
          }
        </p>
      </div>

      {start.isIdle && (
        <button
          className="btn-primary"
          onClick={() => start.mutate()}
        >
          Start Training All Models
        </button>
      )}

      {status?.status === "complete" && (
        <div className="flex items-center justify-center gap-2 text-success-700">
          <CheckCircle size={18} /> Training complete
        </div>
      )}
    </div>
  );
}
