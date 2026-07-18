/**
 * ExperimentPage — experiment overview with status-based navigation.
 */

import { Link, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { ArrowRight, Loader2 } from "lucide-react";
import { experimentApi } from "@/services/api";

const STEPS = [
  { status: "preprocessing", label: "Preprocessing", path: "preprocessing" },
  { status: "training", label: "Training", path: "training" },
  { status: "complete", label: "Results", path: "results" },
];

export function ExperimentPage() {
  const { experimentId } = useParams<{ experimentId: string }>();

  const { data: exp, isLoading } = useQuery({
    queryKey: ["experiment", experimentId],
    queryFn: () => experimentApi.get(experimentId!),
    enabled: !!experimentId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === "complete" || status === "error" ? false : 3000;
    },
  });

  if (isLoading) {
    return <div className="flex justify-center py-20"><Loader2 className="animate-spin text-gray-400" size={32} /></div>;
  }
  if (!exp) return <div className="text-center py-20 text-gray-500">Experiment not found.</div>;

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">{exp.name}</h1>
        <p className="text-gray-500 mt-1">{exp.description || "No description."}</p>
      </div>

      <div className="card p-6 grid grid-cols-3 gap-6 text-sm">
        {[
          { label: "Status", value: exp.status },
          { label: "Task Type", value: exp.task_type?.replace("_", " ") ?? "—" },
          { label: "Target Column", value: exp.target_column ?? "—" },
        ].map(({ label, value }) => (
          <div key={label}>
            <div className="text-gray-500 text-xs uppercase font-medium mb-1">{label}</div>
            <div className="font-semibold text-gray-900 capitalize">{value}</div>
          </div>
        ))}
      </div>

      <div className="card p-6 space-y-3">
        <h2 className="font-semibold text-gray-900">Pipeline Steps</h2>
        {STEPS.map(({ label, path }) => (
          <Link
            key={path}
            to={`/experiments/${experimentId}/${path}`}
            className="flex items-center justify-between p-3 rounded-lg border border-gray-200 hover:border-primary-400 hover:bg-primary-50 transition-colors"
          >
            <span className="font-medium text-gray-700">{label}</span>
            <ArrowRight size={16} className="text-gray-400" />
          </Link>
        ))}
      </div>
    </div>
  );
}
