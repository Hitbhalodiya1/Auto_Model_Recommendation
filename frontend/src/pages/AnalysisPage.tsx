/**
 * AnalysisPage — shows dataset analysis results and allows creating an experiment.
 */

import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, CheckCircle, Loader2 } from "lucide-react";
import { datasetApi, experimentApi } from "@/services/api";

export function AnalysisPage() {
  const { datasetId } = useParams<{ datasetId: string }>();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const [expName, setExpName] = useState("");
  const [targetCol, setTargetCol] = useState("");

  const { data: dataset } = useQuery({
    queryKey: ["dataset", datasetId],
    queryFn: () => datasetApi.get(datasetId!),
    enabled: !!datasetId,
  });

  const analysisQuery = useQuery({
    queryKey: ["dataset-analysis", datasetId],
    queryFn: () => datasetApi.getAnalysis(datasetId!),
    enabled: !!datasetId,
    refetchInterval: (query) => {
      const data = query.state.data;
      return !data ? 3000 : false;
    },
  });

  // fallback to the dataset embedded analysis if available
  const effectiveAnalysis = analysisQuery.data ?? dataset?.analysis;

  const createExp = useMutation({
    mutationFn: () =>
      experimentApi.create({
        name: expName,
        dataset_id: datasetId!,
        target_column: targetCol,
      }),
    onSuccess: (exp) => navigate(`/experiments/${exp.id}/preprocessing`),
  });

  const deleteDataset = useMutation({
    mutationFn: (id: string) => datasetApi.delete(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["datasets"] });
      navigate("/datasets");
    },
  });

  if (!effectiveAnalysis) {
    const p = analysisQuery.data?.progress ?? 0;
    const stepsCompleted = analysisQuery.data?.steps_completed ?? 0;
    const stepsTotal = analysisQuery.data?.steps_total ?? null;
    return (
      <div className="flex flex-col items-center py-12">
        <div className="w-full max-w-2xl">
          <div className="flex items-center justify-between mb-2">
            <div className="text-sm text-gray-700">Analysis in progress…</div>
            <div className="text-sm text-gray-600">{p}%</div>
          </div>
          <div className="w-full bg-gray-200 h-3 rounded overflow-hidden">
            <div className="bg-primary-600 h-3" style={{ width: `${p}%` }} />
          </div>
          {stepsTotal ? (
            <div className="text-xs text-gray-500 mt-2">{stepsCompleted}/{stepsTotal} steps completed</div>
          ) : null}
          <div className="mt-4">
            <button
              className="px-3 py-2 bg-white border rounded text-sm"
              onClick={() => datasetId && datasetApi.triggerAnalysis(datasetId)}
            >
              Re-run Analysis
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (!effectiveAnalysis) {
    return (
      <div className="text-center py-20 text-gray-500">
        Analysis not available yet. Please wait for analysis to complete or click Re-run.
        <div className="mt-4">
          <button
            className="px-3 py-2 bg-white border rounded text-sm"
            onClick={() => datasetId && datasetApi.triggerAnalysis(datasetId)}
          >
            Re-run Analysis
          </button>
        </div>
      </div>
    );
  }

  const qualityColor =
    (effectiveAnalysis?.quality_score ?? 0) >= 80
      ? "text-success-700"
      : (effectiveAnalysis?.quality_score ?? 0) >= 60
      ? "text-warning-700"
      : "text-danger-700";

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <div>
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold text-gray-900">Dataset Analysis</h1>
          <div>
            <button
              className="px-3 py-2 text-sm text-danger-600 hover:underline"
              onClick={() => datasetId && deleteDataset.mutate(datasetId)}
            >
              Delete Dataset
            </button>
          </div>
        </div>
        <p className="text-gray-500 mt-1">{dataset?.original_name}</p>
      </div>

      {/* Quality Score */}
      <div className="card p-6 flex items-center gap-6">
        <div className="text-center">
          <div className={`text-5xl font-bold ${qualityColor}`}>
            {effectiveAnalysis?.quality_score ?? 0}
          </div>
          <div className="text-sm text-gray-500 mt-1">Quality Score</div>
        </div>
        <div className="flex-1 grid grid-cols-4 gap-4">
            {[
            { label: "Rows", value: (effectiveAnalysis?.row_count ?? 0).toLocaleString() },
            { label: "Columns", value: effectiveAnalysis?.column_count ?? 0 },
            { label: "Missing %", value: `${((effectiveAnalysis?.missing_value_pct ?? 0) * 100).toFixed(1)}%` },
            { label: "Duplicates", value: effectiveAnalysis?.duplicate_row_count ?? 0 },
          ].map(({ label, value }) => (
            <div key={label} className="text-center">
              <div className="text-xl font-semibold text-gray-900">{value}</div>
              <div className="text-xs text-gray-500 mt-0.5">{label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Detected task */}
      <div className="card p-4 flex items-center gap-3">
        <CheckCircle size={18} className="text-success-500 shrink-0" />
        <div>
          <span className="font-medium">Detected task:</span>{" "}
          <span className="text-primary-700 font-semibold">{(effectiveAnalysis?.task_type ?? "unknown").replace("_", " ")}</span>
          {effectiveAnalysis?.suggested_target_column && (
            <> · Suggested target: <code className="bg-gray-100 px-1 rounded text-sm">{effectiveAnalysis.suggested_target_column}</code></>
          )}
        </div>
      </div>

      {/* Warnings */}
      {effectiveAnalysis?.warnings && effectiveAnalysis.warnings.length > 0 && (
        <div className="card p-4 space-y-2">
          <h3 className="font-semibold text-gray-900 flex items-center gap-2">
            <AlertTriangle size={16} className="text-warning-500" /> Warnings
          </h3>
          <ul className="text-sm text-gray-700 space-y-1">
            {effectiveAnalysis.warnings.map((w, i) => <li key={i} className="flex gap-2"><span>•</span>{w}</li>)}
          </ul>
        </div>
      )}

      {/* Recommendations */}
      {effectiveAnalysis?.recommendations && effectiveAnalysis.recommendations.length > 0 && (
        <div className="card p-4 space-y-2">
          <h3 className="font-semibold text-gray-900">Preprocessing Recommendations</h3>
          <ul className="text-sm text-gray-700 space-y-1">
            {effectiveAnalysis.recommendations.map((r, i) => <li key={i} className="flex gap-2"><span className="text-primary-500">→</span>{r}</li>)}
          </ul>
        </div>
      )}

      {/* Create Experiment */}
      <div className="card p-6 space-y-4">
        <h3 className="font-semibold text-gray-900">Create Experiment</h3>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Experiment Name</label>
            <input
              type="text"
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
              value={expName}
              onChange={(e) => setExpName(e.target.value)}
              placeholder="e.g. My First Experiment"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Target Column</label>
            <select
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
              value={targetCol}
              onChange={(e) => setTargetCol(e.target.value)}
            >
              <option value="">Select target column…</option>
              {effectiveAnalysis?.column_profiles?.map((cp) => (
                <option key={cp.name} value={cp.name}>{cp.name}</option>
              ))}
            </select>
          </div>
        </div>
        <button
          className="btn-primary"
          disabled={!expName || !targetCol || createExp.isPending}
          onClick={() => createExp.mutate()}
        >
          {createExp.isPending ? <Loader2 className="animate-spin" size={16} /> : "Create Experiment →"}
        </button>
      </div>
    </div>
  );
}
