/**
 * HistoryPage — list all past experiments.
 */

import { Link, useNavigate } from "react-router-dom";
import { useQuery, useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { ArrowRight, FlaskConical, Loader2, X } from "lucide-react";
import { clsx } from "clsx";
import { experimentApi, datasetApi } from "@/services/api";
import type { Dataset, Experiment } from "@/types";

const STATUS_STYLES: Record<string, string> = {
  created: "badge-gray",
  preprocessing: "badge-warning",
  training: "badge-warning",
  evaluating: "badge-warning",
  complete: "badge-success",
  error: "badge-danger",
};

export function HistoryPage() {
  const { data: experiments, isLoading } = useQuery({
    queryKey: ["experiments"],
    queryFn: () => experimentApi.list(),
  });

  const [showPicker, setShowPicker] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [expName, setExpName] = useState("");
  const [targetCol, setTargetCol] = useState("");
  const { data: datasets } = useQuery({ queryKey: ["datasets"], queryFn: () => datasetApi.list() });
  const navigate = useNavigate();

  const createExp = useMutation({
    mutationFn: () =>
      experimentApi.create({
        name: expName,
        dataset_id: selectedDataset?.id ?? "",
        target_column: targetCol,
      }),
    onSuccess: (exp) => {
      setShowPicker(false);
      setSelectedDataset(null);
      setExpName("");
      setTargetCol("");
      navigate(`/experiments/${exp.id}/preprocessing`);
    },
  });

  const analyzedDatasets = datasets?.filter((d: Dataset) => d.status === "analyzed") || [];

  const handleDatasetSelect = (dataset: Dataset) => {
    setSelectedDataset(dataset);
    setExpName(`${dataset.original_name.replace(/\.[^/.]+$/, "")} Experiment`);
    if (dataset.analysis?.suggested_target_column) {
      setTargetCol(dataset.analysis.suggested_target_column);
    }
  };

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Experiment History</h1>
          <p className="text-gray-500 mt-1">All past and ongoing ML experiments.</p>
        </div>
        <div>
          <button className="btn-primary" onClick={() => setShowPicker(true)}>New Experiment</button>
        </div>
      </div>

      <div className="card overflow-hidden">
        {isLoading ? (
          <div className="flex justify-center py-12">
            <Loader2 className="animate-spin text-gray-400" size={24} />
          </div>
        ) : !experiments?.length ? (
          <div className="flex flex-col items-center py-16 text-gray-400">
            <FlaskConical size={40} className="mb-3" />
            <p>No experiments yet.</p>
            <Link to="/datasets" className="mt-4 btn-primary text-sm">Upload a Dataset to Start</Link>
          </div>
        ) : (
          <div className="divide-y divide-gray-100">
            {experiments.map((exp: Experiment) => (
              <Link
                key={exp.id}
                to={`/experiments/${exp.id}`}
                className="flex items-center justify-between px-6 py-4 hover:bg-gray-50 transition-colors"
              >
                <div>
                  <div className="font-medium text-gray-900">{exp.name}</div>
                  <div className="text-sm text-gray-500 mt-0.5">
                    {exp.task_type?.replace("_", " ") ?? "—"} ·{" "}
                    Target: {exp.target_column ?? "—"} ·{" "}
                    {new Date(exp.created_at).toLocaleDateString()}
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <span className={clsx("badge", STATUS_STYLES[exp.status] ?? "badge-gray")}>
                    {exp.status}
                  </span>
                  <ArrowRight size={16} className="text-gray-400" />
                </div>
              </Link>
            ))}
          </div>
        )}
      </div>

      {showPicker && (
        <div className="fixed inset-0 z-30 flex items-center justify-center bg-black/40">
          <div className="bg-white rounded-lg w-[500px] p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-semibold text-lg">Create New Experiment</h3>
              <button onClick={() => { setShowPicker(false); setSelectedDataset(null); }} className="text-gray-400 hover:text-gray-600">
                <X size={20} />
              </button>
            </div>

            {!selectedDataset ? (
              <>
                <p className="text-sm text-gray-500 mb-3">Select an analyzed dataset to create an experiment:</p>
                <div className="max-h-64 overflow-auto border rounded-lg">
                  {analyzedDatasets.length ? (
                    analyzedDatasets.map((d: Dataset) => (
                      <button
                        key={d.id}
                        className="w-full text-left px-4 py-3 hover:bg-gray-50 border-b last:border-b-0"
                        onClick={() => handleDatasetSelect(d)}
                      >
                        <div className="font-medium text-gray-900">{d.original_name}</div>
                        <div className="text-sm text-gray-500 mt-0.5">
                          {d.row_count?.toLocaleString()} rows · {d.column_count} columns
                        </div>
                      </button>
                    ))
                  ) : (
                    <div className="p-4 text-sm text-gray-500">
                      No analyzed datasets available. Please upload and analyze a dataset first.
                    </div>
                  )}
                </div>
                <div className="mt-4 text-right">
                  <button className="px-4 py-2 text-sm text-gray-600 hover:text-gray-800" onClick={() => setShowPicker(false)}>
                    Cancel
                  </button>
                </div>
              </>
            ) : (
              <>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Dataset</label>
                    <div className="px-3 py-2 bg-gray-50 rounded text-sm text-gray-900">
                      {selectedDataset.original_name}
                    </div>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Experiment Name</label>
                    <input
                      type="text"
                      value={expName}
                      onChange={(e) => setExpName(e.target.value)}
                      className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                      placeholder="Enter experiment name"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Target Column</label>
                    <select
                      value={targetCol}
                      onChange={(e) => setTargetCol(e.target.value)}
                      className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                    >
                      <option value="">Select target column</option>
                      {selectedDataset.analysis?.column_profiles?.map((col: { name: string; feature_type: string }) => (
                        <option key={col.name} value={col.name}>
                          {col.name} ({col.feature_type})
                        </option>
                      ))}
                    </select>
                  </div>

                  <div className="flex gap-3 pt-2">
                    <button
                      onClick={() => setSelectedDataset(null)}
                      className="flex-1 px-4 py-2 border rounded-lg text-gray-700 hover:bg-gray-50"
                    >
                      Back
                    </button>
                    <button
                      onClick={() => createExp.mutate()}
                      disabled={!expName || !targetCol || createExp.isPending}
                      className="flex-1 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {createExp.isPending ? "Creating..." : "Create Experiment"}
                    </button>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
