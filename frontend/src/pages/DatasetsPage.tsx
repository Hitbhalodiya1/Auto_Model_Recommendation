/**
 * DatasetsPage — upload new datasets and view existing ones.
 */

import { useCallback, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Database, Loader2, Upload } from "lucide-react";
import { clsx } from "clsx";
import { datasetApi } from "@/services/api";
import type { Dataset } from "@/types";

function StatusBadge({ status }: { status: Dataset["status"] }) {
  const styles = {
    uploaded: "badge-gray",
    analyzing: "badge-warning",
    analyzed: "badge-success",
    error: "badge-danger",
  };
  return <span className={clsx("badge", styles[status])}>{status}</span>;
}

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export function DatasetsPage() {
  const navigate = useNavigate();
  const qc = useQueryClient();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const { data: datasets, isLoading } = useQuery({
    queryKey: ["datasets"],
    queryFn: () => datasetApi.list(),
    refetchInterval: (query) => {
      const data = query.state.data;
      if (!data) return false;
      const hasAnalyzing = data.some((d) => d.status === "analyzing" || d.status === "uploaded");
      return hasAnalyzing ? 3000 : false;
    },
  });

  const upload = useMutation({
    mutationFn: (file: File) => datasetApi.upload(file),
    onSuccess: (dataset: Dataset) => {
      qc.invalidateQueries({ queryKey: ["datasets"] });
      const id = dataset?.id;
      if (id) {
        navigate(`/datasets/${id}/analyze`);
      }
    },
    onError: (error: Error) => {
      const message = (error as any).response?.data?.message || error.message || "Upload failed";
      alert(`Upload failed: ${message}`);
    },
  });

  const deleteMut = useMutation({
    mutationFn: (id: string) => datasetApi.delete(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["datasets"] });
      alert("Dataset deleted successfully");
    },
    onError: (error: Error) => {
      const message = (error as any).response?.data?.message || error.message || "Delete failed";
      alert(`Delete failed: ${message}`);
    },
  });

  const [openMenu, setOpenMenu] = useState<string | null>(null);

  const handleFile = useCallback(
    (file: File) => {
      upload.mutate(file);
    },
    [upload]
  );

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Datasets</h1>
        <p className="text-gray-500 mt-1">Upload and manage your ML datasets.</p>
      </div>

      {/* Upload zone */}
      <div
        className={clsx(
          "card p-10 border-2 border-dashed text-center cursor-pointer transition-colors",
          dragOver ? "border-primary-400 bg-primary-50" : "border-gray-300 hover:border-primary-400"
        )}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        role="button"
        aria-label="Upload dataset"
      >
        <input
          ref={inputRef}
          type="file"
          accept=".csv,.xlsx,.xls"
          className="hidden"
          onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
        />
        {upload.isPending ? (
          <div className="flex flex-col items-center gap-3 text-gray-500">
            <Loader2 className="animate-spin text-primary-500" size={32} />
            <span>Uploading…</span>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3 text-gray-500">
            <Upload size={32} className="text-gray-400" />
            <div>
              <p className="font-medium text-gray-700">Drop your CSV or Excel file here</p>
              <p className="text-sm mt-1">or click to browse · Max 500 MB</p>
            </div>
          </div>
        )}
      </div>

      {/* Dataset list */}
      <div className="card overflow-hidden">
        {isLoading ? (
          <div className="flex justify-center py-12">
            <Loader2 className="animate-spin text-gray-400" size={24} />
          </div>
        ) : !datasets?.length ? (
          <div className="flex flex-col items-center py-16 text-gray-400">
            <Database size={40} className="mb-3" />
            <p>No datasets yet. Upload one above to get started.</p>
          </div>
        ) : (
          <table className="w-full text-sm">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                {["Name", "Rows", "Columns", "Size", "Status", ""].map((h) => (
                  <th key={h} className="px-4 py-3 text-left font-medium text-gray-600">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {datasets.map((d) => (
                <tr
                  key={d.id}
                  className="hover:bg-gray-50 cursor-pointer"
                  onClick={() => d.status === "analyzed" && navigate(`/datasets/${d.id}/analyze`)}
                >
                  <td className="px-4 py-3 font-medium text-gray-900">{d.original_name}</td>
                  <td className="px-4 py-3 text-gray-600">{d.row_count?.toLocaleString() ?? "—"}</td>
                  <td className="px-4 py-3 text-gray-600">{d.column_count ?? "—"}</td>
                  <td className="px-4 py-3 text-gray-600">{formatBytes(d.file_size)}</td>
                  <td className="px-4 py-3"><StatusBadge status={d.status} /></td>
                  <td className="px-4 py-3 text-right">
                    <div className="relative inline-block text-left">
                      <button
                        onClick={(e) => { e.stopPropagation(); setOpenMenu(openMenu === d.id ? null : d.id); }}
                        className="px-2 py-1 text-gray-600 hover:text-gray-900 rounded transition-colors"
                        aria-label="Dataset actions"
                      >
                        ⋯
                      </button>
                      {openMenu === d.id && (
                        <div
                          className="absolute right-0 mt-2 w-44 bg-white border border-gray-200 rounded shadow-lg z-10"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <button
                            className="w-full text-left px-3 py-2 hover:bg-gray-50"
                            onClick={() => { setOpenMenu(null); navigate(`/datasets/${d.id}/analyze`); }}
                          >
                            View Analysis
                          </button>
                          <button
                            className="w-full text-left px-3 py-2 hover:bg-gray-50"
                            onClick={() => { setOpenMenu(null); datasetApi.triggerAnalysis(d.id).then(() => qc.invalidateQueries({ queryKey: ["datasets"] })); }}
                          >
                            Re-run Analysis
                          </button>
                          <button
                            className="w-full text-left px-3 py-2 hover:bg-gray-50 text-primary-700"
                            onClick={() => { setOpenMenu(null); navigate(`/datasets/${d.id}/analyze`); }}
                          >
                            Create Experiment →
                          </button>
                          <div className="border-t border-gray-100" />
                          <button
                            className="w-full text-left px-3 py-2 hover:bg-gray-50 text-danger-600"
                            onClick={() => { setOpenMenu(null); deleteMut.mutate(d.id); }}
                          >
                            Delete
                          </button>
                        </div>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
