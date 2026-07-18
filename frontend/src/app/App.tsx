/**
 * App root — sets up React Router and renders the application shell.
 */

import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AppLayout } from "@/components/layout/AppLayout";
import { HomePage } from "@/pages/HomePage";
import { DatasetsPage } from "@/pages/DatasetsPage";
import { ExperimentPage } from "@/pages/ExperimentPage";
import { AnalysisPage } from "@/pages/AnalysisPage";
import { PreprocessingPage } from "@/pages/PreprocessingPage";
import { TrainingPage } from "@/pages/TrainingPage";
import { ResultsPage } from "@/pages/ResultsPage";
import { HistoryPage } from "@/pages/HistoryPage";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<AppLayout />}>
          <Route index element={<HomePage />} />
          <Route path="datasets" element={<DatasetsPage />} />
          <Route path="datasets/:datasetId/analyze" element={<AnalysisPage />} />
          <Route path="experiments" element={<HistoryPage />} />
          <Route path="experiments/:experimentId" element={<ExperimentPage />} />
          <Route path="experiments/:experimentId/preprocessing" element={<PreprocessingPage />} />
          <Route path="experiments/:experimentId/training" element={<TrainingPage />} />
          <Route path="experiments/:experimentId/results" element={<ResultsPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
