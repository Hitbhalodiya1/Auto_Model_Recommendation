/**
 * HomePage — landing page with platform overview and quick-start CTA.
 */

import { Link } from "react-router-dom";
import { ArrowRight, BarChart3, Brain, Database, FlaskConical } from "lucide-react";

const FEATURES = [
  {
    icon: Database,
    title: "Smart Dataset Analysis",
    description: "Automatically profile your data: detect types, missing values, outliers, and class imbalance.",
  },
  {
    icon: FlaskConical,
    title: "Intelligent Preprocessing",
    description: "Get step-by-step preprocessing recommendations with plain-language explanations for every decision.",
  },
  {
    icon: Brain,
    title: "Parallel Model Training",
    description: "Train all compatible algorithms simultaneously. Compare configurations, hyperparameters, and performance.",
  },
  {
    icon: BarChart3,
    title: "AI-Powered Recommendation",
    description: "Receive a ranked recommendation with composite scoring across performance, speed, and interpretability.",
  },
];

export function HomePage() {
  return (
    <div className="max-w-5xl mx-auto">
      {/* Hero */}
      <div className="text-center py-16">
        <div className="inline-flex items-center gap-2 px-3 py-1 bg-primary-50 text-primary-700 rounded-full text-sm font-medium mb-6">
          <Brain size={14} />
          AutoML · No Code Required
        </div>
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Intelligent ML Model Recommendation
        </h1>
        <p className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
          Upload a dataset and AutoRec automatically analyzes, preprocesses, trains,
          and recommends the best machine learning model — with full explanations.
        </p>
        <div className="flex items-center justify-center gap-4">
          <Link to="/datasets" className="btn-primary flex items-center gap-2">
            Get Started <ArrowRight size={16} />
          </Link>
          <Link to="/experiments" className="btn-secondary">
            View Experiments
          </Link>
        </div>
      </div>

      {/* Features */}
      <div className="grid grid-cols-2 gap-6 pb-16">
        {FEATURES.map(({ icon: Icon, title, description }) => (
          <div key={title} className="card p-6">
            <div className="w-10 h-10 bg-primary-50 rounded-lg flex items-center justify-center mb-4">
              <Icon size={20} className="text-primary-600" />
            </div>
            <h3 className="font-semibold text-gray-900 mb-2">{title}</h3>
            <p className="text-sm text-gray-600">{description}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
