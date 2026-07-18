/**
 * Main application layout with sidebar and top navigation.
 */

import { Link, Outlet, useLocation } from "react-router-dom";
import { clsx } from "clsx";
import {
  BarChart3,
  Brain,
  Database,
  FlaskConical,
  History,
  Home,
  Menu,
  X,
} from "lucide-react";
import { useUIStore } from "@/store/ui-store";

const NAV_ITEMS = [
  { path: "/", label: "Home", icon: Home, exact: true },
  { path: "/datasets", label: "Datasets", icon: Database },
  { path: "/experiments", label: "Experiments", icon: History },
];

export function AppLayout() {
  const location = useLocation();
  const { sidebarOpen, toggleSidebar } = useUIStore();

  const isActive = (path: string, exact?: boolean) => {
    if (exact) return location.pathname === path;
    return location.pathname.startsWith(path);
  };

  return (
    <div className="flex h-screen bg-gray-50 overflow-hidden">
      {/* Sidebar */}
      <aside
        className={clsx(
          "flex flex-col bg-gray-900 text-white transition-all duration-200 shrink-0",
          sidebarOpen ? "w-60" : "w-16"
        )}
      >
        {/* Logo */}
        <div className="flex items-center gap-3 px-4 h-16 border-b border-gray-700">
          <Brain className="text-primary-400 shrink-0" size={24} />
          {sidebarOpen && (
            <span className="font-semibold text-lg tracking-tight">AutoRec</span>
          )}
        </div>

        {/* Nav items */}
        <nav className="flex-1 py-4 space-y-1 px-2">
          {NAV_ITEMS.map(({ path, label, icon: Icon, exact }) => (
            <Link
              key={path}
              to={path}
              className={clsx(
                "flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors",
                isActive(path, exact)
                  ? "bg-primary-600 text-white"
                  : "text-gray-400 hover:bg-gray-800 hover:text-white"
              )}
            >
              <Icon size={18} className="shrink-0" />
              {sidebarOpen && <span>{label}</span>}
            </Link>
          ))}
        </nav>

        {/* Sidebar toggle */}
        <div className="p-3 border-t border-gray-700">
          <button
            onClick={toggleSidebar}
            className="flex items-center justify-center w-full p-2 rounded-lg text-gray-400 hover:bg-gray-800 hover:text-white transition-colors"
            aria-label={sidebarOpen ? "Collapse sidebar" : "Expand sidebar"}
          >
            {sidebarOpen ? <X size={18} /> : <Menu size={18} />}
          </button>
        </div>
      </aside>

      {/* Main content */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Top bar */}
        <header className="h-16 bg-white border-b border-gray-200 flex items-center px-6">
          <div className="flex items-center gap-2 text-sm text-gray-500">
            <BarChart3 size={16} className="text-primary-500" />
            <span className="font-medium text-gray-900">AutoRec</span>
            <span>/</span>
            <span className="capitalize">
              {location.pathname.split("/").filter(Boolean)[0] ?? "home"}
            </span>
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1 overflow-y-auto p-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
