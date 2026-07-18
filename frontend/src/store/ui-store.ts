/**
 * Zustand global store for AutoRec UI state.
 * Server data (datasets, experiments, etc.) lives in TanStack Query cache.
 * This store holds UI state: active experiment, notifications, etc.
 */

import { create } from "zustand";

interface Notification {
  id: string;
  type: "success" | "error" | "info" | "warning";
  title: string;
  message?: string;
}

interface UIStore {
  // Active experiment being worked on
  activeExperimentId: string | null;
  setActiveExperimentId: (id: string | null) => void;

  // Active dataset
  activeDatasetId: string | null;
  setActiveDatasetId: (id: string | null) => void;

  // Notifications
  notifications: Notification[];
  addNotification: (n: Omit<Notification, "id">) => void;
  removeNotification: (id: string) => void;

  // Sidebar state
  sidebarOpen: boolean;
  toggleSidebar: () => void;
}

export const useUIStore = create<UIStore>((set) => ({
  activeExperimentId: null,
  setActiveExperimentId: (id) => set({ activeExperimentId: id }),

  activeDatasetId: null,
  setActiveDatasetId: (id) => set({ activeDatasetId: id }),

  notifications: [],
  addNotification: (n) =>
    set((state) => ({
      notifications: [
        ...state.notifications,
        { ...n, id: crypto.randomUUID() },
      ],
    })),
  removeNotification: (id) =>
    set((state) => ({
      notifications: state.notifications.filter((n) => n.id !== id),
    })),

  sidebarOpen: true,
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
}));
