import { act, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./App";
import type { BackendEvent } from "./ui-types";

const apiMock = vi.hoisted(() => ({
  subscribeBackendEvents: vi.fn(),
  runtimeStatus: vi.fn(),
  listOllamaModels: vi.fn(),
  listManagedModels: vi.fn(),
  addManagedModel: vi.fn(),
  downloadManagedModel: vi.fn(),
  listProjects: vi.fn(),
  kanbanState: vi.fn(),
  listTasks: vi.fn(),
  getProjectState: vi.fn(),
  createProject: vi.fn(),
  submitGoal: vi.fn(),
  submitTask: vi.fn(),
  approvePlan: vi.fn(),
  rejectPlan: vi.fn(),
  approveTaskReview: vi.fn(),
  rejectTaskReview: vi.fn(),
  startRun: vi.fn(),
  setTaskStatus: vi.fn(),
  setActiveOllamaModel: vi.fn(),
  setActiveManagedModel: vi.fn(),
  searchProjectContext: vi.fn(),
}));

vi.mock("./api", () => ({
  createCrytexApi: vi.fn(() => apiMock),
}));

const project = {
  id: "project-1",
  name: "Project Alpha",
  root_path: "A:/tmp/project-alpha",
  created_at: 1,
  updated_at: 1,
  metadata: {},
};

const task = {
  id: "task-1",
  project_id: "project-1",
  parent_id: null,
  title: "Implement streaming proof",
  description: null,
  kind: "codegen",
  status: "InProgress" as const,
  assigned_agent: "coder",
  priority: 1,
  created_at: 1,
  started_at: null,
  finished_at: null,
  payload: {},
  result: null,
  iteration_count: 0,
  priority_score: 0,
  critic_score: null,
  human_score: null,
  prompt_version_id: null,
  lora_adapter_id: null,
  trace_id: "trace-1",
};

function projectState() {
  return {
    project,
    kanban: { project_id: project.id, columns: [] },
    tasks: [task],
    recent_logs: [],
    latest_snapshot: null,
    metrics: {
      timestamp: 1,
      cpu_usage_percent: 0,
      memory_used_mb: 0,
      memory_total_mb: 0,
      swap_used_mb: 0,
      swap_total_mb: 0,
      disk_used_gb: 0,
      disk_total_gb: 0,
      network_rx_mb: 0,
      network_tx_mb: 0,
      tasks_completed: 0,
      tasks_failed: 0,
      average_latency_ms: 0,
      cache_hits: 0,
      cache_misses: 0,
      gpus: [],
    },
  };
}

describe("App backend events", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useRealTimers();
    apiMock.runtimeStatus.mockResolvedValue({
      tauri_runtime: true,
      executor_mode: "real",
      planning_mode: "orchestrator",
      active_backend: "ollama",
      active_model: "qwen3.5:9b",
      ollama_url: "http://127.0.0.1:11434",
      real_agent_execution: true,
    });
    apiMock.listOllamaModels.mockResolvedValue({
      ollama_url: "http://127.0.0.1:11434",
      active_model: "qwen3.5:9b",
      available: true,
      models: [],
      error: null,
    });
    apiMock.listManagedModels.mockResolvedValue({ models: [] });
    apiMock.listProjects.mockResolvedValue([project]);
    apiMock.kanbanState.mockResolvedValue({ project_id: project.id, columns: [] });
    apiMock.listTasks.mockResolvedValue([task]);
    apiMock.getProjectState.mockResolvedValue(projectState());
  });

  it("should render live backend events and refresh the active project", async () => {
    let eventHandler: ((event: BackendEvent) => void) | null = null;
    apiMock.subscribeBackendEvents.mockImplementation(async (handler) => {
      eventHandler = handler;
      return vi.fn();
    });

    render(<App />);

    expect((await screen.findAllByText("Project Alpha")).length).toBeGreaterThan(0);

    act(() => {
      eventHandler?.({
        type: "RunObserved",
        project_id: "project-1",
        payload: {
          action: "task_execution_started",
          task_id: "task-1",
          trace_id: "trace-1",
        },
      });
    });

    expect(await screen.findByText("RunObserved")).toBeInTheDocument();
    expect(screen.getByText(/task_execution_started/)).toBeInTheDocument();

    await waitFor(() => expect(apiMock.getProjectState).toHaveBeenCalledTimes(2));
  });
});
