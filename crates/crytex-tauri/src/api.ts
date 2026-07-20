import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type {
  BackendEvent,
  AddManagedModelRequest,
  CommandRecord,
  CreateProjectRequest,
  DownloadManagedModelRequest,
  ExportRunDiagnosticsRequest,
  GoalPlanResponse,
  KanbanState,
  ManagedModelRecord,
  ManagedModelRuntimeProofReport,
  ManagedModelsResponse,
  OllamaModelsResponse,
  PlanDecisionRequest,
  PlanDecisionResponse,
  Project,
  ProveManagedModelRuntimeRequest,
  SearchProjectContextRequest,
  SearchProjectContextResponse,
  ProjectState,
  RuntimeStatus,
  RunDiagnosticsReport,
  SetActiveOllamaModelRequest,
  SetActiveManagedModelRequest,
  SetTaskStatusRequest,
  StartRunRequest,
  StartRunResponse,
  SubmitGoalRequest,
  SubmitTaskRequest,
  Task,
  TaskReviewDecisionRequest,
  TaskReviewDecisionResponse,
} from "./ui-types";

type CommandSink = (record: CommandRecord) => void;
type BackendEventSink = (event: BackendEvent) => void;

function isTauriRuntime(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

function errorMessage(error: unknown): string {
  if (typeof error === "string") return error;
  if (error && typeof error === "object" && "message" in error) {
    return String((error as { message: unknown }).message);
  }
  return "Unknown command error";
}

async function callCommand<T>(
  name: string,
  payload: Record<string, unknown>,
  sink: CommandSink,
): Promise<T> {
  const startedAt = Date.now();
  try {
    if (!isTauriRuntime()) {
      throw new Error("Tauri runtime unavailable. Start the desktop app to execute backend commands.");
    }
    const response = await invoke<T>(name, payload);
    sink({
      id: `${startedAt}-${name}`,
      name,
      startedAt,
      finishedAt: Date.now(),
      durationMs: Date.now() - startedAt,
      request: payload,
      response,
    });
    return response;
  } catch (error) {
    const message = errorMessage(error);
    sink({
      id: `${startedAt}-${name}`,
      name,
      startedAt,
      finishedAt: Date.now(),
      durationMs: Date.now() - startedAt,
      request: payload,
      error: message,
    });
    throw new Error(message);
  }
}

export function createCrytexApi(sink: CommandSink) {
  return {
    subscribeBackendEvents: async (handler: BackendEventSink) => {
      if (!isTauriRuntime()) return () => {};
      return listen<BackendEvent>("crytex://event", (event) => handler(event.payload));
    },
    runtimeStatus: () => callCommand<RuntimeStatus>("runtime_status", {}, sink),
    listOllamaModels: () => callCommand<OllamaModelsResponse>("list_ollama_models", {}, sink),
    listManagedModels: () => callCommand<ManagedModelsResponse>("list_managed_models", {}, sink),
    addManagedModel: (request: AddManagedModelRequest) =>
      callCommand<ManagedModelRecord>("add_managed_model", { request }, sink),
    downloadManagedModel: (request: DownloadManagedModelRequest) =>
      callCommand<ManagedModelRecord>("download_managed_model", { request }, sink),
    setActiveOllamaModel: (request: SetActiveOllamaModelRequest) =>
      callCommand<RuntimeStatus>("set_active_ollama_model", { request }, sink),
    setActiveManagedModel: (request: SetActiveManagedModelRequest) =>
      callCommand<RuntimeStatus>("set_active_managed_model", { request }, sink),
    proveManagedModelRuntime: (request: ProveManagedModelRuntimeRequest) =>
      callCommand<ManagedModelRuntimeProofReport>("prove_managed_model_runtime", { request }, sink),
    listProjects: () => callCommand<Project[]>("list_projects", {}, sink),
    createProject: (request: CreateProjectRequest) =>
      callCommand<Project>("create_project", { request }, sink),
    kanbanState: (projectId: string) =>
      callCommand<KanbanState>("kanban_state", { projectId }, sink),
    listTasks: (projectId: string) =>
      callCommand<Task[]>("list_tasks", { projectId }, sink),
    submitTask: (request: SubmitTaskRequest) =>
      callCommand<Task>("submit_task", { request }, sink),
    submitGoal: (request: SubmitGoalRequest) =>
      callCommand<GoalPlanResponse>("submit_goal", { request }, sink),
    approvePlan: (request: PlanDecisionRequest) =>
      callCommand<PlanDecisionResponse>("approve_plan", { request }, sink),
    rejectPlan: (request: PlanDecisionRequest) =>
      callCommand<PlanDecisionResponse>("reject_plan", { request }, sink),
    approveTaskReview: (request: TaskReviewDecisionRequest) =>
      callCommand<TaskReviewDecisionResponse>("approve_task_review", { request }, sink),
    rejectTaskReview: (request: TaskReviewDecisionRequest) =>
      callCommand<TaskReviewDecisionResponse>("reject_task_review", { request }, sink),
    startRun: (request: StartRunRequest) =>
      callCommand<StartRunResponse>("start_run", { request }, sink),
    exportRunDiagnostics: (request: ExportRunDiagnosticsRequest) =>
      callCommand<RunDiagnosticsReport>("export_run_diagnostics", { request }, sink),
    setTaskStatus: (request: SetTaskStatusRequest) =>
      callCommand<Task>("set_task_status", { request }, sink),
    getProjectState: (projectId: string) =>
      callCommand<ProjectState>("get_project_state", { projectId }, sink),
    searchProjectContext: (request: SearchProjectContextRequest) =>
      callCommand<SearchProjectContextResponse>("search_project_context", { request }, sink),
  };
}
