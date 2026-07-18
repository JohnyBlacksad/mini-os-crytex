export type RustTaskStatus =
  | "Backlog"
  | "Pending"
  | "InProgress"
  | "Review"
  | "Completed"
  | "Failed"
  | "Cancelled";

export type TaskStatusKey =
  | "backlog"
  | "pending"
  | "in_progress"
  | "review"
  | "completed"
  | "failed"
  | "cancelled";

export type Project = {
  id: string;
  name: string;
  root_path: string;
  created_at: number;
  updated_at: number;
  metadata: unknown;
};

export type Task = {
  id: string;
  project_id: string;
  parent_id: string | null;
  title: string;
  description: string | null;
  kind: string;
  status: RustTaskStatus | TaskStatusKey;
  assigned_agent: string | null;
  priority: number;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  payload: unknown;
  result: unknown | null;
  iteration_count: number;
  priority_score: number;
  critic_score: number | null;
  human_score: number | null;
  prompt_version_id: string | null;
  lora_adapter_id: string | null;
  trace_id: string;
};

export type KanbanTaskCard = {
  id: string;
  title: string;
  kind: string;
  status: RustTaskStatus | TaskStatusKey;
  priority: number;
  assigned_agent: string | null;
};

export type KanbanColumn = {
  status: RustTaskStatus | TaskStatusKey;
  title: string;
  tasks: KanbanTaskCard[];
};

export type KanbanState = {
  project_id: string;
  columns: KanbanColumn[];
};

export type AgentLog = {
  id: string;
  project_id: string | null;
  task_id: string | null;
  agent: string;
  action: string;
  message: string | null;
  level: string;
  timestamp: number;
  metadata: unknown;
};

export type MetricsSnapshot = {
  timestamp: number;
  cpu_usage_percent: number;
  memory_used_mb: number;
  memory_total_mb: number;
  swap_used_mb: number;
  swap_total_mb: number;
  disk_used_gb: number;
  disk_total_gb: number;
  network_rx_mb: number;
  network_tx_mb: number;
  tasks_completed: number;
  tasks_failed: number;
  average_latency_ms: number;
  cache_hits: number;
  cache_misses: number;
  gpus: unknown[];
};

export type ProjectState = {
  project: Project;
  kanban: KanbanState;
  tasks: Task[];
  recent_logs: AgentLog[];
  latest_snapshot: unknown | null;
  metrics: MetricsSnapshot;
};

export type RuntimeStatus = {
  tauri_runtime: boolean;
  executor_mode: string;
  planning_mode: string;
  active_backend: string | null;
  active_model: string | null;
  ollama_url: string | null;
  real_agent_execution: boolean;
  backend_capabilities: BackendCapabilityReport[];
  cuda_toolchain: CudaToolchainStatus | null;
  compatibility_notes: RuntimeCompatibilityNote[];
};

export type RuntimeCompatibilityNote = {
  code: string;
  severity: string;
  message: string;
};

export type CudaToolchainStatus = {
  gpu_detected: boolean;
  nvcc_available: boolean;
  msvc_cl_available: boolean;
  msvc_cl_path: string | null;
  nvcc_ccbin: string | null;
  recommended_nvcc_ccbin: string | null;
  ready: boolean;
  diagnostics: string[];
};

export type BackendCapabilityReport = {
  id: string;
  name: string;
  generate: boolean;
  chat: boolean;
  embed: boolean;
  rerank: boolean;
  lora: boolean;
  hot_swap: boolean;
};

export type OllamaModelRecord = {
  id: string;
  name: string;
  active: boolean;
};

export type OllamaModelsResponse = {
  ollama_url: string;
  active_model: string | null;
  available: boolean;
  models: OllamaModelRecord[];
  error: string | null;
};

export type SetActiveOllamaModelRequest = {
  ollama_url: string;
  model: string;
};

export type ManagedModelStatus = "Available" | "Downloaded" | { Downloading: number } | { Error: string };

export type RecommendedModelConfig = {
  backend: string;
  quantization: string;
  gpu_layers: number | null;
  context_size: number;
};

export type ManagedModelRecord = {
  id: string;
  name: string;
  repo: string | null;
  filename: string | null;
  local_path: string | null;
  quantization: string | null;
  preferred_backend: string;
  params_b: number | null;
  status: ManagedModelStatus;
  recommended: RecommendedModelConfig;
};

export type ManagedModelsResponse = {
  models: ManagedModelRecord[];
};

export type DownloadManagedModelRequest = {
  model_id: string;
};

export type SetActiveManagedModelRequest = {
  model_id: string;
};

export type AddManagedModelRequest = {
  id: string;
  name: string;
  repo: string;
  filename: string;
  quantization: string | null;
  backend: string | null;
  params_b: number | null;
};

export type CreateProjectRequest = {
  name: string;
  root_path: string;
};

export type SubmitTaskRequest = {
  project_id: string;
  parent_id: string | null;
  title: string;
  description: string | null;
  kind: string;
  assigned_agent: string | null;
  priority: number;
  payload: unknown;
  trace_id: string | null;
};

export type SubmitGoalRequest = {
  project_id: string;
  goal: string;
  context: unknown;
  trace_id: string | null;
};

export type GoalPlanResponse = {
  goal: Task;
  generated_tasks: Task[];
};

export type PlanDecisionRequest = {
  goal_task_id: string;
  comment: string | null;
};

export type PlanDecisionResponse = {
  goal: Task;
  generated_tasks: Task[];
};

export type TaskReviewDecisionRequest = {
  task_id: string;
  comment: string | null;
};

export type TaskReviewDecisionResponse = {
  task: Task;
  ready_tasks: Task[];
};

export type StartRunRequest = {
  project_id: string;
  max_steps: number;
};

export type StartRunResponse = {
  run_id: string;
  project_id: string;
  started_at: number;
  review_tasks: Task[];
  remaining_ready_tasks: Task[];
};

export type ExportRunDiagnosticsRequest = {
  project_id: string;
  run_id: string;
  trace_id: string | null;
};

export type RunDiagnosticTask = {
  id: string;
  parent_id: string | null;
  title: string;
  kind: string;
  agent: string | null;
  status: string;
  trace_id: string;
  result_source: string | null;
  review_decision: string | null;
  critic_feedback: string | null;
  critic_score: number | null;
  human_score: number | null;
};

export type RunDiagnosticEvent = {
  action: string;
  task_id: string | null;
  level: string;
  timestamp: number;
  metadata: unknown;
};

export type RunDiagnosticLoraEvolution = {
  task_id: string | null;
  training_job_id: string | null;
  adapter_id: string | null;
  accepted: boolean | null;
  reason: string | null;
  baseline_run_id: string | null;
  challenger_run_id: string | null;
  winner: string | null;
  mc_nemar_p_value: number | null;
  baseline_pass_rate: number | null;
  challenger_pass_rate: number | null;
  metadata: unknown;
};

export type RunDiagnosticsReport = {
  project_id: string;
  run_id: string;
  trace_ids: string[];
  runtime: RuntimeStatus;
  tasks: RunDiagnosticTask[];
  events: RunDiagnosticEvent[];
  review_task_ids: string[];
  critic_feedback: string[];
  remediation_events: RunDiagnosticEvent[];
  lora_evolution: RunDiagnosticLoraEvolution[];
  rag_context_sent_to_model: boolean;
  human_reward_recorded: boolean;
};

export type SearchProjectContextRequest = {
  project_id: string;
  query: string;
  limit: number;
};

export type ProjectContextHit = {
  id: string;
  score: number;
  collection: string;
  source: string | null;
  relative_path: string | null;
  text: string | null;
  start_line: number | null;
  end_line: number | null;
};

export type SearchProjectContextResponse = {
  project_id: string;
  query: string;
  hits: ProjectContextHit[];
};

export type SetTaskStatusRequest = {
  task_id: string;
  status: RustTaskStatus;
};

export type CommandRecord = {
  id: string;
  name: string;
  startedAt: number;
  finishedAt: number;
  durationMs: number;
  request: unknown;
  response?: unknown;
  error?: string;
};

export type BackendEvent = {
  type: string;
  project_id: string | null;
  payload: Record<string, unknown>;
};

export const statusOrder: TaskStatusKey[] = [
  "backlog",
  "pending",
  "in_progress",
  "review",
  "completed",
  "failed",
  "cancelled",
];

export const statusLabels: Record<TaskStatusKey, string> = {
  backlog: "Backlog",
  pending: "Pending",
  in_progress: "In Progress",
  review: "Review",
  completed: "Completed",
  failed: "Failed",
  cancelled: "Cancelled",
};

export const rustStatusByKey: Record<TaskStatusKey, RustTaskStatus> = {
  backlog: "Backlog",
  pending: "Pending",
  in_progress: "InProgress",
  review: "Review",
  completed: "Completed",
  failed: "Failed",
  cancelled: "Cancelled",
};

export function normalizeStatus(status: RustTaskStatus | TaskStatusKey): TaskStatusKey {
  const match = Object.entries(rustStatusByKey).find(([, rust]) => rust === status);
  return (match?.[0] as TaskStatusKey | undefined) ?? (status as TaskStatusKey);
}

export function compactId(id?: string | null): string {
  if (!id) return "none";
  return id.length > 12 ? `${id.slice(0, 6)}...${id.slice(-4)}` : id;
}
