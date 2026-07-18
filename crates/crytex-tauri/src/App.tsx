import {
  Activity,
  AlertTriangle,
  Bot,
  Check,
  CircleDot,
  Database,
  Eye,
  FolderPlus,
  GitBranch,
  GitPullRequest,
  LayoutDashboard,
  ListChecks,
  Loader2,
  Network,
  Play,
  Plus,
  RefreshCcw,
  Search,
  Settings,
  TerminalSquare,
  X,
} from "lucide-react";
import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { createCrytexApi } from "./api";
import type { AddManagedModelRequest, BackendEvent, CommandRecord, CreateProjectRequest, KanbanColumn, ManagedModelsResponse, OllamaModelsResponse, Project, ProjectState, RuntimeStatus, SearchProjectContextResponse, StartRunResponse, SubmitTaskRequest, Task, TaskStatusKey } from "./ui-types";
import { compactId, normalizeStatus, rustStatusByKey, statusLabels, statusOrder } from "./ui-types";

type Section = "workspace" | "goals" | "tasks" | "observe" | "runs" | "rag" | "models" | "evolution" | "settings";
type InspectorTab = "overview" | "payload" | "result" | "scores" | "timing" | "debug";
type ObserveTab = "events" | "logs" | "traces" | "metrics" | "errors" | "raw";

const navItems = [
  { id: "workspace", label: "Workspace", icon: LayoutDashboard, enabled: true },
  { id: "goals", label: "Goals", icon: GitPullRequest, enabled: true },
  { id: "tasks", label: "Generated Tasks", icon: ListChecks, enabled: true },
  { id: "observe", label: "Observe", icon: Activity, enabled: true },
  { id: "runs", label: "Runs", icon: Bot, enabled: true },
  { id: "rag", label: "Index/RAG", icon: Database, enabled: true },
  { id: "models", label: "Models", icon: Network, enabled: true },
  { id: "evolution", label: "Evolution", icon: GitBranch, enabled: true },
  { id: "settings", label: "Settings", icon: Settings, enabled: false },
] as const;

const taskKinds = ["codegen", "research", "summarization", "qa", "security", "review", "generic", "sandbox"];
const agentKinds = ["", "architect", "coder", "researcher", "qa", "security", "critic", "summarizer"];

function nowLabel(date?: number): string {
  if (!date) return "never";
  return new Intl.DateTimeFormat("en", { hour: "2-digit", minute: "2-digit", second: "2-digit" }).format(date);
}

function jsonBlock(value: unknown): string {
  return JSON.stringify(value ?? null, null, 2);
}

function stringOrNull(value: FormDataEntryValue | null): string | null {
  const text = String(value ?? "").trim();
  return text.length > 0 ? text : null;
}

function isUserGoal(task: Task): boolean {
  return task.parent_id === null && task.assigned_agent === "architect" && (task.payload as { source?: unknown })?.source === "user_goal";
}

function taskCounts(tasks: Task[]): Record<TaskStatusKey, number> {
  return statusOrder.reduce(
    (acc, status) => ({ ...acc, [status]: tasks.filter((task) => normalizeStatus(task.status) === status).length }),
    {} as Record<TaskStatusKey, number>,
  );
}

function sortCommands(records: CommandRecord[]): CommandRecord[] {
  return [...records].sort((a, b) => b.startedAt - a.startedAt);
}

function runtimeBadge(status: RuntimeStatus | null): string {
  if (!status) return "runtime unknown";
  const model = status.active_model ? ` / ${status.active_model}` : "";
  return `${status.executor_mode}${model}`;
}

function backendEventProjectId(event: BackendEvent, taskProjectIndex: Map<string, string>): string | null {
  if (event.project_id) return event.project_id;
  const projectId = event.payload.project_id;
  if (typeof projectId === "string") return projectId;
  const taskId = event.payload.task_id;
  return typeof taskId === "string" ? taskProjectIndex.get(taskId) ?? null : null;
}

export function App() {
  const [section, setSection] = useState<Section>("workspace");
  const [projects, setProjects] = useState<Project[]>([]);
  const [activeProjectId, setActiveProjectId] = useState<string | null>(null);
  const [projectSearch, setProjectSearch] = useState("");
  const [tasks, setTasks] = useState<Task[]>([]);
  const [kanbanColumns, setKanbanColumns] = useState<KanbanColumn[]>([]);
  const [projectState, setProjectState] = useState<ProjectState | null>(null);
  const [runtimeStatus, setRuntimeStatus] = useState<RuntimeStatus | null>(null);
  const [ollamaModels, setOllamaModels] = useState<OllamaModelsResponse | null>(null);
  const [managedModels, setManagedModels] = useState<ManagedModelsResponse | null>(null);
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
  const [commandRecords, setCommandRecords] = useState<CommandRecord[]>([]);
  const [observeOpen, setObserveOpen] = useState(true);
  const [observeTab, setObserveTab] = useState<ObserveTab>("events");
  const [inspectorTab, setInspectorTab] = useState<InspectorTab>("overview");
  const [busy, setBusy] = useState(false);
  const [projectFormOpen, setProjectFormOpen] = useState(false);
  const [goalFormOpen, setGoalFormOpen] = useState(false);
  const [taskFormOpen, setTaskFormOpen] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState<number | null>(null);
  const [lastRun, setLastRun] = useState<StartRunResponse | null>(null);
  const [ragResult, setRagResult] = useState<SearchProjectContextResponse | null>(null);
  const [backendEvents, setBackendEvents] = useState<BackendEvent[]>([]);
  const [liveStatus, setLiveStatus] = useState("idle");
  const activeProjectIdRef = useRef<string | null>(null);
  const taskProjectIndexRef = useRef<Map<string, string>>(new Map());
  const refreshTimerRef = useRef<number | null>(null);

  const api = useMemo(
    () =>
      createCrytexApi((record) => {
        setCommandRecords((current) => [record, ...current].slice(0, 120));
        if (record.error) setObserveTab("errors");
      }),
    [],
  );

  const activeProject = projects.find((project) => project.id === activeProjectId) ?? null;
  const selectedTask = tasks.find((task) => task.id === selectedTaskId) ?? null;
  const goals = tasks.filter(isUserGoal);
  const selectedGoal =
    (selectedTask && isUserGoal(selectedTask) ? selectedTask : null) ??
    tasks.find((task) => task.id === selectedTask?.parent_id && isUserGoal(task)) ??
    goals.find((task) => normalizeStatus(task.status) === "review") ??
    goals[0] ??
    null;
  const selectedPlanTasks = selectedGoal ? tasks.filter((task) => task.parent_id === selectedGoal.id) : [];
  const counts = taskCounts(tasks);

  async function refresh(projectId = activeProjectId) {
    setBusy(true);
    try {
      const [nextRuntimeStatus, nextModels, nextManagedModels, nextProjects] = await Promise.all([
        api.runtimeStatus(),
        api.listOllamaModels(),
        api.listManagedModels(),
        api.listProjects(),
      ]);
      setRuntimeStatus(nextRuntimeStatus);
      setOllamaModels(nextModels);
      setManagedModels(nextManagedModels);
      setProjects(nextProjects);
      const nextProjectId = projectId ?? nextProjects[0]?.id ?? null;
      setActiveProjectId(nextProjectId);
      if (!nextProjectId) {
        setTasks([]);
        setKanbanColumns([]);
        setProjectState(null);
        return;
      }
      const [kanban, nextTasks, nextState] = await Promise.all([
        api.kanbanState(nextProjectId),
        api.listTasks(nextProjectId),
        api.getProjectState(nextProjectId),
      ]);
      setKanbanColumns(kanban.columns);
      setTasks(nextTasks);
      setProjectState(nextState);
      setSelectedTaskId((current) => (nextTasks.some((task) => task.id === current) ? current : null));
      setLastRefresh(Date.now());
    } catch (error) {
      setFormError(error instanceof Error ? error.message : "Refresh failed");
    } finally {
      setBusy(false);
    }
  }

  async function downloadManagedModel(modelId: string) {
    try {
      await api.downloadManagedModel({ model_id: modelId });
      const models = await api.listManagedModels();
      setManagedModels(models);
      setObserveOpen(true);
      setObserveTab("events");
    } catch (error) {
      setFormError(error instanceof Error ? error.message : "Managed model download failed");
      setObserveOpen(true);
      setObserveTab("errors");
    }
  }

  async function addManagedModel(request: AddManagedModelRequest) {
    try {
      await api.addManagedModel(request);
      const models = await api.listManagedModels();
      setManagedModels(models);
    } catch (error) {
      setFormError(error instanceof Error ? error.message : "Managed model creation failed");
      setObserveOpen(true);
      setObserveTab("errors");
    }
  }

  async function useManagedModel(modelId: string) {
    try {
      const status = await api.setActiveManagedModel({ model_id: modelId });
      setRuntimeStatus(status);
      setObserveOpen(true);
      setObserveTab("events");
    } catch (error) {
      setFormError(error instanceof Error ? error.message : "Managed model selection failed");
      setObserveOpen(true);
      setObserveTab("errors");
    }
  }

  useEffect(() => {
    void refresh(null);
  }, []);

  useEffect(() => {
    activeProjectIdRef.current = activeProjectId;
  }, [activeProjectId]);

  useEffect(() => {
    taskProjectIndexRef.current = new Map(tasks.map((task) => [task.id, task.project_id]));
  }, [tasks]);

  useEffect(() => {
    let disposed = false;
    void api.subscribeBackendEvents((event) => {
      if (disposed) return;
      setBackendEvents((current) => [event, ...current].slice(0, 120));
      setLiveStatus(`${event.type} / ${nowLabel(Date.now())}`);
      const activeProjectId = activeProjectIdRef.current;
      const eventProjectId = backendEventProjectId(event, taskProjectIndexRef.current);
      if (!activeProjectId || (eventProjectId && eventProjectId !== activeProjectId)) return;
      if (refreshTimerRef.current) window.clearTimeout(refreshTimerRef.current);
      refreshTimerRef.current = window.setTimeout(() => {
        void refresh(activeProjectIdRef.current);
      }, 250);
    }).then((unlisten) => {
      if (disposed) unlisten();
    });
    return () => {
      disposed = true;
      if (refreshTimerRef.current) window.clearTimeout(refreshTimerRef.current);
    };
  }, [api]);

  function openProjectForm() {
    setFormError(null);
    setProjectFormOpen(true);
  }

  function openGoalForm() {
    setFormError(null);
    setGoalFormOpen(true);
  }

  function openTaskForm() {
    setFormError(null);
    setTaskFormOpen(true);
  }

  async function createProject(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setFormError(null);
    const form = new FormData(event.currentTarget);
    const request: CreateProjectRequest = {
      name: String(form.get("name") ?? "").trim(),
      root_path: String(form.get("rootPath") ?? "").trim(),
    };
    if (!request.name || !request.root_path) {
      setFormError("Project name and root path are required.");
      return;
    }
    try {
      const project = await api.createProject(request);
      setProjectFormOpen(false);
      await refresh(project.id);
    } catch (error) {
      setFormError(error instanceof Error ? error.message : "Project creation failed");
    }
  }

  async function submitGoal(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!activeProjectId) return;
    setFormError(null);
    const form = new FormData(event.currentTarget);
    const goal = String(form.get("goal") ?? "").trim();
    if (!goal) {
      setFormError("Goal is required.");
      return;
    }
    try {
      const response = await api.submitGoal({
        project_id: activeProjectId,
        goal,
        context: { source: "ui_agent_console", project: activeProject?.name ?? null, selected_task_id: selectedTaskId, section },
        trace_id: stringOrNull(form.get("traceId")),
      });
      setGoalFormOpen(false);
      setSelectedTaskId(response.goal.id);
      setSection("goals");
      await refresh(activeProjectId);
      event.currentTarget.reset();
    } catch (error) {
      setFormError(error instanceof Error ? error.message : "Goal submission failed");
      setObserveOpen(true);
      setObserveTab("errors");
    }
  }

  async function submitTask(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!activeProjectId) return;
    setFormError(null);
    const form = new FormData(event.currentTarget);
    let payload: unknown;
    try {
      payload = JSON.parse(String(form.get("payload") || "{}"));
    } catch {
      setFormError("Payload must be valid JSON.");
      return;
    }
    const request: SubmitTaskRequest = {
      project_id: activeProjectId,
      parent_id: stringOrNull(form.get("parentId")),
      title: String(form.get("title") ?? "").trim(),
      description: stringOrNull(form.get("description")),
      kind: String(form.get("kind") ?? "generic"),
      assigned_agent: stringOrNull(form.get("agent")),
      priority: Number(form.get("priority") ?? 0),
      payload,
      trace_id: stringOrNull(form.get("traceId")),
    };
    if (!request.title || !request.kind || !Number.isInteger(request.priority)) {
      setFormError("Title, kind, and integer priority are required.");
      return;
    }
    try {
      const task = await api.submitTask(request);
      setTaskFormOpen(false);
      setSelectedTaskId(task.id);
      await refresh(activeProjectId);
    } catch (error) {
      setFormError(error instanceof Error ? error.message : "Task submission failed");
    }
  }

  async function approveSelectedPlan() {
    if (!selectedGoal) return;
    try {
      const response = await api.approvePlan({ goal_task_id: selectedGoal.id, comment: "Approved from Crytex UI" });
      setSelectedTaskId(response.goal.id);
      await refresh(response.goal.project_id);
    } catch (error) {
      setFormError(error instanceof Error ? error.message : "Plan approval failed");
      setObserveOpen(true);
      setObserveTab("errors");
    }
  }

  async function rejectSelectedPlan() {
    if (!selectedGoal) return;
    const comment = window.prompt("Why is this plan rejected?", "Needs revision before execution");
    if (comment === null) return;
    try {
      const response = await api.rejectPlan({ goal_task_id: selectedGoal.id, comment });
      setSelectedTaskId(response.goal.id);
      await refresh(response.goal.project_id);
    } catch (error) {
      setFormError(error instanceof Error ? error.message : "Plan rejection failed");
      setObserveOpen(true);
      setObserveTab("errors");
    }
  }

  async function approveSelectedTaskReview() {
    if (!selectedTask || normalizeStatus(selectedTask.status) !== "review" || isUserGoal(selectedTask)) return;
    try {
      const response = await api.approveTaskReview({
        task_id: selectedTask.id,
        comment: "Approved from Crytex UI",
      });
      setSelectedTaskId(response.ready_tasks[0]?.id ?? response.task.id);
      await refresh(response.task.project_id);
    } catch (error) {
      setFormError(error instanceof Error ? error.message : "Task review approval failed");
      setObserveOpen(true);
      setObserveTab("errors");
    }
  }

  async function rejectSelectedTaskReview() {
    if (!selectedTask || normalizeStatus(selectedTask.status) !== "review" || isUserGoal(selectedTask)) return;
    const comment = window.prompt("Why is this result rejected?", "Needs another iteration");
    if (comment === null) return;
    try {
      const response = await api.rejectTaskReview({
        task_id: selectedTask.id,
        comment,
      });
      setSelectedTaskId(response.task.id);
      await refresh(response.task.project_id);
    } catch (error) {
      setFormError(error instanceof Error ? error.message : "Task review rejection failed");
      setObserveOpen(true);
      setObserveTab("errors");
    }
  }

  async function startProjectRun() {
    if (!activeProjectId) return;
    try {
      const response = await api.startRun({ project_id: activeProjectId, max_steps: 20 });
      setLastRun(response);
      setSection("runs");
      setObserveOpen(true);
      setObserveTab("events");
      await refresh(activeProjectId);
    } catch (error) {
      setFormError(error instanceof Error ? error.message : "Run start failed");
      setObserveOpen(true);
      setObserveTab("errors");
    }
  }

  async function selectOllamaModel(ollamaUrl: string, model: string) {
    try {
      const status = await api.setActiveOllamaModel({ ollama_url: ollamaUrl, model });
      setRuntimeStatus(status);
      const models = await api.listOllamaModels();
      setOllamaModels(models);
    } catch (error) {
      setFormError(error instanceof Error ? error.message : "Model selection failed");
      setObserveOpen(true);
      setObserveTab("errors");
    }
  }

  async function searchRag(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!activeProjectId) return;
    setFormError(null);
    const form = new FormData(event.currentTarget);
    const query = String(form.get("query") ?? "").trim();
    const limit = Number(form.get("limit") ?? 8);
    if (!query) {
      setFormError("RAG query is required.");
      return;
    }
    try {
      const result = await api.searchProjectContext({
        project_id: activeProjectId,
        query,
        limit: Number.isFinite(limit) ? limit : 8,
      });
      setRagResult(result);
    } catch (error) {
      setFormError(error instanceof Error ? error.message : "RAG search failed");
      setObserveOpen(true);
      setObserveTab("errors");
    }
  }

  async function changeStatus(status: TaskStatusKey) {
    if (!selectedTask) return;
    try {
      const task = await api.setTaskStatus({ task_id: selectedTask.id, status: rustStatusByKey[status] });
      setSelectedTaskId(task.id);
      await refresh(task.project_id);
    } catch (error) {
      setFormError(error instanceof Error ? error.message : "Status change failed");
      setObserveOpen(true);
      setObserveTab("errors");
    }
  }

  const filteredProjects = projects.filter((project) =>
    `${project.name} ${project.root_path}`.toLowerCase().includes(projectSearch.toLowerCase()),
  );

  return (
    <div className="app-shell">
      <nav className="nav-rail">
        <div className="mark">CX</div>
        {navItems.map((item) => (
          <button key={item.id} className={`rail-button ${section === item.id ? "active" : ""}`} disabled={!item.enabled} title={item.enabled ? item.label : `${item.label} not wired`} onClick={() => item.enabled && setSection(item.id as Section)} aria-label={item.label}>
            <item.icon size={19} />
          </button>
        ))}
      </nav>

      <aside className="project-sidebar">
        <div className="sidebar-title">
          <span>Projects</span>
          <button className="icon-button" title="Create project" onClick={openProjectForm}><FolderPlus size={16} /></button>
        </div>
        <label className="search-box">
          <Search size={15} />
          <input value={projectSearch} onChange={(event) => setProjectSearch(event.target.value)} placeholder="Search" />
        </label>
        <div className="project-list">
          {filteredProjects.map((project) => (
            <button key={project.id} className={`project-item ${project.id === activeProjectId ? "active" : ""}`} onClick={() => void refresh(project.id)}>
              <span className="project-row"><CircleDot size={13} className={project.id === activeProjectId ? "ok" : "muted"} /><strong>{project.name}</strong></span>
              <span className="path-text">{project.root_path}</span>
              <span className="project-meta">{project.id === activeProjectId ? `${tasks.length} tasks` : compactId(project.id)}</span>
            </button>
          ))}
        </div>
      </aside>

      <main className="workbench">
        <header className="topbar">
          <div className="project-heading"><strong>{activeProject?.name ?? "No project"}</strong><span>{activeProject?.root_path ?? "Create a project to begin manual testing"}</span></div>
          <div className="status-strip"><span className={`badge runtime-${runtimeStatus?.real_agent_execution ? "real" : "stub"}`}>{runtimeBadge(runtimeStatus)}</span><span className="badge">{runtimeStatus?.active_backend ?? "backend none"}</span><span className="badge live-badge">{liveStatus}</span><span className="badge">qdrant edge</span><span className="badge">refresh {nowLabel(lastRefresh ?? undefined)}</span></div>
          <div className="topbar-actions">
            <button className="secondary-button" onClick={() => void refresh()} disabled={busy} title="Refresh">{busy ? <Loader2 className="spin" size={15} /> : <RefreshCcw size={15} />}Refresh</button>
            <button className="primary-button" onClick={openGoalForm} disabled={!activeProject}><Plus size={15} />New goal</button>
            <button className="secondary-button" onClick={() => void startProjectRun()} disabled={!activeProject}><Play size={15} />Start run</button>
            <button className="secondary-button" onClick={openTaskForm} disabled={!activeProject} title="Debug/admin task creation"><Plus size={15} />Debug task</button>
            <button className="secondary-button" onClick={() => setObserveOpen((open) => !open)}><TerminalSquare size={15} />Observe</button>
          </div>
        </header>

        <section className={`workspace ${observeOpen ? "with-observe" : ""}`}>
          <div className="content-pane">
            {section === "observe" ? <ObservePanel tab={observeTab} setTab={setObserveTab} records={commandRecords} backendEvents={backendEvents} state={projectState} selectedTask={selectedTask} /> :
            projects.length === 0 ? <EmptyProjects onOpen={openProjectForm} /> :
            section === "workspace" ? <WorkspacePanel tasks={tasks} logs={projectState?.recent_logs ?? []} selectedTaskId={selectedTaskId} onSelect={setSelectedTaskId} onSubmitGoal={submitGoal} onNewGoal={openGoalForm} onStartRun={() => void startProjectRun()} onApproveTaskReview={() => void approveSelectedTaskReview()} onRejectTaskReview={() => void rejectSelectedTaskReview()} lastRun={lastRun} /> :
            section === "goals" ? <GoalsPanel goals={goals} selectedGoal={selectedGoal} planTasks={selectedPlanTasks} selectedTaskId={selectedTaskId} onSelect={setSelectedTaskId} onNewGoal={openGoalForm} onApprove={() => void approveSelectedPlan()} onReject={() => void rejectSelectedPlan()} /> :
            section === "runs" ? <RunsPanel lastRun={lastRun} tasks={tasks} logs={projectState?.recent_logs ?? []} selectedTaskId={selectedTaskId} onSelect={setSelectedTaskId} onStartRun={() => void startProjectRun()} /> :
            section === "rag" ? <RagPanel project={activeProject} result={ragResult} onSearch={searchRag} /> :
            section === "models" ? <ModelsPanel status={runtimeStatus} inventory={ollamaModels} managed={managedModels} onRefresh={() => void refresh()} onSelect={(url, model) => void selectOllamaModel(url, model)} onDownloadManaged={(modelId) => void downloadManagedModel(modelId)} onUseManaged={(modelId) => void useManagedModel(modelId)} onAddManaged={(request) => void addManagedModel(request)} /> :
            section === "evolution" ? <PlaceholderPanel title="Evolution" items={["Prompt Evolution crate is tested", "LoRA Evolution crate is tested", "Experience dataset UI/IPC pending", "Promote/rollback flows pending"]} /> :
            <GeneratedTasksBoard columns={kanbanColumns} tasks={tasks} selectedTaskId={selectedTaskId} onSelect={setSelectedTaskId} />}
          </div>
          <Inspector project={activeProject} state={projectState} task={selectedTask} tab={inspectorTab} setTab={setInspectorTab} onStatus={changeStatus} counts={counts} selectedGoal={selectedGoal} planTasks={selectedPlanTasks} onApprovePlan={approveSelectedPlan} onRejectPlan={rejectSelectedPlan} onApproveTaskReview={approveSelectedTaskReview} onRejectTaskReview={rejectSelectedTaskReview} />
        </section>

        {observeOpen && <div className="observe-drawer"><ObservePanel tab={observeTab} setTab={setObserveTab} records={commandRecords} backendEvents={backendEvents} state={projectState} selectedTask={selectedTask} compact /></div>}
      </main>

      {projectFormOpen && <Modal title="Create project" onClose={() => setProjectFormOpen(false)} error={formError}><form className="form-grid" onSubmit={createProject}><label>Name<input name="name" autoFocus placeholder="Crytex workspace" /></label><label>Root path<input name="rootPath" placeholder="A:\\Projects\\mini-os-crytex" /></label><div className="modal-actions"><button className="secondary-button" type="button" onClick={() => setProjectFormOpen(false)}>Cancel</button><button className="primary-button" type="submit"><Check size={15} />Create</button></div></form></Modal>}
      {goalFormOpen && activeProject && <Modal title="New goal" onClose={() => setGoalFormOpen(false)} error={formError}><form className="form-grid" onSubmit={submitGoal}><label>Goal<textarea name="goal" autoFocus rows={5} placeholder="Describe what the AI teamlead should achieve. The architect will create the task graph." /></label><label>Trace id<input name="traceId" placeholder="optional trace id" /></label><div className="modal-actions"><button className="secondary-button" type="button" onClick={() => setGoalFormOpen(false)}>Cancel</button><button className="primary-button" type="submit"><GitPullRequest size={15} />Submit goal</button></div></form></Modal>}
      {taskFormOpen && activeProject && <Modal title="Create debug task" onClose={() => setTaskFormOpen(false)} error={formError}><form className="form-grid two" onSubmit={submitTask}><label className="wide">Title<input name="title" autoFocus placeholder="Debug-only manual task" /></label><label className="wide">Description<textarea name="description" rows={3} /></label><label>Kind<select name="kind">{taskKinds.map((kind) => <option key={kind}>{kind}</option>)}</select></label><label>Agent<select name="agent">{agentKinds.map((agent) => <option key={agent} value={agent}>{agent || "unassigned"}</option>)}</select></label><label>Priority<input name="priority" type="number" defaultValue={1} /></label><label>Parent task<input name="parentId" placeholder="optional task id" /></label><label className="wide">Trace id<input name="traceId" placeholder="optional trace id" /></label><label className="wide">Payload JSON<textarea name="payload" rows={7} defaultValue={"{\n  \"source\": \"debug_admin\"\n}"} /></label><div className="modal-actions wide"><button className="secondary-button" type="button" onClick={() => setTaskFormOpen(false)}>Cancel</button><button className="primary-button" type="submit"><Plus size={15} />Submit</button></div></form></Modal>}
    </div>
  );
}

function EmptyProjects({ onOpen }: { onOpen: () => void }) {
  return <div className="empty-state"><Database size={24} /><strong>No projects</strong><button className="primary-button" onClick={onOpen}><FolderPlus size={15} />Create project</button></div>;
}

function WorkspacePanel({ tasks, logs, selectedTaskId, onSelect, onSubmitGoal, onNewGoal, onStartRun, onApproveTaskReview, onRejectTaskReview, lastRun }: { tasks: Task[]; logs: ProjectState["recent_logs"]; selectedTaskId: string | null; onSelect: (id: string) => void; onSubmitGoal: (event: FormEvent<HTMLFormElement>) => void; onNewGoal: () => void; onStartRun: () => void; onApproveTaskReview: () => void; onRejectTaskReview: () => void; lastRun: StartRunResponse | null }) {
  const goals = tasks.filter(isUserGoal);
  const generated = tasks.filter((task) => task.parent_id !== null);
  const reviewGoals = goals.filter((task) => normalizeStatus(task.status) === "review");
  const selected = tasks.find((task) => task.id === selectedTaskId) ?? null;
  const activeGoal =
    (selected?.parent_id ? goals.find((goal) => goal.id === selected.parent_id) : null) ??
    (selected && isUserGoal(selected) ? selected : null) ??
    goals.find((goal) => normalizeStatus(goal.status) === "review") ??
    goals[0] ??
    null;
  const activeChain = activeGoal ? generated.filter((task) => task.parent_id === activeGoal.id) : generated;
  const reviewTasks = tasks.filter((task) => normalizeStatus(task.status) === "review" && !isUserGoal(task));
  const latestReview = reviewTasks[0] ?? lastRun?.review_tasks[0] ?? null;
  const criticalActions = new Set([
    "run_started",
    "task_execution_started",
    "task_execution_finished",
    "critic_rejected",
    "remediation_plan_created",
    "human_review_ready",
    "human_review_approved",
    "human_review_rejected",
    "llm_request",
    "tool_call",
    "tool_result",
  ]);
  const hotLogs = logs.filter((log) => criticalActions.has(log.action)).slice(0, 12);

  return (
    <div className="workspace-grid agent-first">
      <section className="agent-console-panel">
        <header>
          <div>
            <strong>Mission Control</strong>
            <span>goal in, AI teamlead decomposes, agents execute</span>
          </div>
          <button className="secondary-button" onClick={onStartRun}><Play size={15} />Start ready work</button>
        </header>
        <form className="goal-composer" onSubmit={onSubmitGoal}>
          <textarea name="goal" placeholder="Tell the AI teamlead what outcome you need. The architect will decompose it into agent tasks." rows={5} />
          <div className="composer-footer">
            <label><span>Trace</span><input name="traceId" placeholder="optional" /></label>
            <button className="primary-button" type="submit"><GitPullRequest size={15} />Submit goal</button>
          </div>
        </form>
        <div className="agent-feed">
          <div className="feed-row"><span>Goals</span><strong>{goals.length ? `${goals.length} goals in workspace` : "No goals submitted yet"}</strong><code>{reviewGoals.length} waiting for plan approval</code></div>
          <div className="feed-row"><span>Runtime</span><strong>{lastRun ? `${lastRun.review_tasks.length} task moved to review` : "No run started"}</strong><code>{lastRun?.run_id ?? "ready for first run"}</code></div>
        </div>
      </section>
      <section className="chain-panel">
        <header><strong>Agent Chain</strong><span>{activeGoal ? activeGoal.title : "No active goal"}</span></header>
        <AgentChain tasks={activeChain} selectedTaskId={selectedTaskId} onSelect={onSelect} />
      </section>
      <section className="review-gate-panel">
        <header><strong>Human Gate</strong><span>{reviewTasks.length} waiting</span></header>
        {latestReview ? <div className="review-card"><span className={`status-dot ${normalizeStatus(latestReview.status)}`} /><strong>{latestReview.title}</strong><code>{latestReview.assigned_agent ?? "unassigned"} / {compactId(latestReview.trace_id)}</code><div className="review-actions"><button className="secondary-button" onClick={() => { onSelect(latestReview.id); onRejectTaskReview(); }}><X size={15} />Reject</button><button className="primary-button" onClick={() => { onSelect(latestReview.id); onApproveTaskReview(); }}><Check size={15} />Approve</button></div></div> : <div className="empty-line">No task is waiting for human review</div>}
      </section>
      <section className="ide-shell">
        <header><strong>IDE</strong><span>file explorer / editor bridge pending IPC</span></header>
        <div className="ide-layout">
          <div className="file-tree"><code>Project</code><span>File tree will bind to crytex-ide and project file IPC.</span></div>
          <div className="editor-surface"><div className="editor-tabs"><button className="active">README.md</button><button disabled>generated.diff</button></div><pre>{`// Built-in IDE placeholder\n// Next IPC: list_project_files -> read_file -> write_file -> LSP diagnostics.\n// The IDE is first-class because the user can work manually too.`}</pre></div>
        </div>
      </section>
      <section className="goal-console">
        <header><strong>Run State</strong><button className="primary-button" onClick={onNewGoal}><GitPullRequest size={15} />New goal</button></header>
        <div className="metric-grid no-margin"><div className="metric"><span>Goals</span><strong>{goals.length}</strong></div><div className="metric"><span>Generated</span><strong>{generated.length}</strong></div><div className="metric"><span>Pending approvals</span><strong>{reviewGoals.length}</strong></div><div className="metric"><span>Last review</span><strong>{lastRun ? lastRun.review_tasks.length : 0}</strong></div></div>
      </section>
      <section className="wide-panel">
        <header><strong>Observe Highlights</strong><span>{hotLogs.length} critical events</span></header>
        <ObserveHighlights logs={hotLogs} />
      </section>
    </div>
  );
}

function AgentChain({ tasks, selectedTaskId, onSelect }: { tasks: Task[]; selectedTaskId: string | null; onSelect: (id: string) => void }) {
  const ordered = [...tasks].sort((a, b) => a.created_at - b.created_at);
  if (ordered.length === 0) return <div className="empty-line">No AI-generated chain yet</div>;
  return <div className="agent-chain">{ordered.map((task, index) => <button key={task.id} className={`chain-node ${selectedTaskId === task.id ? "active" : ""}`} onClick={() => onSelect(task.id)}><span className="chain-index">{index + 1}</span><div><strong>{task.assigned_agent ?? task.kind}</strong><span>{task.title}</span></div><code>{normalizeStatus(task.status)}</code></button>)}</div>;
}

function ObserveHighlights({ logs }: { logs: ProjectState["recent_logs"] }) {
  if (logs.length === 0) return <div className="empty-line">Run events will appear here after execution</div>;
  return <div className="observe-highlights">{logs.map((log) => <div className={`observe-event ${log.level}`} key={log.id}><Eye size={14} /><span>{log.action}</span><code>{compactId((log.metadata as { trace_id?: string })?.trace_id ?? log.task_id)}</code><small>{nowLabel(log.timestamp)}</small></div>)}</div>;
}
function GoalsPanel({ goals, selectedGoal, planTasks, selectedTaskId, onSelect, onNewGoal, onApprove, onReject }: { goals: Task[]; selectedGoal: Task | null; planTasks: Task[]; selectedTaskId: string | null; onSelect: (id: string) => void; onNewGoal: () => void; onApprove: () => void; onReject: () => void }) {
  const canDecide = selectedGoal ? normalizeStatus(selectedGoal.status) === "review" : false;
  return <div className="split-panel"><section className="list-panel"><header><strong>Goals</strong><button className="primary-button" onClick={onNewGoal}><Plus size={15} />New goal</button></header><TaskList tasks={goals} selectedTaskId={selectedTaskId} onSelect={onSelect} empty="No goals submitted yet" /></section><section className="detail-panel"><header><div><strong>Architect Plan</strong><span>{selectedGoal ? selectedGoal.title : "Select or submit a goal"}</span></div><div className="topbar-actions"><button className="secondary-button" onClick={onReject} disabled={!canDecide}><X size={15} />Reject</button><button className="primary-button" onClick={onApprove} disabled={!canDecide}><Check size={15} />Approve</button></div></header>{selectedGoal ? <><InfoGrid rows={[["Goal status", normalizeStatus(selectedGoal.status)], ["Trace", selectedGoal.trace_id], ["Plan tasks", String(planTasks.length)], ["Human score", selectedGoal.human_score?.toString() ?? "pending"]]} /><TaskList tasks={planTasks} selectedTaskId={selectedTaskId} onSelect={onSelect} empty="No generated plan tasks" /></> : <div className="empty-line">No active goal</div>}</section></div>;
}

function RunsPanel({ lastRun, tasks, logs, selectedTaskId, onSelect, onStartRun }: { lastRun: StartRunResponse | null; tasks: Task[]; logs: ProjectState["recent_logs"]; selectedTaskId: string | null; onSelect: (id: string) => void; onStartRun: () => void }) {
  const reviewedByStub = tasks.filter((task) => (task.result as { source?: unknown } | null)?.source === "tauri_stub_run");
  const reviewTasks = lastRun?.review_tasks ?? reviewedByStub;
  const runActions = logs.filter((log) => ["run_started", "task_execution_started", "task_execution_finished", "critic_rejected", "remediation_plan_created", "human_review_ready", "human_review_approved", "human_review_rejected"].includes(log.action));
  const ioActions = logs.filter((log) => ["llm_request", "llm_response", "tool_call", "tool_result"].includes(log.action));
  return (
    <div className="run-screen">
      <section className="detail-panel run-summary">
        <header>
          <div><strong>Run Cockpit</strong><span>Start execution, watch the chain, stop at human review</span></div>
          <button className="primary-button" onClick={onStartRun}><Play size={15} />Start run</button>
        </header>
        <InfoGrid rows={[["Run id", lastRun?.run_id ?? "none"], ["Started", lastRun ? nowLabel(lastRun.started_at) : "never"], ["Human review gates", String(reviewTasks.length)], ["Remaining ready", String(lastRun?.remaining_ready_tasks.length ?? 0)], ["Run events", String(runActions.length)], ["LLM/tool events", String(ioActions.length)]]} />
        <div className="run-checklist">
          <RunCheck label="Goal decomposed" passed={tasks.some((task) => isUserGoal(task)) && tasks.some((task) => task.parent_id !== null)} />
          <RunCheck label="Agents executed" passed={runActions.some((log) => log.action === "task_execution_finished")} />
          <RunCheck label="Critic reviewed" passed={runActions.some((log) => log.action === "critic_rejected" || log.action === "human_review_ready")} />
          <RunCheck label="Human gate visible" passed={reviewTasks.length > 0 || runActions.some((log) => log.action === "human_review_ready")} />
          <RunCheck label="Reward recorded" passed={runActions.some((log) => log.action === "human_review_approved")} />
        </div>
      </section>

      <section className="detail-panel run-review">
        <header><strong>Review Queue</strong><span>{reviewTasks.length} task results need a human decision</span></header>
        <TaskList tasks={reviewTasks} selectedTaskId={selectedTaskId} onSelect={onSelect} empty="No reviewed run tasks yet" />
      </section>

      <section className="detail-panel run-traces">
        <header><strong>Trace Explorer</strong><span>LLM calls, tools, critic feedback, remediation, reward</span></header>
        <TraceList tasks={tasks} logs={logs} />
      </section>
    </div>
  );
}

function RunCheck({ label, passed }: { label: string; passed: boolean }) {
  return <div className={`run-check ${passed ? "passed" : ""}`}>{passed ? <Check size={14} /> : <CircleDot size={14} />}<span>{label}</span></div>;
}

function RagPanel({ project, result, onSearch }: { project: Project | null; result: SearchProjectContextResponse | null; onSearch: (event: FormEvent<HTMLFormElement>) => void }) {
  return (
    <div className="detail-panel padded rag-panel">
      <header>
        <div><strong>Index/RAG</strong><span>{project ? "Qdrant Edge project context search" : "Create a project to index context"}</span></div>
      </header>
      <form className="rag-search" onSubmit={onSearch}>
        <label className="wide">Query<input name="query" placeholder="Search indexed code/docs context" disabled={!project} /></label>
        <label>Limit<input name="limit" type="number" min={1} max={20} defaultValue={8} disabled={!project} /></label>
        <button className="primary-button" type="submit" disabled={!project}><Search size={15} />Search</button>
      </form>
      <div className="placeholder-list">
        <div className="placeholder-row"><Database size={15} /><span>Initial project files are indexed on project creation.</span></div>
        <div className="placeholder-row"><AlertTriangle size={15} /><span>Watcher auto-start and embedding/reranker selectors are next backend/UI tasks.</span></div>
      </div>
      {!result ? <div className="empty-line">No RAG search has been run in this session</div> :
        <div className="rag-results">
          <header><strong>{result.hits.length} hits</strong><span>{result.query}</span></header>
          {result.hits.length === 0 ? <div className="empty-line">No indexed chunks matched this query</div> : result.hits.map((hit) => (
            <article className="rag-hit" key={`${hit.collection}-${hit.id}`}>
              <div><strong>{hit.relative_path ?? hit.source ?? hit.collection}</strong><code>{hit.collection} / {hit.score.toFixed(3)}</code></div>
              <pre>{hit.text ?? "No chunk text"}</pre>
            </article>
          ))}
        </div>}
    </div>
  );
}

function PlaceholderPanel({ title, items }: { title: string; items: string[] }) {
  return <div className="detail-panel padded"><header><strong>{title}</strong><span>planned first-class screen</span></header><div className="placeholder-list">{items.map((item) => <div className="placeholder-row" key={item}><AlertTriangle size={15} /><span>{item}</span></div>)}</div></div>;
}

function ModelsPanel({ status, inventory, managed, onRefresh, onSelect, onDownloadManaged, onUseManaged, onAddManaged }: { status: RuntimeStatus | null; inventory: OllamaModelsResponse | null; managed: ManagedModelsResponse | null; onRefresh: () => void; onSelect: (ollamaUrl: string, model: string) => void; onDownloadManaged: (modelId: string) => void; onUseManaged: (modelId: string) => void; onAddManaged: (request: AddManagedModelRequest) => void }) {
  const [ollamaUrl, setOllamaUrl] = useState(inventory?.ollama_url ?? status?.ollama_url ?? "http://localhost:11434");
  useEffect(() => {
    setOllamaUrl(inventory?.ollama_url ?? status?.ollama_url ?? "http://localhost:11434");
  }, [inventory?.ollama_url, status?.ollama_url]);
  const rows: [string, string][] = [
    ["Execution mode", status?.executor_mode ?? "unknown"],
    ["Planning mode", status?.planning_mode ?? "unknown"],
    ["Backend", status?.active_backend ?? "none"],
    ["Active model", status?.active_model ?? "none"],
    ["Ollama URL", status?.ollama_url ?? "not configured"],
    ["Real agent execution", status?.real_agent_execution ? "yes" : "no"],
  ];
  const models = inventory?.models ?? [];
  const managedModels = managed?.models ?? [];
  function submitManagedModel(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const paramsRaw = String(form.get("paramsB") ?? "").trim();
    onAddManaged({
      id: String(form.get("id") ?? "").trim(),
      name: String(form.get("name") ?? "").trim(),
      repo: String(form.get("repo") ?? "").trim(),
      filename: String(form.get("filename") ?? "").trim(),
      quantization: stringOrNull(form.get("quantization")),
      backend: stringOrNull(form.get("backend")),
      params_b: paramsRaw ? Number(paramsRaw) : null,
    });
    event.currentTarget.reset();
  }
  return (
    <div className="models-screen">
      <section className="detail-panel padded">
        <header>
          <div><strong>Runtime</strong><span>Current backend used by Start run</span></div>
          <button className="secondary-button" onClick={onRefresh}><RefreshCcw size={15} />Refresh</button>
        </header>
        <div className="runtime-card">
          <div className={`runtime-orb ${status?.real_agent_execution ? "real" : "stub"}`} />
          <div>
            <strong>{runtimeBadge(status)}</strong>
            <span>{status?.real_agent_execution ? "Tasks execute through agent services and model-backed inference." : "Tasks are still using deterministic stub execution."}</span>
          </div>
        </div>
        <InfoGrid rows={rows} />
      </section>
      <section className="detail-panel padded">
        <header><strong>Ollama Inventory</strong><span>{inventory?.available ? `${models.length} models` : "not connected"}</span></header>
        <div className="model-url-row">
          <label>URL<input value={ollamaUrl} onChange={(event) => setOllamaUrl(event.target.value)} /></label>
          <button className="secondary-button" onClick={onRefresh}><RefreshCcw size={15} />Reload</button>
        </div>
        {inventory?.error && <div className="form-error inline"><AlertTriangle size={15} />{inventory.error}</div>}
        <div className="model-list">
          {models.length === 0 ? <div className="empty-line">No Ollama models reported</div> : models.map((model) => (
            <div className={`model-row ${model.active ? "active" : ""}`} key={model.id}>
              <Network size={16} />
              <div><strong>{model.name}</strong><code>{model.id}</code></div>
              {model.active ? <span className="model-active">active</span> : <button className="primary-button" onClick={() => onSelect(ollamaUrl.trim() || inventory?.ollama_url || "http://localhost:11434", model.id)}>Use</button>}
            </div>
          ))}
        </div>
      </section>
      <section className="detail-panel padded">
        <header><strong>Managed Models</strong><span>{managedModels.length} manifest entries</span></header>
        <form className="model-add-form" onSubmit={submitManagedModel}>
          <input name="id" placeholder="id" />
          <input name="name" placeholder="name" />
          <input name="repo" placeholder="Hugging Face repo" />
          <input name="filename" placeholder="filename.gguf" />
          <select name="quantization" defaultValue="Q4_K_M">
            <option value="Q4_K_M">Q4_K_M</option>
            <option value="Q5_K_M">Q5_K_M</option>
            <option value="Q8_0">Q8_0</option>
            <option value="FP16">FP16</option>
            <option value="Q3_K_S">Q3_K_S</option>
            <option value="Q2_K">Q2_K</option>
          </select>
          <select name="backend" defaultValue="mistral_rs">
            <option value="mistral_rs">mistral_rs</option>
            <option value="ollama">ollama</option>
            <option value="onnx">onnx</option>
          </select>
          <input name="paramsB" placeholder="params B" type="number" step="0.1" />
          <button className="secondary-button" type="submit"><Plus size={15} />Add</button>
        </form>
        <div className="model-list">
          {managedModels.length === 0 ? <div className="empty-line">No managed models in manifest</div> : managedModels.map((model) => {
            const downloaded = modelStatusLabel(model.status) === "downloaded";
            return (
              <div className={`model-row ${downloaded ? "active" : ""}`} key={model.id}>
                <Database size={16} />
                <div>
                  <strong>{model.name}</strong>
                  <code>{model.id}</code>
                  <span>{model.repo ?? "local registry"} / {model.filename ?? "no filename"}</span>
                  <span>{model.recommended.backend} / {model.recommended.quantization} / ctx {model.recommended.context_size}</span>
                </div>
                {downloaded ? <button className="primary-button" onClick={() => onUseManaged(model.id)}>Use local</button> : <button className="primary-button" onClick={() => onDownloadManaged(model.id)}>Download</button>}
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
}

function modelStatusLabel(status: unknown): string {
  if (status === "Downloaded") return "downloaded";
  if (status === "Available") return "available";
  if (status && typeof status === "object" && "Downloading" in status) return "downloading";
  if (status && typeof status === "object" && "Error" in status) return "error";
  return "unknown";
}

function GeneratedTasksBoard({ columns, tasks, selectedTaskId, onSelect }: { columns: KanbanColumn[]; tasks: Task[]; selectedTaskId: string | null; onSelect: (id: string) => void }) {
  const columnsByStatus = new Map(columns.map((column) => [normalizeStatus(column.status), column]));
  return <div className="kanban-board">{statusOrder.map((status) => { const column = columnsByStatus.get(status); const cards = (column?.tasks ?? []).map((card) => tasks.find((task) => task.id === card.id)).filter(Boolean) as Task[]; return <section className="kanban-column" key={status}><header><span>{statusLabels[status]}</span><code>{status}</code><b>{cards.length}</b></header><div className="card-stack">{cards.map((task) => <button key={task.id} className={`task-card ${selectedTaskId === task.id ? "active" : ""}`} onClick={() => onSelect(task.id)}><span className={`status-dot ${normalizeStatus(task.status)}`} /><strong>{task.title}</strong><span className="task-meta">{task.kind} / {task.assigned_agent ?? "unassigned"}</span><span className="task-footer"><code>P{task.priority}</code><code>{compactId(task.trace_id)}</code></span></button>)}</div></section>; })}</div>;
}

function TaskList({ tasks, selectedTaskId, onSelect, empty = "No tasks" }: { tasks: Task[]; selectedTaskId: string | null; onSelect: (id: string) => void; empty?: string }) {
  if (tasks.length === 0) return <div className="empty-line">{empty}</div>;
  return <div className="task-list">{tasks.map((task) => <button key={task.id} className={`list-task ${selectedTaskId === task.id ? "active" : ""}`} onClick={() => onSelect(task.id)}><span className={`status-dot ${normalizeStatus(task.status)}`} /><strong>{task.title}</strong><span>{normalizeStatus(task.status)} / {task.assigned_agent ?? "unassigned"} / {task.kind}</span><code>{compactId(task.id)}</code></button>)}</div>;
}

function Inspector({ project, state, task, tab, setTab, onStatus, counts, selectedGoal, planTasks, onApprovePlan, onRejectPlan, onApproveTaskReview, onRejectTaskReview }: { project: Project | null; state: ProjectState | null; task: Task | null; tab: InspectorTab; setTab: (tab: InspectorTab) => void; onStatus: (status: TaskStatusKey) => void; counts: Record<TaskStatusKey, number>; selectedGoal: Task | null; planTasks: Task[]; onApprovePlan: () => void; onRejectPlan: () => void; onApproveTaskReview: () => void; onRejectTaskReview: () => void }) {
  const tabs: InspectorTab[] = ["overview", "payload", "result", "scores", "timing", "debug"];
  const canDecidePlan = selectedGoal ? normalizeStatus(selectedGoal.status) === "review" : false;
  const canDecideTaskReview = task ? normalizeStatus(task.status) === "review" && !isUserGoal(task) : false;
  return <aside className="inspector"><header><span>{task ? "Task inspector" : "Project state"}</span><code>{compactId(task?.id ?? project?.id)}</code></header>{task && <div className="tab-row">{tabs.map((item) => <button key={item} className={tab === item ? "active" : ""} onClick={() => setTab(item)}>{item}</button>)}</div>}{!task ? <div className="inspector-body"><InfoGrid rows={[["Project", project?.name ?? "none"], ["Root", project?.root_path ?? "none"], ["Tasks", String(state?.tasks.length ?? 0)], ["Logs", String(state?.recent_logs.length ?? 0)], ["Snapshot", state?.latest_snapshot ? "available" : "none"]]} /><div className="metric-grid">{statusOrder.map((status) => <div className="metric" key={status}><span>{statusLabels[status]}</span><strong>{counts[status]}</strong></div>)}</div><PlanDecisionBox selectedGoal={selectedGoal} planTasks={planTasks} canDecidePlan={canDecidePlan} onApprovePlan={onApprovePlan} onRejectPlan={onRejectPlan} /></div> : <div className="inspector-body">{tab === "overview" && <><InfoGrid rows={[["Title", task.title], ["Status", normalizeStatus(task.status)], ["Kind", task.kind], ["Agent", task.assigned_agent ?? "unassigned"], ["Priority", String(task.priority)], ["Trace", task.trace_id], ["Parent", task.parent_id ?? "none"], ["Iterations", String(task.iteration_count)]]} /><label className="status-menu">Status<select value={normalizeStatus(task.status)} onChange={(event) => void onStatus(event.target.value as TaskStatusKey)}>{statusOrder.map((status) => <option key={status} value={status}>{statusLabels[status]}</option>)}</select></label>{isUserGoal(task) && <PlanDecisionBox selectedGoal={selectedGoal} planTasks={planTasks} canDecidePlan={canDecidePlan} onApprovePlan={onApprovePlan} onRejectPlan={onRejectPlan} />}{!isUserGoal(task) && <TaskReviewDecisionBox task={task} canDecideTaskReview={canDecideTaskReview} onApproveTaskReview={onApproveTaskReview} onRejectTaskReview={onRejectTaskReview} />}</>}{tab === "payload" && <pre>{jsonBlock(task.payload)}</pre>}{tab === "result" && <><TaskReviewDecisionBox task={task} canDecideTaskReview={canDecideTaskReview} onApproveTaskReview={onApproveTaskReview} onRejectTaskReview={onRejectTaskReview} /><pre>{jsonBlock(task.result ?? "No result yet")}</pre></>}{tab === "scores" && <InfoGrid rows={[["Priority score", String(task.priority_score)], ["Critic score", task.critic_score?.toString() ?? "not scored"], ["Human score", task.human_score?.toString() ?? "not scored"], ["Iterations", String(task.iteration_count)]]} />}{tab === "timing" && <InfoGrid rows={[["Created", nowLabel(task.created_at)], ["Started", nowLabel(task.started_at ?? undefined)], ["Finished", nowLabel(task.finished_at ?? undefined)]]} />}{tab === "debug" && <pre>{jsonBlock(task)}</pre>}</div>}</aside>;
}

function PlanDecisionBox({ selectedGoal, planTasks, canDecidePlan, onApprovePlan, onRejectPlan }: { selectedGoal: Task | null; planTasks: Task[]; canDecidePlan: boolean; onApprovePlan: () => void; onRejectPlan: () => void }) {
  return <div className="decision-box"><span>Selected plan</span><strong>{selectedGoal?.title ?? "none"}</strong><code>{selectedGoal ? `${normalizeStatus(selectedGoal.status)} / ${planTasks.length} tasks` : "no goal selected"}</code><div className="decision-actions"><button className="secondary-button" onClick={onRejectPlan} disabled={!canDecidePlan}><X size={15} />Reject</button><button className="primary-button" onClick={onApprovePlan} disabled={!canDecidePlan}><Check size={15} />Approve</button></div></div>;
}

function TaskReviewDecisionBox({ task, canDecideTaskReview, onApproveTaskReview, onRejectTaskReview }: { task: Task; canDecideTaskReview: boolean; onApproveTaskReview: () => void; onRejectTaskReview: () => void }) {
  return <div className="decision-box"><span>Task review</span><strong>{task.title}</strong><code>{normalizeStatus(task.status)} / {task.assigned_agent ?? "unassigned"} / {task.result ? "result attached" : "no result"}</code><div className="decision-actions"><button className="secondary-button" onClick={onRejectTaskReview} disabled={!canDecideTaskReview}><X size={15} />Reject result</button><button className="primary-button" onClick={onApproveTaskReview} disabled={!canDecideTaskReview}><Check size={15} />Approve result</button></div></div>;
}

function ObservePanel({ tab, setTab, records, backendEvents, state, selectedTask, compact }: { tab: ObserveTab; setTab: (tab: ObserveTab) => void; records: CommandRecord[]; backendEvents: BackendEvent[]; state: ProjectState | null; selectedTask: Task | null; compact?: boolean }) {
  const tabs: ObserveTab[] = ["events", "logs", "traces", "metrics", "errors", "raw"];
  const sorted = sortCommands(records);
  const errors = sorted.filter((record) => record.error);
  return <section className={`observe-panel ${compact ? "compact" : ""}`}><header><div><strong>Observe</strong><span>{backendEvents.length} events / {sorted.length} commands / {errors.length} errors</span></div><div className="tab-row">{tabs.map((item) => <button key={item} className={tab === item ? "active" : ""} onClick={() => setTab(item)}>{item}</button>)}</div></header>{tab === "events" && <BackendEventTable events={backendEvents} />}{tab === "errors" && <CommandTable records={errors} />}{tab === "logs" && <LogTable logs={state?.recent_logs ?? []} />}{tab === "traces" && <TraceList tasks={state?.tasks ?? []} logs={state?.recent_logs ?? []} />}{tab === "metrics" && <MetricsView metrics={state?.metrics ?? null} />}{tab === "raw" && <pre>{jsonBlock({ selectedTask, backendEvents, lastCommand: sorted[0] ?? null, projectState: state })}</pre>}</section>;
}

function CommandTable({ records }: { records: CommandRecord[] }) {
  if (records.length === 0) return <div className="empty-line">No command records</div>;
  return <div className="table">{records.map((record) => <div className={`table-row ${record.error ? "error" : ""}`} key={record.id}><code>{nowLabel(record.startedAt)}</code><span>{record.name}</span><code>{record.durationMs}ms</code><span>{record.error ?? "ok"}</span></div>)}</div>;
}

function BackendEventTable({ events }: { events: BackendEvent[] }) {
  if (events.length === 0) return <div className="empty-line">No live backend events yet</div>;
  return <div className="table">{events.map((event, index) => <div className="table-row backend-event-row" key={`${event.type}-${index}`}><code>{event.type}</code><span>{event.project_id ?? String(event.payload.task_id ?? event.payload.model_id ?? "runtime")}</span><code>{compactId(String(event.payload.task_id ?? event.payload.snapshot_id ?? event.payload.lora_id ?? ""))}</code><span>{jsonBlock(event.payload)}</span></div>)}</div>;
}

function LogTable({ logs }: { logs: ProjectState["recent_logs"] }) {
  if (logs.length === 0) return <div className="empty-line">No backend logs available</div>;
  return <div className="table">{logs.map((log) => <div className={`table-row ${log.level}`} key={log.id}><code>{nowLabel(log.timestamp)}</code><span>{log.agent}</span><span>{log.action}</span><span>{log.message ?? compactId(log.task_id)}</span></div>)}</div>;
}

function TraceList({ tasks, logs }: { tasks: Task[]; logs: ProjectState["recent_logs"] }) {
  const [activeTraceId, setActiveTraceId] = useState<string | null>(null);
  const grouped = new Map<string, { trace: string; tasks: Task[]; logs: ProjectState["recent_logs"] }>();
  tasks.forEach((task) => {
    const entry = grouped.get(task.trace_id) ?? { trace: task.trace_id, tasks: [], logs: [] };
    entry.tasks.push(task);
    grouped.set(task.trace_id, entry);
  });
  logs.forEach((log) => {
    const trace = String((log.metadata as { trace_id?: unknown })?.trace_id ?? "");
    if (!trace) return;
    const entry = grouped.get(trace) ?? { trace, tasks: [], logs: [] };
    entry.logs.push(log);
    grouped.set(trace, entry);
  });
  const traces = [...grouped.values()].sort((a, b) => newestTimestamp(b) - newestTimestamp(a));
  if (traces.length === 0) return <div className="empty-line">No traces yet</div>;
  const active = traces.find((trace) => trace.trace === activeTraceId) ?? traces[0];
  const orderedLogs = [...active.logs].sort((a, b) => a.timestamp - b.timestamp);
  return (
    <div className="trace-workbench">
      <div className="trace-sidebar">
        {traces.map((trace) => (
          <button className={`trace-summary ${trace.trace === active.trace ? "active" : ""}`} key={trace.trace} onClick={() => setActiveTraceId(trace.trace)}>
            <code>{trace.trace}</code>
            <span>{trace.tasks.length} tasks / {trace.logs.length} events</span>
            <small>{traceStage(trace.logs)}</small>
          </button>
        ))}
      </div>
      <div className="timeline">
        <header>
          <div><strong>{active.trace}</strong><span>{active.tasks.length} tasks, {orderedLogs.length} observed events</span></div>
          <code>{traceStage(active.logs)}</code>
        </header>
        <div className="timeline-events">
          {orderedLogs.map((log) => <TimelineEvent log={log} key={log.id} />)}
        </div>
      </div>
    </div>
  );
}

function newestTimestamp(trace: { tasks: Task[]; logs: ProjectState["recent_logs"] }): number {
  const taskTimes = trace.tasks.map((task) => task.finished_at ?? task.started_at ?? task.created_at);
  const logTimes = trace.logs.map((log) => log.timestamp);
  return Math.max(0, ...taskTimes, ...logTimes);
}

function traceStage(logs: ProjectState["recent_logs"]): string {
  const actions = new Set(logs.map((log) => log.action));
  if (actions.has("human_review_approved")) return "approved";
  if (actions.has("human_review_rejected")) return "human rejected";
  if (actions.has("human_review_ready")) return "human gate";
  if (actions.has("remediation_plan_created")) return "remediation";
  if (actions.has("critic_rejected")) return "critic rejected";
  if (actions.has("task_execution_started")) return "running";
  if (actions.has("run_started")) return "started";
  return "observed";
}

function TimelineEvent({ log }: { log: ProjectState["recent_logs"][number] }) {
  const meta = (log.metadata ?? {}) as Record<string, unknown>;
  const feedback = typeof meta.feedback === "string" ? meta.feedback : null;
  const failureType = typeof meta.failure_type === "string" ? meta.failure_type : null;
  const reward = typeof meta.reward === "number" ? meta.reward.toFixed(2) : null;
  const model = typeof meta.model === "string" ? meta.model : null;
  const toolName = typeof meta.tool_name === "string" ? meta.tool_name : null;
  const duration = typeof meta.duration_ms === "number" ? `${meta.duration_ms}ms` : null;
  const usage = meta.usage && typeof meta.usage === "object" ? meta.usage as Record<string, unknown> : null;
  const totalTokens = typeof usage?.total_tokens === "number" ? `${usage.total_tokens} tokens` : null;
  const reviewDecision = typeof meta.review_decision === "string" ? meta.review_decision : null;
  const tags = [model, toolName, duration, totalTokens, reviewDecision].filter((tag): tag is string => Boolean(tag));
  return (
    <details className={`timeline-event ${timelineTone(log.action)} ${log.level}`} open={["critic_rejected", "remediation_plan_created", "human_review_ready"].includes(log.action)}>
      <summary>
        <span className="timeline-dot" />
        <div>
          <strong>{log.action}</strong>
          <span>{log.agent} / {compactId(log.task_id)}</span>
        </div>
        <code>{nowLabel(log.timestamp)}</code>
      </summary>
      <div className="timeline-body">
        {tags.length > 0 && <div className="timeline-tags">{tags.map((tag) => <code key={tag}>{tag}</code>)}</div>}
        {feedback && <div className="feedback-line"><strong>Feedback</strong><span>{feedback}</span></div>}
        {failureType && <div className="feedback-line"><strong>Failure</strong><span>{failureType}</span></div>}
        {reward && <div className="feedback-line"><strong>Reward</strong><span>{reward}</span></div>}
        <pre>{jsonBlock(meta)}</pre>
      </div>
    </details>
  );
}

function timelineTone(action: string): string {
  if (action.includes("rejected") || action.includes("error")) return "danger";
  if (action.includes("remediation")) return "warn";
  if (action.includes("approved") || action.includes("ready")) return "success";
  if (action.includes("tool") || action.includes("llm")) return "io";
  return "normal";
}

function MetricsView({ metrics }: { metrics: ProjectState["metrics"] | null }) {
  if (!metrics) return <div className="empty-line">No metrics snapshot</div>;
  return <div className="metric-grid"><div className="metric"><span>Completed</span><strong>{metrics.tasks_completed}</strong></div><div className="metric"><span>Failed</span><strong>{metrics.tasks_failed}</strong></div><div className="metric"><span>Latency</span><strong>{metrics.average_latency_ms}ms</strong></div><div className="metric"><span>Cache hits</span><strong>{metrics.cache_hits}</strong></div><div className="metric"><span>CPU</span><strong>{metrics.cpu_usage_percent.toFixed(1)}%</strong></div><div className="metric"><span>Memory</span><strong>{metrics.memory_used_mb}/{metrics.memory_total_mb} MB</strong></div></div>;
}

function InfoGrid({ rows }: { rows: [string, string][] }) {
  return <div className="info-grid">{rows.map(([label, value]) => <div key={label}><span>{label}</span><code>{value}</code></div>)}</div>;
}

function Modal({ title, children, onClose, error }: { title: string; children: React.ReactNode; onClose: () => void; error: string | null }) {
  return <div className="modal-backdrop"><div className="modal"><header><strong>{title}</strong><button className="icon-button" onClick={onClose} title="Close"><X size={17} /></button></header>{error && <div className="form-error"><AlertTriangle size={15} />{error}</div>}{children}</div></div>;
}

