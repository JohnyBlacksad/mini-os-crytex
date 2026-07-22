//! Runtime state managed by the Tauri desktop shell.

use crate::commands::{
    self, AddManagedModelCommand, AgentTaskExecutor, BackendE2eEvidenceGate,
    BackendE2eMatrixCommand, BackendE2eMatrixReport, BackendE2eScenarioKind,
    BackendE2eScenarioReport, CreateProjectCommand, DownloadManagedModelCommand,
    EvaluatePromptChallengerCommand, EvaluatePromptChallengerResponse, ExportRunDiagnosticsCommand,
    GoalPlanResponse, ManagedModelRecord, ManagedModelRuntimeProofReport, ManagedModelsResponse,
    OllamaModelsResponse, PlanDecisionCommand, PlanDecisionResponse,
    ProveManagedModelRuntimeCommand, RunDiagnosticsReport, RuntimeStatus,
    SearchProjectContextCommand, SearchProjectContextResponse, SetActiveManagedModelCommand,
    SetActiveOllamaModelCommand, SetTaskStatusCommand, StartRunCommand, StartRunResponse,
    StubTaskExecutor, SubmitGoalCommand, SubmitTaskCommand, TaskExecutor,
    TaskReviewDecisionCommand, TaskReviewDecisionResponse, TauriCommandError,
    TrainLoraAdapterCommand, TrainLoraAdapterResponse,
};
use async_trait::async_trait;
use crytex_agents::{
    architect::ArchitectAgent, coder::CoderAgent, critic::CriticAgent, qa::QaAgent,
    researcher::ResearcherAgent, security::SecurityAgent, summarizer::SummarizerAgent,
};
use crytex_core::bus::{Event, EventBus};
use crytex_core::indexer::ProjectIndexer;
use crytex_core::metrics::{MetricsService, MetricsServiceImpl};
use crytex_core::models::{KanbanState, Project, Task, TaskDependency, TaskStatus};
use crytex_core::persistence::ProjectSnapshotRepository;
use crytex_core::services::{
    Agent, AgentRole, AgentService, AgentServiceError, AuditLogService, AuditLogServiceImpl,
    ContextAssembler, DeviceKind, Embedder, EventHandler, EventService, EventServiceImpl,
    HardwareDetector, InferenceService, InferenceServiceError, InferenceServiceImpl,
    LoraBenchmarkGate, LoraEvolutionService, LoraEvolutionServiceImpl, MockEmbedder,
    MockSparseEmbedder, ModelCompatibilityPlanner, ModelManager, ModelManagerImpl, Orchestrator,
    OrchestratorImpl, ProjectService, ProjectServiceImpl, ProjectWatcher, PromptBenchmarkGate,
    PromptEvolutionService, RagChunkEvidence, RecordRewardRequest, RewardService,
    RuntimeFeatureSet, SparseEmbedder, SystemHardwareDetector, TaskService, TaskServiceImpl,
    ToolDescription, ToolService, ToolServiceError, VectorStore, detect_cuda_toolchain_status,
};
use crytex_core::state_export::ProjectState;
use crytex_inference::{
    BackendCapabilityReport, BackendInfo, BackendRegistry, InferenceRequest, InferenceResponse,
    LoRAAdapter, ModelInfo,
};
use crytex_inference_candle::CandleLoraTrainer;
use crytex_inference_mistral::MistralRsBackend;
use crytex_inference_ollama::OllamaBackend;
use crytex_storage::{Storage, vector::EdgeVectorStore};
use serde_json::Value;
use std::collections::HashMap;
use std::env;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::{RwLock, oneshot};
use tokio::task::JoinHandle;

/// Shared runtime dependencies exposed to Tauri command wrappers.
#[derive(Clone)]
pub struct CrytexAppState {
    project_service: Arc<dyn ProjectService>,
    task_service: Arc<dyn TaskService>,
    audit_service: Arc<dyn AuditLogService>,
    snapshot_repo: Arc<dyn ProjectSnapshotRepository>,
    metrics_service: Arc<dyn MetricsService>,
    event_service: Arc<dyn EventService>,
    model_manager: Arc<dyn ModelManager>,
    orchestrator: Arc<dyn Orchestrator>,
    task_executor: Arc<RwLock<Arc<dyn TaskExecutor>>>,
    active_inference: Arc<RwLock<Option<Arc<dyn InferenceService>>>>,
    reward_service: Arc<RewardService>,
    runtime_status: Arc<RwLock<RuntimeStatus>>,
    embedder: Arc<dyn Embedder>,
    vector_store: Arc<dyn VectorStore>,
    project_indexer: ProjectIndexer,
    context_assembler: Arc<ContextAssembler>,
    lora_evolution: Arc<dyn LoraEvolutionService>,
    prompt_evolution: Arc<PromptEvolutionService<Storage, Storage>>,
    prompt_benchmark_gate: Option<Arc<dyn PromptBenchmarkGate>>,
    watcher_shutdowns: Arc<RwLock<HashMap<String, oneshot::Sender<()>>>>,
    watcher_tasks: Arc<RwLock<HashMap<String, JoinHandle<()>>>>,
}

type TaskExecutorFactory = Box<
    dyn FnOnce(
            Arc<dyn ProjectService>,
            Arc<dyn AuditLogService>,
            Arc<ContextAssembler>,
        ) -> Arc<dyn TaskExecutor>
        + Send,
>;

impl CrytexAppState {
    /// Build a local SQLite-backed runtime state for manual desktop testing.
    pub async fn new_sqlite(db_path: impl AsRef<Path>) -> Result<Self, TauriCommandError> {
        let runtime_status = runtime_status_from_env();
        let planning_inference = configured_planning_inference_service();
        let planning_tools = planning_inference
            .as_ref()
            .map(|_| Arc::new(EmptyToolService) as Arc<dyn ToolService>);
        let reranker = configured_reranker_from_env()?;
        Self::new_sqlite_with_executor_and_planning(
            db_path,
            None,
            planning_inference,
            planning_tools,
            reranker,
            runtime_status,
        )
        .await
    }

    /// Build a SQLite-backed runtime state with an explicitly injected task executor.
    pub async fn new_sqlite_with_executor(
        db_path: impl AsRef<Path>,
        task_executor: Arc<dyn TaskExecutor>,
    ) -> Result<Self, TauriCommandError> {
        Self::new_sqlite_with_executor_and_planning(
            db_path,
            Some(task_executor),
            None,
            None,
            None,
            custom_executor_runtime_status(),
        )
        .await
    }

    /// Build a SQLite-backed runtime whose task executor uses real agents, Ollama,
    /// and project-scoped coding tools.
    pub async fn new_sqlite_with_ollama_agent_executor(
        db_path: impl AsRef<Path>,
        ollama_url: impl Into<String>,
        model: impl Into<String>,
    ) -> Result<Self, TauriCommandError> {
        let ollama_url = ollama_url.into();
        let model = model.into();
        let inference = ollama_inference_service(ollama_url.clone(), model.clone());
        let runtime_status = ollama_agent_runtime_status(ollama_url, model, "deterministic");
        let reranker = configured_reranker_from_env()?;
        Self::new_sqlite_with_executor_factory_and_planning(
            db_path,
            Box::new(move |project_service, audit_service, context_assembler| {
                Arc::new(
                    AgentTaskExecutor::new_project_scoped(
                        Arc::new(
                            StaticAgentService::with_default_agents(Some(context_assembler))
                                .with_audit(audit_service.clone()),
                        ),
                        inference,
                        project_service,
                    )
                    .with_audit(audit_service),
                ) as Arc<dyn TaskExecutor>
            }),
            None,
            None,
            None,
            None,
            None,
            None,
            reranker,
            runtime_status,
        )
        .await
    }

    /// Build state with explicit task execution and planning dependencies.
    pub async fn new_sqlite_with_executor_and_planning(
        db_path: impl AsRef<Path>,
        task_executor: Option<Arc<dyn TaskExecutor>>,
        planning_inference: Option<Arc<dyn InferenceService>>,
        planning_tools: Option<Arc<dyn ToolService>>,
        reranker: Option<Arc<dyn crytex_core::services::Reranker>>,
        runtime_status: RuntimeStatus,
    ) -> Result<Self, TauriCommandError> {
        Self::new_sqlite_with_executor_factory_and_planning(
            db_path,
            Box::new(move |project_service, audit_service, context_assembler| {
                configured_task_executor(
                    task_executor,
                    project_service,
                    audit_service,
                    context_assembler,
                )
            }),
            planning_inference,
            planning_tools,
            None,
            None,
            None,
            None,
            reranker,
            runtime_status,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg(test)]
    async fn new_sqlite_with_executor_factory_planning_and_reranker(
        db_path: impl AsRef<Path>,
        task_executor_factory: TaskExecutorFactory,
        reranker: Arc<dyn crytex_core::services::Reranker>,
        runtime_status: RuntimeStatus,
    ) -> Result<Self, TauriCommandError> {
        Self::new_sqlite_with_executor_factory_and_planning(
            db_path,
            task_executor_factory,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(reranker),
            runtime_status,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn new_sqlite_with_executor_factory_and_planning(
        db_path: impl AsRef<Path>,
        task_executor_factory: TaskExecutorFactory,
        planning_inference: Option<Arc<dyn InferenceService>>,
        planning_tools: Option<Arc<dyn ToolService>>,
        lora_evolution: Option<Arc<dyn LoraEvolutionService>>,
        lora_benchmark_gate: Option<Arc<dyn LoraBenchmarkGate>>,
        prompt_benchmark_gate: Option<Arc<dyn PromptBenchmarkGate>>,
        model_manager: Option<Arc<dyn ModelManager>>,
        reranker: Option<Arc<dyn crytex_core::services::Reranker>>,
        runtime_status: RuntimeStatus,
    ) -> Result<Self, TauriCommandError> {
        let db_path_buf = db_path.as_ref().to_path_buf();
        let vector_path = vector_store_path_for_db(&db_path_buf);
        let app_data_dir = db_path_buf
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let model_config_dir = app_data_dir.join("models-config");
        let model_cache_dir = app_data_dir.join("models-cache");
        let db_path = db_path_buf.display().to_string();
        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new(384));
        let sparse_embedder: Arc<dyn SparseEmbedder> = Arc::new(MockSparseEmbedder);
        let vector_store: Arc<dyn VectorStore> = Arc::new(EdgeVectorStore::new(vector_path)?);
        let project_indexer = ProjectIndexer::new(embedder.clone(), vector_store.clone())
            .with_sparse_embedder(sparse_embedder.clone());
        let mut context_assembler = ContextAssembler::new(embedder.clone(), vector_store.clone())
            .with_sparse_embedder(sparse_embedder);
        if let Some(reranker) = reranker {
            context_assembler = context_assembler.with_reranker(reranker);
        }
        let context_assembler = Arc::new(context_assembler);
        let storage = Arc::new(
            Storage::new(&db_path)
                .await
                .map_err(|err| TauriCommandError::Bootstrap(err.to_string()))?,
        );
        let event_bus = Arc::new(EventBus::new());
        let event_service: Arc<dyn EventService> = Arc::new(EventServiceImpl::new(event_bus));
        let model_manager: Arc<dyn ModelManager> = model_manager.unwrap_or_else(|| {
            Arc::new(ModelManagerImpl::new_standard(
                model_config_dir,
                model_cache_dir,
                event_service.clone(),
                Arc::new(SystemHardwareDetector::new()),
            ))
        });
        let audit_service: Arc<dyn AuditLogService> =
            Arc::new(AuditLogServiceImpl::new(storage.clone()));
        event_service
            .start_handler(Arc::new(RunObservedAuditBridge::new(audit_service.clone())))
            .await;
        let project_service: Arc<dyn ProjectService> =
            Arc::new(ProjectServiceImpl::new(storage.clone()));
        let task_service: Arc<dyn TaskService> = Arc::new(TaskServiceImpl::new(
            storage.clone(),
            event_service.clone(),
            audit_service.clone(),
        ));
        let orchestrator =
            configured_orchestrator(task_service.clone(), planning_inference, planning_tools);
        let snapshot_repo: Arc<dyn ProjectSnapshotRepository> = storage.clone();
        let metrics_service: Arc<dyn MetricsService> =
            Arc::new(MetricsServiceImpl::new(storage.clone()));
        let base_task_executor = task_executor_factory(
            project_service.clone(),
            audit_service.clone(),
            context_assembler.clone(),
        );
        let reward_service = Arc::new(RewardService::new(storage.clone()));
        let prompt_evolution = Arc::new(PromptEvolutionService::new(
            storage.clone(),
            storage.clone(),
        ));
        let lora_evolution = lora_evolution.unwrap_or_else(|| {
            let mut service = LoraEvolutionServiceImpl::new(
                task_service.clone(),
                storage.clone(),
                storage.clone(),
                storage.clone(),
                Arc::new(NoopLoraInferenceService),
                event_service.clone(),
                Arc::new(CandleLoraTrainer::new()),
                app_data_dir.join("lora-adapters"),
                runtime_status
                    .active_model
                    .clone()
                    .unwrap_or_else(|| "tauri-managed-model".to_string()),
            )
            .with_threshold(50)
            .with_min_human_score(1.0)
            .with_validation_reward_threshold(0.4)
            .with_validation_loss_threshold(8.0)
            .with_experience_repo(storage.clone())
            .with_training_job_repo(storage.clone())
            .with_vector_index(embedder.clone(), vector_store.clone());
            if let Some(gate) = lora_benchmark_gate {
                service = service.with_benchmark_gate(gate);
            }
            Arc::new(service) as Arc<dyn LoraEvolutionService>
        });
        let task_executor: Arc<dyn TaskExecutor> = Arc::new(LoraSelectingTaskExecutor::new(
            base_task_executor,
            task_service.clone(),
            lora_evolution.clone(),
        ));

        task_service.load_all_tasks().await?;
        repair_stub_completed_tasks(task_service.clone()).await?;

        Ok(Self {
            project_service,
            task_service,
            audit_service,
            snapshot_repo,
            metrics_service,
            event_service,
            model_manager,
            orchestrator,
            task_executor: Arc::new(RwLock::new(task_executor)),
            active_inference: Arc::new(RwLock::new(None)),
            reward_service,
            runtime_status: Arc::new(RwLock::new(runtime_status)),
            embedder,
            vector_store,
            project_indexer,
            context_assembler,
            lora_evolution,
            prompt_evolution,
            prompt_benchmark_gate,
            watcher_shutdowns: Arc::new(RwLock::new(HashMap::new())),
            watcher_tasks: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    pub async fn list_projects(&self) -> Result<Vec<Project>, TauriCommandError> {
        commands::list_projects(self.project_service.clone()).await
    }

    pub async fn create_project(
        &self,
        request: CreateProjectCommand,
    ) -> Result<Project, TauriCommandError> {
        let project = commands::create_project(self.project_service.clone(), request).await?;
        self.project_indexer
            .index(&project.id, Path::new(&project.root_path))
            .await?;
        self.start_project_watcher(&project).await?;
        Ok(project)
    }

    async fn start_project_watcher(&self, project: &Project) -> Result<(), TauriCommandError> {
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let (ready_tx, ready_rx) = oneshot::channel();
        self.watcher_shutdowns
            .write()
            .await
            .insert(project.id.clone(), shutdown_tx);

        let project_id = project.id.clone();
        let root_path = PathBuf::from(&project.root_path);
        let watcher = ProjectWatcher::new(self.project_indexer.clone(), self.event_service.clone());

        let watcher_task = tokio::spawn(async move {
            if let Err(error) = watcher
                .watch_with_ready(project_id.clone(), root_path, shutdown_rx, ready_tx)
                .await
            {
                eprintln!(
                    "project watcher stopped with error: project_id={project_id} error={error}"
                );
            }
        });
        self.watcher_tasks
            .write()
            .await
            .insert(project.id.clone(), watcher_task);

        match ready_rx.await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(error)) => Err(TauriCommandError::Bootstrap(format!(
                "project watcher failed to start: {error}"
            ))),
            Err(error) => Err(TauriCommandError::Bootstrap(format!(
                "project watcher readiness channel closed: {error}"
            ))),
        }
    }

    pub async fn shutdown_project_watchers(&self) {
        let shutdowns = self
            .watcher_shutdowns
            .write()
            .await
            .drain()
            .map(|(_, shutdown)| shutdown)
            .collect::<Vec<_>>();
        for shutdown in shutdowns {
            let _ = shutdown.send(());
        }

        let tasks = self
            .watcher_tasks
            .write()
            .await
            .drain()
            .map(|(_, task)| task)
            .collect::<Vec<_>>();
        for task in tasks {
            let _ = task.await;
        }
    }

    pub async fn kanban_state(&self, project_id: &str) -> Result<KanbanState, TauriCommandError> {
        commands::kanban_state(self.project_service.clone(), project_id).await
    }

    pub async fn list_tasks(&self, project_id: &str) -> Result<Vec<Task>, TauriCommandError> {
        commands::list_tasks(self.task_service.clone(), project_id).await
    }

    pub async fn submit_task(&self, request: SubmitTaskCommand) -> Result<Task, TauriCommandError> {
        commands::submit_task(self.task_service.clone(), request).await
    }

    pub async fn submit_goal(
        &self,
        request: SubmitGoalCommand,
    ) -> Result<GoalPlanResponse, TauriCommandError> {
        commands::submit_goal(
            self.task_service.clone(),
            self.orchestrator.clone(),
            request,
        )
        .await
    }

    pub async fn set_task_status(
        &self,
        request: SetTaskStatusCommand,
    ) -> Result<Task, TauriCommandError> {
        commands::set_task_status(self.task_service.clone(), request).await
    }

    pub async fn approve_plan(
        &self,
        request: PlanDecisionCommand,
    ) -> Result<PlanDecisionResponse, TauriCommandError> {
        commands::approve_plan(self.task_service.clone(), request).await
    }

    pub async fn reject_plan(
        &self,
        request: PlanDecisionCommand,
    ) -> Result<PlanDecisionResponse, TauriCommandError> {
        commands::reject_plan(self.task_service.clone(), request).await
    }

    pub async fn approve_task_review(
        &self,
        request: TaskReviewDecisionCommand,
    ) -> Result<TaskReviewDecisionResponse, TauriCommandError> {
        let comment = request.comment.clone();
        let response = commands::approve_task_review(self.task_service.clone(), request).await?;
        self.record_human_review_feedback(&response.task, 1.0, comment.as_deref(), "approved")
            .await?;
        self.record_lora_feedback_after_review(&response.task, true)
            .await?;
        Ok(response)
    }

    pub async fn reject_task_review(
        &self,
        request: TaskReviewDecisionCommand,
    ) -> Result<TaskReviewDecisionResponse, TauriCommandError> {
        let comment = request.comment.clone();
        let response = commands::reject_task_review(self.task_service.clone(), request).await?;
        self.record_human_review_feedback(&response.task, 0.0, comment.as_deref(), "rejected")
            .await?;
        self.record_lora_feedback_after_review(&response.task, false)
            .await?;
        Ok(response)
    }

    pub async fn start_run(
        &self,
        request: StartRunCommand,
    ) -> Result<StartRunResponse, TauriCommandError> {
        let runtime_status = self.runtime_status.read().await;
        if !runtime_status.ready_to_run {
            return Err(TauriCommandError::Bootstrap(
                "no inference backend configured".to_string(),
            ));
        }
        drop(runtime_status);
        commands::start_run_with_orchestrator(
            self.task_service.clone(),
            Some(self.orchestrator.clone()),
            Some(self.audit_service.clone()),
            Some(self.event_service.clone()),
            self.task_executor.read().await.clone(),
            request,
        )
        .await
    }

    pub async fn run_backend_e2e_matrix(
        &self,
        request: BackendE2eMatrixCommand,
    ) -> Result<BackendE2eMatrixReport, TauriCommandError> {
        let existing_ready_tasks = self
            .task_service
            .list_ready()
            .await?
            .into_iter()
            .filter(|task| task.project_id == request.project_id)
            .map(|task| task.id)
            .collect::<Vec<_>>();
        if !existing_ready_tasks.is_empty() {
            return Err(TauriCommandError::Bootstrap(format!(
                "backend e2e matrix requires an isolated project with no ready tasks; ready_task_ids={existing_ready_tasks:?}"
            )));
        }
        let trace_id = request
            .trace_id
            .clone()
            .unwrap_or_else(|| format!("backend-e2e-{}", timestamp_millis()));
        let scenarios = matrix_scenarios_or_default(request.scenarios);
        let mut reports = Vec::with_capacity(scenarios.len());

        for scenario in scenarios {
            reports.push(
                self.run_backend_e2e_scenario(
                    &request.project_id,
                    &trace_id,
                    scenario,
                    request.max_steps.max(1),
                )
                .await?,
            );
        }

        let passed = reports
            .iter()
            .flat_map(|report| report.gates.iter())
            .all(|gate| gate.passed);
        Ok(BackendE2eMatrixReport {
            project_id: request.project_id,
            trace_id,
            passed,
            scenarios: reports,
        })
    }

    async fn run_backend_e2e_scenario(
        &self,
        project_id: &str,
        matrix_trace_id: &str,
        scenario: BackendE2eScenarioKind,
        max_steps: usize,
    ) -> Result<BackendE2eScenarioReport, TauriCommandError> {
        let scenario_trace_id = format!("{matrix_trace_id}:{scenario:?}").to_lowercase();
        self.seed_backend_e2e_scenario(project_id, &scenario_trace_id, &scenario)
            .await?;
        let run = self
            .start_run(StartRunCommand {
                project_id: project_id.to_string(),
                max_steps,
            })
            .await?;
        for review_task in &run.review_tasks {
            let _ = self
                .approve_task_review(TaskReviewDecisionCommand {
                    task_id: review_task.id.clone(),
                    comment: Some(format!("backend e2e matrix accepted {scenario:?}")),
                })
                .await;
        }
        let diagnostics = self
            .export_run_diagnostics(ExportRunDiagnosticsCommand {
                project_id: project_id.to_string(),
                run_id: run.run_id.clone(),
                trace_id: Some(scenario_trace_id.clone()),
            })
            .await?;
        let gates = backend_e2e_gates(&scenario, &diagnostics);
        Ok(BackendE2eScenarioReport {
            scenario,
            trace_id: scenario_trace_id,
            run_id: run.run_id,
            review_task_ids: diagnostics.review_task_ids.clone(),
            gates,
            diagnostics,
        })
    }

    async fn seed_backend_e2e_scenario(
        &self,
        project_id: &str,
        trace_id: &str,
        scenario: &BackendE2eScenarioKind,
    ) -> Result<(), TauriCommandError> {
        let root = self
            .submit_task(SubmitTaskCommand {
                project_id: project_id.to_string(),
                parent_id: None,
                title: format!("Backend e2e matrix {scenario:?}"),
                description: Some(
                    "Synthetic backend proof root created by the matrix runner".into(),
                ),
                kind: "goal".into(),
                assigned_agent: Some("architect".into()),
                priority: 100,
                payload: serde_json::json!({
                    "source": "backend_e2e_matrix",
                    "scenario": format!("{scenario:?}"),
                }),
                trace_id: Some(trace_id.to_string()),
            })
            .await?;
        let root = self
            .set_task_status(SetTaskStatusCommand {
                task_id: root.id,
                status: TaskStatus::Completed,
            })
            .await?;

        match scenario {
            BackendE2eScenarioKind::HappyPath => {
                self.seed_backend_e2e_chain(project_id, &root.id, trace_id, None)
                    .await?;
            }
            BackendE2eScenarioKind::RejectRemediation => {
                self.seed_backend_e2e_chain(
                    project_id,
                    &root.id,
                    trace_id,
                    Some("backend_e2e_reject"),
                )
                .await?;
            }
            BackendE2eScenarioKind::Failure => {
                let _ = self
                    .submit_task(SubmitTaskCommand {
                        project_id: project_id.to_string(),
                        parent_id: Some(root.id),
                        title: "Backend e2e forced executor failure".into(),
                        description: Some(
                            "The matrix runner forces this task to fail to verify diagnostics"
                                .into(),
                        ),
                        kind: "codegen".into(),
                        assigned_agent: Some("coder".into()),
                        priority: 100,
                        payload: serde_json::json!({
                            "source": "backend_e2e_matrix",
                            "backend_e2e_force_failure": true,
                            "prompt": "Force an executor failure and keep diagnostic evidence"
                        }),
                        trace_id: Some(trace_id.to_string()),
                    })
                    .await?;
            }
        }
        Ok(())
    }

    async fn seed_backend_e2e_chain(
        &self,
        project_id: &str,
        parent_id: &str,
        trace_id: &str,
        critic_mode: Option<&str>,
    ) -> Result<(), TauriCommandError> {
        let mut previous: Option<Task> = None;
        for (agent, kind, priority) in [
            ("architect", "design", 100),
            ("coder", "codegen", 99),
            ("qa", "qa", 98),
            ("security", "security", 97),
            ("critic", "review", 96),
        ] {
            let mut payload = serde_json::json!({
                "source": "backend_e2e_matrix",
                "prompt": format!("Run backend e2e {agent} step with typed artifact output"),
            });
            if agent == "critic" && critic_mode == Some("backend_e2e_reject") {
                payload["backend_e2e_review_decision"] = serde_json::json!("reject");
                if let Some(previous) = &previous {
                    payload["backend_e2e_target_task_id"] = serde_json::json!(previous.id);
                }
            }
            let task = self
                .submit_task(SubmitTaskCommand {
                    project_id: project_id.to_string(),
                    parent_id: Some(parent_id.to_string()),
                    title: format!("Backend e2e {agent} step"),
                    description: Some(format!("Matrix runner {agent} step for trace {trace_id}")),
                    kind: kind.to_string(),
                    assigned_agent: Some(agent.to_string()),
                    priority,
                    payload,
                    trace_id: Some(trace_id.to_string()),
                })
                .await?;
            if let Some(previous) = previous {
                self.task_service
                    .add_dependency(TaskDependency {
                        task_id: task.id.clone(),
                        depends_on: previous.id,
                        dep_type: "backend_e2e_chain".into(),
                    })
                    .await?;
            }
            previous = Some(task);
        }
        Ok(())
    }

    pub async fn get_project_state(
        &self,
        project_id: &str,
    ) -> Result<ProjectState, TauriCommandError> {
        commands::get_project_state(
            self.project_service.clone(),
            self.task_service.clone(),
            self.audit_service.clone(),
            self.snapshot_repo.clone(),
            self.metrics_service.clone(),
            project_id,
        )
        .await
    }

    pub async fn export_run_diagnostics(
        &self,
        request: ExportRunDiagnosticsCommand,
    ) -> Result<RunDiagnosticsReport, TauriCommandError> {
        let state = self.get_project_state(&request.project_id).await?;
        let runtime = self.runtime_status.read().await.clone();
        commands::build_run_diagnostics(request, state, runtime)
    }

    pub async fn search_project_context(
        &self,
        request: SearchProjectContextCommand,
    ) -> Result<SearchProjectContextResponse, TauriCommandError> {
        commands::search_project_context(self.embedder.clone(), self.vector_store.clone(), request)
            .await
    }

    pub async fn subscribe_to_events(
        &self,
    ) -> Result<tokio::sync::broadcast::Receiver<crytex_core::bus::Event>, TauriCommandError> {
        commands::subscribe_to_events(self.event_service.clone()).await
    }

    pub async fn runtime_status(&self) -> Result<RuntimeStatus, TauriCommandError> {
        commands::runtime_status(self.runtime_status.read().await.clone()).await
    }

    pub async fn list_ollama_models(&self) -> Result<OllamaModelsResponse, TauriCommandError> {
        let status = self.runtime_status.read().await.clone();
        let ollama_url = status.ollama_url.clone().unwrap_or_else(default_ollama_url);
        commands::list_ollama_models(
            Arc::new(OllamaBackend::new(
                ollama_url.clone(),
                status
                    .active_model
                    .clone()
                    .unwrap_or_else(|| "default".to_string()),
            )),
            ollama_url,
            status.active_model,
        )
        .await
    }

    pub async fn list_managed_models(&self) -> Result<ManagedModelsResponse, TauriCommandError> {
        commands::list_managed_models(self.model_manager.clone()).await
    }

    pub async fn download_managed_model(
        &self,
        request: DownloadManagedModelCommand,
    ) -> Result<ManagedModelRecord, TauriCommandError> {
        let record = commands::download_managed_model(self.model_manager.clone(), request).await?;
        self.event_service.publish(Event::RunObserved {
            project_id: "runtime".to_string(),
            task_id: None,
            trace_id: format!("model-download-{}", record.id),
            action: "managed_model_downloaded".to_string(),
            metadata: serde_json::json!({
                "model_id": record.id.clone(),
                "repo": record.repo.clone(),
                "filename": record.filename.clone(),
                "local_path": record.local_path.clone(),
                "status": record.status.clone(),
                "preferred_backend": record.preferred_backend.clone(),
                "recommended": record.recommended.clone(),
            }),
        });
        Ok(record)
    }

    pub async fn add_managed_model(
        &self,
        request: AddManagedModelCommand,
    ) -> Result<ManagedModelRecord, TauriCommandError> {
        commands::add_managed_model(self.model_manager.clone(), request).await
    }

    pub async fn set_active_managed_model(
        &self,
        request: SetActiveManagedModelCommand,
    ) -> Result<RuntimeStatus, TauriCommandError> {
        let model = self.model_manager.get_model(&request.model_id)?;
        let local_path = model.local_path.clone().ok_or_else(|| {
            TauriCommandError::Bootstrap(format!(
                "managed model {} is not downloaded",
                request.model_id
            ))
        })?;
        let recommended = self.model_manager.recommend_config(&request.model_id)?;
        let inference = mistral_inference_service(
            local_path.display().to_string(),
            recommended.context_size,
            recommended.gpu_layers,
        );
        let active_inference = inference.clone();
        let executor = Arc::new(
            AgentTaskExecutor::new_project_scoped(
                Arc::new(
                    StaticAgentService::with_default_agents(Some(self.context_assembler.clone()))
                        .with_audit(self.audit_service.clone()),
                ),
                inference,
                self.project_service.clone(),
            )
            .with_audit(self.audit_service.clone()),
        ) as Arc<dyn TaskExecutor>;
        let status = managed_model_runtime_status(
            &model,
            local_path.display().to_string(),
            self.runtime_status.read().await.planning_mode.as_str(),
        );

        *self.task_executor.write().await = executor;
        *self.active_inference.write().await = Some(active_inference);
        *self.runtime_status.write().await = status.clone();
        self.event_service.publish(Event::RuntimeSelected {
            backend: "mistralrs".to_string(),
            model_id: model.id.clone(),
            model_path: Some(local_path.display().to_string()),
            endpoint_url: None,
            context_size: Some(recommended.context_size),
            gpu_layers: recommended.gpu_layers,
            quantization: Some(recommended.quantization.as_str().to_string()),
        });
        self.event_service.publish(Event::RunObserved {
            project_id: "runtime".to_string(),
            task_id: None,
            trace_id: format!("model-activate-{}", model.id),
            action: "managed_model_activated".to_string(),
            metadata: serde_json::json!({
                "model_id": model.id,
                "backend": status.active_backend.clone(),
                "model_path": status.active_model.clone(),
                "executor_mode": status.executor_mode.clone(),
                "planning_mode": status.planning_mode.clone(),
                "real_agent_execution": status.real_agent_execution,
                "recommended_context_size": recommended.context_size,
                "recommended_gpu_layers": recommended.gpu_layers,
                "recommended_quantization": recommended.quantization.as_str(),
            }),
        });
        commands::runtime_status(status).await
    }

    pub async fn prove_managed_model_runtime(
        &self,
        request: ProveManagedModelRuntimeCommand,
    ) -> Result<ManagedModelRuntimeProofReport, TauriCommandError> {
        let inference = self.active_inference.read().await.clone().ok_or_else(|| {
            TauriCommandError::Bootstrap(
                "no active inference runtime configured for managed model proof".to_string(),
            )
        })?;
        let runtime = self.runtime_status.read().await.clone();
        let report = commands::prove_managed_model_runtime(
            self.model_manager.clone(),
            inference,
            runtime,
            request,
        )
        .await?;
        self.event_service.publish(Event::RunObserved {
            project_id: "runtime".to_string(),
            task_id: None,
            trace_id: report.trace_id.clone(),
            action: "model_runtime_proved".to_string(),
            metadata: serde_json::json!({
                "model_id": report.model.id.clone(),
                "backend": report.runtime.active_backend.clone(),
                "downloaded": report.downloaded,
                "activated": report.activated,
                "generated": report.generated,
                "passed": report.runtime_probe.passed,
                "failure_reasons": report.failure_reasons.clone(),
                "generated_preview": report.runtime_probe.generated_preview.clone(),
            }),
        });
        Ok(report)
    }

    pub async fn set_active_ollama_model(
        &self,
        ollama_url: &str,
        model: &str,
    ) -> Result<RuntimeStatus, TauriCommandError> {
        let inference = ollama_inference_service(ollama_url.to_string(), model.to_string());
        let active_inference = inference.clone();
        let executor = Arc::new(
            AgentTaskExecutor::new_project_scoped(
                Arc::new(
                    StaticAgentService::with_default_agents(Some(self.context_assembler.clone()))
                        .with_audit(self.audit_service.clone()),
                ),
                inference,
                self.project_service.clone(),
            )
            .with_audit(self.audit_service.clone()),
        ) as Arc<dyn TaskExecutor>;
        let status = ollama_agent_runtime_status(
            ollama_url.to_string(),
            model.to_string(),
            self.runtime_status.read().await.planning_mode.as_str(),
        );

        *self.task_executor.write().await = executor;
        *self.active_inference.write().await = Some(active_inference);
        *self.runtime_status.write().await = status.clone();
        self.event_service.publish(Event::RuntimeSelected {
            backend: "ollama".to_string(),
            model_id: model.to_string(),
            model_path: None,
            endpoint_url: Some(ollama_url.to_string()),
            context_size: None,
            gpu_layers: None,
            quantization: None,
        });
        commands::runtime_status(status).await
    }

    pub async fn set_active_ollama_model_from_command(
        &self,
        request: SetActiveOllamaModelCommand,
    ) -> Result<RuntimeStatus, TauriCommandError> {
        self.set_active_ollama_model(&request.ollama_url, &request.model)
            .await
    }

    async fn record_human_review_feedback(
        &self,
        task: &Task,
        human_score: f64,
        comment: Option<&str>,
        decision: &str,
    ) -> Result<(), TauriCommandError> {
        let text = task.result.as_ref().map(|result| result.to_string());
        let reward = self
            .reward_service
            .record(RecordRewardRequest {
                task_id: &task.id,
                project_id: Some(&task.project_id),
                prompt_version_id: task.prompt_version_id.as_deref(),
                critic_score: task.critic_score,
                human_score: Some(human_score),
                text: text.as_deref(),
                comment,
            })
            .await?;
        let action = format!("human_review_{decision}");
        let _ = self
            .audit_service
            .log(
                crytex_core::services::AuditLogEntry::new("human", action)
                    .project_id(&task.project_id)
                    .task_id(&task.id)
                    .trace_id(&task.trace_id)
                    .level(crytex_core::models::AuditLogLevel::Info)
                    .metadata(serde_json::json!({
                        "human_score": human_score,
                        "reward": reward,
                        "comment": comment,
                        "trace_id": task.trace_id,
                        "prompt_version_id": task.prompt_version_id,
                    })),
            )
            .await;
        Ok(())
    }

    async fn record_lora_feedback_after_review(
        &self,
        task: &Task,
        approved: bool,
    ) -> Result<(), TauriCommandError> {
        if task.kind == "review" || task.assigned_agent.as_deref() == Some("critic") {
            return Ok(());
        }

        if approved {
            self.lora_evolution.collect_golden_example(&task.id).await?;
        } else {
            self.lora_evolution
                .collect_counter_example(&task.id)
                .await?;
            return Ok(());
        }
        if self.lora_evolution.should_train(&task.kind).await? {
            let _ = self.lora_evolution.train_and_register(&task.kind).await?;
        }
        Ok(())
    }

    pub async fn train_lora_adapter(
        &self,
        request: TrainLoraAdapterCommand,
    ) -> Result<TrainLoraAdapterResponse, TauriCommandError> {
        let adapter = if let Some(role) = request.agent_role.as_deref() {
            let role = AgentRole::from_agent(role).ok_or_else(|| {
                TauriCommandError::Bootstrap(format!("unknown LoRA agent role: {role}"))
            })?;
            self.lora_evolution
                .train_and_register_for_role(role)
                .await?
        } else {
            self.lora_evolution
                .train_and_register(&request.task_kind)
                .await?
        };
        let benchmark_gate = adapter.metrics.get("benchmark_gate").cloned();

        Ok(TrainLoraAdapterResponse {
            promoted: adapter.active,
            benchmark_gate,
            metrics: adapter.metrics.clone(),
            adapter,
        })
    }

    pub async fn evaluate_prompt_challenger(
        &self,
        request: EvaluatePromptChallengerCommand,
    ) -> Result<EvaluatePromptChallengerResponse, TauriCommandError> {
        let gate = self.prompt_benchmark_gate.clone().ok_or_else(|| {
            TauriCommandError::Bootstrap(
                "prompt benchmark gate is not configured for this runtime".to_string(),
            )
        })?;
        let baseline = self
            .prompt_evolution
            .active_version(&request.agent)
            .await?
            .ok_or_else(|| {
                TauriCommandError::Bootstrap(format!(
                    "no active baseline prompt for agent: {}",
                    request.agent
                ))
            })?;
        let decision = self
            .prompt_evolution
            .evaluate_challenger_with_benchmark(
                &request.challenger_prompt_version_id,
                gate.as_ref(),
            )
            .await?;
        let versions = self.prompt_evolution.list_versions(&request.agent).await?;
        let challenger = versions
            .iter()
            .find(|version| version.id == request.challenger_prompt_version_id)
            .cloned()
            .ok_or_else(|| {
                TauriCommandError::Bootstrap(format!(
                    "prompt version not found after benchmark: {}",
                    request.challenger_prompt_version_id
                ))
            })?;
        let active = self
            .prompt_evolution
            .active_version(&request.agent)
            .await?
            .ok_or_else(|| {
                TauriCommandError::Bootstrap(format!(
                    "no active prompt after benchmark for agent: {}",
                    request.agent
                ))
            })?;
        let benchmark_gate = challenger
            .metrics
            .get("prompt_benchmark_gate")
            .cloned()
            .unwrap_or_else(|| decision.metadata.clone());
        let action = if decision.accepted {
            "prompt_evolution_promoted"
        } else {
            "prompt_evolution_rejected"
        };
        let trace_id = request
            .trace_id
            .clone()
            .unwrap_or_else(|| format!("prompt-evolution-{}", challenger.id));
        let project_id = request.project_id;
        let task_id = request.task_id;
        let metadata = serde_json::json!({
            "run_id": request.run_id,
            "trace_id": trace_id.clone(),
            "agent": request.agent,
            "baseline_prompt_version_id": baseline.id,
            "challenger_prompt_version_id": challenger.id,
            "prompt_benchmark_gate": benchmark_gate,
        });
        let mut entry = crytex_core::services::AuditLogEntry::new("event", action)
            .project_id(&project_id)
            .trace_id(&trace_id)
            .level(crytex_core::models::AuditLogLevel::Info)
            .metadata(metadata.clone());
        if let Some(task_id) = task_id.as_deref() {
            entry = entry.task_id(task_id);
        }
        self.audit_service
            .log(entry)
            .await
            .map_err(|error| TauriCommandError::Bootstrap(error.to_string()))?;
        self.event_service.publish(Event::RunObserved {
            project_id,
            task_id,
            trace_id: trace_id.clone(),
            action: action.to_string(),
            metadata,
        });

        Ok(EvaluatePromptChallengerResponse {
            promoted: decision.accepted,
            benchmark_gate,
            challenger,
            active,
        })
    }
}

fn matrix_scenarios_or_default(
    scenarios: Vec<BackendE2eScenarioKind>,
) -> Vec<BackendE2eScenarioKind> {
    if scenarios.is_empty() {
        return vec![
            BackendE2eScenarioKind::HappyPath,
            BackendE2eScenarioKind::RejectRemediation,
            BackendE2eScenarioKind::Failure,
        ];
    }
    scenarios
}

fn backend_e2e_gates(
    scenario: &BackendE2eScenarioKind,
    diagnostics: &RunDiagnosticsReport,
) -> Vec<BackendE2eEvidenceGate> {
    let mut gates = common_backend_e2e_gates(diagnostics);
    match scenario {
        BackendE2eScenarioKind::HappyPath => {
            gates.push(gate(
                "human_review_gate",
                !diagnostics.review_task_ids.is_empty(),
                "happy path must stop at human review",
            ));
            gates.push(gate(
                "artifact_lineage",
                !diagnostics.artifact_lineage.is_empty(),
                "happy path must pass typed artifacts between agent tasks",
            ));
            gates.push(gate(
                "human_reward",
                diagnostics.human_reward_recorded,
                "happy path must record human approval reward evidence",
            ));
        }
        BackendE2eScenarioKind::RejectRemediation => {
            gates.push(gate(
                "critic_rejection",
                diagnostics
                    .events
                    .iter()
                    .any(|event| event.action == "critic_rejected"),
                "reject scenario must record critic_rejected",
            ));
            gates.push(gate(
                "critic_feedback",
                !diagnostics.critic_feedback.is_empty(),
                "reject scenario must expose critic feedback",
            ));
            gates.push(gate(
                "remediation_plan",
                diagnostics
                    .events
                    .iter()
                    .any(|event| event.action == "remediation_plan_created"),
                "reject scenario must create remediation plan",
            ));
            gates.push(gate(
                "remediation_events",
                !diagnostics.remediation_events.is_empty(),
                "reject scenario must export remediation diagnostics",
            ));
            gates.push(gate(
                "final_human_review_gate",
                !diagnostics.review_task_ids.is_empty(),
                "remediation path must return to human review",
            ));
        }
        BackendE2eScenarioKind::Failure => {
            gates.push(gate(
                "failed_task_status",
                diagnostics.tasks.iter().any(|task| task.status == "Failed"),
                "failure scenario must persist failed task status",
            ));
            gates.push(gate(
                "failure_event",
                diagnostics
                    .events
                    .iter()
                    .any(|event| event.action == "task_execution_failed"),
                "failure scenario must emit task_execution_failed",
            ));
            gates.push(gate(
                "failure_result_source",
                diagnostics
                    .tasks
                    .iter()
                    .any(|task| task.result_source.as_deref() == Some("task_executor_error")),
                "failure scenario must preserve executor error as task result source",
            ));
        }
    }
    gates
}

fn common_backend_e2e_gates(diagnostics: &RunDiagnosticsReport) -> Vec<BackendE2eEvidenceGate> {
    vec![
        gate(
            "trace_id",
            !diagnostics.trace_ids.is_empty()
                && diagnostics
                    .tasks
                    .iter()
                    .all(|task| diagnostics.trace_ids.contains(&task.trace_id)),
            "diagnostics must include a trace id for every scenario task",
        ),
        gate(
            "run_started",
            diagnostics
                .events
                .iter()
                .any(|event| event.action == "run_started"),
            "diagnostics must include run_started",
        ),
        gate(
            "task_execution_started",
            diagnostics
                .events
                .iter()
                .any(|event| event.action == "task_execution_started"),
            "diagnostics must include task_execution_started",
        ),
        gate(
            "task_execution_finished_or_failed",
            diagnostics.events.iter().any(|event| {
                matches!(
                    event.action.as_str(),
                    "task_execution_finished" | "task_execution_failed"
                )
            }),
            "diagnostics must include task completion or failure evidence",
        ),
    ]
}

fn gate(name: &str, passed: bool, message: &str) -> BackendE2eEvidenceGate {
    BackendE2eEvidenceGate {
        name: name.to_string(),
        passed,
        message: message.to_string(),
    }
}

fn timestamp_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or_default()
}

fn rag_chunks_json(chunks: &[RagChunkEvidence]) -> Vec<Value> {
    chunks
        .iter()
        .map(|chunk| {
            serde_json::json!({
                "id": chunk.id,
                "score": chunk.score,
                "source": chunk.source,
                "relative_path": chunk.relative_path,
                "symbol_id": chunk.symbol_id,
                "related_symbols": chunk.related_symbols,
                "text_preview": chunk.text_preview,
                "retrieval_sources": chunk.retrieval_sources,
                "selection_reason": chunk.selection_reason,
            })
        })
        .collect()
}

/// Attaches the currently promoted LoRA adapter to a task before agent execution.
struct LoraSelectingTaskExecutor {
    inner: Arc<dyn TaskExecutor>,
    task_service: Arc<dyn TaskService>,
    lora_evolution: Arc<dyn LoraEvolutionService>,
}

impl LoraSelectingTaskExecutor {
    fn new(
        inner: Arc<dyn TaskExecutor>,
        task_service: Arc<dyn TaskService>,
        lora_evolution: Arc<dyn LoraEvolutionService>,
    ) -> Self {
        Self {
            inner,
            task_service,
            lora_evolution,
        }
    }

    async fn task_with_selected_lora(&self, task: &Task) -> Result<Task, TauriCommandError> {
        if task.lora_adapter_id.is_some() {
            return Ok(task.clone());
        }

        let Some(adapter_id) = self
            .lora_evolution
            .select_lora(task, &task.project_id)
            .await?
        else {
            return Ok(task.clone());
        };

        let mut task = task.clone();
        task.lora_adapter_id = Some(adapter_id);
        self.task_service.update_task(&task).await?;
        Ok(task)
    }
}

#[async_trait]
impl TaskExecutor for LoraSelectingTaskExecutor {
    async fn execute(&self, task: &Task, run_id: &str) -> Result<Value, TauriCommandError> {
        let task = self.task_with_selected_lora(task).await?;
        self.inner.execute(&task, run_id).await
    }
}

struct NoopLoraInferenceService;

#[async_trait]
impl InferenceService for NoopLoraInferenceService {
    async fn generate(
        &self,
        _request: InferenceRequest,
    ) -> Result<InferenceResponse, InferenceServiceError> {
        Err(InferenceServiceError::UnsupportedOperation(
            "noop lora inference only registers adapters".to_string(),
        ))
    }

    async fn embed(&self, _text: &str) -> Result<Vec<f32>, InferenceServiceError> {
        Err(InferenceServiceError::UnsupportedOperation(
            "noop lora inference does not embed".to_string(),
        ))
    }

    fn available_backends(&self) -> Vec<BackendInfo> {
        Vec::new()
    }

    async fn register_lora(&self, _lora: LoRAAdapter) -> Result<(), InferenceServiceError> {
        Ok(())
    }

    async fn swap_lora(&self, _lora_id: &str) -> Result<(), InferenceServiceError> {
        Ok(())
    }

    async fn list_models(
        &self,
        _backend_id: Option<&str>,
    ) -> Result<Vec<ModelInfo>, InferenceServiceError> {
        Ok(Vec::new())
    }
}

struct RunObservedAuditBridge {
    audit_service: Arc<dyn AuditLogService>,
}

impl RunObservedAuditBridge {
    fn new(audit_service: Arc<dyn AuditLogService>) -> Self {
        Self { audit_service }
    }
}

#[async_trait]
impl EventHandler for RunObservedAuditBridge {
    async fn handle(&self, event: Event) {
        let Event::RunObserved {
            project_id,
            task_id,
            trace_id,
            action,
            mut metadata,
        } = event
        else {
            return;
        };
        metadata["trace_id"] = Value::String(trace_id.clone());

        let mut entry = crytex_core::services::AuditLogEntry::new("event", action)
            .project_id(&project_id)
            .trace_id(&trace_id)
            .level(crytex_core::models::AuditLogLevel::Info)
            .metadata(metadata);
        if let Some(task_id) = task_id {
            entry = entry.task_id(&task_id);
        }
        let _ = self.audit_service.log(entry).await;
    }
}

struct EmptyToolService;

#[async_trait]
impl ToolService for EmptyToolService {
    async fn invoke(&self, name: &str, _args: Value) -> Result<Value, ToolServiceError> {
        Err(ToolServiceError::NotFound(name.to_string()))
    }

    fn list_tools(&self) -> Vec<ToolDescription> {
        vec![]
    }
}

struct StaticAgentService {
    agents: HashMap<String, Arc<dyn Agent>>,
    context_assembler: Option<Arc<ContextAssembler>>,
    audit_service: Option<Arc<dyn AuditLogService>>,
}

impl StaticAgentService {
    fn with_default_agents(context_assembler: Option<Arc<ContextAssembler>>) -> Self {
        let mut agents: HashMap<String, Arc<dyn Agent>> = HashMap::new();
        for agent in [
            Arc::new(ArchitectAgent::new()) as Arc<dyn Agent>,
            Arc::new(CoderAgent::new()) as Arc<dyn Agent>,
            Arc::new(QaAgent::new()) as Arc<dyn Agent>,
            Arc::new(SecurityAgent::new()) as Arc<dyn Agent>,
            Arc::new(CriticAgent::new()) as Arc<dyn Agent>,
            Arc::new(ResearcherAgent::new()) as Arc<dyn Agent>,
            Arc::new(SummarizerAgent::new()) as Arc<dyn Agent>,
        ] {
            agents.insert(agent.name().to_string(), agent);
        }
        Self {
            agents,
            context_assembler,
            audit_service: None,
        }
    }

    fn with_audit(mut self, audit_service: Arc<dyn AuditLogService>) -> Self {
        self.audit_service = Some(audit_service);
        self
    }

    fn default_agent_for_kind(kind: &str) -> &'static str {
        match kind {
            "architecture" | "design" => "architect",
            "research" => "researcher",
            "summarization" => "summarizer",
            "qa" => "qa",
            "security" => "security",
            "review" => "critic",
            _ => "coder",
        }
    }
}

#[async_trait]
impl AgentService for StaticAgentService {
    async fn register(&self, _agent: Arc<dyn Agent>) {}

    async fn find(&self, name: &str) -> Option<Arc<dyn Agent>> {
        self.agents.get(name).cloned()
    }

    async fn list(&self) -> Vec<String> {
        self.agents.keys().cloned().collect()
    }

    fn route(&self, task: &Task) -> Option<String> {
        Some(
            task.assigned_agent
                .clone()
                .unwrap_or_else(|| Self::default_agent_for_kind(&task.kind).to_string()),
        )
    }

    async fn execute(
        &self,
        task: &Task,
        inference: Arc<dyn InferenceService>,
        tools: Arc<dyn ToolService>,
    ) -> Result<Value, AgentServiceError> {
        let agent_name = self
            .route(task)
            .ok_or_else(|| AgentServiceError::AgentNotFound(task.kind.clone()))?;
        let agent = self
            .find(&agent_name)
            .await
            .ok_or_else(|| AgentServiceError::AgentNotFound(agent_name.clone()))?;
        let mut task = task.clone();
        if let Some(assembler) = &self.context_assembler {
            let user_query = format!(
                "{} {}",
                task.title,
                task.description.as_deref().unwrap_or_default()
            );
            let request = crytex_core::services::ContextRequest {
                system_prompt: "Relevant project context for the current task.".into(),
                user_query,
                project_id: Some(task.project_id.clone()),
                history: Vec::new(),
                token_budget: 4_096,
                top_k: 5,
                summarize_threshold_ratio: 0.6,
            };
            if let Ok(assembly) = assembler.assemble_with_evidence(request).await {
                if !assembly.rag.chunks.is_empty() {
                    let chunks = rag_chunks_json(&assembly.rag.chunks);
                    let retrieval_candidates = rag_chunks_json(&assembly.rag.retrieval_candidates);
                    let reranked_chunks = rag_chunks_json(&assembly.rag.reranked_chunks);
                    if let Some(audit_service) = &self.audit_service {
                        let _ = audit_service
                            .log(
                                crytex_core::services::AuditLogEntry::new(
                                    &agent_name,
                                    "rag_context_assembled",
                                )
                                .project_id(&task.project_id)
                                .task_id(&task.id)
                                .trace_id(&task.trace_id)
                                .level(crytex_core::models::AuditLogLevel::Info)
                                .metadata(serde_json::json!({
                                    "query": assembly.rag.query,
                                    "project_id": assembly.rag.project_id,
                                    "trace_id": task.trace_id,
                                    "rerank_applied": assembly.rag.rerank_applied,
                                    "retrieval_candidates": retrieval_candidates,
                                    "reranked_chunks": reranked_chunks,
                                    "chunks": chunks,
                                })),
                            )
                            .await;
                    }
                }

                let context = assembly
                    .messages
                    .iter()
                    .map(|message| format!("{}: {}", message.role, message.content))
                    .collect::<Vec<_>>()
                    .join("\n\n");
                task.payload["assembled_context"] = Value::String(context);
            }
        }

        Ok(agent.execute(&task, inference, tools).await?)
    }
}

fn configured_orchestrator(
    task_service: Arc<dyn TaskService>,
    planning_inference: Option<Arc<dyn InferenceService>>,
    planning_tools: Option<Arc<dyn ToolService>>,
) -> Arc<dyn Orchestrator> {
    let orchestrator = OrchestratorImpl::new(task_service);
    match (planning_inference, planning_tools) {
        (Some(inference), Some(tools)) => Arc::new(
            orchestrator
                .with_planning_agent(Arc::new(ArchitectAgent::new()))
                .with_inference(inference)
                .with_tools(tools),
        ),
        _ => Arc::new(orchestrator),
    }
}

fn configured_planning_inference_service() -> Option<Arc<dyn InferenceService>> {
    env::var("CRYTEX_TAURI_OLLAMA_MODEL")
        .ok()
        .filter(|model| !model.trim().is_empty())
        .map(|model| {
            let url = env::var("CRYTEX_TAURI_OLLAMA_URL").unwrap_or_else(|_| default_ollama_url());
            ollama_inference_service(url, model)
        })
}

fn configured_reranker_from_env()
-> Result<Option<Arc<dyn crytex_core::services::Reranker>>, TauriCommandError> {
    configured_reranker_from_model_name(env::var("CRYTEX_TAURI_RERANK_MODEL").ok())
}

fn configured_reranker_from_model_name(
    model_name: Option<String>,
) -> Result<Option<Arc<dyn crytex_core::services::Reranker>>, TauriCommandError> {
    let Some(model_name) = model_name.map(|value| value.trim().to_string()) else {
        return Ok(None);
    };
    if model_name.is_empty() {
        return Ok(None);
    }

    let reranker = crytex_inference_onnx::OnnxReranker::from_name(&model_name)
        .map_err(|err| TauriCommandError::Bootstrap(err.to_string()))?;
    Ok(Some(Arc::new(reranker)))
}

fn runtime_status_from_env() -> RuntimeStatus {
    let status = env::var("CRYTEX_TAURI_OLLAMA_MODEL")
        .ok()
        .filter(|model| !model.trim().is_empty())
        .map(|model| {
            let url = env::var("CRYTEX_TAURI_OLLAMA_URL").unwrap_or_else(|_| default_ollama_url());
            ollama_agent_runtime_status(url, model, "ollama_agent")
        })
        .unwrap_or_else(stub_runtime_status);
    with_cuda_toolchain_status(status)
}

fn with_cuda_toolchain_status(mut status: RuntimeStatus) -> RuntimeStatus {
    status.cuda_toolchain = Some(detect_cuda_toolchain_status(&SystemHardwareDetector::new()));
    status
}

fn default_ollama_url() -> String {
    "http://localhost:11434".to_string()
}

fn stub_runtime_status() -> RuntimeStatus {
    RuntimeStatus {
        tauri_runtime: true,
        executor_mode: "stub".to_string(),
        planning_mode: "deterministic".to_string(),
        active_backend: None,
        active_model: None,
        ollama_url: None,
        real_agent_execution: false,
        ready_to_run: false,
        missing_requirements: vec!["no inference backend configured".to_string()],
        backend_capabilities: vec![],
        cuda_toolchain: None,
        compatibility_notes: vec![],
        model_compatibility: None,
    }
}

fn custom_executor_runtime_status() -> RuntimeStatus {
    RuntimeStatus {
        tauri_runtime: true,
        executor_mode: "custom_executor".to_string(),
        planning_mode: "deterministic".to_string(),
        active_backend: None,
        active_model: None,
        ollama_url: None,
        real_agent_execution: true,
        ready_to_run: true,
        missing_requirements: vec![],
        backend_capabilities: vec![],
        cuda_toolchain: None,
        compatibility_notes: vec![],
        model_compatibility: None,
    }
}

fn ollama_agent_runtime_status(
    ollama_url: String,
    model: String,
    planning_mode: &str,
) -> RuntimeStatus {
    RuntimeStatus {
        tauri_runtime: true,
        executor_mode: "ollama_agent".to_string(),
        planning_mode: planning_mode.to_string(),
        active_backend: Some("ollama".to_string()),
        active_model: Some(model),
        ollama_url: Some(ollama_url),
        real_agent_execution: true,
        ready_to_run: true,
        missing_requirements: vec![],
        backend_capabilities: vec![backend_capability_report(
            "ollama",
            "Ollama",
            &["generate", "chat"],
        )],
        cuda_toolchain: None,
        compatibility_notes: vec![],
        model_compatibility: None,
    }
}

fn managed_model_runtime_status(
    model: &crytex_core::services::ManagedModel,
    model_path: String,
    planning_mode: &str,
) -> RuntimeStatus {
    let is_gguf = is_gguf_runtime_model(&model_path);
    let capabilities = if is_gguf {
        vec!["generate", "chat", "gguf"]
    } else {
        vec!["generate", "chat", "lora"]
    };
    let compatibility_notes = mistral_runtime_compatibility_notes(is_gguf);
    let detector = SystemHardwareDetector::new();
    let device = detector.detect();
    let model_compatibility = Some(ModelCompatibilityPlanner::plan(
        model,
        &device,
        &RuntimeFeatureSet {
            cuda_available: matches!(device, DeviceKind::Cuda { .. }),
            metal_available: matches!(device, DeviceKind::Metal { .. }),
            gdn_cuda_available: MistralRsBackend::cuda_gdn_kernel_available(),
            cuda_unquantized_moe_fallback_available: true,
        },
    ));
    RuntimeStatus {
        tauri_runtime: true,
        executor_mode: "mistralrs_agent".to_string(),
        planning_mode: planning_mode.to_string(),
        active_backend: Some("mistralrs".to_string()),
        active_model: Some(model_path),
        ollama_url: None,
        real_agent_execution: true,
        ready_to_run: true,
        missing_requirements: vec![],
        backend_capabilities: vec![backend_capability_report(
            "mistralrs",
            "mistral.rs",
            &capabilities,
        )],
        cuda_toolchain: None,
        compatibility_notes,
        model_compatibility,
    }
}

fn mistral_runtime_compatibility_notes(is_gguf: bool) -> Vec<commands::RuntimeCompatibilityNote> {
    if !is_gguf {
        return vec![];
    }

    if MistralRsBackend::cuda_gdn_kernel_available() {
        return vec![commands::RuntimeCompatibilityNote {
            code: "mistralrs_cuda_gdn_kernel".to_string(),
            severity: "info".to_string(),
            message: "Mistral GGUF CUDA can compile and call the full CUDA kernel set, including GDN kernels; the tiny Qwen3-Next GDN smoke is proven on GPU, while broader model-family compatibility is still reported by model/runtime diagnostics.".to_string(),
        }];
    }

    vec![commands::RuntimeCompatibilityNote {
        code: "mistralrs_cuda_gdn_kernel".to_string(),
        severity: "warning".to_string(),
        message: "Mistral GGUF CUDA is running in non-GDN compatibility mode: normal GGUF generation works on GPU, but this build does not include the patched GDN CUDA kernel path.".to_string(),
    }]
}

fn backend_capability_report(
    id: &str,
    name: &str,
    capabilities: &[&str],
) -> BackendCapabilityReport {
    BackendInfo {
        id: id.to_string(),
        name: name.to_string(),
        capabilities: capabilities
            .iter()
            .map(|capability| capability.to_string())
            .collect(),
    }
    .capability_report()
}

fn is_gguf_runtime_model(model_path: &str) -> bool {
    let path = Path::new(model_path);
    if path
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
    {
        return true;
    }

    path.is_dir()
        && std::fs::read_dir(path)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(Result::ok)
            .any(|entry| {
                entry
                    .path()
                    .extension()
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
            })
}

fn mistral_inference_service(
    model_path: impl Into<String>,
    context_size: usize,
    gpu_layers: Option<usize>,
) -> Arc<dyn InferenceService> {
    let backend = Arc::new(MistralRsBackend::new(
        model_path.into(),
        context_size,
        gpu_layers,
    ));
    let mut registry = BackendRegistry::new("mistralrs");
    registry.register("mistralrs", backend);
    Arc::new(InferenceServiceImpl::new(
        Arc::new(registry),
        Some("mistralrs".to_string()),
    ))
}

fn configured_task_executor(
    explicit_executor: Option<Arc<dyn TaskExecutor>>,
    project_service: Arc<dyn ProjectService>,
    audit_service: Arc<dyn AuditLogService>,
    context_assembler: Arc<ContextAssembler>,
) -> Arc<dyn TaskExecutor> {
    if let Some(executor) = explicit_executor {
        return executor;
    }

    env::var("CRYTEX_TAURI_OLLAMA_MODEL")
        .ok()
        .filter(|model| !model.trim().is_empty())
        .map(|model| {
            let url = env::var("CRYTEX_TAURI_OLLAMA_URL").unwrap_or_else(|_| default_ollama_url());
            let inference = ollama_inference_service(url, model);
            Arc::new(
                AgentTaskExecutor::new_project_scoped(
                    Arc::new(StaticAgentService::with_default_agents(Some(
                        context_assembler,
                    ))),
                    inference,
                    project_service,
                )
                .with_audit(audit_service),
            ) as Arc<dyn TaskExecutor>
        })
        .unwrap_or_else(|| Arc::new(StubTaskExecutor))
}

fn vector_store_path_for_db(db_path: &Path) -> PathBuf {
    match db_path.parent() {
        Some(parent) => parent.join("crytex-vectors"),
        None => PathBuf::from("crytex-vectors"),
    }
}

fn ollama_inference_service(
    ollama_url: impl Into<String>,
    model: impl Into<String>,
) -> Arc<dyn InferenceService> {
    let backend = Arc::new(OllamaBackend::new(ollama_url.into(), model.into()));
    let mut registry = BackendRegistry::new("ollama");
    registry.register("ollama", backend);
    Arc::new(InferenceServiceImpl::new(
        Arc::new(registry),
        Some("ollama".to_string()),
    ))
}

async fn repair_stub_completed_tasks(
    task_service: Arc<dyn TaskService>,
) -> Result<(), TauriCommandError> {
    let tasks = task_service.load_all_tasks().await?;
    for mut task in tasks {
        let is_stub_completed = task.status == crytex_core::models::TaskStatus::Completed
            && task
                .result
                .as_ref()
                .and_then(|result| result.get("source"))
                .and_then(|source| source.as_str())
                == Some("tauri_stub_run");
        if is_stub_completed {
            task.status = crytex_core::models::TaskStatus::Review;
            task.finished_at = None;
            task_service.update_task(&task).await?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use crytex_bench::{
        BenchLoraBenchmarkGate, DefaultBenchmarkHarness, ExactMatchScorer,
        repository::MemoryBenchmarkResultRepository,
        runner::{BenchmarkRunOutput, BenchmarkRunner},
    };
    use crytex_core::models::{BenchmarkCase, BenchmarkVariant};
    use crytex_core::persistence::BenchmarkResultRepository;
    use crytex_core::services::{
        AgentRole, InferenceService, InferenceServiceError, LoraBenchmarkDecision,
        LoraBenchmarkRequest, LoraEvolutionError, ManagedModel, ManifestEntry, ModelManagerError,
        ModelStatus, PromptBenchmarkDecision, PromptBenchmarkGate, PromptBenchmarkRequest,
        Quantization, RecommendedConfig, ToolDescription, ToolService, ToolServiceError,
    };
    use crytex_inference::{
        BackendInfo, InferenceRequest, InferenceResponse, LoRAAdapter, ModelInfo, TokenUsage,
    };
    use serde_json::json;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn runtime_status_includes_cuda_toolchain_preflight() {
        let status = with_cuda_toolchain_status(stub_runtime_status());

        assert!(status.cuda_toolchain.is_some());
    }

    #[test]
    fn managed_gguf_runtime_status_reports_cuda_gdn_compatibility() {
        let model = ManagedModel {
            id: "tinyllama-q2".into(),
            name: "TinyLlama Q2".into(),
            repo: Some("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF".into()),
            filename: Some("tinyllama-1.1b-chat-v1.0.Q2_K.gguf".into()),
            local_path: Some(PathBuf::from("C:\\models\\tinyllama.gguf")),
            quantization: None,
            preferred_backend: crytex_core::config::BackendKind::MistralRs,
            params_b: Some(1.1),
            status: ModelStatus::Downloaded,
        };
        let status =
            managed_model_runtime_status(&model, "C:\\models\\tinyllama.gguf".into(), "test");

        assert!(status.compatibility_notes.iter().any(|note| {
            note.code == "mistralrs_cuda_gdn_kernel"
                && note.message.contains("GDN")
                && note.message.contains("GGUF")
        }));
        assert!(status.model_compatibility.is_some());
        assert_eq!(
            status.model_compatibility.unwrap().format,
            crytex_core::services::ModelFormat::Gguf
        );
    }

    #[tokio::test]
    async fn backend_e2e_matrix_runner_proves_happy_reject_remediation_and_failure() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("crytex-backend-e2e-matrix.db");
        let state = CrytexAppState::new_sqlite_with_executor(&db_path, Arc::new(StubTaskExecutor))
            .await
            .unwrap();

        let project_root = dir.path().join("project");
        std::fs::create_dir_all(&project_root).unwrap();
        let project = state
            .create_project(CreateProjectCommand {
                name: "Backend E2E Matrix".into(),
                root_path: project_root.display().to_string(),
            })
            .await
            .unwrap();

        let report = state
            .run_backend_e2e_matrix(BackendE2eMatrixCommand {
                project_id: project.id.clone(),
                trace_id: Some("trace-backend-e2e-matrix".into()),
                max_steps: 20,
                scenarios: vec![],
            })
            .await
            .unwrap();

        assert!(report.passed, "{:#?}", report.scenarios);
        assert_eq!(report.scenarios.len(), 3);
        for scenario in &report.scenarios {
            assert!(
                scenario.gates.iter().all(|gate| gate.passed),
                "{:?} gates failed: {:#?}",
                scenario.scenario,
                scenario.gates
            );
            assert_eq!(
                scenario.diagnostics.trace_ids,
                vec![scenario.trace_id.clone()]
            );
            assert!(!scenario.diagnostics.tasks.is_empty());
        }
        assert!(report.scenarios.iter().any(|scenario| {
            scenario.scenario == BackendE2eScenarioKind::HappyPath
                && scenario.diagnostics.human_reward_recorded
                && !scenario.diagnostics.artifact_lineage.is_empty()
        }));
        assert!(report.scenarios.iter().any(|scenario| {
            scenario.scenario == BackendE2eScenarioKind::RejectRemediation
                && scenario
                    .diagnostics
                    .events
                    .iter()
                    .any(|event| event.action == "critic_rejected")
                && scenario
                    .diagnostics
                    .events
                    .iter()
                    .any(|event| event.action == "remediation_plan_created")
        }));
        assert!(report.scenarios.iter().any(|scenario| {
            scenario.scenario == BackendE2eScenarioKind::Failure
                && scenario
                    .diagnostics
                    .events
                    .iter()
                    .any(|event| event.action == "task_execution_failed")
                && scenario
                    .diagnostics
                    .tasks
                    .iter()
                    .any(|task| task.status == "Failed")
        }));

        state.shutdown_project_watchers().await;
    }

    #[tokio::test]
    async fn backend_e2e_matrix_runner_requires_isolated_ready_queue() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("crytex-backend-e2e-matrix-guard.db");
        let state = CrytexAppState::new_sqlite_with_executor(&db_path, Arc::new(StubTaskExecutor))
            .await
            .unwrap();

        let project_root = dir.path().join("project");
        std::fs::create_dir_all(&project_root).unwrap();
        let project = state
            .create_project(CreateProjectCommand {
                name: "Backend E2E Matrix Guard".into(),
                root_path: project_root.display().to_string(),
            })
            .await
            .unwrap();
        let existing = state
            .submit_task(SubmitTaskCommand {
                project_id: project.id.clone(),
                parent_id: None,
                title: "Existing ready task".into(),
                description: None,
                kind: "codegen".into(),
                assigned_agent: Some("coder".into()),
                priority: 1,
                payload: json!({}),
                trace_id: Some("trace-existing-ready".into()),
            })
            .await
            .unwrap();

        let error = state
            .run_backend_e2e_matrix(BackendE2eMatrixCommand {
                project_id: project.id.clone(),
                trace_id: Some("trace-backend-e2e-matrix-guard".into()),
                max_steps: 20,
                scenarios: vec![BackendE2eScenarioKind::HappyPath],
            })
            .await
            .unwrap_err();

        assert!(error.to_string().contains("isolated project"));
        assert!(error.to_string().contains(&existing.id));
        state.shutdown_project_watchers().await;
    }

    #[test]
    fn configured_reranker_from_model_name_builds_onnx_backend_without_eager_loading() {
        let reranker =
            configured_reranker_from_model_name(Some("BAAI/bge-reranker-base".to_string()))
                .expect("known reranker should configure")
                .expect("reranker should be enabled");

        assert_eq!(Arc::strong_count(&reranker), 1);
    }

    struct DownloadableManagedModelManager {
        model: Mutex<Option<ManagedModel>>,
        local_path: PathBuf,
    }

    impl DownloadableManagedModelManager {
        fn new(local_path: PathBuf) -> Self {
            Self {
                model: Mutex::new(None),
                local_path,
            }
        }

        fn current_model(&self) -> Result<ManagedModel, ModelManagerError> {
            self.model
                .lock()
                .unwrap()
                .clone()
                .ok_or_else(|| ModelManagerError::NotFound("managed model".into()))
        }
    }

    #[async_trait]
    impl ModelManager for DownloadableManagedModelManager {
        fn list_models(&self) -> Result<Vec<ManagedModel>, ModelManagerError> {
            Ok(self.model.lock().unwrap().clone().into_iter().collect())
        }

        fn get_model(&self, id: &str) -> Result<ManagedModel, ModelManagerError> {
            self.current_model().and_then(|model| {
                (model.id == id)
                    .then_some(model)
                    .ok_or_else(|| ModelManagerError::NotFound(id.into()))
            })
        }

        async fn download_model(&self, id: &str) -> Result<ManagedModel, ModelManagerError> {
            let mut model = self.get_model(id)?;
            if let Some(parent) = self.local_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&self.local_path, b"mock gguf model")?;
            model.local_path = Some(self.local_path.clone());
            model.status = ModelStatus::Downloaded;
            *self.model.lock().unwrap() = Some(model.clone());
            Ok(model)
        }

        fn add_model(&self, entry: ManifestEntry) -> Result<ManagedModel, ModelManagerError> {
            let id = entry
                .id
                .ok_or_else(|| ModelManagerError::Download("id required".into()))?;
            let model = ManagedModel {
                id,
                name: entry.name.unwrap_or_else(|| "unnamed".into()),
                repo: entry.repo,
                filename: entry.filename,
                local_path: None,
                quantization: entry
                    .quantization
                    .as_deref()
                    .and_then(|value| value.parse().ok()),
                preferred_backend: crytex_core::config::BackendKind::MistralRs,
                params_b: entry.params_b,
                status: ModelStatus::Available,
            };
            *self.model.lock().unwrap() = Some(model.clone());
            Ok(model)
        }

        fn recommend_config(&self, id: &str) -> Result<RecommendedConfig, ModelManagerError> {
            let model = self.get_model(id)?;
            Ok(RecommendedConfig {
                backend: model.preferred_backend,
                quantization: model.quantization.unwrap_or(Quantization::Q4KM),
                gpu_layers: None,
                context_size: 8192,
            })
        }
    }

    struct PlanningInference;

    #[async_trait]
    impl InferenceService for PlanningInference {
        async fn generate(
            &self,
            _request: InferenceRequest,
        ) -> Result<InferenceResponse, InferenceServiceError> {
            Ok(InferenceResponse {
                content: serde_json::json!({
                    "summary": "Two generated tasks",
                    "plan": {
                        "subtasks": [
                            {
                                "title": "Inspect project",
                                "description": "Read the project shape",
                                "kind": "research",
                                "agent": "researcher",
                                "prompt": "Inspect the project"
                            },
                            {
                                "title": "Implement change",
                                "description": "Apply the requested change",
                                "kind": "codegen",
                                "agent": "coder",
                                "prompt": "Implement the change"
                            }
                        ]
                    }
                })
                .to_string(),
                usage: TokenUsage {
                    prompt_tokens: 10,
                    completion_tokens: 20,
                    total_tokens: 30,
                },
                finish_reason: "stop".into(),
            })
        }

        async fn embed(&self, _text: &str) -> Result<Vec<f32>, InferenceServiceError> {
            Ok(vec![])
        }

        fn available_backends(&self) -> Vec<BackendInfo> {
            vec![]
        }

        async fn register_lora(&self, _lora: LoRAAdapter) -> Result<(), InferenceServiceError> {
            Ok(())
        }

        async fn swap_lora(&self, _lora_id: &str) -> Result<(), InferenceServiceError> {
            Ok(())
        }

        async fn list_models(
            &self,
            _backend_id: Option<&str>,
        ) -> Result<Vec<ModelInfo>, InferenceServiceError> {
            Ok(vec![])
        }
    }

    struct CapturingInference {
        requests: Arc<Mutex<Vec<InferenceRequest>>>,
    }

    impl CapturingInference {
        fn new(requests: Arc<Mutex<Vec<InferenceRequest>>>) -> Self {
            Self { requests }
        }
    }

    #[async_trait]
    impl InferenceService for CapturingInference {
        async fn generate(
            &self,
            request: InferenceRequest,
        ) -> Result<InferenceResponse, InferenceServiceError> {
            self.requests.lock().unwrap().push(request);
            Ok(InferenceResponse {
                content: json!({
                    "files_changed": [],
                    "test_results": null,
                    "summary": "captured request"
                })
                .to_string(),
                usage: TokenUsage {
                    prompt_tokens: 10,
                    completion_tokens: 5,
                    total_tokens: 15,
                },
                finish_reason: "stop".into(),
            })
        }

        async fn embed(&self, _text: &str) -> Result<Vec<f32>, InferenceServiceError> {
            Ok(vec![])
        }

        fn available_backends(&self) -> Vec<BackendInfo> {
            vec![]
        }

        async fn register_lora(&self, _lora: LoRAAdapter) -> Result<(), InferenceServiceError> {
            Ok(())
        }

        async fn swap_lora(&self, _lora_id: &str) -> Result<(), InferenceServiceError> {
            Ok(())
        }

        async fn list_models(
            &self,
            _backend_id: Option<&str>,
        ) -> Result<Vec<ModelInfo>, InferenceServiceError> {
            Ok(vec![])
        }
    }

    struct RuntimeProofInference;

    #[async_trait]
    impl InferenceService for RuntimeProofInference {
        async fn generate(
            &self,
            _request: InferenceRequest,
        ) -> Result<InferenceResponse, InferenceServiceError> {
            Ok(InferenceResponse {
                content: "CRYTEX_PROBE_OK managed model generated through mistral.rs".into(),
                usage: TokenUsage {
                    prompt_tokens: 11,
                    completion_tokens: 7,
                    total_tokens: 18,
                },
                finish_reason: "stop".into(),
            })
        }

        async fn embed(&self, _text: &str) -> Result<Vec<f32>, InferenceServiceError> {
            Ok(vec![])
        }

        fn available_backends(&self) -> Vec<BackendInfo> {
            vec![BackendInfo {
                id: "mistralrs".into(),
                name: "mistral.rs".into(),
                capabilities: vec!["generate".into(), "chat".into()],
            }]
        }

        async fn register_lora(&self, _lora: LoRAAdapter) -> Result<(), InferenceServiceError> {
            Ok(())
        }

        async fn swap_lora(&self, _lora_id: &str) -> Result<(), InferenceServiceError> {
            Ok(())
        }

        async fn list_models(
            &self,
            _backend_id: Option<&str>,
        ) -> Result<Vec<ModelInfo>, InferenceServiceError> {
            Ok(vec![])
        }
    }

    #[derive(Debug)]
    struct PreferRerankTarget;

    #[async_trait]
    impl crytex_core::services::Reranker for PreferRerankTarget {
        async fn rerank(
            &self,
            _query: &str,
            passages: &[crytex_core::services::RerankPassage],
        ) -> Result<Vec<crytex_core::services::RerankResult>, crytex_core::services::RerankerError>
        {
            let mut results = passages
                .iter()
                .map(|passage| crytex_core::services::RerankResult {
                    id: passage.id.clone(),
                    score: if passage.text.contains("RERANK_TARGET_CONTEXT") {
                        10.0
                    } else {
                        1.0
                    },
                    text: passage.text.clone(),
                    payload: passage.payload.clone(),
                })
                .collect::<Vec<_>>();
            results.sort_by(|left, right| right.score.partial_cmp(&left.score).unwrap());
            Ok(results)
        }
    }

    struct NoopToolService;

    #[async_trait]
    impl ToolService for NoopToolService {
        async fn invoke(
            &self,
            _name: &str,
            _args: serde_json::Value,
        ) -> Result<serde_json::Value, ToolServiceError> {
            Ok(serde_json::Value::Null)
        }

        fn list_tools(&self) -> Vec<ToolDescription> {
            vec![]
        }
    }

    #[derive(Default)]
    struct RecordingLoraEvolution {
        should_train: AtomicBool,
        collected: Mutex<Vec<String>>,
        counter_examples: Mutex<Vec<String>>,
        trained: Mutex<Vec<String>>,
    }

    impl RecordingLoraEvolution {
        fn new(should_train: bool) -> Self {
            Self {
                should_train: AtomicBool::new(should_train),
                collected: Mutex::new(Vec::new()),
                counter_examples: Mutex::new(Vec::new()),
                trained: Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait]
    impl LoraEvolutionService for RecordingLoraEvolution {
        async fn collect_golden_example(&self, task_id: &str) -> Result<(), LoraEvolutionError> {
            self.collected.lock().unwrap().push(task_id.to_string());
            Ok(())
        }

        async fn collect_counter_example(&self, task_id: &str) -> Result<(), LoraEvolutionError> {
            self.counter_examples
                .lock()
                .unwrap()
                .push(task_id.to_string());
            Ok(())
        }

        async fn should_train(&self, _task_kind: &str) -> Result<bool, LoraEvolutionError> {
            Ok(self.should_train.load(Ordering::SeqCst))
        }

        async fn train_and_register(
            &self,
            task_kind: &str,
        ) -> Result<crytex_core::models::LoraAdapter, LoraEvolutionError> {
            self.trained.lock().unwrap().push(task_kind.to_string());
            Ok(crytex_core::models::LoraAdapter {
                id: format!("{task_kind}-mock-lora"),
                project_id: None,
                name: format!("{task_kind}-mock-lora"),
                file_path: "mock.safetensors".into(),
                base_model: "mock-base".into(),
                task_kind: Some(task_kind.to_string()),
                agent_role: None,
                metrics: json!({
                    "benchmark_gate": {
                        "accepted": true,
                        "reason": "recording gate accepted"
                    }
                }),
                created_at: 0,
                active: true,
            })
        }

        async fn should_train_for_role(
            &self,
            _role: AgentRole,
        ) -> Result<bool, LoraEvolutionError> {
            Ok(false)
        }

        async fn train_and_register_for_role(
            &self,
            role: AgentRole,
        ) -> Result<crytex_core::models::LoraAdapter, LoraEvolutionError> {
            Err(LoraEvolutionError::ValidationFailed(
                role.as_str().to_string(),
                "role training not used in this test".to_string(),
            ))
        }

        async fn select_lora(
            &self,
            _task: &Task,
            _project_id: &str,
        ) -> Result<Option<String>, LoraEvolutionError> {
            Ok(None)
        }

        async fn select_lora_by_role(
            &self,
            _role: AgentRole,
            _project_id: &str,
        ) -> Result<Option<String>, LoraEvolutionError> {
            Ok(None)
        }
    }

    #[derive(Default)]
    struct AcceptingPromptBenchmarkGate {
        requests: Mutex<Vec<PromptBenchmarkRequest>>,
    }

    #[async_trait]
    impl PromptBenchmarkGate for AcceptingPromptBenchmarkGate {
        async fn evaluate(
            &self,
            request: PromptBenchmarkRequest,
        ) -> Result<PromptBenchmarkDecision, crytex_core::services::PromptEvolutionError> {
            self.requests.lock().unwrap().push(request.clone());
            Ok(PromptBenchmarkDecision {
                accepted: true,
                reason: "challenger passed held-out prompt benchmark".into(),
                baseline_score: 0.25,
                challenger_score: 0.75,
                metadata: json!({
                    "held_out": true,
                    "baseline_run_id": "prompt-baseline-run",
                    "challenger_run_id": "prompt-challenger-run",
                    "winner": "Challenger",
                    "delta_pass_rate": 0.5,
                    "mc_nemar_p_value": 0.03125,
                    "baseline_pass_rate": 0.25,
                    "challenger_pass_rate": 0.75
                }),
            })
        }
    }

    #[derive(Default)]
    struct AcceptingLoraBenchmarkGate {
        requests: Mutex<Vec<LoraBenchmarkRequest>>,
    }

    #[async_trait]
    impl LoraBenchmarkGate for AcceptingLoraBenchmarkGate {
        async fn evaluate(
            &self,
            request: LoraBenchmarkRequest,
        ) -> Result<LoraBenchmarkDecision, LoraEvolutionError> {
            self.requests.lock().unwrap().push(request);
            Ok(LoraBenchmarkDecision {
                accepted: true,
                reason: "challenger passed held-out benchmark".into(),
                metadata: json!({
                    "baseline_run_id": "bench-baseline-product",
                    "challenger_run_id": "bench-challenger-product",
                    "winner": "Challenger",
                    "mc_nemar_p_value": 0.03125,
                    "baseline_pass_rate": 0.4,
                    "challenger_pass_rate": 0.8
                }),
            })
        }
    }

    struct HeldOutSensitiveBenchmarkRunner;

    #[async_trait]
    impl BenchmarkRunner for HeldOutSensitiveBenchmarkRunner {
        async fn run(
            &self,
            case: &BenchmarkCase,
            variant: &BenchmarkVariant,
        ) -> Result<BenchmarkRunOutput, crytex_bench::BenchError> {
            let answer = if variant.lora_adapter_id.as_deref() == Some("codegen-v1") {
                case.expected
                    .as_ref()
                    .cloned()
                    .unwrap_or_else(|| json!({ "answer": "missing expected answer" }))
            } else {
                json!({ "answer": "baseline missed held-out answer" })
            };
            Ok(BenchmarkRunOutput {
                task_id: None,
                result: answer,
                latency_ms: 1,
                token_usage: None,
            })
        }
    }

    async fn write_held_out_lora_golden_set(dir: &Path) -> PathBuf {
        let path = dir.join("heldout-lora.jsonl");
        let lines = (0..6)
            .map(|idx| {
                json!({
                    "id": format!("heldout-{idx}"),
                    "input": {
                        "prompt": format!("solve isolated held-out benchmark case {idx}")
                    },
                    "expected": {
                        "answer": format!("correct isolated held-out answer {idx}")
                    },
                    "tags": ["heldout", "lora"],
                    "metadata": {
                        "source": "tauri-product-path-test"
                    }
                })
                .to_string()
            })
            .collect::<Vec<_>>()
            .join("\n");
        tokio::fs::write(&path, lines).await.unwrap();
        path
    }

    struct ChainExecutor;

    #[async_trait]
    impl TaskExecutor for ChainExecutor {
        async fn execute(
            &self,
            task: &Task,
            run_id: &str,
        ) -> Result<serde_json::Value, TauriCommandError> {
            let upstream = task
                .payload
                .get("upstream_artifacts")
                .and_then(|value| value.as_array())
                .cloned()
                .unwrap_or_default();
            let decision = if task.assigned_agent.as_deref() == Some("critic") {
                serde_json::json!({
                    "review_decision": "pass",
                    "checked_artifact_count": upstream.len(),
                    "summary": "critic passed chain to human review"
                })
            } else {
                serde_json::json!({
                    "artifact": valid_role_artifact(task, upstream.len())
                })
            };
            Ok(serde_json::json!({
                "source": "chain_executor",
                "run_id": run_id,
                "task_id": task.id,
                "agent": task.assigned_agent,
                "kind": task.kind,
                "upstream_artifacts_seen": upstream,
                "agent_result": decision
            }))
        }
    }

    struct RejectThenPassChainExecutor {
        rejected_once: AtomicBool,
    }

    impl RejectThenPassChainExecutor {
        fn new() -> Self {
            Self {
                rejected_once: AtomicBool::new(false),
            }
        }
    }

    #[async_trait]
    impl TaskExecutor for RejectThenPassChainExecutor {
        async fn execute(
            &self,
            task: &Task,
            run_id: &str,
        ) -> Result<serde_json::Value, TauriCommandError> {
            let upstream = task
                .payload
                .get("upstream_artifacts")
                .and_then(|value| value.as_array())
                .cloned()
                .unwrap_or_default();
            let agent_result = if task.assigned_agent.as_deref() == Some("critic")
                && !self.rejected_once.swap(true, Ordering::SeqCst)
            {
                let target_task_id = upstream
                    .iter()
                    .find(|artifact| artifact["agent"] == "coder")
                    .and_then(|artifact| artifact["task_id"].as_str())
                    .expect("critic should see coder artifact");
                json!({
                    "review_decision": "reject",
                    "target_task_id": target_task_id,
                    "blocking_issues": [{
                        "severity": "high",
                        "reason": "coder artifact missed the acceptance criteria",
                        "expected": "return a corrected patch artifact"
                    }],
                    "feedback": "coder artifact missed the acceptance criteria"
                })
            } else if task.assigned_agent.as_deref() == Some("critic") {
                json!({
                    "review_decision": "pass",
                    "checked_artifact_count": upstream.len(),
                    "summary": "critic passed corrected chain to human review"
                })
            } else {
                json!({
                    "artifact": valid_role_artifact(task, upstream.len())
                })
            };
            Ok(json!({
                "source": "reject_then_pass_chain_executor",
                "run_id": run_id,
                "task_id": task.id,
                "agent": task.assigned_agent,
                "upstream_artifacts_seen": upstream,
                "agent_result": agent_result
            }))
        }
    }

    fn valid_role_artifact(task: &Task, upstream_count: usize) -> serde_json::Value {
        match task.assigned_agent.as_deref() {
            Some("coder") => json!({
                "producer": "coder",
                "files_changed": ["src/lib.rs"],
                "summary": "implemented the requested behavior",
                "test_results": "cargo test passed",
                "upstream_count": upstream_count
            }),
            Some("qa") => json!({
                "producer": "qa",
                "summary": "verified the patch against acceptance criteria",
                "test_results": "targeted tests passed",
                "upstream_count": upstream_count
            }),
            Some("security") => json!({
                "producer": "security",
                "summary": "reviewed the patch for unsafe behavior",
                "risk": "low",
                "upstream_count": upstream_count
            }),
            Some("architect") => json!({
                "producer": "architect",
                "summary": "designed an atomic implementation plan",
                "content": "architecture artifact",
                "upstream_count": upstream_count
            }),
            _ => json!({
                "producer": task.assigned_agent,
                "summary": format!("{} artifact", task.kind),
                "upstream_count": upstream_count
            }),
        }
    }

    #[tokio::test]
    async fn sqlite_state_reports_stub_runtime_when_no_model_executor_is_configured() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("crytex-ui.db");
        let state = CrytexAppState::new_sqlite(&db_path).await.unwrap();

        let status = state.runtime_status().await.unwrap();

        assert_eq!(status.executor_mode, "stub");
        assert_eq!(status.planning_mode, "deterministic");
        assert_eq!(status.active_backend.as_deref(), None);
        assert_eq!(status.active_model.as_deref(), None);
        assert_eq!(status.ollama_url.as_deref(), None);
        assert!(!status.real_agent_execution);
        assert!(status.backend_capabilities.is_empty());
    }

    #[tokio::test]
    async fn sqlite_state_reports_ollama_agent_runtime_when_model_executor_is_configured() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("crytex-ui.db");
        let state = CrytexAppState::new_sqlite_with_ollama_agent_executor(
            &db_path,
            "http://127.0.0.1:11434",
            "qwen3.5:9b",
        )
        .await
        .unwrap();

        let status = state.runtime_status().await.unwrap();

        assert_eq!(status.executor_mode, "ollama_agent");
        assert_eq!(status.planning_mode, "deterministic");
        assert_eq!(status.active_backend.as_deref(), Some("ollama"));
        assert_eq!(status.active_model.as_deref(), Some("qwen3.5:9b"));
        assert_eq!(status.ollama_url.as_deref(), Some("http://127.0.0.1:11434"));
        assert!(status.real_agent_execution);
        assert_eq!(status.backend_capabilities.len(), 1);
        assert_eq!(status.backend_capabilities[0].id, "ollama");
        assert!(status.backend_capabilities[0].generate);
        assert!(status.backend_capabilities[0].chat);
        assert!(!status.backend_capabilities[0].lora);
        assert!(!status.backend_capabilities[0].hot_swap);
    }

    #[tokio::test]
    async fn sqlite_state_switches_active_ollama_model_for_future_runs() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("crytex-ui.db");
        let state = CrytexAppState::new_sqlite(&db_path).await.unwrap();

        let status = state
            .set_active_ollama_model("http://127.0.0.1:11434", "qwen3.5:9b")
            .await
            .unwrap();

        assert_eq!(status.executor_mode, "ollama_agent");
        assert_eq!(status.active_backend.as_deref(), Some("ollama"));
        assert_eq!(status.active_model.as_deref(), Some("qwen3.5:9b"));
        assert_eq!(status.ollama_url.as_deref(), Some("http://127.0.0.1:11434"));
        assert!(status.real_agent_execution);
        assert_eq!(status.backend_capabilities.len(), 1);
        assert_eq!(status.backend_capabilities[0].id, "ollama");
        assert!(status.backend_capabilities[0].generate);
        assert!(status.backend_capabilities[0].chat);
        assert!(!status.backend_capabilities[0].lora);
        assert!(!status.backend_capabilities[0].hot_swap);

        let current = state.runtime_status().await.unwrap();
        assert_eq!(current, status);
    }

    #[tokio::test]
    async fn sqlite_state_emits_runtime_selected_event_for_ollama_model_switch() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("crytex-ui.db");
        let state = CrytexAppState::new_sqlite(&db_path).await.unwrap();
        let mut events = state.subscribe_to_events().await.unwrap();

        state
            .set_active_ollama_model("http://127.0.0.1:11434", "qwen3.5:9b")
            .await
            .unwrap();

        let event = events.recv().await.unwrap();

        assert!(matches!(
            event,
            crytex_core::bus::Event::RuntimeSelected {
                backend,
                model_id,
                model_path: None,
                endpoint_url,
                context_size: None,
                gpu_layers: None,
                quantization: None,
            } if backend == "ollama"
                && model_id == "qwen3.5:9b"
                && endpoint_url.as_deref() == Some("http://127.0.0.1:11434")
        ));
    }

    #[tokio::test]
    async fn sqlite_state_switches_active_downloaded_managed_model_for_future_runs() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("crytex-ui.db");
        let state = CrytexAppState::new_sqlite(&db_path).await.unwrap();

        state
            .add_managed_model(AddManagedModelCommand {
                id: "local-qwen".into(),
                name: "Local Qwen".into(),
                repo: "Qwen/Qwen2.5-Coder-9B-Instruct-GGUF".into(),
                filename: "local-qwen.gguf".into(),
                quantization: Some("Q4_K_M".into()),
                backend: Some("mistral_rs".into()),
                params_b: Some(9.0),
            })
            .await
            .unwrap();

        let model_path = dir
            .path()
            .join("models-cache")
            .join("models")
            .join("local-qwen")
            .join("local-qwen.gguf");
        let model_path_text = model_path.display().to_string();
        std::fs::create_dir_all(model_path.parent().unwrap()).unwrap();
        std::fs::write(&model_path, b"gguf").unwrap();
        let registry_path = dir.path().join("models-cache").join("registry.toml");
        std::fs::write(
            registry_path,
            format!(
                "[models.local-qwen]\nlocal_path = \"{}\"\nsize = 4\n",
                model_path_text.replace('\\', "\\\\")
            ),
        )
        .unwrap();

        let status = state
            .set_active_managed_model(SetActiveManagedModelCommand {
                model_id: "local-qwen".into(),
            })
            .await
            .unwrap();

        assert_eq!(status.executor_mode, "mistralrs_agent");
        assert_eq!(status.active_backend.as_deref(), Some("mistralrs"));
        assert_eq!(
            status.active_model.as_deref(),
            Some(model_path_text.as_str())
        );
        assert_eq!(status.ollama_url, None);
        assert!(status.real_agent_execution);
        assert_eq!(status.backend_capabilities.len(), 1);
        assert_eq!(status.backend_capabilities[0].id, "mistralrs");
        assert!(status.backend_capabilities[0].generate);
        assert!(status.backend_capabilities[0].chat);
        assert!(!status.backend_capabilities[0].lora);
        assert!(!status.backend_capabilities[0].hot_swap);

        let current = state.runtime_status().await.unwrap();
        assert_eq!(current, status);
    }

    #[tokio::test]
    async fn sqlite_state_adds_downloads_lists_and_activates_managed_model() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("crytex-ui.db");
        let model_path = dir.path().join("hf-cache").join("local-qwen.gguf");
        let model_manager = Arc::new(DownloadableManagedModelManager::new(model_path.clone()));
        let state = CrytexAppState::new_sqlite_with_executor_factory_and_planning(
            &db_path,
            Box::new(|_, _, _| Arc::new(StubTaskExecutor)),
            None,
            None,
            None,
            None,
            None,
            Some(model_manager),
            None,
            stub_runtime_status(),
        )
        .await
        .unwrap();

        let added = state
            .add_managed_model(AddManagedModelCommand {
                id: "local-qwen".into(),
                name: "Local Qwen".into(),
                repo: "Qwen/Qwen2.5-Coder-9B-Instruct-GGUF".into(),
                filename: "local-qwen.gguf".into(),
                quantization: Some("Q4_K_M".into()),
                backend: Some("mistral_rs".into()),
                params_b: Some(9.0),
            })
            .await
            .unwrap();

        assert!(matches!(added.status, ModelStatus::Available));
        assert!(added.local_path.is_none());

        let downloaded = state
            .download_managed_model(DownloadManagedModelCommand {
                model_id: "local-qwen".into(),
            })
            .await
            .unwrap();

        assert!(matches!(downloaded.status, ModelStatus::Downloaded));
        assert_eq!(downloaded.local_path.as_deref(), Some(model_path.as_path()));
        assert!(model_path.exists());

        let listed = state.list_managed_models().await.unwrap();
        let listed_model = listed
            .models
            .iter()
            .find(|model| model.id == "local-qwen")
            .unwrap();
        assert!(matches!(listed_model.status, ModelStatus::Downloaded));
        assert_eq!(
            listed_model.local_path.as_deref(),
            Some(model_path.as_path())
        );

        let status = state
            .set_active_managed_model(SetActiveManagedModelCommand {
                model_id: "local-qwen".into(),
            })
            .await
            .unwrap();

        assert_eq!(status.active_backend.as_deref(), Some("mistralrs"));
        assert_eq!(
            status.active_model.as_deref(),
            Some(model_path.display().to_string().as_str())
        );
        assert_eq!(status.backend_capabilities.len(), 1);
        assert_eq!(status.backend_capabilities[0].id, "mistralrs");
        assert!(status.backend_capabilities[0].generate);
        assert!(status.backend_capabilities[0].chat);
        assert!(!status.backend_capabilities[0].hot_swap);
    }

    #[tokio::test]
    async fn sqlite_state_exports_download_and_activation_diagnostics_for_managed_model() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("crytex-ui.db");
        let model_path = dir.path().join("hf-cache").join("local-qwen.gguf");
        let model_manager = Arc::new(DownloadableManagedModelManager::new(model_path.clone()));
        let state = CrytexAppState::new_sqlite_with_executor_factory_and_planning(
            &db_path,
            Box::new(|_, _, _| Arc::new(StubTaskExecutor)),
            None,
            None,
            None,
            None,
            None,
            Some(model_manager),
            None,
            stub_runtime_status(),
        )
        .await
        .unwrap();
        let mut events = state.subscribe_to_events().await.unwrap();

        state
            .add_managed_model(AddManagedModelCommand {
                id: "local-qwen".into(),
                name: "Local Qwen".into(),
                repo: "Qwen/Qwen2.5-Coder-9B-Instruct-GGUF".into(),
                filename: "local-qwen.gguf".into(),
                quantization: Some("Q4_K_M".into()),
                backend: Some("mistral_rs".into()),
                params_b: Some(9.0),
            })
            .await
            .unwrap();
        state
            .download_managed_model(DownloadManagedModelCommand {
                model_id: "local-qwen".into(),
            })
            .await
            .unwrap();
        state
            .set_active_managed_model(SetActiveManagedModelCommand {
                model_id: "local-qwen".into(),
            })
            .await
            .unwrap();

        let mut observed_actions = Vec::new();
        while observed_actions.len() < 2 {
            if let crytex_core::bus::Event::RunObserved {
                action, metadata, ..
            } = events.recv().await.unwrap()
            {
                observed_actions.push((action, metadata));
            }
        }

        let downloaded = observed_actions
            .iter()
            .find(|(action, _)| action == "managed_model_downloaded")
            .expect("download diagnostics should be emitted");
        assert_eq!(downloaded.1["model_id"], "local-qwen");
        assert_eq!(downloaded.1["local_path"], model_path.display().to_string());

        let activated = observed_actions
            .iter()
            .find(|(action, _)| action == "managed_model_activated")
            .expect("activation diagnostics should be emitted");
        assert_eq!(activated.1["model_id"], "local-qwen");
        assert_eq!(activated.1["backend"], "mistralrs");
        assert_eq!(activated.1["model_path"], model_path.display().to_string());
        assert_eq!(activated.1["real_agent_execution"], true);
    }

    #[tokio::test]
    async fn sqlite_state_proves_downloaded_active_managed_model_generates_and_exports_diagnostics()
    {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("crytex-ui.db");
        let model_path = dir.path().join("hf-cache").join("local-qwen.gguf");
        let model_manager = Arc::new(DownloadableManagedModelManager::new(model_path.clone()));
        let state = CrytexAppState::new_sqlite_with_executor_factory_and_planning(
            &db_path,
            Box::new(|_, _, _| Arc::new(StubTaskExecutor)),
            None,
            None,
            None,
            None,
            None,
            Some(model_manager),
            None,
            stub_runtime_status(),
        )
        .await
        .unwrap();

        state
            .add_managed_model(AddManagedModelCommand {
                id: "local-qwen".into(),
                name: "Local Qwen".into(),
                repo: "Qwen/Qwen2.5-Coder-9B-Instruct-GGUF".into(),
                filename: "local-qwen.gguf".into(),
                quantization: Some("Q4_K_M".into()),
                backend: Some("mistral_rs".into()),
                params_b: Some(9.0),
            })
            .await
            .unwrap();
        state
            .download_managed_model(DownloadManagedModelCommand {
                model_id: "local-qwen".into(),
            })
            .await
            .unwrap();
        let status = state
            .set_active_managed_model(SetActiveManagedModelCommand {
                model_id: "local-qwen".into(),
            })
            .await
            .unwrap();
        *state.active_inference.write().await = Some(Arc::new(RuntimeProofInference));
        let mut events = state.subscribe_to_events().await.unwrap();

        let report = state
            .prove_managed_model_runtime(ProveManagedModelRuntimeCommand {
                model_id: "local-qwen".into(),
                trace_id: Some("trace-managed-proof".into()),
                max_tokens: Some(16),
                timeout_seconds: Some(5),
            })
            .await
            .unwrap();

        assert!(report.downloaded);
        assert!(report.activated);
        assert!(report.generated);
        assert!(report.runtime_probe.passed);
        assert!(report.failure_reasons.is_empty());
        assert_eq!(report.trace_id, "trace-managed-proof");
        assert_eq!(report.runtime, status);
        assert_eq!(report.runtime.active_backend.as_deref(), Some("mistralrs"));
        assert_eq!(
            report.runtime.active_model.as_deref(),
            Some(model_path.display().to_string().as_str())
        );
        assert_eq!(
            report.runtime_probe.generated_preview.as_deref(),
            Some("CRYTEX_PROBE_OK managed model generated through mistral.rs")
        );

        let event = loop {
            let event = events.recv().await.unwrap();
            if matches!(
                event,
                crytex_core::bus::Event::RunObserved {
                    ref action,
                    ..
                } if action == "model_runtime_proved"
            ) {
                break event;
            }
        };
        let crytex_core::bus::Event::RunObserved {
            project_id,
            task_id,
            trace_id,
            action,
            metadata,
        } = event
        else {
            unreachable!("event was filtered to model_runtime_proved");
        };

        assert_eq!(project_id, "runtime");
        assert_eq!(task_id, None);
        assert_eq!(trace_id, "trace-managed-proof");
        assert_eq!(action, "model_runtime_proved");
        assert_eq!(metadata["model_id"], "local-qwen");
        assert_eq!(metadata["backend"], "mistralrs");
        assert_eq!(metadata["downloaded"], true);
        assert_eq!(metadata["activated"], true);
        assert_eq!(metadata["generated"], true);
        assert_eq!(metadata["passed"], true);
        assert_eq!(
            metadata["generated_preview"],
            "CRYTEX_PROBE_OK managed model generated through mistral.rs"
        );
    }

    #[tokio::test]
    async fn sqlite_state_emits_runtime_selected_event_for_managed_model_switch() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("crytex-ui.db");
        let state = CrytexAppState::new_sqlite(&db_path).await.unwrap();
        let mut events = state.subscribe_to_events().await.unwrap();

        state
            .add_managed_model(AddManagedModelCommand {
                id: "local-qwen".into(),
                name: "Local Qwen".into(),
                repo: "Qwen/Qwen2.5-Coder-9B-Instruct-GGUF".into(),
                filename: "local-qwen.gguf".into(),
                quantization: Some("Q4_K_M".into()),
                backend: Some("mistral_rs".into()),
                params_b: Some(9.0),
            })
            .await
            .unwrap();

        let model_path = dir
            .path()
            .join("models-cache")
            .join("models")
            .join("local-qwen")
            .join("local-qwen.gguf");
        let model_path_text = model_path.display().to_string();
        std::fs::create_dir_all(model_path.parent().unwrap()).unwrap();
        std::fs::write(&model_path, b"gguf").unwrap();
        let registry_path = dir.path().join("models-cache").join("registry.toml");
        std::fs::write(
            registry_path,
            format!(
                "[models.local-qwen]\nlocal_path = \"{}\"\nsize = 4\n",
                model_path_text.replace('\\', "\\\\")
            ),
        )
        .unwrap();

        let recommended = state.model_manager.recommend_config("local-qwen").unwrap();

        state
            .set_active_managed_model(SetActiveManagedModelCommand {
                model_id: "local-qwen".into(),
            })
            .await
            .unwrap();

        let event = loop {
            let event = events.recv().await.unwrap();
            if matches!(event, crytex_core::bus::Event::RuntimeSelected { .. }) {
                break event;
            }
        };

        let crytex_core::bus::Event::RuntimeSelected {
            backend,
            model_id,
            model_path,
            endpoint_url,
            context_size,
            gpu_layers,
            quantization,
        } = event
        else {
            unreachable!("event was filtered to RuntimeSelected");
        };

        assert_eq!(backend, "mistralrs");
        assert_eq!(model_id, "local-qwen");
        assert_eq!(model_path.as_deref(), Some(model_path_text.as_str()));
        assert_eq!(endpoint_url, None);
        assert_eq!(context_size, Some(recommended.context_size));
        assert_eq!(gpu_layers, recommended.gpu_layers);
        assert_eq!(
            quantization.as_deref(),
            Some(recommended.quantization.as_str())
        );
    }

    #[tokio::test]
    async fn sqlite_state_supports_project_task_and_state_commands() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("crytex-ui.db");
        let state = CrytexAppState::new_sqlite(&db_path).await.unwrap();

        let project_root = dir.path().join("project");
        std::fs::create_dir_all(&project_root).unwrap();

        let project = state
            .create_project(CreateProjectCommand {
                name: "Manual Test".into(),
                root_path: project_root.display().to_string(),
            })
            .await
            .unwrap();

        let task = state
            .submit_task(SubmitTaskCommand {
                project_id: project.id.clone(),
                parent_id: None,
                title: "Build Tauri UI".into(),
                description: Some("wire manual test surface".into()),
                kind: "codegen".into(),
                assigned_agent: Some("coder".into()),
                priority: 5,
                payload: json!({ "screen": "tasks" }),
                trace_id: Some("trace-ui-test".into()),
            })
            .await
            .unwrap();

        let goal_plan = state
            .submit_goal(SubmitGoalCommand {
                project_id: project.id.clone(),
                goal: "Polish backend for goal-first UI".into(),
                context: json!({ "active_screen": "workspace" }),
                trace_id: Some("trace-goal-test".into()),
            })
            .await
            .unwrap();

        assert_eq!(goal_plan.goal.kind, "codegen");
        assert_eq!(goal_plan.goal.assigned_agent, Some("architect".into()));
        assert_eq!(
            goal_plan.goal.status,
            crytex_core::models::TaskStatus::Review
        );
        assert_eq!(goal_plan.generated_tasks.len(), 5);
        assert!(
            goal_plan
                .generated_tasks
                .iter()
                .all(|generated| generated.parent_id == Some(goal_plan.goal.id.clone()))
        );
        assert!(
            goal_plan
                .generated_tasks
                .iter()
                .all(|generated| generated.status == crytex_core::models::TaskStatus::Backlog)
        );

        let approved = state
            .approve_plan(PlanDecisionCommand {
                goal_task_id: goal_plan.goal.id.clone(),
                comment: Some("ship it".into()),
            })
            .await
            .unwrap();
        assert_eq!(
            approved.goal.status,
            crytex_core::models::TaskStatus::Completed
        );
        assert!(
            approved
                .generated_tasks
                .iter()
                .all(|generated| generated.status == crytex_core::models::TaskStatus::Pending)
        );

        let kanban = state.kanban_state(&project.id).await.unwrap();
        let pending = kanban
            .columns
            .iter()
            .find(|column| column.status.as_str() == "pending")
            .unwrap();
        assert!(pending.tasks.iter().any(|card| card.id == task.id));

        let exported = state.get_project_state(&project.id).await.unwrap();
        assert_eq!(exported.project.id, project.id);
        assert_eq!(exported.tasks.len(), 7);
        assert!(exported.recent_logs.len() >= 7);
        state.shutdown_project_watchers().await;
    }

    #[tokio::test]
    async fn sqlite_state_indexes_project_on_create_and_searches_context() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("crytex-ui.db");
        let state = CrytexAppState::new_sqlite(&db_path).await.unwrap();

        let project_root = dir.path().join("project");
        std::fs::create_dir_all(project_root.join("docs")).unwrap();
        std::fs::write(
            project_root.join("docs/architecture.md"),
            "# Crytex RAG\n\nCRYTEX_RAG_SENTINEL context must be retrievable.\n",
        )
        .unwrap();

        let project = state
            .create_project(CreateProjectCommand {
                name: "Indexed Project".into(),
                root_path: project_root.display().to_string(),
            })
            .await
            .unwrap();

        let response = state
            .search_project_context(SearchProjectContextCommand {
                project_id: project.id.clone(),
                query: "CRYTEX_RAG_SENTINEL".into(),
                limit: 5,
            })
            .await
            .unwrap();

        assert!(
            response.hits.iter().any(|hit| {
                hit.collection == "doc_chunks"
                    && matches!(
                        hit.relative_path.as_deref(),
                        Some("docs\\architecture.md") | Some("docs/architecture.md")
                    )
            }),
            "expected indexed docs/architecture.md in hits: {:?}",
            response.hits
        );
        assert!(
            response.hits.iter().any(|hit| {
                hit.text
                    .as_deref()
                    .is_some_and(|text| text.contains("CRYTEX_RAG_SENTINEL"))
            }),
            "expected retrieved text to include sentinel: {:?}",
            response.hits
        );
        state.shutdown_project_watchers().await;
    }

    #[tokio::test]
    async fn sqlite_state_starts_project_watcher_after_create_project() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("crytex-ui.db");
        let state = CrytexAppState::new_sqlite(&db_path).await.unwrap();

        let project_root = dir.path().join("project");
        std::fs::create_dir_all(project_root.join("src")).unwrap();

        let project = state
            .create_project(CreateProjectCommand {
                name: "Watched Project".into(),
                root_path: project_root.display().to_string(),
            })
            .await
            .unwrap();

        std::fs::write(
            project_root.join("src/live_context.rs"),
            "pub fn watcher_sentinel() -> &'static str { \"WATCHER_SENTINEL auto indexed\" }\n",
        )
        .unwrap();

        let mut indexed = None;
        for _ in 0..20 {
            let response = state
                .search_project_context(SearchProjectContextCommand {
                    project_id: project.id.clone(),
                    query: "WATCHER_SENTINEL auto indexed".into(),
                    limit: 5,
                })
                .await
                .unwrap();

            if response.hits.iter().any(|hit| {
                hit.text
                    .as_deref()
                    .is_some_and(|text| text.contains("WATCHER_SENTINEL"))
            }) {
                indexed = Some(response);
                break;
            }

            tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        }

        assert!(
            indexed.is_some(),
            "expected Tauri app state watcher to index newly created file"
        );
        state.shutdown_project_watchers().await;
    }

    #[tokio::test]
    async fn sqlite_state_injects_indexed_rag_context_into_agent_run_prompt() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("crytex-ui.db");
        let captured_requests = Arc::new(Mutex::new(Vec::new()));
        let inference = Arc::new(CapturingInference::new(captured_requests.clone()));
        let state = CrytexAppState::new_sqlite_with_executor_factory_and_planning(
            &db_path,
            Box::new(move |project_service, audit_service, context_assembler| {
                Arc::new(
                    AgentTaskExecutor::new_project_scoped(
                        Arc::new(
                            StaticAgentService::with_default_agents(Some(context_assembler))
                                .with_audit(audit_service.clone()),
                        ),
                        inference,
                        project_service,
                    )
                    .with_audit(audit_service),
                ) as Arc<dyn TaskExecutor>
            }),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            custom_executor_runtime_status(),
        )
        .await
        .unwrap();

        let project_root = dir.path().join("project");
        std::fs::create_dir_all(project_root.join("docs")).unwrap();
        std::fs::write(
            project_root.join("docs/runtime.md"),
            "Payment retry adapter design note.\n\nRAG_ONLY_SECRET_CONTEXT must be preserved for agent execution.\n",
        )
        .unwrap();

        let project = state
            .create_project(CreateProjectCommand {
                name: "Agent RAG".into(),
                root_path: project_root.display().to_string(),
            })
            .await
            .unwrap();

        state
            .submit_task(SubmitTaskCommand {
                project_id: project.id.clone(),
                parent_id: None,
                title: "Implement payment retry adapter".into(),
                description: Some("Use the relevant project context before coding".into()),
                kind: "codegen".into(),
                assigned_agent: Some("coder".into()),
                priority: 5,
                payload: json!({}),
                trace_id: Some("trace-agent-rag".into()),
            })
            .await
            .unwrap();

        state
            .start_run(StartRunCommand {
                project_id: project.id.clone(),
                max_steps: 1,
            })
            .await
            .unwrap();

        let prompt = captured_requests
            .lock()
            .unwrap()
            .iter()
            .flat_map(|request| request.messages.iter())
            .map(|message| message.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        assert!(
            prompt.contains("RAG_ONLY_SECRET_CONTEXT"),
            "agent prompt should include indexed RAG context, got: {prompt}"
        );
        state.shutdown_project_watchers().await;
    }

    #[tokio::test]
    async fn sqlite_state_applies_reranker_before_injecting_rag_context_into_agent_prompt() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("crytex-ui.db");
        let captured_requests = Arc::new(Mutex::new(Vec::new()));
        let inference = Arc::new(CapturingInference::new(captured_requests.clone()));
        let reranker = Arc::new(PreferRerankTarget);
        let state = CrytexAppState::new_sqlite_with_executor_factory_planning_and_reranker(
            &db_path,
            Box::new(move |project_service, audit_service, context_assembler| {
                Arc::new(
                    AgentTaskExecutor::new_project_scoped(
                        Arc::new(
                            StaticAgentService::with_default_agents(Some(context_assembler))
                                .with_audit(audit_service.clone()),
                        ),
                        inference,
                        project_service,
                    )
                    .with_audit(audit_service),
                ) as Arc<dyn TaskExecutor>
            }),
            reranker,
            custom_executor_runtime_status(),
        )
        .await
        .unwrap();

        let project_root = dir.path().join("project");
        std::fs::create_dir_all(project_root.join("docs")).unwrap();
        std::fs::write(
            project_root.join("docs/dense-winner.md"),
            "Payment retry adapter generic dense context. RERANK_BASELINE_CONTEXT.",
        )
        .unwrap();
        std::fs::write(
            project_root.join("docs/rerank-target.md"),
            "Payment retry adapter precise reranked context. RERANK_TARGET_CONTEXT.",
        )
        .unwrap();

        let project = state
            .create_project(CreateProjectCommand {
                name: "Agent Rerank RAG".into(),
                root_path: project_root.display().to_string(),
            })
            .await
            .unwrap();

        state
            .submit_task(SubmitTaskCommand {
                project_id: project.id.clone(),
                parent_id: None,
                title: "Implement payment retry adapter".into(),
                description: Some("Use reranked relevant project context first".into()),
                kind: "codegen".into(),
                assigned_agent: Some("coder".into()),
                priority: 5,
                payload: json!({}),
                trace_id: Some("trace-agent-rerank-rag".into()),
            })
            .await
            .unwrap();

        state
            .start_run(StartRunCommand {
                project_id: project.id.clone(),
                max_steps: 1,
            })
            .await
            .unwrap();

        let prompt = captured_requests
            .lock()
            .unwrap()
            .iter()
            .flat_map(|request| request.messages.iter())
            .map(|message| message.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        let target_pos = prompt
            .find("RERANK_TARGET_CONTEXT")
            .expect("reranked target context should be injected into prompt");
        let baseline_pos = prompt
            .find("RERANK_BASELINE_CONTEXT")
            .expect("baseline context should be injected into prompt");
        assert!(
            target_pos < baseline_pos,
            "reranker should order target context before baseline context in agent prompt: {prompt}"
        );
        let exported = state.get_project_state(&project.id).await.unwrap();
        let exported_actions = exported
            .recent_logs
            .iter()
            .map(|log| format!("{}:{:?}", log.action, log.metadata.get("trace_id")))
            .collect::<Vec<_>>();
        let rag_event = exported
            .recent_logs
            .iter()
            .find(|log| {
                log.action == "rag_context_assembled"
                    && log.metadata["trace_id"] == "trace-agent-rerank-rag"
            })
            .unwrap_or_else(|| {
                panic!("RAG diagnostics event should be exported: {exported_actions:?}")
            });
        assert_eq!(rag_event.metadata["rerank_applied"], true);
        let retrieval_paths = rag_event.metadata["retrieval_candidates"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|candidate| candidate["relative_path"].as_str())
            .collect::<Vec<_>>();
        assert!(retrieval_paths.contains(&"docs/dense-winner.md"));
        assert!(retrieval_paths.contains(&"docs/rerank-target.md"));
        assert_eq!(
            rag_event.metadata["reranked_chunks"][0]["relative_path"],
            "docs/rerank-target.md"
        );
        assert!(
            rag_event.metadata["chunks"][0]["selection_reason"]
                .as_str()
                .unwrap_or_default()
                .contains("retrieval evidence")
        );
        state.shutdown_project_watchers().await;
    }

    #[tokio::test]
    async fn sqlite_state_exports_graph_rag_metadata_to_prompt_and_diagnostics() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("crytex-ui.db");
        let captured_requests = Arc::new(Mutex::new(Vec::new()));
        let inference = Arc::new(CapturingInference::new(captured_requests.clone()));
        let state = CrytexAppState::new_sqlite_with_executor_factory_and_planning(
            &db_path,
            Box::new(move |project_service, audit_service, context_assembler| {
                Arc::new(
                    AgentTaskExecutor::new_project_scoped(
                        Arc::new(
                            StaticAgentService::with_default_agents(Some(context_assembler))
                                .with_audit(audit_service.clone()),
                        ),
                        inference,
                        project_service,
                    )
                    .with_audit(audit_service),
                ) as Arc<dyn TaskExecutor>
            }),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            custom_executor_runtime_status(),
        )
        .await
        .unwrap();

        let project_root = dir.path().join("project");
        std::fs::create_dir_all(project_root.join("src")).unwrap();
        std::fs::write(
            project_root.join("src/lib.rs"),
            r#"
pub fn payment_retry_helper() -> bool {
    true
}

pub fn payment_retry_caller() -> bool {
    payment_retry_helper()
}
"#,
        )
        .unwrap();

        let project = state
            .create_project(CreateProjectCommand {
                name: "Graph RAG Metadata".into(),
                root_path: project_root.display().to_string(),
            })
            .await
            .unwrap();

        state
            .submit_task(SubmitTaskCommand {
                project_id: project.id.clone(),
                parent_id: None,
                title: "Use payment_retry_helper graph context".into(),
                description: Some("Explain how the caller reaches helper".into()),
                kind: "codegen".into(),
                assigned_agent: Some("coder".into()),
                priority: 5,
                payload: json!({}),
                trace_id: Some("trace-agent-graph-rag".into()),
            })
            .await
            .unwrap();

        state
            .start_run(StartRunCommand {
                project_id: project.id.clone(),
                max_steps: 1,
            })
            .await
            .unwrap();

        let prompt = captured_requests
            .lock()
            .unwrap()
            .iter()
            .flat_map(|request| request.messages.iter())
            .map(|message| message.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        assert!(
            prompt.contains("file=src/lib.rs"),
            "agent prompt should expose source file metadata: {prompt}"
        );
        assert!(
            prompt.contains("symbol=rust:Function:src/lib.rs:"),
            "agent prompt should expose graph symbol metadata: {prompt}"
        );
        assert!(
            prompt.contains("related=rust:Function:src/lib.rs:"),
            "agent prompt should expose related symbol metadata: {prompt}"
        );

        let exported = state.get_project_state(&project.id).await.unwrap();
        let rag_event = exported
            .recent_logs
            .iter()
            .find(|log| {
                log.action == "rag_context_assembled"
                    && log.metadata["trace_id"] == "trace-agent-graph-rag"
            })
            .expect("RAG diagnostics event should be exported");
        let selected_chunk = rag_event.metadata["chunks"]
            .as_array()
            .and_then(|chunks| chunks.first())
            .expect("RAG diagnostics should include selected chunk metadata");
        assert_eq!(
            selected_chunk["relative_path"],
            serde_json::json!("src/lib.rs")
        );
        assert!(
            selected_chunk["symbol_id"]
                .as_str()
                .unwrap_or_default()
                .starts_with("rust:Function:src/lib.rs:"),
            "diagnostics should expose graph symbol metadata: {selected_chunk:?}"
        );
        assert!(
            selected_chunk["related_symbols"]
                .as_array()
                .into_iter()
                .flatten()
                .any(|symbol| symbol
                    .as_str()
                    .unwrap_or_default()
                    .starts_with("rust:Function:src/lib.rs:")),
            "diagnostics should expose related symbol metadata: {selected_chunk:?}"
        );
        state.shutdown_project_watchers().await;
    }

    #[tokio::test]
    async fn sqlite_state_uses_promoted_lora_for_next_agent_request() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("crytex-ui.db");
        let captured_requests = Arc::new(Mutex::new(Vec::new()));
        let inference = Arc::new(CapturingInference::new(captured_requests.clone()));
        let state = CrytexAppState::new_sqlite_with_executor(&db_path, Arc::new(StubTaskExecutor))
            .await
            .unwrap();

        let project_root = dir.path().join("project");
        std::fs::create_dir_all(&project_root).unwrap();
        let project = state
            .project_service
            .create(crytex_core::services::CreateProjectRequest {
                name: "Promoted LoRA Agent Request",
                root_path: &project_root,
            })
            .await
            .unwrap();

        for index in 0..50 {
            let task = state
                .submit_task(SubmitTaskCommand {
                    project_id: project.id.clone(),
                    parent_id: None,
                    title: format!("Create LoRA selection training example {index:02}"),
                    description: Some(
                        "Detailed successful codegen output used as golden data".into(),
                    ),
                    kind: "codegen".into(),
                    assigned_agent: Some("coder".into()),
                    priority: 5,
                    payload: json!({ "example_index": index }),
                    trace_id: Some("trace-lora-selection-training".into()),
                })
                .await
                .unwrap();

            let run = state
                .start_run(StartRunCommand {
                    project_id: project.id.clone(),
                    max_steps: 10,
                })
                .await
                .unwrap();
            assert_eq!(run.review_tasks.len(), 1);
            assert_eq!(run.review_tasks[0].id, task.id);

            state
                .approve_task_review(TaskReviewDecisionCommand {
                    task_id: task.id,
                    comment: Some("accepted for LoRA selection proof".into()),
                })
                .await
                .unwrap();
        }

        let capturing_executor = Arc::new(
            AgentTaskExecutor::new_project_scoped(
                Arc::new(StaticAgentService::with_default_agents(Some(
                    state.context_assembler.clone(),
                ))),
                inference,
                state.project_service.clone(),
            )
            .with_audit(state.audit_service.clone()),
        ) as Arc<dyn TaskExecutor>;
        *state.task_executor.write().await = Arc::new(LoraSelectingTaskExecutor::new(
            capturing_executor,
            state.task_service.clone(),
            state.lora_evolution.clone(),
        ));

        let next_task = state
            .submit_task(SubmitTaskCommand {
                project_id: project.id.clone(),
                parent_id: None,
                title: "Use promoted LoRA on the next codegen request".into(),
                description: Some("The next agent run should attach codegen-v1".into()),
                kind: "codegen".into(),
                assigned_agent: Some("coder".into()),
                priority: 5,
                payload: json!({}),
                trace_id: Some("trace-lora-selected-next-run".into()),
            })
            .await
            .unwrap();

        state
            .start_run(StartRunCommand {
                project_id: project.id.clone(),
                max_steps: 10,
            })
            .await
            .unwrap();

        {
            let requests = captured_requests.lock().unwrap();
            assert!(!requests.is_empty(), "expected next task to call inference");
            assert!(
                requests
                    .iter()
                    .any(|request| request.lora_adapter_id.as_deref() == Some("codegen-v1")),
                "expected promoted codegen-v1 adapter in next inference request, got: {:?}",
                requests
                    .iter()
                    .map(|request| request.lora_adapter_id.clone())
                    .collect::<Vec<_>>()
            );
        }
        let exported = state.get_project_state(&project.id).await.unwrap();
        assert!(
            exported.tasks.iter().any(|task| task.id == next_task.id),
            "next task should remain visible in project state"
        );
        state.shutdown_project_watchers().await;
    }

    #[tokio::test]
    async fn sqlite_state_uses_planning_agent_when_inference_is_configured() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("crytex-ui.db");
        let state = CrytexAppState::new_sqlite_with_executor_and_planning(
            &db_path,
            Some(Arc::new(StubTaskExecutor)),
            Some(Arc::new(PlanningInference)),
            Some(Arc::new(NoopToolService)),
            None,
            RuntimeStatus {
                planning_mode: "planning_agent".to_string(),
                ..stub_runtime_status()
            },
        )
        .await
        .unwrap();

        let project_root = dir.path().join("project");
        std::fs::create_dir_all(&project_root).unwrap();
        let project = state
            .create_project(CreateProjectCommand {
                name: "Planning".into(),
                root_path: project_root.display().to_string(),
            })
            .await
            .unwrap();

        let response = state
            .submit_goal(SubmitGoalCommand {
                project_id: project.id,
                goal: "Ship a real planning path".into(),
                context: json!({}),
                trace_id: Some("trace-planning".into()),
            })
            .await
            .unwrap();

        assert_eq!(response.generated_tasks.len(), 2);
        assert_eq!(
            response.generated_tasks[0].assigned_agent.as_deref(),
            Some("researcher")
        );
        assert_eq!(
            response.generated_tasks[1].assigned_agent.as_deref(),
            Some("coder")
        );
        assert_eq!(
            response.generated_tasks[0].title,
            "Inspect project: Ship a real planning path"
        );
        state.shutdown_project_watchers().await;
    }

    #[tokio::test]
    async fn sqlite_state_moves_stub_run_tasks_to_review_after_plan_approval() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("crytex-ui.db");
        let state = CrytexAppState::new_sqlite_with_executor(&db_path, Arc::new(StubTaskExecutor))
            .await
            .unwrap();

        let project_root = dir.path().join("project");
        std::fs::create_dir_all(&project_root).unwrap();

        let project = state
            .create_project(CreateProjectCommand {
                name: "Manual Test".into(),
                root_path: project_root.display().to_string(),
            })
            .await
            .unwrap();

        let goal_plan = state
            .submit_goal(SubmitGoalCommand {
                project_id: project.id.clone(),
                goal: "Run approved generated plan".into(),
                context: json!({ "active_screen": "workspace" }),
                trace_id: Some("trace-run-test".into()),
            })
            .await
            .unwrap();

        state
            .approve_plan(PlanDecisionCommand {
                goal_task_id: goal_plan.goal.id.clone(),
                comment: Some("approved for stub execution".into()),
            })
            .await
            .unwrap();

        let run = state
            .start_run(StartRunCommand {
                project_id: project.id.clone(),
                max_steps: 10,
            })
            .await
            .unwrap();

        assert_eq!(run.project_id, project.id);
        assert_eq!(run.review_tasks.len(), 1);
        assert_eq!(run.remaining_ready_tasks.len(), 0);
        assert!(
            run.review_tasks
                .iter()
                .all(|task| task.status == crytex_core::models::TaskStatus::Review)
        );
        assert!(run.review_tasks.iter().all(|task| {
            task.result
                .as_ref()
                .and_then(|result| result.get("source"))
                .and_then(|source| source.as_str())
                == Some("tauri_stub_run")
        }));

        let exported = state.get_project_state(&project.id).await.unwrap();
        assert_eq!(
            run.review_tasks[0].assigned_agent.as_deref(),
            Some("critic")
        );
        assert!(
            exported.tasks.iter().any(|task| {
                task.parent_id == Some(goal_plan.goal.id.clone())
                    && task.assigned_agent.as_deref() == Some("coder")
                    && task.status == crytex_core::models::TaskStatus::Completed
            }),
            "generated chain should auto-complete intermediate agent tasks before critic review"
        );
        state.shutdown_project_watchers().await;
    }

    #[tokio::test]
    async fn sqlite_state_runs_generated_agent_chain_until_critic_human_review_gate() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("crytex-ui.db");
        let state = CrytexAppState::new_sqlite_with_executor(&db_path, Arc::new(ChainExecutor))
            .await
            .unwrap();

        let project_root = dir.path().join("project");
        std::fs::create_dir_all(&project_root).unwrap();

        let project = state
            .create_project(CreateProjectCommand {
                name: "Agent Chain".into(),
                root_path: project_root.display().to_string(),
            })
            .await
            .unwrap();

        let goal_plan = state
            .submit_goal(SubmitGoalCommand {
                project_id: project.id.clone(),
                goal: "Implement a feature through the full agent chain".into(),
                context: json!({ "expected": "chain handoff" }),
                trace_id: Some("trace-agent-chain".into()),
            })
            .await
            .unwrap();

        state
            .approve_plan(PlanDecisionCommand {
                goal_task_id: goal_plan.goal.id.clone(),
                comment: Some("approved chain plan".into()),
            })
            .await
            .unwrap();

        let run = state
            .start_run(StartRunCommand {
                project_id: project.id.clone(),
                max_steps: 10,
            })
            .await
            .unwrap();

        assert_eq!(run.review_tasks.len(), 1);
        assert_eq!(
            run.review_tasks[0].assigned_agent.as_deref(),
            Some("critic")
        );
        assert_eq!(
            run.review_tasks[0].status,
            crytex_core::models::TaskStatus::Review
        );
        assert_eq!(
            run.review_tasks[0].result.as_ref().unwrap()["agent_result"]["review_decision"],
            "pass"
        );
        assert_eq!(
            run.review_tasks[0].result.as_ref().unwrap()["agent_result"]["checked_artifact_count"],
            4
        );

        let exported = state.get_project_state(&project.id).await.unwrap();
        let generated = exported
            .tasks
            .iter()
            .filter(|task| task.parent_id.as_deref() == Some(goal_plan.goal.id.as_str()))
            .collect::<Vec<_>>();
        assert_eq!(generated.len(), 5);

        for agent in ["architect", "coder", "qa", "security"] {
            let task = generated
                .iter()
                .find(|task| task.assigned_agent.as_deref() == Some(agent))
                .unwrap();
            assert_eq!(task.status, crytex_core::models::TaskStatus::Completed);
            assert_eq!(task.result.as_ref().unwrap()["source"], "chain_executor");
        }

        let coder = generated
            .iter()
            .find(|task| task.assigned_agent.as_deref() == Some("coder"))
            .unwrap();
        assert_eq!(
            coder.payload["upstream_artifacts"]
                .as_array()
                .expect("coder should receive architect artifact")
                .len(),
            1
        );
        assert_eq!(coder.payload["upstream_artifacts"][0]["agent"], "architect");
        assert_eq!(coder.payload["upstream_artifacts"][0]["schema_version"], 1);
        assert_eq!(
            coder.payload["upstream_artifacts"][0]["source_agent"],
            "architect"
        );
        assert_eq!(
            coder.payload["upstream_artifacts"][0]["source_task_id"],
            coder.payload["upstream_artifacts"][0]["task_id"]
        );
        assert_eq!(
            coder.payload["upstream_artifacts"][0]["artifact_kind"],
            "design_artifact"
        );
        assert!(
            coder.payload["upstream_artifacts"][0]["artifact_id"]
                .as_str()
                .is_some_and(|id| id.starts_with("artifact-"))
        );
        assert_eq!(
            coder.payload["upstream_artifacts"][0]["content"]["producer"],
            "architect"
        );
        let diagnostics = state
            .export_run_diagnostics(ExportRunDiagnosticsCommand {
                project_id: project.id.clone(),
                run_id: run.run_id,
                trace_id: Some("trace-agent-chain".into()),
            })
            .await
            .unwrap();
        assert!(
            diagnostics.artifact_lineage.iter().any(|artifact| {
                artifact.source_agent.as_deref() == Some("architect")
                    && artifact.artifact_kind == "design_artifact"
                    && Some(artifact.source_task_id.as_str())
                        == coder.payload["upstream_artifacts"][0]["task_id"].as_str()
            }),
            "run diagnostics should export the typed artifact lineage"
        );
        state.shutdown_project_watchers().await;
    }

    #[tokio::test]
    async fn sqlite_state_reviewer_rejection_creates_remediation_chain_with_feedback() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("crytex-ui.db");
        let state = CrytexAppState::new_sqlite_with_executor(
            &db_path,
            Arc::new(RejectThenPassChainExecutor::new()),
        )
        .await
        .unwrap();

        let project_root = dir.path().join("project");
        std::fs::create_dir_all(&project_root).unwrap();

        let project = state
            .create_project(CreateProjectCommand {
                name: "Rejected Chain".into(),
                root_path: project_root.display().to_string(),
            })
            .await
            .unwrap();

        let goal_plan = state
            .submit_goal(SubmitGoalCommand {
                project_id: project.id.clone(),
                goal: "Implement with reviewer-driven retry".into(),
                context: json!({ "expected": "critic routes feedback to coder" }),
                trace_id: Some("trace-retry-chain".into()),
            })
            .await
            .unwrap();

        state
            .approve_plan(PlanDecisionCommand {
                goal_task_id: goal_plan.goal.id.clone(),
                comment: Some("approved retry chain".into()),
            })
            .await
            .unwrap();

        let run = state
            .start_run(StartRunCommand {
                project_id: project.id.clone(),
                max_steps: 10,
            })
            .await
            .unwrap();

        assert_eq!(run.review_tasks.len(), 1);
        assert_eq!(
            run.review_tasks[0].assigned_agent.as_deref(),
            Some("critic")
        );
        assert_eq!(
            run.review_tasks[0].result.as_ref().unwrap()["agent_result"]["review_decision"],
            "pass"
        );

        let exported = state.get_project_state(&project.id).await.unwrap();
        let runner_actions = [
            "run_started",
            "task_execution_started",
            "task_execution_finished",
            "critic_rejected",
            "remediation_plan_created",
            "human_review_ready",
        ];
        let log_actions = exported
            .recent_logs
            .iter()
            .map(|log| log.action.as_str())
            .collect::<Vec<_>>();
        for action in runner_actions {
            assert!(
                log_actions.contains(&action),
                "missing runner action {action}; exported actions: {log_actions:?}"
            );
        }
        assert!(
            exported
                .recent_logs
                .iter()
                .filter(|log| runner_actions.contains(&log.action.as_str()))
                .all(|log| {
                    log.metadata["run_id"].as_str().is_some()
                        && log.metadata["trace_id"] == "trace-retry-chain"
                })
        );
        let original_generated = exported
            .tasks
            .iter()
            .filter(|task| task.parent_id.as_deref() == Some(goal_plan.goal.id.as_str()))
            .collect::<Vec<_>>();
        let original_critic = original_generated
            .iter()
            .find(|task| {
                task.assigned_agent.as_deref() == Some("critic")
                    && task
                        .result
                        .as_ref()
                        .is_some_and(|result| result["agent_result"]["review_decision"] == "reject")
            })
            .unwrap();
        assert_eq!(
            original_critic.status,
            crytex_core::models::TaskStatus::Failed
        );

        let remediation_parent = original_generated
            .iter()
            .find(|task| {
                task.kind == "debug"
                    && task.payload["source"] == "reviewer_rejection"
                    && task.payload["reviewer_task_id"] == original_critic.id
            })
            .unwrap();
        assert_eq!(
            remediation_parent.status,
            crytex_core::models::TaskStatus::Completed
        );
        assert_eq!(
            remediation_parent.payload["feedback"],
            "coder artifact missed the acceptance criteria"
        );

        let remediation_tasks = exported
            .tasks
            .iter()
            .filter(|task| task.parent_id.as_deref() == Some(remediation_parent.id.as_str()))
            .collect::<Vec<_>>();
        assert_eq!(remediation_tasks.len(), 4);

        let debug = remediation_tasks
            .iter()
            .find(|task| task.kind == "debug")
            .unwrap();
        assert_eq!(debug.assigned_agent.as_deref(), Some("coder"));
        assert_eq!(
            debug.payload["critic_report"]["feedback"],
            "coder artifact missed the acceptance criteria"
        );

        for agent in ["coder", "qa", "critic"] {
            let task = remediation_tasks
                .iter()
                .find(|task| task.assigned_agent.as_deref() == Some(agent))
                .unwrap();
            let expected_status = if agent == "critic" {
                crytex_core::models::TaskStatus::Review
            } else {
                crytex_core::models::TaskStatus::Completed
            };
            assert_eq!(task.status, expected_status);
        }

        let critic = remediation_tasks
            .iter()
            .find(|task| task.assigned_agent.as_deref() == Some("critic"))
            .unwrap();
        assert_eq!(
            critic.result.as_ref().unwrap()["agent_result"]["review_decision"],
            "pass"
        );
        let qa = remediation_tasks
            .iter()
            .find(|task| task.assigned_agent.as_deref() == Some("qa"))
            .unwrap();
        assert_eq!(qa.status, crytex_core::models::TaskStatus::Completed);
        assert_eq!(
            qa.payload["upstream_artifacts"]
                .as_array()
                .expect("qa should receive remediation artifacts")
                .len(),
            2
        );
        for agent in ["architect", "security"] {
            let stale_original = original_generated
                .iter()
                .find(|task| task.assigned_agent.as_deref() == Some(agent))
                .unwrap();
            assert_eq!(
                stale_original.status,
                crytex_core::models::TaskStatus::Completed
            );
        }
        state.shutdown_project_watchers().await;
    }

    #[tokio::test]
    async fn sqlite_state_approves_reviewed_task_and_unblocks_next_generated_task() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("crytex-ui.db");
        let state = CrytexAppState::new_sqlite_with_executor(&db_path, Arc::new(StubTaskExecutor))
            .await
            .unwrap();

        let project_root = dir.path().join("project");
        std::fs::create_dir_all(&project_root).unwrap();

        let project = state
            .create_project(CreateProjectCommand {
                name: "Manual Test".into(),
                root_path: project_root.display().to_string(),
            })
            .await
            .unwrap();

        let goal_plan = state
            .submit_goal(SubmitGoalCommand {
                project_id: project.id.clone(),
                goal: "Run generated chain after review approval".into(),
                context: json!({ "active_screen": "workspace" }),
                trace_id: Some("trace-review-approval".into()),
            })
            .await
            .unwrap();

        state
            .approve_plan(PlanDecisionCommand {
                goal_task_id: goal_plan.goal.id.clone(),
                comment: Some("approved generated chain".into()),
            })
            .await
            .unwrap();

        let first_run = state
            .start_run(StartRunCommand {
                project_id: project.id.clone(),
                max_steps: 10,
            })
            .await
            .unwrap();
        assert_eq!(first_run.review_tasks.len(), 1);
        assert_eq!(
            first_run.review_tasks[0].assigned_agent.as_deref(),
            Some("critic")
        );

        let approved_review = state
            .approve_task_review(TaskReviewDecisionCommand {
                task_id: first_run.review_tasks[0].id.clone(),
                comment: Some("critic gate accepted".into()),
            })
            .await
            .unwrap();
        assert_eq!(
            approved_review.task.status,
            crytex_core::models::TaskStatus::Completed
        );
        assert_eq!(approved_review.task.human_score, Some(1.0));
        assert!(approved_review.ready_tasks.is_empty());

        let exported = state.get_project_state(&project.id).await.unwrap();
        assert!(
            exported.recent_logs.iter().any(|log| {
                log.action == "human_review_approved"
                    && log.task_id.as_deref() == Some(first_run.review_tasks[0].id.as_str())
                    && log.metadata["human_score"] == 1.0
                    && log.metadata["reward"].as_f64().is_some()
                    && log.metadata["comment"] == "critic gate accepted"
            }),
            "human approval should emit an observable reward/evolution signal"
        );
        state.shutdown_project_watchers().await;
    }

    #[tokio::test]
    async fn sqlite_state_approval_triggers_lora_evolution_when_threshold_is_met() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("crytex-ui.db");
        let lora_evolution = Arc::new(RecordingLoraEvolution::new(true));
        let state = CrytexAppState::new_sqlite_with_executor_factory_and_planning(
            &db_path,
            Box::new(|_, _, _| Arc::new(StubTaskExecutor)),
            None,
            None,
            Some(lora_evolution.clone()),
            None,
            None,
            None,
            None,
            custom_executor_runtime_status(),
        )
        .await
        .unwrap();

        let project_root = dir.path().join("project");
        std::fs::create_dir_all(&project_root).unwrap();
        let project = state
            .create_project(CreateProjectCommand {
                name: "LoRA Trigger Test".into(),
                root_path: project_root.display().to_string(),
            })
            .await
            .unwrap();
        let task = state
            .submit_task(SubmitTaskCommand {
                project_id: project.id.clone(),
                parent_id: None,
                title: "Implement approval-triggered LoRA evolution".into(),
                description: Some("prove the product path invokes evolution".into()),
                kind: "codegen".into(),
                assigned_agent: Some("coder".into()),
                priority: 5,
                payload: json!({}),
                trace_id: Some("trace-lora-trigger".into()),
            })
            .await
            .unwrap();

        let run = state
            .start_run(StartRunCommand {
                project_id: project.id.clone(),
                max_steps: 1,
            })
            .await
            .unwrap();
        assert_eq!(run.review_tasks[0].id, task.id);

        state
            .approve_task_review(TaskReviewDecisionCommand {
                task_id: task.id.clone(),
                comment: Some("good enough for golden data".into()),
            })
            .await
            .unwrap();

        assert_eq!(
            lora_evolution.collected.lock().unwrap().as_slice(),
            [task.id]
        );
        assert_eq!(
            lora_evolution.trained.lock().unwrap().as_slice(),
            ["codegen"]
        );
        state.shutdown_project_watchers().await;
    }

    #[tokio::test]
    async fn sqlite_state_train_lora_adapter_command_returns_promoted_benchmark_gate_result() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("crytex-ui.db");
        let lora_evolution = Arc::new(RecordingLoraEvolution::new(true));
        let state = CrytexAppState::new_sqlite_with_executor_factory_and_planning(
            &db_path,
            Box::new(|_, _, _| Arc::new(StubTaskExecutor)),
            None,
            None,
            Some(lora_evolution.clone()),
            None,
            None,
            None,
            None,
            custom_executor_runtime_status(),
        )
        .await
        .unwrap();

        let response = state
            .train_lora_adapter(TrainLoraAdapterCommand {
                task_kind: "codegen".into(),
                agent_role: None,
            })
            .await
            .unwrap();

        assert_eq!(
            lora_evolution.trained.lock().unwrap().as_slice(),
            ["codegen"]
        );
        assert_eq!(response.adapter.id, "codegen-mock-lora");
        assert!(response.promoted);
        assert_eq!(
            response.benchmark_gate.as_ref().unwrap()["accepted"],
            serde_json::Value::Bool(true)
        );
        assert_eq!(
            response.metrics["benchmark_gate"]["reason"],
            "recording gate accepted"
        );
    }

    #[tokio::test]
    async fn sqlite_state_persists_run_observed_events_for_diagnostics() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("crytex-ui.db");
        let state = CrytexAppState::new_sqlite_with_executor(&db_path, Arc::new(StubTaskExecutor))
            .await
            .unwrap();

        let project_root = dir.path().join("project");
        std::fs::create_dir_all(&project_root).unwrap();
        let project = state
            .project_service
            .create(crytex_core::services::CreateProjectRequest {
                name: "Run Observed Bridge",
                root_path: &project_root,
            })
            .await
            .unwrap();
        let task = state
            .task_service
            .submit(crytex_core::services::CreateTaskRequest {
                project_id: project.id.clone(),
                parent_id: None,
                title: "LoRA observed task".into(),
                description: None,
                kind: "codegen".into(),
                assigned_agent: Some("coder".into()),
                priority: 5,
                payload: json!({}),
                trace_id: Some("trace-lora-observed".into()),
            })
            .await
            .unwrap();

        state.event_service.publish(Event::RunObserved {
            project_id: project.id.clone(),
            task_id: Some(task.id.clone()),
            trace_id: "trace-lora-observed".into(),
            action: "lora_evolution_promoted".into(),
            metadata: json!({
                "run_id": "run-lora",
                "training_job_id": "job-lora",
                "adapter_id": "adapter-lora",
                "benchmark_gate": {
                    "accepted": true,
                    "winner": "challenger"
                }
            }),
        });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let exported = state.get_project_state(&project.id).await.unwrap();
        assert!(exported.recent_logs.iter().any(|log| {
            log.action == "lora_evolution_promoted"
                && log.task_id.as_deref() == Some(task.id.as_str())
                && log.metadata["trace_id"] == "trace-lora-observed"
                && log.metadata["run_id"] == "run-lora"
                && log.metadata["benchmark_gate"]["accepted"] == true
        }));
        let report = state
            .export_run_diagnostics(ExportRunDiagnosticsCommand {
                project_id: project.id.clone(),
                run_id: "run-lora".into(),
                trace_id: Some("trace-lora-observed".into()),
            })
            .await
            .unwrap();
        assert_eq!(report.lora_evolution.len(), 1);
        assert_eq!(
            report.lora_evolution[0].training_job_id.as_deref(),
            Some("job-lora")
        );
        assert_eq!(
            report.lora_evolution[0].adapter_id.as_deref(),
            Some("adapter-lora")
        );
        assert_eq!(report.lora_evolution[0].accepted, Some(true));
    }

    #[tokio::test]
    async fn sqlite_state_approval_triggered_lora_service_decision_reaches_diagnostics() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("crytex-ui.db");
        let state = CrytexAppState::new_sqlite_with_executor(&db_path, Arc::new(StubTaskExecutor))
            .await
            .unwrap();

        let project_root = dir.path().join("project");
        std::fs::create_dir_all(&project_root).unwrap();
        let project = state
            .project_service
            .create(crytex_core::services::CreateProjectRequest {
                name: "Approval Triggered LoRA Diagnostics",
                root_path: &project_root,
            })
            .await
            .unwrap();

        let mut triggering_task_id = String::new();
        let mut triggering_run_id = String::new();
        let trace_id = "trace-lora-threshold".to_string();

        for index in 0..50 {
            let task = state
                .submit_task(SubmitTaskCommand {
                    project_id: project.id.clone(),
                    parent_id: None,
                    title: format!("Implement observable LoRA training example number {index:02}"),
                    description: Some(
                        "Produce a sufficiently detailed artifact for the golden dataset".into(),
                    ),
                    kind: "codegen".into(),
                    assigned_agent: Some("coder".into()),
                    priority: 5,
                    payload: json!({ "example_index": index }),
                    trace_id: Some(trace_id.clone()),
                })
                .await
                .unwrap();

            let run = state
                .start_run(StartRunCommand {
                    project_id: project.id.clone(),
                    max_steps: 1,
                })
                .await
                .unwrap();
            assert_eq!(run.review_tasks.len(), 1);
            assert_eq!(run.review_tasks[0].id, task.id);

            state
                .approve_task_review(TaskReviewDecisionCommand {
                    task_id: task.id.clone(),
                    comment: Some("accepted as high-quality training signal".into()),
                })
                .await
                .unwrap();

            triggering_task_id = task.id;
            triggering_run_id = run.run_id;
        }

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let report = state
            .export_run_diagnostics(ExportRunDiagnosticsCommand {
                project_id: project.id.clone(),
                run_id: triggering_run_id.clone(),
                trace_id: Some(trace_id.clone()),
            })
            .await
            .unwrap();

        assert_eq!(report.lora_evolution.len(), 1);
        let decision = &report.lora_evolution[0];
        assert_eq!(
            decision.task_id.as_deref(),
            Some(triggering_task_id.as_str())
        );
        assert!(decision.training_job_id.is_some());
        assert_eq!(decision.adapter_id.as_deref(), Some("codegen-v1"));
        assert_eq!(decision.accepted, Some(true));
        assert_eq!(
            decision.metadata["run_id"].as_str(),
            Some(triggering_run_id.as_str())
        );
        assert_eq!(
            decision.metadata["trace_id"].as_str(),
            Some(trace_id.as_str())
        );
        assert_eq!(decision.metadata["task_kind"].as_str(), Some("codegen"));
        assert_eq!(
            decision.metadata["triggering_task_id"].as_str(),
            Some(triggering_task_id.as_str())
        );
        assert_eq!(
            decision.metadata["training_example_count"].as_u64(),
            Some(50)
        );
    }

    #[tokio::test]
    async fn sqlite_state_prompt_benchmark_gate_promotes_and_reaches_diagnostics() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("crytex-ui.db");
        let gate = Arc::new(AcceptingPromptBenchmarkGate::default());
        let state = CrytexAppState::new_sqlite_with_executor_factory_and_planning(
            &db_path,
            Box::new(|_, _, _| Arc::new(StubTaskExecutor)),
            None,
            None,
            None,
            None,
            Some(gate.clone()),
            None,
            None,
            custom_executor_runtime_status(),
        )
        .await
        .unwrap();

        let project_root = dir.path().join("project");
        std::fs::create_dir_all(&project_root).unwrap();
        let project = state
            .project_service
            .create(crytex_core::services::CreateProjectRequest {
                name: "Prompt Benchmark Diagnostics",
                root_path: &project_root,
            })
            .await
            .unwrap();
        let baseline = state
            .prompt_evolution
            .seed_agent("coder", "baseline prompt")
            .await
            .unwrap();
        let challenger = state
            .prompt_evolution
            .mutate(
                &baseline.id,
                crytex_core::services::MutationOperator::AddConstraint,
            )
            .await
            .unwrap();

        let response = state
            .evaluate_prompt_challenger(EvaluatePromptChallengerCommand {
                project_id: project.id.clone(),
                run_id: Some("run-prompt".into()),
                trace_id: Some("trace-prompt".into()),
                task_id: None,
                agent: "coder".into(),
                challenger_prompt_version_id: challenger.id.clone(),
            })
            .await
            .unwrap();

        assert!(response.promoted);
        assert_eq!(response.active.id, challenger.id);
        assert_eq!(
            response.benchmark_gate["winner"],
            serde_json::Value::String("Challenger".into())
        );
        {
            let requests = gate.requests.lock().unwrap();
            assert_eq!(requests.len(), 1);
            assert_eq!(requests[0].baseline_prompt_version_id, baseline.id);
            assert_eq!(requests[0].challenger_prompt_version_id, challenger.id);
        }

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let report = state
            .export_run_diagnostics(ExportRunDiagnosticsCommand {
                project_id: project.id.clone(),
                run_id: "run-prompt".into(),
                trace_id: Some("trace-prompt".into()),
            })
            .await
            .unwrap();

        assert_eq!(report.prompt_evolution.len(), 1);
        let decision = &report.prompt_evolution[0];
        assert_eq!(decision.task_id, None);
        assert_eq!(decision.agent.as_deref(), Some("coder"));
        assert_eq!(
            decision.baseline_prompt_version_id.as_deref(),
            Some(baseline.id.as_str())
        );
        assert_eq!(
            decision.challenger_prompt_version_id.as_deref(),
            Some(challenger.id.as_str())
        );
        assert_eq!(decision.accepted, Some(true));
        assert_eq!(
            decision.baseline_run_id.as_deref(),
            Some("prompt-baseline-run")
        );
        assert_eq!(
            decision.challenger_run_id.as_deref(),
            Some("prompt-challenger-run")
        );
        assert_eq!(decision.baseline_pass_rate, Some(0.25));
        assert_eq!(decision.challenger_pass_rate, Some(0.75));
    }

    #[tokio::test]
    async fn sqlite_state_approval_triggered_lora_benchmark_gate_reaches_diagnostics() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("crytex-ui.db");
        let gate = Arc::new(AcceptingLoraBenchmarkGate::default());
        let state = CrytexAppState::new_sqlite_with_executor_factory_and_planning(
            &db_path,
            Box::new(|_, _, _| Arc::new(StubTaskExecutor)),
            None,
            None,
            None,
            Some(gate.clone()),
            None,
            None,
            None,
            custom_executor_runtime_status(),
        )
        .await
        .unwrap();

        let project_root = dir.path().join("project");
        std::fs::create_dir_all(&project_root).unwrap();
        let project = state
            .project_service
            .create(crytex_core::services::CreateProjectRequest {
                name: "Approval Triggered LoRA Benchmark",
                root_path: &project_root,
            })
            .await
            .unwrap();

        let mut triggering_task_id = String::new();
        let mut triggering_run_id = String::new();
        let trace_id = "trace-lora-benchmark-product".to_string();

        for index in 0..50 {
            let task = state
                .submit_task(SubmitTaskCommand {
                    project_id: project.id.clone(),
                    parent_id: None,
                    title: format!("Implement benchmarked LoRA training example number {index:02}"),
                    description: Some(
                        "Produce a detailed artifact that can be evaluated by the benchmark gate"
                            .into(),
                    ),
                    kind: "codegen".into(),
                    assigned_agent: Some("coder".into()),
                    priority: 5,
                    payload: json!({ "example_index": index }),
                    trace_id: Some(trace_id.clone()),
                })
                .await
                .unwrap();

            let run = state
                .start_run(StartRunCommand {
                    project_id: project.id.clone(),
                    max_steps: 1,
                })
                .await
                .unwrap();
            assert_eq!(run.review_tasks.len(), 1);
            assert_eq!(run.review_tasks[0].id, task.id);

            state
                .approve_task_review(TaskReviewDecisionCommand {
                    task_id: task.id.clone(),
                    comment: Some("accepted as benchmark-gated training signal".into()),
                })
                .await
                .unwrap();

            triggering_task_id = task.id;
            triggering_run_id = run.run_id;
        }

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        {
            let gate_requests = gate.requests.lock().unwrap();
            assert_eq!(gate_requests.len(), 1);
            assert_eq!(gate_requests[0].task_kind, "codegen");
            assert_eq!(gate_requests[0].challenger_adapter_id, "codegen-v1");
        }

        let report = state
            .export_run_diagnostics(ExportRunDiagnosticsCommand {
                project_id: project.id.clone(),
                run_id: triggering_run_id.clone(),
                trace_id: Some(trace_id.clone()),
            })
            .await
            .unwrap();

        assert_eq!(report.lora_evolution.len(), 1);
        let decision = &report.lora_evolution[0];
        assert_eq!(
            decision.task_id.as_deref(),
            Some(triggering_task_id.as_str())
        );
        assert_eq!(decision.adapter_id.as_deref(), Some("codegen-v1"));
        assert_eq!(decision.accepted, Some(true));
        assert_eq!(
            decision.reason.as_deref(),
            Some("challenger passed held-out benchmark")
        );
        assert_eq!(
            decision.baseline_run_id.as_deref(),
            Some("bench-baseline-product")
        );
        assert_eq!(
            decision.challenger_run_id.as_deref(),
            Some("bench-challenger-product")
        );
        assert_eq!(decision.winner.as_deref(), Some("Challenger"));
        assert_eq!(decision.mc_nemar_p_value, Some(0.03125));
        assert_eq!(decision.baseline_pass_rate, Some(0.4));
        assert_eq!(decision.challenger_pass_rate, Some(0.8));
        assert_eq!(
            decision.metadata["benchmark_gate"]["challenger_run_id"].as_str(),
            Some("bench-challenger-product")
        );
        assert_eq!(
            decision.metadata["run_id"].as_str(),
            Some(triggering_run_id.as_str())
        );
    }

    #[tokio::test]
    async fn sqlite_state_approval_triggered_real_bench_gate_uses_held_out_corpus() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("crytex-ui.db");
        let golden_set_path = write_held_out_lora_golden_set(dir.path()).await;
        let benchmark_repo: Arc<dyn BenchmarkResultRepository> =
            Arc::new(MemoryBenchmarkResultRepository::new());
        let event_service: Arc<dyn EventService> =
            Arc::new(EventServiceImpl::new(Arc::new(EventBus::new())));
        let harness = Arc::new(DefaultBenchmarkHarness::new(
            benchmark_repo.clone(),
            event_service,
        ));
        let gate = Arc::new(
            BenchLoraBenchmarkGate::new(
                harness,
                benchmark_repo.clone(),
                golden_set_path.clone(),
                Arc::new(HeldOutSensitiveBenchmarkRunner),
                Arc::new(ExactMatchScorer),
            )
            .with_significance_level(0.05)
            .with_min_delta_pass_rate(0.5),
        );
        let state = CrytexAppState::new_sqlite_with_executor_factory_and_planning(
            &db_path,
            Box::new(|_, _, _| Arc::new(StubTaskExecutor)),
            None,
            None,
            None,
            Some(gate),
            None,
            None,
            None,
            custom_executor_runtime_status(),
        )
        .await
        .unwrap();

        let project_root = dir.path().join("project");
        std::fs::create_dir_all(&project_root).unwrap();
        let project = state
            .project_service
            .create(crytex_core::services::CreateProjectRequest {
                name: "Real Bench Gate LoRA Product Path",
                root_path: &project_root,
            })
            .await
            .unwrap();

        let mut triggering_run_id = String::new();
        let trace_id = "trace-real-bench-gate-product".to_string();

        for index in 0..50 {
            let task = state
                .submit_task(SubmitTaskCommand {
                    project_id: project.id.clone(),
                    parent_id: None,
                    title: format!("Implement held-out-gated LoRA training example {index:02}"),
                    description: Some(
                        "Training example text is intentionally different from held-out benchmark cases"
                            .into(),
                    ),
                    kind: "codegen".into(),
                    assigned_agent: Some("coder".into()),
                    priority: 5,
                    payload: json!({ "example_index": index }),
                    trace_id: Some(trace_id.clone()),
                })
                .await
                .unwrap();

            let run = state
                .start_run(StartRunCommand {
                    project_id: project.id.clone(),
                    max_steps: 1,
                })
                .await
                .unwrap();
            assert_eq!(run.review_tasks.len(), 1);
            assert_eq!(run.review_tasks[0].id, task.id);

            state
                .approve_task_review(TaskReviewDecisionCommand {
                    task_id: task.id,
                    comment: Some("accepted for held-out benchmark-gated LoRA training".into()),
                })
                .await
                .unwrap();

            triggering_run_id = run.run_id;
        }

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let benchmark_runs = benchmark_repo.list_runs(10).await.unwrap();
        assert_eq!(benchmark_runs.len(), 2);
        assert!(benchmark_runs.iter().any(|run| {
            run.name == "codegen baseline"
                && run.golden_set_path == golden_set_path
                && run.pass_rate == 0.0
        }));
        assert!(benchmark_runs.iter().any(|run| {
            run.name.starts_with("codegen challenger codegen-v1")
                && run.golden_set_path == golden_set_path
                && run.pass_rate == 1.0
        }));

        let report = state
            .export_run_diagnostics(ExportRunDiagnosticsCommand {
                project_id: project.id.clone(),
                run_id: triggering_run_id.clone(),
                trace_id: Some(trace_id.clone()),
            })
            .await
            .unwrap();

        assert_eq!(report.lora_evolution.len(), 1);
        let decision = &report.lora_evolution[0];
        assert_eq!(decision.adapter_id.as_deref(), Some("codegen-v1"));
        assert_eq!(decision.accepted, Some(true));
        assert_eq!(decision.winner.as_deref(), Some("Challenger"));
        assert_eq!(decision.baseline_pass_rate, Some(0.0));
        assert_eq!(decision.challenger_pass_rate, Some(1.0));
        assert!(decision.mc_nemar_p_value.is_some_and(|value| value <= 0.05));
        assert!(
            decision
                .baseline_run_id
                .as_deref()
                .is_some_and(|id| !id.is_empty())
        );
        assert!(
            decision
                .challenger_run_id
                .as_deref()
                .is_some_and(|id| !id.is_empty())
        );
        assert_eq!(
            decision.metadata["benchmark_gate"]["delta_pass_rate"].as_f64(),
            Some(1.0)
        );
        assert_eq!(
            decision.metadata["run_id"].as_str(),
            Some(triggering_run_id.as_str())
        );
    }

    #[tokio::test]
    async fn sqlite_state_rejects_reviewed_task_with_feedback_for_retry() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("crytex-ui.db");
        let state = CrytexAppState::new_sqlite_with_executor(&db_path, Arc::new(StubTaskExecutor))
            .await
            .unwrap();

        let project_root = dir.path().join("project");
        std::fs::create_dir_all(&project_root).unwrap();

        let project = state
            .create_project(CreateProjectCommand {
                name: "Manual Test".into(),
                root_path: project_root.display().to_string(),
            })
            .await
            .unwrap();

        let task = state
            .submit_task(SubmitTaskCommand {
                project_id: project.id.clone(),
                parent_id: None,
                title: "Needs review rejection".into(),
                description: None,
                kind: "codegen".into(),
                assigned_agent: Some("coder".into()),
                priority: 5,
                payload: json!({}),
                trace_id: Some("trace-review-reject".into()),
            })
            .await
            .unwrap();

        let run = state
            .start_run(StartRunCommand {
                project_id: project.id.clone(),
                max_steps: 1,
            })
            .await
            .unwrap();
        assert_eq!(run.review_tasks[0].id, task.id);

        let rejected = state
            .reject_task_review(TaskReviewDecisionCommand {
                task_id: run.review_tasks[0].id.clone(),
                comment: Some("result was too vague".into()),
            })
            .await
            .unwrap();

        assert_eq!(
            rejected.task.status,
            crytex_core::models::TaskStatus::Pending
        );
        assert_eq!(rejected.task.human_score, Some(0.0));
        assert_eq!(rejected.task.iteration_count, 1);
        assert_eq!(
            rejected.task.payload["retry_feedback"][0]["comment"],
            "result was too vague"
        );
        assert_eq!(rejected.ready_tasks.len(), 1);
        assert_eq!(rejected.ready_tasks[0].id, rejected.task.id);

        let exported = state.get_project_state(&project.id).await.unwrap();
        assert!(
            exported.recent_logs.iter().any(|log| {
                log.action == "human_review_rejected"
                    && log.task_id.as_deref() == Some(run.review_tasks[0].id.as_str())
                    && log.metadata["human_score"] == 0.0
                    && log.metadata["reward"].as_f64().is_some()
                    && log.metadata["comment"] == "result was too vague"
            }),
            "human rejection should emit an observable reward/evolution signal"
        );
        state.shutdown_project_watchers().await;
    }

    #[tokio::test]
    async fn sqlite_state_rejection_records_lora_counter_example_without_training() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("crytex-ui.db");
        let lora_evolution = Arc::new(RecordingLoraEvolution::new(true));
        let state = CrytexAppState::new_sqlite_with_executor_factory_and_planning(
            &db_path,
            Box::new(|_, _, _| Arc::new(StubTaskExecutor)),
            None,
            None,
            Some(lora_evolution.clone()),
            None,
            None,
            None,
            None,
            custom_executor_runtime_status(),
        )
        .await
        .unwrap();

        let project_root = dir.path().join("project");
        std::fs::create_dir_all(&project_root).unwrap();
        let project = state
            .create_project(CreateProjectCommand {
                name: "LoRA Counter Test".into(),
                root_path: project_root.display().to_string(),
            })
            .await
            .unwrap();
        let task = state
            .submit_task(SubmitTaskCommand {
                project_id: project.id.clone(),
                parent_id: None,
                title: "Generate a bad implementation".into(),
                description: Some("force human rejection".into()),
                kind: "codegen".into(),
                assigned_agent: Some("coder".into()),
                priority: 5,
                payload: json!({}),
                trace_id: Some("trace-lora-counter".into()),
            })
            .await
            .unwrap();

        let run = state
            .start_run(StartRunCommand {
                project_id: project.id.clone(),
                max_steps: 1,
            })
            .await
            .unwrap();
        assert_eq!(run.review_tasks[0].id, task.id);

        let rejected = state
            .reject_task_review(TaskReviewDecisionCommand {
                task_id: task.id.clone(),
                comment: Some("does not satisfy the task".into()),
            })
            .await
            .unwrap();

        assert_eq!(
            rejected.task.status,
            crytex_core::models::TaskStatus::Pending
        );
        assert_eq!(
            lora_evolution.counter_examples.lock().unwrap().as_slice(),
            [task.id]
        );
        assert!(lora_evolution.collected.lock().unwrap().is_empty());
        assert!(lora_evolution.trained.lock().unwrap().is_empty());
        state.shutdown_project_watchers().await;
    }

    #[tokio::test]
    async fn sqlite_state_rejects_goal_plan_with_feedback() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("crytex-ui.db");
        let state = CrytexAppState::new_sqlite_with_executor(&db_path, Arc::new(StubTaskExecutor))
            .await
            .unwrap();

        let project_root = dir.path().join("project");
        std::fs::create_dir_all(&project_root).unwrap();

        let project = state
            .create_project(CreateProjectCommand {
                name: "Manual Test".into(),
                root_path: project_root.display().to_string(),
            })
            .await
            .unwrap();

        let goal_plan = state
            .submit_goal(SubmitGoalCommand {
                project_id: project.id.clone(),
                goal: "Build the wrong thing".into(),
                context: json!({ "active_screen": "workspace" }),
                trace_id: Some("trace-reject-plan".into()),
            })
            .await
            .unwrap();

        let rejected = state
            .reject_plan(PlanDecisionCommand {
                goal_task_id: goal_plan.goal.id.clone(),
                comment: Some("split IDE work from model setup".into()),
            })
            .await
            .unwrap();

        assert_eq!(
            rejected.goal.status,
            crytex_core::models::TaskStatus::Pending
        );
        assert_eq!(rejected.goal.iteration_count, 1);
        assert_eq!(
            rejected.goal.payload["retry_feedback"][0]["comment"],
            "split IDE work from model setup"
        );
        assert!(
            rejected
                .generated_tasks
                .iter()
                .all(|generated| generated.status == crytex_core::models::TaskStatus::Cancelled)
        );
        state.shutdown_project_watchers().await;
    }

    #[tokio::test]
    async fn sqlite_state_repairs_old_stub_completed_tasks_on_startup() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("crytex-ui.db");

        {
            let state = CrytexAppState::new_sqlite(&db_path).await.unwrap();
            let project_root = dir.path().join("project");
            std::fs::create_dir_all(&project_root).unwrap();

            let project = state
                .create_project(CreateProjectCommand {
                    name: "Manual Test".into(),
                    root_path: project_root.display().to_string(),
                })
                .await
                .unwrap();

            let stub_task = state
                .submit_task(SubmitTaskCommand {
                    project_id: project.id.clone(),
                    parent_id: None,
                    title: "Old stub completed task".into(),
                    description: None,
                    kind: "codegen".into(),
                    assigned_agent: Some("coder".into()),
                    priority: 5,
                    payload: json!({}),
                    trace_id: Some("trace-old-stub".into()),
                })
                .await
                .unwrap();
            let real_task = state
                .submit_task(SubmitTaskCommand {
                    project_id: project.id.clone(),
                    parent_id: None,
                    title: "Real completed task".into(),
                    description: None,
                    kind: "codegen".into(),
                    assigned_agent: Some("coder".into()),
                    priority: 5,
                    payload: json!({}),
                    trace_id: Some("trace-real-completed".into()),
                })
                .await
                .unwrap();

            state
                .task_service
                .set_result(&stub_task.id, json!({ "source": "tauri_stub_run" }))
                .await
                .unwrap();
            state
                .task_service
                .set_result(&real_task.id, json!({ "source": "real_agent" }))
                .await
                .unwrap();
            state.shutdown_project_watchers().await;
        }

        let repaired = CrytexAppState::new_sqlite(&db_path).await.unwrap();
        let tasks = repaired.task_service.load_all_tasks().await.unwrap();
        let stub = tasks
            .iter()
            .find(|task| task.title == "Old stub completed task")
            .unwrap();
        let real = tasks
            .iter()
            .find(|task| task.title == "Real completed task")
            .unwrap();

        assert_eq!(stub.status, crytex_core::models::TaskStatus::Review);
        assert!(stub.finished_at.is_none());
        assert_eq!(real.status, crytex_core::models::TaskStatus::Completed);
        repaired.shutdown_project_watchers().await;
    }

    static ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    #[tokio::test]
    async fn app_state_fails_to_start_run_without_configured_backend() {
        let _guard = ENV_LOCK.lock().await;
        // SAFETY: env vars are process-global; this test holds a static lock to
        // avoid races with other tests that may read the same variable.
        let previous = std::env::var("CRYTEX_TAURI_OLLAMA_MODEL").ok();
        unsafe { std::env::remove_var("CRYTEX_TAURI_OLLAMA_MODEL") };

        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("crytex-no-backend.db");
        let state = CrytexAppState::new_sqlite(&db_path).await.unwrap();

        let project_root = dir.path().join("project");
        std::fs::create_dir_all(&project_root).unwrap();
        let project = state
            .create_project(CreateProjectCommand {
                name: "No Backend".into(),
                root_path: project_root.display().to_string(),
            })
            .await
            .unwrap();

        let result = state
            .start_run(StartRunCommand {
                project_id: project.id,
                max_steps: 10,
            })
            .await;

        if let Some(value) = previous {
            // SAFETY: same lock-held reasoning as above.
            unsafe { std::env::set_var("CRYTEX_TAURI_OLLAMA_MODEL", value) };
        }
        state.shutdown_project_watchers().await;

        let err = result.expect_err("start_run should fail without configured backend");
        let message = err.to_string();
        assert!(
            message.contains("no inference backend configured"),
            "unexpected error: {message}"
        );
    }
}
