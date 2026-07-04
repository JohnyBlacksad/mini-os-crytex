use std::sync::Arc;

use crate::{
    config::CrytexConfig,
    metrics::MetricsService,
    persistence::Persistence,
    services::{
        AgentService, AlertService, AuditLogService, EventService, InferenceService,
        LoraEvolutionService, LoraRouter, MemoryBankService, ModelManager, ProjectService,
        TaskService, ToolService,
    },
    tracing::TraceContext,
};

/// Composition root for the Crytex kernel.
///
/// `AppContext` holds all domain services and cross-cutting concerns.
/// It does not create dependencies internally; they are injected by the
/// application entry point (e.g. `crytex-kernel`).
#[derive(Clone)]
pub struct AppContext {
    pub config: CrytexConfig,
    pub trace_context: TraceContext,
    pub event_service: Arc<dyn EventService>,
    pub persistence: Arc<dyn Persistence>,
    pub project_service: Arc<dyn ProjectService>,
    pub task_service: Arc<dyn TaskService>,
    pub audit_service: Arc<dyn AuditLogService>,
    pub agent_service: Arc<dyn AgentService>,
    pub inference_service: Arc<dyn InferenceService>,
    pub model_manager: Arc<dyn ModelManager>,
    pub tool_service: Arc<dyn ToolService>,
    pub metrics_service: Arc<dyn MetricsService>,
    pub alert_service: Arc<dyn AlertService>,
    pub lora_evolution: Arc<dyn LoraEvolutionService>,
    pub lora_router: Arc<dyn LoraRouter>,
    pub memory_bank: Arc<dyn MemoryBankService>,
}

impl AppContext {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: CrytexConfig,
        trace_context: TraceContext,
        event_service: Arc<dyn EventService>,
        persistence: Arc<dyn Persistence>,
        project_service: Arc<dyn ProjectService>,
        task_service: Arc<dyn TaskService>,
        audit_service: Arc<dyn AuditLogService>,
        agent_service: Arc<dyn AgentService>,
        inference_service: Arc<dyn InferenceService>,
        model_manager: Arc<dyn ModelManager>,
        tool_service: Arc<dyn ToolService>,
        metrics_service: Arc<dyn MetricsService>,
        alert_service: Arc<dyn AlertService>,
        lora_evolution: Arc<dyn LoraEvolutionService>,
        lora_router: Arc<dyn LoraRouter>,
        memory_bank: Arc<dyn MemoryBankService>,
    ) -> Self {
        Self {
            config,
            trace_context,
            event_service,
            persistence,
            project_service,
            task_service,
            audit_service,
            agent_service,
            inference_service,
            model_manager,
            tool_service,
            metrics_service,
            alert_service,
            lora_evolution,
            lora_router,
            memory_bank,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::Event;
    use crate::config::BackendKind;
    use crate::config::CrytexConfig;
    use crate::metrics::{MetricsError, MetricsService, MetricsSnapshot};
    use crate::models::{
        AgentLog, Artifact, BenchmarkResult, BenchmarkRun, BenchmarkRunSummary, Experience,
        KanbanState, LoraAdapter, MemoryEntry, Project, ProjectSnapshot, PromptVersion, Task,
        TaskDependency, TaskStatus, TrainingExample, TrainingJob,
    };
    use crate::persistence::{
        ArtifactRepository, BenchmarkResultRepository, LogRepository, PersistenceError,
        ProjectRepository, ProjectSnapshotRepository, TaskRepository, TrainingJobRepository,
    };
    use crate::services::{
        AgentRole, AgentService, AgentServiceError, Alert, AlertService, AuditError, AuditLogEntry,
        AuditLogService, CreateProjectRequest, CreateTaskRequest, EventService, InferenceService,
        InferenceServiceError, LoraEvolutionError, LoraEvolutionService, LoraRouter,
        LoraRouterError, ManagedModel, MemoryBankError, MemoryBankService, ModelManagerError,
        ProjectError, ProjectService, Quantization, RecommendedConfig, TaskError, TaskService,
    };
    use crate::services::{ToolDescription, ToolService, ToolServiceError};
    use async_trait::async_trait;
    use std::sync::Arc;

    struct MockPersistence;

    #[async_trait]
    impl ProjectRepository for MockPersistence {
        async fn insert_project(&self, _project: &Project) -> Result<(), PersistenceError> {
            Ok(())
        }
        async fn get_project(&self, _id: &str) -> Result<Option<Project>, PersistenceError> {
            Ok(None)
        }
        async fn list_projects(&self) -> Result<Vec<Project>, PersistenceError> {
            Ok(vec![])
        }
    }

    #[async_trait]
    impl TaskRepository for MockPersistence {
        async fn insert_task(&self, _task: &Task) -> Result<(), PersistenceError> {
            Ok(())
        }
        async fn update_task(&self, _task: &Task) -> Result<(), PersistenceError> {
            Ok(())
        }
        async fn update_task_status(
            &self,
            _id: &str,
            _status: TaskStatus,
            _result: Option<serde_json::Value>,
        ) -> Result<(), PersistenceError> {
            Ok(())
        }
        async fn get_task(&self, _id: &str) -> Result<Option<Task>, PersistenceError> {
            Ok(None)
        }
        async fn list_tasks_by_project(
            &self,
            _project_id: &str,
        ) -> Result<Vec<Task>, PersistenceError> {
            Ok(vec![])
        }
        async fn list_all_tasks(&self) -> Result<Vec<Task>, PersistenceError> {
            Ok(vec![])
        }
        async fn list_ready_tasks(&self) -> Result<Vec<Task>, PersistenceError> {
            Ok(vec![])
        }
        async fn add_dependency(&self, _dep: &TaskDependency) -> Result<(), PersistenceError> {
            Ok(())
        }
    }

    #[async_trait]
    impl ArtifactRepository for MockPersistence {
        async fn insert_artifact(&self, _artifact: &Artifact) -> Result<(), PersistenceError> {
            Ok(())
        }
        async fn list_artifacts_by_task(
            &self,
            _task_id: &str,
        ) -> Result<Vec<Artifact>, PersistenceError> {
            Ok(vec![])
        }
    }

    #[async_trait]
    impl LogRepository for MockPersistence {
        async fn insert_agent_log(&self, _log: &AgentLog) -> Result<(), PersistenceError> {
            Ok(())
        }
        async fn list_logs_by_task(
            &self,
            _task_id: &str,
        ) -> Result<Vec<AgentLog>, PersistenceError> {
            Ok(vec![])
        }
        async fn list_logs_by_project(
            &self,
            _project_id: &str,
        ) -> Result<Vec<AgentLog>, PersistenceError> {
            Ok(vec![])
        }
    }

    #[async_trait]
    impl crate::persistence::ExperienceRepository for MockPersistence {
        async fn insert_experience(&self, _exp: &Experience) -> Result<(), PersistenceError> {
            Ok(())
        }
        async fn list_experiences_by_task(
            &self,
            _task_id: &str,
        ) -> Result<Vec<Experience>, PersistenceError> {
            Ok(vec![])
        }
        async fn list_experiences_by_prompt_version(
            &self,
            _prompt_version_id: &str,
        ) -> Result<Vec<Experience>, PersistenceError> {
            Ok(vec![])
        }
    }

    #[async_trait]
    impl crate::persistence::PromptVersionRepository for MockPersistence {
        async fn insert_prompt_version(
            &self,
            _version: &PromptVersion,
        ) -> Result<(), PersistenceError> {
            Ok(())
        }
        async fn update_prompt_version(
            &self,
            _version: &PromptVersion,
        ) -> Result<(), PersistenceError> {
            Ok(())
        }
        async fn get_prompt_version(
            &self,
            _id: &str,
        ) -> Result<Option<PromptVersion>, PersistenceError> {
            Ok(None)
        }
        async fn list_prompt_versions_by_agent(
            &self,
            _agent: &str,
        ) -> Result<Vec<PromptVersion>, PersistenceError> {
            Ok(vec![])
        }
        async fn get_active_prompt_version(
            &self,
            _agent: &str,
        ) -> Result<Option<PromptVersion>, PersistenceError> {
            Ok(None)
        }
        async fn set_active_prompt_version(
            &self,
            _id: &str,
            _agent: &str,
        ) -> Result<(), PersistenceError> {
            Ok(())
        }
    }

    #[async_trait]
    impl crate::persistence::TrainingExampleRepository for MockPersistence {
        async fn insert_training_example(
            &self,
            _example: &TrainingExample,
        ) -> Result<(), PersistenceError> {
            Ok(())
        }
        async fn list_training_examples_by_kind(
            &self,
            _task_kind: &str,
        ) -> Result<Vec<TrainingExample>, PersistenceError> {
            Ok(vec![])
        }
        async fn count_training_examples_by_kind(
            &self,
            _task_kind: &str,
        ) -> Result<usize, PersistenceError> {
            Ok(0)
        }
        async fn list_training_examples_by_project(
            &self,
            _project_id: &str,
        ) -> Result<Vec<TrainingExample>, PersistenceError> {
            Ok(vec![])
        }
        async fn list_training_examples_by_role(
            &self,
            _agent_role: &str,
        ) -> Result<Vec<TrainingExample>, PersistenceError> {
            Ok(vec![])
        }
        async fn count_training_examples_by_role(
            &self,
            _agent_role: &str,
        ) -> Result<usize, PersistenceError> {
            Ok(0)
        }
    }

    #[async_trait]
    impl crate::persistence::LoraAdapterRepository for MockPersistence {
        async fn insert_lora_adapter(
            &self,
            _adapter: &LoraAdapter,
        ) -> Result<(), PersistenceError> {
            Ok(())
        }
        async fn get_lora_adapter(
            &self,
            _id: &str,
        ) -> Result<Option<LoraAdapter>, PersistenceError> {
            Ok(None)
        }
        async fn list_lora_adapters_by_kind(
            &self,
            _task_kind: &str,
        ) -> Result<Vec<LoraAdapter>, PersistenceError> {
            Ok(vec![])
        }
        async fn list_lora_adapters_by_project(
            &self,
            _project_id: &str,
        ) -> Result<Vec<LoraAdapter>, PersistenceError> {
            Ok(vec![])
        }
        async fn list_lora_adapters_by_role(
            &self,
            _agent_role: &str,
        ) -> Result<Vec<LoraAdapter>, PersistenceError> {
            Ok(vec![])
        }
        async fn set_lora_adapter_active(
            &self,
            _id: &str,
            _active: bool,
        ) -> Result<(), PersistenceError> {
            Ok(())
        }
    }

    #[async_trait]
    impl crate::persistence::MemoryEntryRepository for MockPersistence {
        async fn insert_memory_entry(&self, _entry: &MemoryEntry) -> Result<(), PersistenceError> {
            Ok(())
        }
        async fn list_memory_entries(
            &self,
            _project_id: Option<&str>,
            _kind: Option<&str>,
            _limit: usize,
        ) -> Result<Vec<MemoryEntry>, PersistenceError> {
            Ok(vec![])
        }
        async fn list_memory_entries_by_session(
            &self,
            _session_id: &str,
        ) -> Result<Vec<MemoryEntry>, PersistenceError> {
            Ok(vec![])
        }
    }

    #[async_trait]
    impl TrainingJobRepository for MockPersistence {
        async fn insert_training_job(&self, _job: &TrainingJob) -> Result<(), PersistenceError> {
            Ok(())
        }
        async fn update_training_job(&self, _job: &TrainingJob) -> Result<(), PersistenceError> {
            Ok(())
        }
        async fn get_training_job(
            &self,
            _id: &str,
        ) -> Result<Option<TrainingJob>, PersistenceError> {
            Ok(None)
        }
        async fn list_training_jobs_by_kind(
            &self,
            _task_kind: &str,
        ) -> Result<Vec<TrainingJob>, PersistenceError> {
            Ok(vec![])
        }
    }

    #[async_trait]
    impl ProjectSnapshotRepository for MockPersistence {
        async fn insert_project_snapshot(
            &self,
            _snapshot: &ProjectSnapshot,
        ) -> Result<(), PersistenceError> {
            Ok(())
        }
        async fn get_project_snapshot(
            &self,
            _id: &str,
        ) -> Result<Option<ProjectSnapshot>, PersistenceError> {
            Ok(None)
        }
        async fn list_project_snapshots(
            &self,
            _project_id: &str,
        ) -> Result<Vec<ProjectSnapshot>, PersistenceError> {
            Ok(vec![])
        }
    }

    #[async_trait]
    impl BenchmarkResultRepository for MockPersistence {
        async fn insert_run(&self, _run: &BenchmarkRun) -> Result<(), PersistenceError> {
            Ok(())
        }
        async fn get_run(&self, _id: &str) -> Result<Option<BenchmarkRun>, PersistenceError> {
            Ok(None)
        }
        async fn list_runs(
            &self,
            _limit: usize,
        ) -> Result<Vec<BenchmarkRunSummary>, PersistenceError> {
            Ok(vec![])
        }
        async fn insert_result(
            &self,
            _run_id: &str,
            _result: &BenchmarkResult,
        ) -> Result<(), PersistenceError> {
            Ok(())
        }
        async fn list_results(
            &self,
            _run_id: &str,
        ) -> Result<Vec<BenchmarkResult>, PersistenceError> {
            Ok(vec![])
        }
    }

    struct MockProjectService;

    #[async_trait]
    impl ProjectService for MockProjectService {
        async fn create(
            &self,
            _request: CreateProjectRequest<'_>,
        ) -> Result<Project, ProjectError> {
            Ok(Project {
                id: "mock".into(),
                name: "mock".into(),
                root_path: "/tmp/mock".into(),
                created_at: 0,
                updated_at: 0,
                metadata: serde_json::Value::Null,
            })
        }

        async fn get(&self, _id: &str) -> Result<Option<Project>, ProjectError> {
            Ok(None)
        }

        async fn list(&self) -> Result<Vec<Project>, ProjectError> {
            Ok(vec![])
        }

        async fn update_metadata(
            &self,
            _id: &str,
            _metadata: serde_json::Value,
        ) -> Result<Project, ProjectError> {
            Ok(Project {
                id: "mock".into(),
                name: "mock".into(),
                root_path: "/tmp/mock".into(),
                created_at: 0,
                updated_at: 0,
                metadata: serde_json::Value::Null,
            })
        }

        async fn kanban_state(&self, _project_id: &str) -> Result<KanbanState, ProjectError> {
            Ok(KanbanState {
                project_id: "mock".into(),
                columns: vec![],
            })
        }
    }

    struct MockTaskService;

    #[async_trait]
    impl TaskService for MockTaskService {
        async fn submit(&self, _request: CreateTaskRequest) -> Result<Task, TaskError> {
            Err(TaskError::NotFound("mock".into()))
        }
        async fn add_dependency(&self, _dep: TaskDependency) -> Result<(), TaskError> {
            Ok(())
        }
        async fn get(&self, _id: &str) -> Result<Option<Task>, TaskError> {
            Ok(None)
        }
        async fn list_by_project(&self, _project_id: &str) -> Result<Vec<Task>, TaskError> {
            Ok(vec![])
        }
        async fn list_ready(&self) -> Result<Vec<Task>, TaskError> {
            Ok(vec![])
        }
        async fn set_status(&self, _id: &str, _status: TaskStatus) -> Result<Task, TaskError> {
            Err(TaskError::NotFound("mock".into()))
        }
        async fn cancel(&self, _id: &str) -> Result<Task, TaskError> {
            Err(TaskError::NotFound("mock".into()))
        }
        async fn set_result(
            &self,
            _id: &str,
            _result: serde_json::Value,
        ) -> Result<Task, TaskError> {
            Err(TaskError::NotFound("mock".into()))
        }
        async fn set_critic_score(&self, _id: &str, _score: f64) -> Result<Task, TaskError> {
            Err(TaskError::NotFound("mock".into()))
        }
        async fn set_human_score(&self, _id: &str, _score: f64) -> Result<Task, TaskError> {
            Err(TaskError::NotFound("mock".into()))
        }
        async fn retry(&self, _id: &str, _feedback: Option<&str>) -> Result<Task, TaskError> {
            Err(TaskError::NotFound("mock".into()))
        }
        async fn load_all_tasks(&self) -> Result<Vec<Task>, TaskError> {
            Ok(vec![])
        }
        async fn update_task(&self, _task: &Task) -> Result<(), TaskError> {
            Ok(())
        }
    }

    struct MockAuditService;

    #[async_trait]
    impl AuditLogService for MockAuditService {
        async fn log(&self, _entry: AuditLogEntry) -> Result<(), AuditError> {
            Ok(())
        }
        async fn list_by_task(&self, _task_id: &str) -> Result<Vec<AgentLog>, AuditError> {
            Ok(vec![])
        }
        async fn list_by_project(&self, _project_id: &str) -> Result<Vec<AgentLog>, AuditError> {
            Ok(vec![])
        }
    }

    struct MockEventService;

    #[async_trait]
    impl EventService for MockEventService {
        fn publish(&self, _event: Event) {}
        fn subscribe(&self) -> tokio::sync::broadcast::Receiver<Event> {
            let (tx, _) = tokio::sync::broadcast::channel(1);
            tx.subscribe()
        }
        async fn start_handler(&self, _handler: Arc<dyn crate::services::EventHandler>) {}
    }

    struct MockAgentService;

    #[async_trait]
    impl AgentService for MockAgentService {
        async fn register(&self, _agent: Arc<dyn crate::services::Agent>) {}
        async fn find(&self, _name: &str) -> Option<Arc<dyn crate::services::Agent>> {
            None
        }
        async fn list(&self) -> Vec<String> {
            vec![]
        }
        fn route(&self, _task: &Task) -> Option<String> {
            None
        }
        async fn execute(
            &self,
            _task: &Task,
            _inference: Arc<dyn InferenceService>,
            _tools: Arc<dyn ToolService>,
        ) -> Result<serde_json::Value, AgentServiceError> {
            Ok(serde_json::Value::Null)
        }
    }

    struct MockInferenceService;

    #[async_trait]
    impl InferenceService for MockInferenceService {
        async fn generate(
            &self,
            _request: crytex_inference::InferenceRequest,
        ) -> Result<crytex_inference::InferenceResponse, InferenceServiceError> {
            Err(InferenceServiceError::NoBackend)
        }
        async fn embed(&self, _text: &str) -> Result<Vec<f32>, InferenceServiceError> {
            Ok(vec![])
        }
        fn available_backends(&self) -> Vec<crytex_inference::BackendInfo> {
            vec![]
        }
        async fn register_lora(
            &self,
            _lora: crytex_inference::LoRAAdapter,
        ) -> Result<(), InferenceServiceError> {
            Ok(())
        }
        async fn swap_lora(&self, _lora_id: &str) -> Result<(), InferenceServiceError> {
            Ok(())
        }
        async fn list_models(
            &self,
            _backend_id: Option<&str>,
        ) -> Result<Vec<crytex_inference::ModelInfo>, InferenceServiceError> {
            Ok(vec![])
        }
    }

    struct MockModelManager;

    #[async_trait]
    impl ModelManager for MockModelManager {
        fn list_models(&self) -> Result<Vec<ManagedModel>, ModelManagerError> {
            Ok(vec![])
        }
        fn get_model(&self, _id: &str) -> Result<ManagedModel, ModelManagerError> {
            Err(ModelManagerError::NotFound("mock".into()))
        }
        async fn download_model(&self, _id: &str) -> Result<ManagedModel, ModelManagerError> {
            Err(ModelManagerError::NotFound("mock".into()))
        }
        fn recommend_config(&self, _id: &str) -> Result<RecommendedConfig, ModelManagerError> {
            Ok(RecommendedConfig {
                backend: BackendKind::MistralRs,
                quantization: Quantization::Q4KM,
                gpu_layers: Some(0),
                context_size: 4096,
            })
        }
    }

    struct MockToolService;

    #[async_trait]
    impl ToolService for MockToolService {
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

    struct MockMetricsService;

    #[async_trait]
    impl MetricsService for MockMetricsService {
        async fn snapshot(&self) -> Result<MetricsSnapshot, MetricsError> {
            Ok(MetricsSnapshot::default())
        }
        async fn record_task_completion(
            &self,
            _latency_ms: u64,
            _success: bool,
        ) -> Result<(), MetricsError> {
            Ok(())
        }
        async fn record_cache_hit(&self) -> Result<(), MetricsError> {
            Ok(())
        }
        async fn record_cache_miss(&self) -> Result<(), MetricsError> {
            Ok(())
        }
        async fn history(
            &self,
            _from: i64,
            _to: i64,
        ) -> Result<Vec<MetricsSnapshot>, MetricsError> {
            Ok(vec![])
        }
    }

    struct MockMemoryBankService;

    #[async_trait]
    impl MemoryBankService for MockMemoryBankService {
        async fn remember(&self, _entry: &MemoryEntry) -> Result<(), MemoryBankError> {
            Ok(())
        }
        async fn recall(
            &self,
            _project_id: Option<&str>,
            _kind: Option<&str>,
            _limit: usize,
        ) -> Result<Vec<MemoryEntry>, MemoryBankError> {
            Ok(vec![])
        }
        async fn recall_semantic(
            &self,
            _project_id: Option<&str>,
            _query: &str,
            _limit: usize,
        ) -> Result<Vec<MemoryEntry>, MemoryBankError> {
            Ok(vec![])
        }
        async fn summarize_session(
            &self,
            _session_id: &str,
        ) -> Result<Option<String>, MemoryBankError> {
            Ok(None)
        }
        async fn mental_model_for_project(
            &self,
            _project_id: &str,
        ) -> Result<serde_json::Value, MemoryBankError> {
            Ok(serde_json::Value::Null)
        }
    }

    struct MockAlertService;

    #[async_trait]
    impl AlertService for MockAlertService {
        async fn check(&self, _snapshot: &MetricsSnapshot) -> Vec<Alert> {
            vec![]
        }
    }

    struct MockLoraEvolution;

    #[async_trait]
    impl LoraEvolutionService for MockLoraEvolution {
        async fn collect_golden_example(&self, _task_id: &str) -> Result<(), LoraEvolutionError> {
            Ok(())
        }
        async fn collect_counter_example(&self, _task_id: &str) -> Result<(), LoraEvolutionError> {
            Ok(())
        }
        async fn should_train(&self, _task_kind: &str) -> Result<bool, LoraEvolutionError> {
            Ok(false)
        }
        async fn train_and_register(
            &self,
            _task_kind: &str,
        ) -> Result<LoraAdapter, LoraEvolutionError> {
            unimplemented!()
        }
        async fn should_train_for_role(
            &self,
            _role: AgentRole,
        ) -> Result<bool, LoraEvolutionError> {
            Ok(false)
        }
        async fn train_and_register_for_role(
            &self,
            _role: AgentRole,
        ) -> Result<LoraAdapter, LoraEvolutionError> {
            unimplemented!()
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

    struct MockLoraRouter;

    #[async_trait]
    impl LoraRouter for MockLoraRouter {
        async fn resolve(
            &self,
            _task: &Task,
            _project_id: &str,
        ) -> Result<Option<String>, LoraRouterError> {
            Ok(None)
        }
        async fn resolve_for_role(
            &self,
            _role: crate::services::AgentRole,
            _project_id: &str,
        ) -> Result<Option<String>, LoraRouterError> {
            Ok(None)
        }
    }

    #[test]
    fn app_context_holds_dependencies() {
        let config = CrytexConfig::default();
        let event_service: Arc<dyn EventService> = Arc::new(MockEventService);
        let persistence: Arc<dyn Persistence> = Arc::new(MockPersistence);
        let project_service: Arc<dyn ProjectService> = Arc::new(MockProjectService);
        let task_service: Arc<dyn TaskService> = Arc::new(MockTaskService);
        let audit_service: Arc<dyn AuditLogService> = Arc::new(MockAuditService);
        let agent_service: Arc<dyn AgentService> = Arc::new(MockAgentService);
        let inference_service: Arc<dyn InferenceService> = Arc::new(MockInferenceService);
        let model_manager: Arc<dyn ModelManager> = Arc::new(MockModelManager);
        let tool_service: Arc<dyn ToolService> = Arc::new(MockToolService);
        let metrics_service: Arc<dyn MetricsService> = Arc::new(MockMetricsService);
        let alert_service: Arc<dyn AlertService> = Arc::new(MockAlertService);

        let lora_evolution: Arc<dyn LoraEvolutionService> = Arc::new(MockLoraEvolution);
        let lora_router: Arc<dyn LoraRouter> = Arc::new(MockLoraRouter);
        let memory_bank: Arc<dyn MemoryBankService> = Arc::new(MockMemoryBankService);

        let trace_context = TraceContext::new();
        let ctx = AppContext::new(
            config.clone(),
            trace_context.clone(),
            event_service.clone(),
            persistence.clone(),
            project_service.clone(),
            task_service.clone(),
            audit_service.clone(),
            agent_service.clone(),
            inference_service.clone(),
            model_manager.clone(),
            tool_service.clone(),
            metrics_service.clone(),
            alert_service.clone(),
            lora_evolution.clone(),
            lora_router.clone(),
            memory_bank.clone(),
        );

        assert_eq!(ctx.config.max_concurrent_tasks, config.max_concurrent_tasks);
        assert_eq!(ctx.trace_context.trace_id, trace_context.trace_id);
    }
}
