use crytex_core::{
    AppContext, CrytexTelemetry,
    bus::Event,
    config::CrytexConfig,
    metrics::MetricsService,
    services::{
        AgentServiceImpl, AlertService, AlertServiceImpl, AlertThresholds, AuditLogServiceImpl,
        Embedder, EventServiceImpl, InferenceServiceImpl, LoraEvolutionService,
        LoraEvolutionServiceImpl, LoraRouter, LoraRouterImpl, MemoryBankService,
        MemoryBankServiceImpl, ModelManagerImpl, ProjectServiceImpl, RetryRateLimitBackend,
        SandboxService, SystemHardwareDetector, TaskServiceImpl, ToolService, VectorStore,
    },
};
use crytex_inference::BackendRegistry;
use crytex_sandbox::SandboxOrchestrator;
use crytex_storage::Storage;
use crytex_storage::vector::memory::MemoryVectorStore;
use crytex_tools::{Capability, ScanningToolService, ToolServiceImpl, TypedToolRegistry};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{info, warn};

#[tokio::main]
async fn main() {
    CrytexTelemetry::init();

    info!("Starting crytex loader...");

    let config = CrytexConfig::load();
    if let Err(e) = config.ensure_dirs() {
        warn!("Failed to create data directories: {}", e);
    }

    let mut registry = BackendRegistry::new("mistral");
    registry.register(
        "mistral",
        Arc::new(RetryRateLimitBackend::default_for(Arc::new(
            crytex_inference_mistral::MistralRsBackend::new("default", 4096, None),
        ))),
    );
    let inference = Arc::new(InferenceServiceImpl::new(
        Arc::new(registry),
        Some("mistral".to_string()),
    ));
    let vector_store: Arc<dyn VectorStore> = Arc::new(MemoryVectorStore::new());
    let embedder: Arc<dyn Embedder> = Arc::new(InferenceEmbedder(inference.clone()));

    let storage = match Storage::new(&config.paths.db_path.to_string_lossy())
        .await
        .map(|s| s.with_experience_vector_store(embedder.clone(), vector_store.clone()))
    {
        Ok(s) => Arc::new(s),
        Err(e) => {
            warn!("Failed to initialize storage: {}", e);
            std::process::exit(1);
        }
    };

    let event_bus = Arc::new(crytex_core::EventBus::new());
    let event_service = Arc::new(EventServiceImpl::new(event_bus));
    let project_service = Arc::new(ProjectServiceImpl::new(storage.clone()));
    let audit_service = Arc::new(AuditLogServiceImpl::new(storage.clone()));
    let task_service = Arc::new(TaskServiceImpl::new(
        storage.clone(),
        event_service.clone(),
        audit_service.clone(),
    ));
    let scanner: Arc<dyn crytex_core::security::SecurityScanner> =
        Arc::new(crytex_core::security::RegexSecurityScanner::new());

    let sandbox_service: Arc<dyn SandboxService> = Arc::new(SandboxOrchestrator::auto().await);
    let tool_registry = TypedToolRegistry::new().with_default_coding_tools().build();
    let project_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    let tool_factory: Arc<dyn Fn(Capability) -> Arc<dyn ToolService> + Send + Sync> = {
        let registry = tool_registry.clone();
        let sandbox = sandbox_service.clone();
        let scanner = scanner.clone();
        let security_config = config.security.clone();
        let project_root = project_root.clone();
        Arc::new(move |permissions| {
            let inner = ToolServiceImpl::new(
                registry.clone(),
                project_root.clone(),
                permissions,
                sandbox.clone(),
                Some(scanner.clone()),
                security_config.clone(),
            );
            Arc::new(ScanningToolService::new(Arc::new(inner), scanner.clone()))
        })
    };

    let agent_service = Arc::new(
        AgentServiceImpl::new(audit_service.clone())
            .with_scanner(scanner.clone())
            .with_tool_factory(tool_factory),
    );
    let config_dir = CrytexConfig::config_path()
        .parent()
        .expect("config path must have a parent")
        .to_path_buf();
    let model_manager: Arc<dyn crytex_core::services::ModelManager> =
        Arc::new(ModelManagerImpl::new_standard(
            &config_dir,
            &config.paths.data_dir,
            event_service.clone(),
            Arc::new(SystemHardwareDetector::new()),
        ));
    let tool_service: Arc<dyn ToolService> = Arc::new(ScanningToolService::new(
        Arc::new(ToolServiceImpl::new(
            tool_registry,
            project_root,
            Capability::all(),
            sandbox_service,
            Some(scanner.clone()),
            config.security.clone(),
        )),
        scanner.clone(),
    ));
    let metrics_service: Arc<dyn MetricsService> = Arc::new(
        crytex_core::metrics::MetricsServiceImpl::new(storage.clone()),
    );
    let alert_service: Arc<dyn AlertService> = Arc::new(AlertServiceImpl::new(
        AlertThresholds::default(),
        event_service.clone(),
    ));

    let adapters_dir = config.paths.data_dir.join("adapters");
    let base_model = config
        .inference
        .default_backend_config()
        .map(|b| b.model.clone())
        .unwrap_or_default();
    let lora_evolution: Arc<dyn LoraEvolutionService> = Arc::new(
        LoraEvolutionServiceImpl::new(
            task_service.clone(),
            storage.clone(),
            storage.clone(),
            storage.clone(),
            inference.clone(),
            event_service.clone(),
            Arc::new(crytex_inference_candle::CandleLoraTrainer::new()),
            adapters_dir,
            base_model,
        )
        .with_threshold(50)
        .with_vector_index(embedder.clone(), vector_store.clone()),
    );
    let lora_router: Arc<dyn LoraRouter> = Arc::new(
        LoraRouterImpl::new(lora_evolution.clone())
            .with_semantic_fallback(embedder.clone(), vector_store.clone()),
    );
    let memory_bank: Arc<dyn MemoryBankService> = Arc::new(
        MemoryBankServiceImpl::new(storage.clone())
            .with_semantic_index(embedder.clone(), vector_store.clone()),
    );

    let ctx = AppContext::new(
        config,
        crytex_core::tracing::TraceContext::new(),
        event_service.clone(),
        storage,
        project_service,
        task_service,
        audit_service,
        agent_service,
        inference,
        model_manager,
        tool_service,
        metrics_service,
        alert_service,
        lora_evolution.clone(),
        lora_router.clone(),
        memory_bank.clone(),
    );

    let mut rx = ctx.event_service.subscribe();
    while let Ok(event) = rx.recv().await {
        match event {
            Event::TaskCreated {
                task_id,
                project_id,
            } => {
                info!("Task {} created in project {}", task_id, project_id);
            }
            Event::TaskStarted { task_id } => {
                info!("Task {} started", task_id);
            }
            Event::TaskCompleted { task_id, result } => {
                info!("Task {} completed: {:?}", task_id, result);
            }
            Event::TaskFailed { task_id, error } => {
                warn!("Task {} failed: {}", task_id, error);
            }
            _ => {}
        }
    }
}

/// Thin wrapper exposing an inference service as an [`Embedder`].
struct InferenceEmbedder(Arc<dyn crytex_core::services::InferenceService>);

#[async_trait::async_trait]
impl Embedder for InferenceEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, crytex_core::services::EmbeddingError> {
        self.0
            .embed(text)
            .await
            .map_err(|e| crytex_core::services::EmbeddingError::EmbeddingFailed(e.to_string()))
    }

    async fn dimension(&self) -> Result<usize, crytex_core::services::EmbeddingError> {
        let vector = self
            .0
            .embed("crytex::embedder_dimension_probe")
            .await
            .map_err(|e| crytex_core::services::EmbeddingError::EmbeddingFailed(e.to_string()))?;
        Ok(vector.len())
    }
}
