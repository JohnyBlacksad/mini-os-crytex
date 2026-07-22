//! Domain services for the Crytex kernel.
//!
//! Each service is defined by a trait (contract) and one or more implementations.
//! This keeps the core decoupled from concrete infrastructure.

pub mod agent_service;
pub mod alert_service;
pub mod artifact_contract;
pub mod audit_log_service;
pub mod bulk_audit_log;
pub mod caching;
pub mod context_assembler;
pub mod critic_council;
pub mod embedder;
pub mod event_service;
pub mod hardware;
pub mod hybrid;
pub mod inference_service;
pub mod kanban_projection;
pub mod lora_evolution;
pub mod lora_router;
pub mod lora_trainer;
pub mod memory_bank;
pub mod model_compatibility;
pub mod model_manager;
pub mod model_runtime_probe;
pub mod orchestrator;
pub mod project_service;
pub mod project_watcher;
pub mod prompt_evolution;
pub mod rate_limit;
pub mod reranker;
pub mod reward_service;
pub mod role;
pub mod role_quality;
pub mod sandbox_service;
pub mod scheduler;
pub mod task_service;
pub mod tool_service;
pub mod vector_store;
pub mod worker;
pub mod workflow;

pub use crate::rag_pipeline::{
    CrashSafeRebuildReport, IncrementalReindexReport, PromptInjectionSeverity,
    RagCandidateEvidence, RagDiagnostics, RagPipeline, RagPipelineError, RagPipelineRequest,
    RagPipelineResponse,
};
pub use agent_service::{Agent, AgentError, AgentService, AgentServiceError, AgentServiceImpl};
pub use alert_service::{Alert, AlertService, AlertServiceImpl, AlertSeverity, AlertThresholds};
pub use artifact_contract::{
    ArtifactContractViolation, artifact_content_from_result, artifact_kind_for_agent,
    requires_agent_artifact_contract, validate_agent_result, validate_artifact_content,
};
pub use audit_log_service::{
    AuditError, AuditEvent, AuditLogEntry, AuditLogService, AuditLogServiceImpl,
};
pub use bulk_audit_log::BulkAuditLogService;
pub use context_assembler::{
    ContextAssembler, ContextAssemblerError, ContextRequest, RagChunkEvidence,
};
pub use critic_council::{CriticCouncil, CriticCouncilError};
pub use embedder::{Embedder, EmbeddingError, MockEmbedder, MockSparseEmbedder, SparseEmbedder};
pub use event_service::{EventHandler, EventService, EventServiceImpl};
pub use hardware::{
    CudaToolchainProbe, CudaToolchainStatus, DeviceKind, HardwareDetector, HardwareRecommendation,
    SystemHardwareDetector, build_cuda_toolchain_status, detect_cuda_toolchain_status,
    recommend_local_device,
};
pub use hybrid::{
    DistributionBasedScoreFusion, FusionStrategy, FusionStrategyKind, HybridRetriever,
    HybridSearchError, RankedList, RankedResult, ReciprocalRankFusion, RetrieverSource,
    build_fusion_strategy,
};
pub use inference_service::{InferenceService, InferenceServiceError, InferenceServiceImpl};
pub use kanban_projection::{
    KanbanBoardProjection, KanbanColumnProjection, KanbanHistoryProjection, KanbanMovement,
    KanbanProjectionError, KanbanProjectionService, KanbanRunSelector, KanbanStatus,
    KanbanTaskProjection,
};
pub use lora_evolution::{
    LoraBenchmarkDecision, LoraBenchmarkGate, LoraBenchmarkRequest, LoraDatasetBalancingReport,
    LoraDatasetInspector, LoraDatasetLeakageReport, LoraDatasetLowInformationReport,
    LoraDatasetReport, LoraEvolutionError, LoraEvolutionService, LoraEvolutionServiceImpl,
    LoraQualityGateName, LoraQualityGateResult, lora_quality_gate, passed_lora_quality_gates,
};
pub use lora_router::{
    LoraRouter, LoraRouterError, LoraRouterImpl, LoraSelection, MemoryRoleAdapterRegistry,
    RoleAdapterRegistry,
};
pub use lora_trainer::{
    AdapterArtifactValidator, AdapterMetadata, LoraMetrics, LoraTrainer, LoraTrainingConfig,
    LoraTrainingError, LoraTrainingObjective, LoraTrainingResult, dataset_hash,
    validate_objective_examples,
};
pub use memory_bank::{
    MemoryBankError, MemoryBankService, MemoryBankServiceImpl, MemoryEntryBuilder,
};
pub use model_compatibility::{
    CompatibilityStatus, ExecutionStrategy, ModelCompatibilityPlan, ModelCompatibilityPlanner,
    ModelFeature, ModelFormat, ModelSupportStatus, RuntimeFeatureSet,
};
pub use model_manager::{
    FileSystemManifestSource, HardwareModelRecommender, HfGgufResolution, HfGgufResolveRequest,
    HfGgufVariant, HfHubDownloader, ManagedModel, ManifestEntry, ModelDownloader, ModelManager,
    ModelManagerError, ModelManagerImpl, ModelManifestSource, ModelRecommender, ModelRegistryStore,
    ModelStatus, Quantization, RecommendedConfig, RegistryEntry, TomlRegistryStore,
};
pub use model_runtime_probe::{
    ModelRuntimeMatrixProbe, ModelRuntimeMatrixReport, ModelRuntimeMatrixRequest,
    ModelRuntimeProbe, ModelRuntimeProbeReport, ModelRuntimeProbeRequest, ProbeStageName,
    ProbeStageReport, ProbeStageStatus, RuntimeMatrixEntryReport, RuntimeMatrixEntryRequest,
    RuntimeMatrixReportWriter, RuntimeMatrixSummary,
};
pub use orchestrator::{Orchestrator, OrchestratorError, OrchestratorImpl};
pub use project_service::{CreateProjectRequest, ProjectError, ProjectService, ProjectServiceImpl};
pub use project_watcher::{ProjectWatcher, WatcherError};
pub use prompt_evolution::{
    FailureRoute, MutationOperator, PromptBenchmarkDecision, PromptBenchmarkGate,
    PromptBenchmarkRequest, PromptDecisionKind, PromptEvolutionDecisionReport,
    PromptEvolutionError, PromptEvolutionService, PromptFailureKind, PromptFailureRouter,
    PromptProposal,
};
pub use rate_limit::{RetryPolicy, RetryRateLimitBackend, TokenBucket};
pub use reranker::{RerankPassage, RerankResult, Reranker, RerankerError};
pub use reward_service::{RecordRewardRequest, RewardService, RewardServiceError};
pub use role::AgentRole;
pub use role_quality::{
    RoleArtifactContract, RoleBenchmarkFixture, RoleLoraHotSwapReport, RoleQualityCatalog,
    RoleQualityContract, RoleQualityGate, RoleQualityProof, RoleQualityProofReport,
    RoleSmokeReport,
};
pub use sandbox_service::{
    ExecutionRequest, ExecutionResult, SandboxMount, SandboxNetwork, SandboxResources,
    SandboxService, SandboxServiceError,
};
pub use scheduler::{Scheduler, SchedulerError, SchedulerImpl};
pub use task_service::{CreateTaskRequest, TaskError, TaskService, TaskServiceImpl};
pub use tool_service::{ToolDescription, ToolService, ToolServiceError};
pub use vector_store::{
    SearchOptions, SearchResult, SparseVector, SparseVectorPoint, VectorPoint, VectorStore,
    VectorStoreError,
};
pub use worker::{TaskHandler, WorkerError, WorkerPool};
pub use workflow::{
    AgentWorkflowNodeExecutor, BackoffStrategy, MemoryWorkflowRepository, TomlWorkflowRepository,
    WorkflowDefinition, WorkflowEdge, WorkflowEngine, WorkflowError, WorkflowNode,
    WorkflowNodeExecutor, WorkflowRepository, WorkflowResult, WorkflowRetryPolicy, WorkflowState,
};

// AuditLogLevel is defined in models.rs; re-export it here for ergonomic service API usage.
pub use crate::models::AuditLogLevel;
