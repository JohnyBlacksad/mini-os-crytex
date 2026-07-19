#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

mod factory;

use crate::factory::{
    create_embedder, create_hybrid_retriever, create_lora_evolution_service, create_lora_router,
    create_memory_bank_service, create_project_indexer, create_reranker, create_sparse_embedder,
    create_vector_store,
};
use async_trait::async_trait;
use clap::{Parser, Subcommand};
use crytex_agents::critics::{
    code::CodeCriticAgent, security::SecurityCriticAgent, style::StyleCriticAgent,
    test::TestCriticAgent,
};
use crytex_agents::{
    architect::ArchitectAgent, coder::CoderAgent, critic::CriticAgent, qa::QaAgent,
    researcher::ResearcherAgent, security::SecurityAgent, summarizer::SummarizerAgent,
};
use crytex_bench::{
    ABTest, AgentBenchmarkRunner, BenchLoraBenchmarkGate, BenchmarkHarness, BenchmarkRunRequest,
    BenchmarkVariant, DefaultBenchmarkHarness, ExactMatchScorer, JsonSchemaScorer, LlmJudgeScorer,
    SandboxTestScorer,
};
use crytex_compress::{
    DiskCcrStore,
    compressors::{
        CodeCompressor, DiffCompressor, JsonCompressor, LogCompressor, SearchCompressor,
        SmartCompressor, TextCompressor, TruncateCompressor,
    },
    content::ContentType,
    pipeline::CompressionPipeline,
    tokenizer::TokenizerEstimator,
};
use crytex_core::persistence::ExperienceRepository;
use crytex_core::services::{SandboxService, ToolService};
use crytex_core::{
    AppContext, CrytexTelemetry,
    bus::Event,
    config::{BackendConfig, BackendKind, CrytexConfig},
    metrics::MetricsService,
    models::{LoraAdapter, ProjectSnapshot, Task, TaskStatus},
    persistence::PromptVersionRepository,
    services::{
        AgentRole, AgentService, AgentServiceImpl, AgentWorkflowNodeExecutor, AlertService,
        AlertServiceImpl, AlertThresholds, BulkAuditLogService, CreateProjectRequest,
        CreateTaskRequest, CriticCouncil, EventServiceImpl, HfGgufResolveRequest,
        InferenceServiceImpl, LoraRouter, ModelManager, ModelManagerImpl, ModelRuntimeMatrixProbe,
        ModelRuntimeMatrixRequest, ModelRuntimeProbe, ModelRuntimeProbeRequest, MutationOperator,
        Orchestrator, OrchestratorImpl, ProjectService, ProjectServiceImpl, ProjectWatcher,
        PromptEvolutionService, Quantization, RecordRewardRequest, RewardService,
        RuntimeFeatureSet, RuntimeMatrixEntryRequest, RuntimeMatrixReportWriter, SchedulerImpl,
        SystemHardwareDetector, TaskHandler, TaskServiceImpl, TomlWorkflowRepository, WorkerError,
        WorkerPool, WorkflowRepository, recommend_local_device,
    },
    state_export::export_project_state,
};
use crytex_doc::graph::{CodeGraph, builder::CodeGraphBuilder};
use crytex_ide::ide_service::start_ide_bridge;
use crytex_inference::BackendRegistry;
use crytex_sandbox::SandboxOrchestrator;
use crytex_storage::Storage;
use crytex_tools::{Capability, ScanningToolService, ToolServiceImpl, TypedToolRegistry};
use serde::Serialize;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{error, info, warn};
use ulid::Ulid;

/// Print an error message and exit the CLI process gracefully.
macro_rules! unwrap_or_exit {
    ($expr:expr, $msg:expr) => {
        match $expr {
            Ok(value) => value,
            Err(err) => {
                eprintln!("{}: {}", $msg, err);
                std::process::exit(1);
            }
        }
    };
}

macro_rules! require_or_exit {
    ($expr:expr, $msg:expr) => {
        match $expr {
            Some(value) => value,
            None => {
                eprintln!("{}", $msg);
                std::process::exit(1);
            }
        }
    };
}

#[derive(Parser)]
#[command(name = "crytex-kernel")]
#[command(about = "Crytex autonomous coding kernel")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Create a new project
    CreateProject {
        #[arg(short, long)]
        name: String,
        #[arg(short, long)]
        path: String,
    },
    /// List all projects
    ListProjects,
    /// Submit a new task
    Submit {
        #[arg(short, long)]
        project: String,
        #[arg(short = 'm', long)]
        prompt: String,
        #[arg(short, long, default_value = "generic")]
        kind: String,
        #[arg(short, long)]
        backend: Option<String>,
    },
    /// List tasks in a project
    ListTasks {
        #[arg(short, long)]
        project: String,
    },
    /// Show task details
    ShowTask {
        #[arg(short, long)]
        id: String,
    },
    /// List configured inference backends
    ListBackends,
    /// List models available from a backend or from the model manager
    ListModels {
        #[arg(short, long)]
        backend: Option<String>,
    },
    /// Download a model from HuggingFace
    DownloadModel {
        #[arg(short, long)]
        id: String,
        #[arg(long)]
        activate: bool,
        #[arg(long, default_value = "local-hf")]
        backend_id: String,
    },
    /// Prove HuggingFace GGUF download, activation, load, and generation as one JSON artifact
    ProveHfModel {
        #[arg(short, long)]
        id: String,
        #[arg(short, long)]
        name: Option<String>,
        #[arg(short, long)]
        repo: String,
        #[arg(short, long)]
        filename: Option<String>,
        #[arg(short, long)]
        quantization: Option<String>,
        #[arg(long)]
        params_b: Option<f32>,
        #[arg(long, default_value = "local-hf-proof")]
        backend_id: String,
        #[arg(long)]
        trace_id: Option<String>,
        #[arg(long, default_value = "16")]
        max_tokens: usize,
        #[arg(long, default_value = "120")]
        timeout_seconds: u64,
        #[arg(long)]
        report_path: Option<PathBuf>,
    },
    /// Add or update a managed HuggingFace/local model entry
    AddModel {
        #[arg(short, long)]
        id: String,
        #[arg(short, long)]
        name: Option<String>,
        #[arg(short, long)]
        repo: Option<String>,
        #[arg(short, long)]
        filename: Option<String>,
        #[arg(short, long)]
        quantization: Option<String>,
        #[arg(short, long, default_value = "mistralrs")]
        backend: String,
        #[arg(long)]
        params_b: Option<f32>,
    },
    /// Show details for a managed model
    ShowModel {
        #[arg(short, long)]
        id: String,
    },
    /// Recommend runtime configuration for a managed model
    RecommendModel {
        #[arg(short, long)]
        id: String,
    },
    /// Resolve the best GGUF file from a HuggingFace model repo
    ResolveHfGguf {
        #[arg(short, long)]
        repo: String,
        #[arg(short, long)]
        quantization: Option<String>,
        #[arg(long)]
        params_b: Option<f32>,
    },
    /// Run metadata, compatibility, and generation smoke probe for a managed model
    ProbeModel {
        #[arg(short, long)]
        id: String,
        #[arg(short, long)]
        backend: Option<String>,
        #[arg(short, long)]
        model: Option<String>,
        #[arg(long)]
        trace_id: Option<String>,
        #[arg(long, default_value = "16")]
        max_tokens: usize,
        #[arg(long)]
        timeout_seconds: Option<u64>,
    },
    /// Run baseline and LoRA runtime probe matrix for a managed model
    ProbeRuntimeMatrix {
        #[arg(short, long)]
        id: String,
        #[arg(short, long)]
        backend: Vec<String>,
        #[arg(short, long)]
        model: Option<String>,
        #[arg(long)]
        lora: Vec<String>,
        #[arg(long)]
        trace_id: Option<String>,
        #[arg(long)]
        report_dir: Option<PathBuf>,
        #[arg(long, default_value = "16")]
        max_tokens: usize,
    },
    /// Switch the default inference backend
    SwitchBackend {
        #[arg(short, long)]
        id: String,
    },
    /// Add an external backend (Ollama, OpenAI-compatible, Anthropic, or custom)
    AddBackend {
        #[arg(short, long)]
        id: String,
        #[arg(short, long)]
        kind: String,
        #[arg(short, long)]
        model: String,
        #[arg(short, long)]
        url: Option<String>,
        #[arg(short, long)]
        api_key: Option<String>,
        #[arg(short = 'H', long)]
        header: Vec<String>,
        #[arg(short = 'g', long)]
        gpu_layers: Option<usize>,
        #[arg(short = 'c', long)]
        context_size: Option<usize>,
    },
    /// Run the execution loop
    Run,
    /// Index a project directory into the vector store
    Index {
        #[arg(short, long)]
        project_id: String,
        #[arg(short, long)]
        path: PathBuf,
    },
    /// Stream metrics snapshots to stdout as NDJSON
    WatchMetrics {
        #[arg(short, long, default_value = "60")]
        interval_secs: u64,
    },
    /// Export full project state as JSON
    State {
        #[arg(short, long)]
        project: String,
        #[arg(long)]
        json: bool,
    },
    /// Approve a task that is in review
    Approve {
        #[arg(short, long)]
        id: String,
        #[arg(long)]
        score: Option<f64>,
    },
    /// Reject a task that is in review and return it to the queue for retry
    Reject {
        #[arg(short, long)]
        id: String,
        #[arg(long)]
        score: Option<f64>,
        #[arg(long)]
        comment: Option<String>,
    },
    /// List prompt versions for an agent
    Prompts {
        #[arg(short, long)]
        agent: String,
        #[arg(long)]
        json: bool,
    },
    /// Evolve the active prompt for an agent
    EvolvePrompt {
        #[arg(short, long)]
        agent: String,
        #[arg(short, long, default_value = "rephrase")]
        operator: String,
    },
    /// Manage LoRA adapters
    Lora {
        #[command(subcommand)]
        command: LoraCommands,
    },
    /// Run a benchmark against a golden set
    Bench {
        #[command(subcommand)]
        command: BenchCommands,
    },
    /// Compare two benchmark runs (A/B test)
    ABTest {
        #[command(subcommand)]
        command: ABTestCommands,
    },
}

#[derive(Subcommand)]
enum LoraCommands {
    /// List registered adapters
    List {
        #[arg(short, long)]
        project: Option<String>,
    },
    /// Register a new adapter from a file
    Register {
        id: String,
        path: String,
        #[arg(long)]
        base_model: String,
        #[arg(long)]
        kind: Option<String>,
    },
    /// Activate an adapter
    Swap { id: String },
    /// Select an adapter for a project and persist the choice as a snapshot
    Select { project: String, adapter: String },
    /// Train a new adapter for a task kind
    Train { kind: String },
    /// Bind an adapter to a canonical agent role
    SelectRole { role: String, adapter: String },
    /// List role -> adapter bindings
    ListRoles,
}

#[derive(Subcommand)]
enum BenchCommands {
    /// Run a benchmark against a golden set
    Run {
        #[arg(short, long)]
        name: String,
        #[arg(short, long)]
        golden_set: PathBuf,
        #[arg(short, long, default_value = "benchmark")]
        kind: String,
        #[arg(short, long)]
        project: String,
        #[arg(short, long)]
        agent: Option<String>,
        #[arg(long)]
        lora: Option<String>,
        #[arg(long)]
        prompt: Option<String>,
        #[arg(long)]
        backend: Option<String>,
        #[arg(short, long, default_value = "exact")]
        scorer: String,
        #[arg(short, long, default_value = "1")]
        concurrency: usize,
    },
    /// List recent benchmark runs
    List {
        #[arg(short, long, default_value = "20")]
        limit: usize,
    },
    /// Show a benchmark run and its per-case results
    Show {
        #[arg(short, long)]
        id: String,
        #[arg(long)]
        json: bool,
    },
    /// Compare two benchmark runs
    Compare {
        #[arg(short, long)]
        baseline: String,
        #[arg(short, long)]
        challenger: String,
        #[arg(long, default_value = "0.05")]
        alpha: f64,
    },
}

#[derive(Subcommand)]
enum ABTestCommands {
    /// Run an A/B test between two benchmark runs
    Run {
        #[arg(short, long)]
        baseline: String,
        #[arg(short, long)]
        challenger: String,
        #[arg(long, default_value = "0.05")]
        alpha: f64,
    },
}

fn parse_backend_kind(kind: &str) -> Result<BackendKind, String> {
    match kind.to_lowercase().as_str() {
        "ollama" => Ok(BackendKind::Ollama),
        "openai" | "open_ai_compatible" | "openai-compatible" => Ok(BackendKind::OpenAiCompatible),
        "anthropic" => Ok(BackendKind::Anthropic),
        "mistral" | "mistralrs" | "mistral.rs" | "llama" | "llamacpp" | "llama.cpp"
        | "llama-gguf" => Ok(BackendKind::MistralRs),
        "onnx" => Ok(BackendKind::Onnx),
        "custom" => Ok(BackendKind::Custom),
        other => Err(format!("unknown backend kind: {}", other)),
    }
}

fn build_manifest_entry(
    id: String,
    name: Option<String>,
    repo: Option<String>,
    filename: Option<String>,
    quantization: Option<String>,
    backend: String,
    params_b: Option<f32>,
) -> Result<crytex_core::services::ManifestEntry, String> {
    parse_backend_kind(&backend)?;
    if let Some(quantization) = quantization.as_deref() {
        quantization
            .parse::<crytex_core::services::Quantization>()
            .map(|_| ())?;
    }
    Ok(crytex_core::services::ManifestEntry {
        id: Some(id),
        name,
        repo,
        filename,
        quantization,
        backend: Some(backend),
        params_b,
    })
}

fn build_hf_proof_manifest_entry(
    id: String,
    name: Option<String>,
    repo: String,
    filename: Option<String>,
    quantization: Option<String>,
    params_b: Option<f32>,
    resolution: Option<&crytex_core::services::HfGgufResolution>,
) -> Result<crytex_core::services::ManifestEntry, String> {
    let resolved_filename =
        filename.or_else(|| resolution.map(|resolution| resolution.selected.filename.clone()));
    let resolved_quantization = quantization.or_else(|| {
        resolution.map(|resolution| resolution.selected.quantization.as_str().to_string())
    });
    build_manifest_entry(
        id,
        name,
        Some(repo),
        resolved_filename,
        resolved_quantization,
        "mistralrs".into(),
        params_b,
    )
}

fn build_downloaded_model_backend_config(
    backend_id: &str,
    model: &crytex_core::services::ManagedModel,
    recommendation: &crytex_core::services::RecommendedConfig,
) -> Result<BackendConfig, String> {
    let local_path = model
        .local_path
        .as_ref()
        .ok_or_else(|| format!("model {} is not downloaded", model.id))?;
    if recommendation.backend != BackendKind::MistralRs {
        return Err(format!(
            "downloaded HF GGUF activation requires mistral.rs, got {:?}",
            recommendation.backend
        ));
    }
    Ok(BackendConfig::mistral_rs(
        backend_id.to_string(),
        local_path.display().to_string(),
        Some(recommendation.context_size),
        recommendation.gpu_layers,
    ))
}

#[derive(Debug, Serialize)]
struct HfModelProofReport {
    trace_id: String,
    model_id: String,
    repo: Option<String>,
    filename: Option<String>,
    local_path: Option<String>,
    backend_id: String,
    recommendation: crytex_core::services::RecommendedConfig,
    runtime_placement: HfRuntimePlacementProof,
    runtime_probe: crytex_core::services::ModelRuntimeProbeReport,
    passed: bool,
}

#[derive(Debug, Serialize)]
struct HfRuntimePlacementProof {
    kind: String,
    gpu_layers: Option<usize>,
    compatibility_strategy: String,
    evidence: String,
}

fn build_hf_model_proof_report(
    backend_id: String,
    model: &crytex_core::services::ManagedModel,
    recommendation: crytex_core::services::RecommendedConfig,
    runtime_probe: crytex_core::services::ModelRuntimeProbeReport,
) -> HfModelProofReport {
    let runtime_placement = build_hf_runtime_placement_proof(&recommendation, &runtime_probe);
    HfModelProofReport {
        trace_id: runtime_probe.trace_id.clone(),
        model_id: model.id.clone(),
        repo: model.repo.clone(),
        filename: model.filename.clone(),
        local_path: model
            .local_path
            .as_ref()
            .map(|path| path.display().to_string()),
        backend_id,
        recommendation,
        runtime_placement,
        passed: runtime_probe.passed,
        runtime_probe,
    }
}

fn build_hf_runtime_placement_proof(
    recommendation: &crytex_core::services::RecommendedConfig,
    runtime_probe: &crytex_core::services::ModelRuntimeProbeReport,
) -> HfRuntimePlacementProof {
    let strategy = format!("{:?}", runtime_probe.compatibility.strategy);
    let (kind, evidence) = match (
        recommendation.backend,
        recommendation.gpu_layers,
        runtime_probe.compatibility.strategy,
    ) {
        (
            BackendKind::MistralRs,
            None,
            crytex_core::services::ExecutionStrategy::CudaFused
            | crytex_core::services::ExecutionStrategy::CudaWithFallback,
        ) => (
            "cuda_auto_device_mapping",
            "mistral.rs selected a CUDA execution strategy; gpu_layers=None means automatic device mapping",
        ),
        (BackendKind::MistralRs, Some(0), _) => (
            "cpu",
            "gpu_layers=0 explicitly selects CPU execution for the local GGUF backend",
        ),
        (BackendKind::MistralRs, Some(999), _) => (
            "cuda_full_offload",
            "gpu_layers=999 pins all supported GGUF layers to CUDA to avoid mixed CPU/GPU auto mapping",
        ),
        (BackendKind::MistralRs, Some(_), _) => (
            "manual_gpu_layers",
            "gpu_layers is explicitly pinned for the local GGUF backend",
        ),
        _ => (
            "backend_default",
            "runtime placement is delegated to the selected backend",
        ),
    };

    HfRuntimePlacementProof {
        kind: kind.into(),
        gpu_layers: recommendation.gpu_layers,
        compatibility_strategy: strategy,
        evidence: evidence.into(),
    }
}

const DEFAULT_CODEGEN_WORKFLOW: &str = r#"
id = "codegen"
name = "Code generation pipeline"
version = "1.0.0"
entry = "architect"
max_concurrency = 4

[[nodes]]
type = "agent"
id = "architect"
agent = "architect"
input = "prompt"
output = "design"

[[nodes]]
type = "agent"
id = "coder"
agent = "coder"
input = "design"
output = "patch"

[[nodes]]
type = "agent"
id = "qa"
agent = "qa"
input = "patch"
output = "test_report"

[[nodes]]
type = "agent"
id = "security"
agent = "security"
input = "patch"
output = "security_report"

[[nodes]]
type = "agent"
id = "critic"
agent = "critic"
input = "patch"
output = "review"

[[nodes]]
type = "end"
id = "end"

[[edges]]
from = "architect"
to = "coder"

[[edges]]
from = "coder"
to = "qa"

[[edges]]
from = "coder"
to = "security"

[[edges]]
from = "qa"
to = "critic"

[[edges]]
from = "security"
to = "critic"

[[edges]]
from = "critic"
to = "end"
"#;

async fn ensure_default_workflow(dir: &std::path::Path) -> std::io::Result<()> {
    if !dir.exists() {
        tokio::fs::create_dir_all(dir).await?;
    }
    let path = dir.join("codegen.toml");
    if !path.exists() {
        tokio::fs::write(&path, DEFAULT_CODEGEN_WORKFLOW).await?;
    }
    Ok(())
}

fn parse_headers(headers: &[String]) -> Result<std::collections::HashMap<String, String>, String> {
    let mut map = std::collections::HashMap::new();
    for h in headers {
        let (key, value) = h
            .split_once('=')
            .ok_or_else(|| format!("header must be in key=value format: {}", h))?;
        map.insert(key.trim().to_string(), value.trim().to_string());
    }
    Ok(map)
}

fn create_inference_service(
    config: &CrytexConfig,
) -> Result<Arc<dyn crytex_core::services::InferenceService>, Box<dyn std::error::Error>> {
    let default_id = config.inference.default_backend.as_deref().unwrap_or("");
    let mut registry = BackendRegistry::new(default_id);

    for backend_config in &config.inference.backends {
        let backend = factory::create_backend(backend_config)?;
        registry.register(backend_config.id.clone(), backend);
    }

    // GPU-first local default: if the user has not configured any backend, create
    // a mistral.rs backend with auto-detected device settings.
    if registry.is_empty() {
        let detector = SystemHardwareDetector::new();
        let rec = recommend_local_device(&detector, None);
        info!("auto-detected device: {}", rec.reason);

        let local_config = BackendConfig::mistral_rs(
            "local",
            "default",
            Some(config.inference.context_token_budget.unwrap_or(4096)),
            rec.gpu_layers,
        );
        let backend = factory::create_backend(&local_config)?;
        registry.register(local_config.id.clone(), backend);
        // Fall back to the auto-created local backend.
        let _ = registry.set_default(&local_config.id);
    }

    let registry = Arc::new(registry);

    let mut service = if let Some(default_backend) = config.inference.default_backend.as_deref() {
        let default_manager = registry
            .get(default_backend)
            .ok_or("default backend not registered")?;

        let estimator = Arc::new(TokenizerEstimator::for_model(
            &config
                .inference
                .default_backend_config()
                .map(|b| b.model.clone())
                .unwrap_or_else(|| "gpt-4".to_string()),
        ));
        let relevance_scorer: Arc<dyn crytex_compress::scoring::RelevanceScorer> = Arc::new(
            crytex_compress::scoring::HybridRelevanceScorer::new(Arc::new(
                crytex_compress::embed::InferenceEmbedder::new(default_manager),
            )),
        );
        let fallback = Arc::new(
            TruncateCompressor::new(estimator.clone()).with_relevance_scorer(relevance_scorer),
        );
        let smart = Arc::new(
            SmartCompressor::new(fallback)
                .with_token_estimator(estimator.clone())
                .with_compressor(ContentType::Diff, Arc::new(DiffCompressor::default()))
                .with_compressor(ContentType::Log, Arc::new(LogCompressor::default()))
                .with_compressor(
                    ContentType::SearchResults,
                    Arc::new(SearchCompressor::default()),
                )
                .with_compressor(ContentType::Json, Arc::new(JsonCompressor::default()))
                .with_compressor(ContentType::SourceCode, Arc::new(CodeCompressor::default()))
                .with_compressor(ContentType::PlainText, Arc::new(TextCompressor::default()))
                .with_ccr_store(Arc::new(DiskCcrStore::new(&config.paths.ccr_dir))),
        );
        let pipeline = Arc::new(CompressionPipeline::with_estimator(smart, estimator));
        let budget = config.inference.context_token_budget.unwrap_or(4096);

        InferenceServiceImpl::new(registry, Some(default_backend.to_string()))
            .with_compression(pipeline, budget)
    } else {
        InferenceServiceImpl::new(registry, None)
    };

    if let Some(embedding_backend) = &config.inference.embedding_backend {
        service = service.with_embedding_backend(embedding_backend.clone());
    }

    Ok(Arc::new(service))
}

async fn create_agent_service(
    audit: Arc<dyn crytex_core::services::AuditLogService>,
    prompt_repo: Arc<dyn PromptVersionRepository>,
    scanner: Arc<dyn crytex_core::security::SecurityScanner>,
    tool_factory: Arc<dyn Fn(Capability) -> Arc<dyn ToolService> + Send + Sync>,
    context_assembler: Option<Arc<crytex_core::services::ContextAssembler>>,
) -> Arc<dyn AgentService> {
    let mut builder = AgentServiceImpl::new(audit)
        .with_prompt_repo(prompt_repo)
        .with_scanner(scanner)
        .with_tool_factory(tool_factory);
    if let Some(assembler) = context_assembler {
        builder = builder.with_context_assembler(assembler);
    }
    let service = Arc::new(builder);
    service.register(Arc::new(ArchitectAgent::new())).await;
    service.register(Arc::new(CoderAgent::new())).await;
    service.register(Arc::new(QaAgent::new())).await;
    service.register(Arc::new(CriticAgent::new())).await;
    service.register(Arc::new(SecurityAgent::new())).await;
    service.register(Arc::new(CodeCriticAgent::new())).await;
    service.register(Arc::new(StyleCriticAgent::new())).await;
    service.register(Arc::new(SecurityCriticAgent::new())).await;
    service.register(Arc::new(TestCriticAgent::new())).await;
    service.register(Arc::new(ResearcherAgent::new())).await;
    service.register(Arc::new(SummarizerAgent::new())).await;
    service
}

async fn seed_prompt_versions(
    prompt_service: &PromptEvolutionService<
        impl PromptVersionRepository,
        impl ExperienceRepository,
    >,
) {
    use crytex_agents::prompts;

    let _ = prompt_service
        .seed_agent("architect", &prompts::architect_system_prompt(&[], None))
        .await;
    let _ = prompt_service
        .seed_agent("coder", &prompts::coder_system_prompt(false, &[], None))
        .await;
    let _ = prompt_service
        .seed_agent("qa", &prompts::qa_system_prompt(&[], None))
        .await;
    let _ = prompt_service
        .seed_agent("critic", &prompts::critic_system_prompt(&[], None))
        .await;
    let _ = prompt_service
        .seed_agent("security", &prompts::security_system_prompt(&[], None))
        .await;
    let _ = prompt_service
        .seed_agent("researcher", &prompts::researcher_system_prompt(&[], None))
        .await;
    let _ = prompt_service
        .seed_agent(
            "summarizer",
            "You are a summarizer. Condense information while preserving key points.",
        )
        .await;
}

struct AgentTaskHandler {
    task_service: Arc<dyn crytex_core::services::TaskService>,
    agent_service: Arc<dyn AgentService>,
    inference: Arc<dyn crytex_core::services::InferenceService>,
    tool_service: Arc<dyn ToolService>,
    critic_council: Option<CriticCouncil>,
    metrics_service: Arc<dyn MetricsService>,
    code_graph: Option<Arc<CodeGraph>>,
    lora_router: Arc<dyn LoraRouter>,
}

#[async_trait]
impl TaskHandler for AgentTaskHandler {
    async fn handle(&self, task: Task) -> Result<(), WorkerError> {
        self.task_service
            .set_status(&task.id, TaskStatus::InProgress)
            .await
            .map_err(|e| WorkerError::Handler(e.to_string()))?;

        let lora_id = self
            .lora_router
            .resolve(&task, &task.project_id)
            .await
            .ok()
            .flatten();
        let mut task = task.clone();
        task.lora_adapter_id = lora_id;

        let routed_agent = self.agent_service.route(&task).unwrap_or_default();
        let task = if routed_agent == "architect" {
            if let Some(graph) = &self.code_graph {
                let mut task = task.clone();
                task.payload["codebase_summary"] = serde_json::Value::String(graph.summary());
                task
            } else {
                task
            }
        } else {
            task
        };

        let start = tokio::time::Instant::now();

        match self
            .agent_service
            .execute(&task, self.inference.clone(), self.tool_service.clone())
            .await
        {
            Ok(result) => {
                if task.kind == "review" {
                    // Store the review result, then run the critic council and move to Review status
                    // for human approval.
                    self.task_service
                        .set_result(&task.id, result)
                        .await
                        .map_err(|e| WorkerError::Handler(e.to_string()))?;

                    if let Some(council) = &self.critic_council {
                        council
                            .evaluate(&task)
                            .await
                            .map_err(|e| WorkerError::Handler(e.to_string()))?;
                    }

                    self.task_service
                        .set_status(&task.id, TaskStatus::Review)
                        .await
                        .map_err(|e| WorkerError::Handler(e.to_string()))?;
                } else {
                    self.task_service
                        .set_result(&task.id, result)
                        .await
                        .map_err(|e| WorkerError::Handler(e.to_string()))?;
                }
                let latency_ms = start.elapsed().as_millis() as u64;
                let _ = self
                    .metrics_service
                    .record_task_completion(latency_ms, true)
                    .await;
                Ok(())
            }
            Err(e) => {
                let latency_ms = start.elapsed().as_millis() as u64;
                let _ = self
                    .metrics_service
                    .record_task_completion(latency_ms, false)
                    .await;
                let _ = self
                    .task_service
                    .set_status(&task.id, TaskStatus::Failed)
                    .await;
                Err(WorkerError::Handler(e.to_string()))
            }
        }
    }
}

#[tokio::main]
#[allow(clippy::expect_used)]
async fn main() {
    CrytexTelemetry::init();

    let cli = Cli::parse();
    let config = CrytexConfig::load();
    if let Err(e) = config.ensure_dirs() {
        warn!("Failed to create data directories: {}", e);
    }

    if let Commands::AddBackend {
        id,
        kind,
        model,
        url,
        api_key,
        header,
        gpu_layers,
        context_size,
    } = &cli.command
    {
        let kind = parse_backend_kind(kind).unwrap_or_else(|e| {
            eprintln!("{}", e);
            std::process::exit(1);
        });
        let headers = parse_headers(header).unwrap_or_else(|e| {
            eprintln!("{}", e);
            std::process::exit(1);
        });
        let backend_config = BackendConfig {
            id: id.clone(),
            kind,
            model: model.clone(),
            url: url.clone(),
            api_key: api_key.clone(),
            headers,
            timeout_seconds: None,
            context_size: *context_size,
            gpu_layers: *gpu_layers,
            supports_lora: false,
        };
        let mut config = config.clone();
        config
            .inference
            .backends
            .retain(|backend| backend.id != *id);
        config.inference.backends.push(backend_config);
        if let Err(e) = config.save() {
            eprintln!("Failed to save config: {}", e);
            std::process::exit(1);
        }
        println!("Backend {} added. Use switch-backend to select it.", id);
        return;
    }

    if let Commands::AddModel {
        id,
        name,
        repo,
        filename,
        quantization,
        backend,
        params_b,
    } = &cli.command
    {
        let event_bus = Arc::new(crytex_core::EventBus::new());
        let event_service = Arc::new(EventServiceImpl::new(event_bus));
        let config_dir = CrytexConfig::config_path()
            .parent()
            .expect("config path must have a parent")
            .to_path_buf();
        let model_manager = ModelManagerImpl::new_standard(
            &config_dir,
            &config.paths.data_dir,
            event_service,
            Arc::new(SystemHardwareDetector::new()),
        );
        let entry = build_manifest_entry(
            id.clone(),
            name.clone(),
            repo.clone(),
            filename.clone(),
            quantization.clone(),
            backend.clone(),
            *params_b,
        )
        .unwrap_or_else(|e| {
            eprintln!("Failed to build model manifest entry: {}", e);
            std::process::exit(1);
        });
        let model = model_manager.add_model(entry).unwrap_or_else(|e| {
            eprintln!("Failed to add model: {}", e);
            std::process::exit(1);
        });
        println!(
            "Model {} added. Use download-model --id {} then probe-model --id {}.",
            model.id, model.id, model.id
        );
        return;
    }

    if let Commands::ProveHfModel {
        id,
        name,
        repo,
        filename,
        quantization,
        params_b,
        backend_id,
        trace_id,
        max_tokens,
        timeout_seconds,
        report_path,
    } = &cli.command
    {
        let event_bus = Arc::new(crytex_core::EventBus::new());
        let event_service = Arc::new(EventServiceImpl::new(event_bus));
        let config_dir = CrytexConfig::config_path()
            .parent()
            .expect("config path must have a parent")
            .to_path_buf();
        let model_manager = ModelManagerImpl::new_standard(
            &config_dir,
            &config.paths.data_dir,
            event_service,
            Arc::new(SystemHardwareDetector::new()),
        );
        let preferred_quantization = quantization
            .as_deref()
            .map(str::parse::<Quantization>)
            .transpose()
            .unwrap_or_else(|e| {
                eprintln!("Failed to parse quantization: {}", e);
                std::process::exit(1);
            });
        let resolved_gguf = if filename.is_none() {
            Some(
                model_manager
                    .resolve_hf_gguf(HfGgufResolveRequest {
                        repo: repo.clone(),
                        preferred_quantization,
                        params_b: *params_b,
                    })
                    .await
                    .unwrap_or_else(|e| {
                        eprintln!("Failed to resolve HF GGUF: {}", e);
                        std::process::exit(1);
                    }),
            )
        } else {
            None
        };
        let entry = build_hf_proof_manifest_entry(
            id.clone(),
            name.clone().or_else(|| Some(id.clone())),
            repo.clone(),
            filename.clone(),
            quantization.clone(),
            *params_b,
            resolved_gguf.as_ref(),
        )
        .unwrap_or_else(|e| {
            eprintln!("Failed to build HF model manifest entry: {}", e);
            std::process::exit(1);
        });
        let _model = model_manager.add_model(entry).unwrap_or_else(|e| {
            eprintln!("Failed to add HF model: {}", e);
            std::process::exit(1);
        });
        let model = model_manager.download_model(id).await.unwrap_or_else(|e| {
            eprintln!("Failed to download HF model: {}", e);
            std::process::exit(1);
        });
        let recommendation = model_manager.recommend_config(id).unwrap_or_else(|e| {
            eprintln!("Failed to recommend HF runtime config: {}", e);
            std::process::exit(1);
        });
        let backend_config =
            build_downloaded_model_backend_config(backend_id, &model, &recommendation)
                .unwrap_or_else(|e| {
                    eprintln!("Failed to build HF backend config: {}", e);
                    std::process::exit(1);
                });
        let mut active_config = config.clone();
        active_config
            .inference
            .backends
            .retain(|backend| backend.id != backend_config.id);
        active_config.inference.default_backend = Some(backend_config.id.clone());
        active_config.inference.backends.push(backend_config);
        if let Err(e) = active_config.save() {
            eprintln!("Failed to save activated HF backend config: {}", e);
            std::process::exit(1);
        }
        let inference = create_inference_service(&active_config).unwrap_or_else(|e| {
            eprintln!("Failed to create inference service for HF proof: {}", e);
            std::process::exit(1);
        });
        let detector = SystemHardwareDetector::new();
        let device = crytex_core::services::HardwareDetector::detect(&detector);
        let runtime = RuntimeFeatureSet::from_device(&device);
        let model_name = model
            .local_path
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| id.clone());
        let runtime_probe = ModelRuntimeProbe::new(inference)
            .probe(
                &model,
                &device,
                &runtime,
                ModelRuntimeProbeRequest {
                    backend_id: Some(backend_id.clone()),
                    model_name,
                    trace_id: trace_id.clone(),
                    max_tokens: *max_tokens,
                    timeout_seconds: Some(*timeout_seconds),
                    lora_adapter_id: None,
                },
            )
            .await;
        let report =
            build_hf_model_proof_report(backend_id.clone(), &model, recommendation, runtime_probe);
        let json = unwrap_or_exit!(
            serde_json::to_string_pretty(&report),
            "Failed to serialize HF proof report"
        );
        if let Some(path) = report_path {
            if let Some(parent) = path.parent()
                && let Err(e) = std::fs::create_dir_all(parent)
            {
                eprintln!("Failed to create HF proof report directory: {}", e);
                std::process::exit(1);
            }
            if let Err(e) = std::fs::write(path, &json) {
                eprintln!("Failed to write HF proof report: {}", e);
                std::process::exit(1);
            }
        }
        println!("{}", json);
        if !report.passed {
            std::process::exit(2);
        }
        return;
    }

    if let Commands::ResolveHfGguf {
        repo,
        quantization,
        params_b,
    } = &cli.command
    {
        let event_bus = Arc::new(crytex_core::EventBus::new());
        let event_service = Arc::new(EventServiceImpl::new(event_bus));
        let config_dir = CrytexConfig::config_path()
            .parent()
            .expect("config path must have a parent")
            .to_path_buf();
        let model_manager = ModelManagerImpl::new_standard(
            &config_dir,
            &config.paths.data_dir,
            event_service,
            Arc::new(SystemHardwareDetector::new()),
        );
        let preferred_quantization = quantization
            .as_deref()
            .map(str::parse::<Quantization>)
            .transpose()
            .unwrap_or_else(|e| {
                eprintln!("Failed to parse quantization: {}", e);
                std::process::exit(1);
            });
        let resolution = model_manager
            .resolve_hf_gguf(HfGgufResolveRequest {
                repo: repo.clone(),
                preferred_quantization,
                params_b: *params_b,
            })
            .await
            .unwrap_or_else(|e| {
                eprintln!("Failed to resolve HF GGUF: {}", e);
                std::process::exit(1);
            });
        let json = unwrap_or_exit!(
            serde_json::to_string_pretty(&resolution),
            "Failed to serialize HF GGUF resolution"
        );
        println!("{}", json);
        return;
    }

    let inference = create_inference_service(&config).expect("Failed to create inference service");

    if let Commands::ProbeModel {
        id,
        backend,
        model,
        trace_id,
        max_tokens,
        timeout_seconds,
    } = &cli.command
    {
        let event_bus = Arc::new(crytex_core::EventBus::new());
        let event_service = Arc::new(EventServiceImpl::new(event_bus));
        let config_dir = CrytexConfig::config_path()
            .parent()
            .expect("config path must have a parent")
            .to_path_buf();
        let model_manager: Arc<dyn crytex_core::services::ModelManager> =
            Arc::new(ModelManagerImpl::new_standard(
                &config_dir,
                &config.paths.data_dir,
                event_service,
                Arc::new(SystemHardwareDetector::new()),
            ));
        let backend_id = backend.clone().or_else(|| {
            config
                .inference
                .default_backend
                .as_ref()
                .map(ToString::to_string)
        });
        let managed_model = match model_manager.get_model(id) {
            Ok(model) => model,
            Err(_error) if model.is_some() => {
                let preferred_backend = backend_id
                    .as_ref()
                    .and_then(|id| config.inference.backend(id))
                    .map(|backend| backend.kind)
                    .unwrap_or(BackendKind::Custom);
                crytex_core::services::ManagedModel {
                    id: id.clone(),
                    name: model.clone().unwrap_or_else(|| id.clone()),
                    repo: None,
                    filename: None,
                    local_path: None,
                    quantization: None,
                    preferred_backend,
                    params_b: None,
                    status: crytex_core::services::ModelStatus::Available,
                }
            }
            Err(error) => {
                eprintln!("Failed to get model: {}", error);
                std::process::exit(1);
            }
        };
        let detector = SystemHardwareDetector::new();
        let device = crytex_core::services::HardwareDetector::detect(&detector);
        let runtime = RuntimeFeatureSet::from_device(&device);
        let model_name = model.clone().unwrap_or_else(|| {
            managed_model
                .local_path
                .as_ref()
                .map(|path| path.display().to_string())
                .or_else(|| {
                    backend_id
                        .as_ref()
                        .and_then(|id| config.inference.backend(id))
                        .map(|backend| backend.model.clone())
                })
                .unwrap_or_else(|| managed_model.id.clone())
        });
        let probe = ModelRuntimeProbe::new(inference);
        let report = probe
            .probe(
                &managed_model,
                &device,
                &runtime,
                ModelRuntimeProbeRequest {
                    backend_id,
                    model_name,
                    trace_id: trace_id.clone(),
                    max_tokens: *max_tokens,
                    timeout_seconds: *timeout_seconds,
                    lora_adapter_id: None,
                },
            )
            .await;
        let json = unwrap_or_exit!(
            serde_json::to_string_pretty(&report),
            "Failed to serialize probe"
        );
        println!("{}", json);
        if !report.passed {
            std::process::exit(2);
        }
        return;
    }

    if let Commands::ProbeRuntimeMatrix {
        id,
        backend,
        model,
        lora,
        trace_id,
        report_dir,
        max_tokens,
    } = &cli.command
    {
        let event_bus = Arc::new(crytex_core::EventBus::new());
        let event_service = Arc::new(EventServiceImpl::new(event_bus));
        let config_dir = CrytexConfig::config_path()
            .parent()
            .expect("config path must have a parent")
            .to_path_buf();
        let model_manager: Arc<dyn crytex_core::services::ModelManager> =
            Arc::new(ModelManagerImpl::new_standard(
                &config_dir,
                &config.paths.data_dir,
                event_service,
                Arc::new(SystemHardwareDetector::new()),
            ));
        let backend_ids = if backend.is_empty() {
            config
                .inference
                .default_backend
                .as_ref()
                .map(|backend| vec![backend.clone()])
                .unwrap_or_default()
        } else {
            backend.clone()
        };
        if backend_ids.is_empty() {
            eprintln!("No backend was provided and no default backend is configured");
            std::process::exit(1);
        }
        let managed_model = match model_manager.get_model(id) {
            Ok(model) => model,
            Err(_error) if model.is_some() => {
                let preferred_backend = backend_ids
                    .first()
                    .and_then(|id| config.inference.backend(id))
                    .map(|backend| backend.kind)
                    .unwrap_or(BackendKind::Custom);
                crytex_core::services::ManagedModel {
                    id: id.clone(),
                    name: model.clone().unwrap_or_else(|| id.clone()),
                    repo: None,
                    filename: None,
                    local_path: None,
                    quantization: None,
                    preferred_backend,
                    params_b: None,
                    status: crytex_core::services::ModelStatus::Available,
                }
            }
            Err(error) => {
                eprintln!("Failed to get model: {}", error);
                std::process::exit(1);
            }
        };
        let detector = SystemHardwareDetector::new();
        let device = crytex_core::services::HardwareDetector::detect(&detector);
        let runtime = RuntimeFeatureSet::from_device(&device);
        let lora_variants = if lora.is_empty() {
            vec![None]
        } else {
            std::iter::once(None)
                .chain(lora.iter().cloned().map(Some))
                .collect::<Vec<_>>()
        };
        let mut entries = Vec::new();
        for backend_id in &backend_ids {
            let runtime_model_name = model.clone().unwrap_or_else(|| {
                managed_model
                    .local_path
                    .as_ref()
                    .map(|path| path.display().to_string())
                    .or_else(|| {
                        config
                            .inference
                            .backend(backend_id)
                            .map(|backend| backend.model.clone())
                    })
                    .unwrap_or_else(|| managed_model.id.clone())
            });
            for lora_adapter_id in &lora_variants {
                let variant = lora_adapter_id.as_deref().unwrap_or("baseline");
                entries.push(RuntimeMatrixEntryRequest {
                    label: format!("{backend_id}:{variant}"),
                    model: managed_model.clone(),
                    backend_id: Some(backend_id.clone()),
                    model_name: runtime_model_name.clone(),
                    lora_adapter_id: lora_adapter_id.clone(),
                    max_tokens: *max_tokens,
                });
            }
        }
        let matrix = ModelRuntimeMatrixProbe::new(inference);
        let report = matrix
            .probe(
                &device,
                &runtime,
                ModelRuntimeMatrixRequest {
                    trace_id: trace_id.clone(),
                    entries,
                },
            )
            .await;
        let json = unwrap_or_exit!(
            serde_json::to_string_pretty(&report),
            "Failed to serialize runtime matrix"
        );
        let report_dir = report_dir
            .clone()
            .unwrap_or_else(|| config.paths.data_dir.join("reports").join("runtime-matrix"));
        let report_path = unwrap_or_exit!(
            RuntimeMatrixReportWriter::write_pretty_json(&report, report_dir),
            "Failed to write runtime matrix report"
        );
        println!("{}", json);
        eprintln!("Runtime matrix report written to {}", report_path.display());
        if !report.passed {
            std::process::exit(2);
        }
        return;
    }

    let storage = Arc::new(
        Storage::new(&config.paths.db_path.to_string_lossy())
            .await
            .expect("Failed to open database"),
    );

    let metrics_service: Arc<dyn MetricsService> = Arc::new(
        crytex_core::metrics::MetricsServiceImpl::new(storage.clone()),
    );

    let vector_store = create_vector_store(&config, Some(metrics_service.clone()));
    let embedder = create_embedder(&config, inference.clone(), Some(metrics_service.clone())).await;
    let sparse_embedder = create_sparse_embedder(&config);
    let reranker = create_reranker(&config);

    let storage = Arc::new(
        (*storage)
            .clone()
            .with_experience_vector_store(embedder.clone(), vector_store.clone()),
    );

    let event_bus = Arc::new(crytex_core::EventBus::new());
    let event_service = Arc::new(EventServiceImpl::new(event_bus));
    let benchmark_repo: Arc<dyn crytex_core::persistence::BenchmarkResultRepository> =
        storage.clone();
    let benchmark_harness = Arc::new(DefaultBenchmarkHarness::new(
        benchmark_repo.clone(),
        event_service.clone(),
    ));
    let project_service = Arc::new(ProjectServiceImpl::new(storage.clone()));
    let audit_service: Arc<dyn crytex_core::services::AuditLogService> = Arc::new(
        BulkAuditLogService::new(storage.clone(), config.paths.data_dir.join("logs")),
    );
    let prompt_service = Arc::new(PromptEvolutionService::new(
        storage.clone(),
        storage.clone(),
    ));
    seed_prompt_versions(&prompt_service).await;
    let task_service = Arc::new(
        TaskServiceImpl::new(
            storage.clone(),
            event_service.clone(),
            audit_service.clone(),
        )
        .with_prompt_repo(storage.clone()),
    );
    let scanner: Arc<dyn crytex_core::security::SecurityScanner> =
        Arc::new(crytex_core::security::RegexSecurityScanner::new());

    let sandbox_service: Arc<dyn SandboxService> = Arc::new(SandboxOrchestrator::auto().await);
    let mut tool_registry = TypedToolRegistry::new()
        .with_default_coding_tools()
        .with_semantic_search(embedder.clone(), vector_store.clone());
    if vector_store.supports_sparse().await
        && let Some(sparse) = sparse_embedder.clone()
    {
        tool_registry = tool_registry.with_sparse_search(sparse, vector_store.clone());
    }
    let tool_registry = tool_registry.build();
    let project_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    let code_graph: Option<Arc<CodeGraph>> =
        match CodeGraphBuilder::new().index_project(&project_root) {
            Ok(graph) => {
                info!(
                    "Indexed codebase graph: {} symbols, {} relations",
                    graph.len(),
                    graph.graph.edge_count()
                );
                Some(Arc::new(graph))
            }
            Err(e) => {
                warn!("Failed to index codebase graph: {e}");
                None
            }
        };

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

    let hybrid_retriever = create_hybrid_retriever(
        &config,
        embedder.clone(),
        vector_store.clone(),
        sparse_embedder.clone(),
    );
    let mut context_assembler =
        crytex_core::services::ContextAssembler::new(embedder.clone(), vector_store.clone())
            .with_hybrid_retriever(hybrid_retriever);
    if let Some(reranker) = reranker {
        context_assembler = context_assembler.with_reranker(reranker);
    }
    let context_assembler: Option<Arc<crytex_core::services::ContextAssembler>> =
        Some(Arc::new(context_assembler));
    let agent_service = create_agent_service(
        audit_service.clone(),
        storage.clone(),
        scanner.clone(),
        tool_factory,
        context_assembler,
    )
    .await;

    let workflow_dir = config.workflow.workflows_dir.clone();
    if let Err(e) = ensure_default_workflow(&workflow_dir).await {
        warn!("Failed to ensure default workflow directory: {e}");
    }
    let workflow_repository: Arc<dyn WorkflowRepository> =
        Arc::new(TomlWorkflowRepository::new(workflow_dir));

    let tool_service: Arc<dyn ToolService> = Arc::new(ScanningToolService::new(
        Arc::new(ToolServiceImpl::new(
            tool_registry,
            project_root,
            Capability::all(),
            sandbox_service.clone(),
            Some(scanner.clone()),
            config.security.clone(),
        )),
        scanner.clone(),
    ));

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
    let lora_benchmark_gate = {
        let golden_set_path = config
            .benchmark
            .golden_sets_dir
            .join("lora_evolution.jsonl");
        if !golden_set_path.exists() {
            warn!(
                "LoRA benchmark gate disabled: held-out golden set not found at {}",
                golden_set_path.display()
            );
            None
        } else {
            match project_service.list().await {
                Ok(projects) => {
                    if let Some(project) = projects.first() {
                        let project_id = project.id.clone();
                        let task_service = task_service.clone();
                        let agent_service = agent_service.clone();
                        let inference = inference.clone();
                        let tool_service = tool_service.clone();
                        let runner_factory = Arc::new(move |task_kind: &str| {
                            Arc::new(AgentBenchmarkRunner::new(
                                project_id.clone(),
                                task_kind.to_string(),
                                task_service.clone(),
                                agent_service.clone(),
                                inference.clone(),
                                tool_service.clone(),
                            )) as Arc<dyn crytex_bench::BenchmarkRunner>
                        });
                        Some(Arc::new(
                            BenchLoraBenchmarkGate::new_with_runner_factory(
                                benchmark_harness.clone(),
                                benchmark_repo.clone(),
                                golden_set_path,
                                runner_factory,
                                Arc::new(ExactMatchScorer),
                            )
                            .with_max_concurrency(config.benchmark.default_concurrency),
                        )
                            as Arc<dyn crytex_core::services::LoraBenchmarkGate>)
                    } else {
                        warn!(
                            "LoRA benchmark gate disabled: no project exists for benchmark tasks"
                        );
                        None
                    }
                }
                Err(e) => {
                    warn!("LoRA benchmark gate disabled: failed to list projects: {e}");
                    None
                }
            }
        }
    };
    let lora_evolution = create_lora_evolution_service(
        storage.clone(),
        task_service.clone(),
        storage.clone(),
        inference.clone(),
        event_service.clone(),
        Some(embedder.clone()),
        Some(vector_store.clone()),
        adapters_dir,
        base_model,
        lora_benchmark_gate,
    );
    let lora_router = create_lora_router(
        lora_evolution.clone(),
        Some(embedder.clone()),
        Some(vector_store.clone()),
    );
    let memory_bank = create_memory_bank_service(
        storage.clone(),
        Some(embedder.clone()),
        Some(vector_store.clone()),
    );

    let ctx = AppContext::new(
        config,
        crytex_core::tracing::TraceContext::new(),
        event_service.clone(),
        storage.clone(),
        project_service.clone(),
        task_service.clone(),
        audit_service.clone(),
        agent_service.clone(),
        inference.clone(),
        model_manager.clone(),
        tool_service.clone(),
        metrics_service.clone(),
        alert_service.clone(),
        lora_evolution.clone(),
        lora_router.clone(),
        memory_bank.clone(),
    );
    let prompt_service = prompt_service.clone();

    match cli.command {
        Commands::CreateProject { name, path } => {
            let request = CreateProjectRequest {
                name: &name,
                root_path: std::path::Path::new(&path),
            };
            let project = unwrap_or_exit!(
                ctx.project_service.create(request).await,
                "Failed to create project"
            );
            println!("Created project {} ({})", project.id, project.name);
        }
        Commands::ListProjects => {
            let projects =
                unwrap_or_exit!(ctx.project_service.list().await, "Failed to list projects");
            if projects.is_empty() {
                println!("No projects");
            } else {
                for p in projects {
                    println!("{}  {}  {}", p.id, p.name, p.root_path);
                }
            }
        }
        Commands::Submit {
            project,
            prompt,
            kind,
            backend,
        } => {
            let default_model = ctx
                .config
                .inference
                .default_backend_config()
                .map(|b| b.model.clone())
                .unwrap_or_default();
            let mut payload = serde_json::json!({
                "prompt": prompt,
                "model": default_model,
            });
            if let Some(backend_id) = backend {
                payload["backend"] = serde_json::Value::String(backend_id);
            }

            let request = CreateTaskRequest {
                project_id: project,
                parent_id: None,
                title: prompt.clone(),
                description: Some(prompt.clone()),
                kind: kind.clone(),
                assigned_agent: None,
                priority: 0,
                payload,
                trace_id: Some(ctx.trace_context.trace_id.clone()),
            };

            let task = unwrap_or_exit!(
                ctx.task_service.submit(request).await,
                "Failed to submit task"
            );
            println!("Submitted task {}", task.id);
        }
        Commands::ListTasks { project } => {
            let tasks = unwrap_or_exit!(
                ctx.task_service.list_by_project(&project).await,
                "Failed to list tasks"
            );
            if tasks.is_empty() {
                println!("No tasks");
            } else {
                for t in tasks {
                    println!(
                        "{}  {:12}  {:12}  {}",
                        t.id,
                        t.kind,
                        t.status.to_string(),
                        t.title
                    );
                }
            }
        }
        Commands::ShowTask { id } => {
            let task = require_or_exit!(
                unwrap_or_exit!(ctx.task_service.get(&id).await, "Failed to get task"),
                "Task not found"
            );
            println!("ID:          {}", task.id);
            println!("Project:     {}", task.project_id);
            println!("Kind:        {}", task.kind);
            println!("Status:      {}", task.status);
            println!("Agent:       {:?}", task.assigned_agent);
            println!("Title:       {}", task.title);
            println!(
                "Result:      {}",
                task.result.unwrap_or(serde_json::Value::Null)
            );
        }
        Commands::Approve { id, score } => {
            let task = require_or_exit!(
                unwrap_or_exit!(ctx.task_service.get(&id).await, "Failed to get task"),
                "Task not found"
            );

            if task.status != TaskStatus::Review {
                eprintln!(
                    "Task {} is not in review status (current: {})",
                    id, task.status
                );
                std::process::exit(1);
            }

            let human_score = score.unwrap_or(5.0);
            unwrap_or_exit!(
                ctx.task_service.set_human_score(&id, human_score).await,
                "Failed to set human score"
            );

            let reward_service = RewardService::new(ctx.persistence.clone());
            let reward = unwrap_or_exit!(
                reward_service
                    .record(RecordRewardRequest {
                        task_id: &id,
                        project_id: Some(&task.project_id),
                        prompt_version_id: task.prompt_version_id.as_deref(),
                        critic_score: task.critic_score,
                        human_score: Some(human_score),
                        text: task.result.as_ref().map(|r| r.as_str().unwrap_or("")),
                        comment: None,
                    })
                    .await,
                "Failed to record reward"
            );

            if let Some(version_id) = task.prompt_version_id.as_ref() {
                let _ = prompt_service.recompute_fitness(version_id).await;
            }

            unwrap_or_exit!(
                ctx.task_service
                    .set_status(&id, TaskStatus::Completed)
                    .await,
                "Failed to complete task"
            );

            if let Err(e) = ctx.lora_evolution.collect_golden_example(&id).await {
                eprintln!("Warning: failed to collect golden example: {}", e);
            }

            println!(
                "Approved task {} with human score {:.1} and reward {:.2}",
                id, human_score, reward
            );
        }
        Commands::Reject { id, score, comment } => {
            let task = require_or_exit!(
                unwrap_or_exit!(ctx.task_service.get(&id).await, "Failed to get task"),
                "Task not found"
            );

            if task.status != TaskStatus::Review {
                eprintln!(
                    "Task {} is not in review status (current: {})",
                    id, task.status
                );
                std::process::exit(1);
            }

            let human_score = score.unwrap_or(1.0);
            unwrap_or_exit!(
                ctx.task_service.set_human_score(&id, human_score).await,
                "Failed to set human score"
            );

            let reward_service = RewardService::new(ctx.persistence.clone());
            let reward = unwrap_or_exit!(
                reward_service
                    .record(RecordRewardRequest {
                        task_id: &id,
                        project_id: Some(&task.project_id),
                        prompt_version_id: task.prompt_version_id.as_deref(),
                        critic_score: task.critic_score,
                        human_score: Some(human_score),
                        text: task.result.as_ref().map(|r| r.as_str().unwrap_or("")),
                        comment: comment.as_deref(),
                    })
                    .await,
                "Failed to record reward"
            );

            if let Some(version_id) = task.prompt_version_id.as_ref() {
                let _ = prompt_service.recompute_fitness(version_id).await;
            }

            unwrap_or_exit!(
                ctx.task_service.retry(&id, comment.as_deref()).await,
                "Failed to retry task"
            );

            println!(
                "Rejected task {} with human score {:.1} and reward {:.2}. Task returned to queue for retry.",
                id, human_score, reward
            );
        }
        Commands::Prompts { agent, json } => {
            let versions = prompt_service
                .list_versions(&agent)
                .await
                .unwrap_or_else(|e| {
                    eprintln!("Failed to list prompt versions: {}", e);
                    std::process::exit(1);
                });
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&versions).unwrap_or_else(|_| "[]".to_string())
                );
            } else {
                if versions.is_empty() {
                    println!("No prompt versions for agent {}", agent);
                } else {
                    println!("Prompt versions for {}:", agent);
                    for v in versions {
                        let active_marker = if v.active { " *" } else { "" };
                        let fitness = v
                            .fitness
                            .map(|f| format!("{:.2}", f))
                            .unwrap_or_else(|| "-".to_string());
                        println!(
                            "{}  parent={}  fitness={}{}",
                            v.id,
                            v.parent_id.as_deref().unwrap_or("-"),
                            fitness,
                            active_marker
                        );
                    }
                }
            }
        }
        Commands::EvolvePrompt { agent, operator } => {
            let op = match operator.as_str() {
                "rephrase" => MutationOperator::Rephrase,
                "constraint" => MutationOperator::AddConstraint,
                "example" => MutationOperator::InjectExample,
                "tone" => MutationOperator::ChangeTone,
                other => {
                    eprintln!("Unknown mutation operator: {}", other);
                    std::process::exit(1);
                }
            };
            let mut rng = rand::thread_rng();
            let child = prompt_service
                .evolve_step(&agent, op, 2, &mut rng)
                .await
                .unwrap_or_else(|e| {
                    eprintln!("Failed to evolve prompt: {}", e);
                    std::process::exit(1);
                });
            println!(
                "Created new prompt version {} for agent {} from parent {}",
                child.id,
                child.agent,
                child.parent_id.as_deref().unwrap_or("-")
            );
        }
        Commands::Lora { command } => match command {
            LoraCommands::List { project } => {
                let adapters = unwrap_or_exit!(
                    if let Some(project_id) = project {
                        ctx.persistence
                            .list_lora_adapters_by_project(&project_id)
                            .await
                    } else {
                        // Without a project filter we list by the common task kinds.
                        let mut all = Vec::new();
                        for kind in ["codegen", "architecture", "security", "qa", "research"] {
                            match ctx.persistence.list_lora_adapters_by_kind(kind).await {
                                Ok(mut list) => all.append(&mut list),
                                Err(_) => continue,
                            }
                        }
                        Ok(all)
                    },
                    "Failed to list adapters"
                );

                if adapters.is_empty() {
                    println!("No adapters found");
                } else {
                    for a in adapters {
                        println!(
                            "{}  {}  {}  active={}  kind={:?}",
                            a.id, a.name, a.base_model, a.active, a.task_kind
                        );
                    }
                }
            }
            LoraCommands::Register {
                id,
                path,
                base_model,
                kind,
            } => {
                let adapter = LoraAdapter {
                    id: id.clone(),
                    project_id: None,
                    name: id.clone(),
                    file_path: path,
                    base_model: base_model.clone(),
                    task_kind: kind,
                    agent_role: None,
                    metrics: serde_json::Value::Null,
                    created_at: chrono::Utc::now().timestamp_millis(),
                    active: true,
                };
                unwrap_or_exit!(
                    ctx.persistence.insert_lora_adapter(&adapter).await,
                    "Failed to save adapter"
                );
                unwrap_or_exit!(
                    ctx.inference_service
                        .register_lora(crytex_inference::LoRAAdapter {
                            id,
                            path: adapter.file_path,
                            base_model,
                        })
                        .await,
                    "Failed to register adapter"
                );
                println!("Registered adapter {}", adapter.id);
            }
            LoraCommands::Swap { id } => {
                unwrap_or_exit!(
                    ctx.inference_service.swap_lora(&id).await,
                    "Failed to swap adapter"
                );
                unwrap_or_exit!(
                    ctx.persistence.set_lora_adapter_active(&id, true).await,
                    "Failed to activate adapter"
                );
                println!("Swapped to adapter {}", id);
            }
            LoraCommands::Select { project, adapter } => {
                let project = require_or_exit!(
                    unwrap_or_exit!(
                        ctx.project_service.get(&project).await,
                        "Failed to load project"
                    ),
                    "Project not found"
                );
                require_or_exit!(
                    unwrap_or_exit!(
                        ctx.persistence.get_lora_adapter(&adapter).await,
                        "Failed to load adapter"
                    ),
                    "Adapter not found"
                );

                unwrap_or_exit!(
                    ctx.inference_service.swap_lora(&adapter).await,
                    "Failed to swap adapter"
                );
                unwrap_or_exit!(
                    ctx.persistence
                        .set_lora_adapter_active(&adapter, true)
                        .await,
                    "Failed to activate adapter"
                );

                let snapshot = ProjectSnapshot {
                    id: Ulid::new().to_string(),
                    project_id: project.id,
                    name: format!("lora-selection-{}", adapter),
                    state_json: serde_json::json!({ "selected_lora_adapter_id": adapter }),
                    created_at: chrono::Utc::now().timestamp_millis(),
                };
                unwrap_or_exit!(
                    ctx.persistence.insert_project_snapshot(&snapshot).await,
                    "Failed to persist project snapshot"
                );
                println!(
                    "Selected adapter {} for project {} (snapshot {})",
                    adapter, project.name, snapshot.id
                );
            }
            LoraCommands::Train { kind } => {
                let adapter = ctx
                    .lora_evolution
                    .train_and_register(&kind)
                    .await
                    .unwrap_or_else(|e| {
                        eprintln!("Failed to train adapter: {}", e);
                        std::process::exit(1);
                    });
                println!(
                    "Trained adapter {} for kind {} -> {}",
                    adapter.id, kind, adapter.file_path
                );
            }
            LoraCommands::SelectRole { role, adapter } => {
                let role = match AgentRole::from_agent(&role) {
                    Some(r) => r,
                    None => {
                        eprintln!("Unknown agent role: {}", role);
                        std::process::exit(1);
                    }
                };
                let mut config = CrytexConfig::load();
                config
                    .role_adapters
                    .insert(role.as_str().to_string(), adapter.clone());
                unwrap_or_exit!(config.save(), "Failed to save config");
                println!("Bound adapter {} to role {}", adapter, role.as_str());
            }
            LoraCommands::ListRoles => {
                let config = CrytexConfig::load();
                if config.role_adapters.is_empty() {
                    println!("No role bindings configured");
                } else {
                    for (role, adapter) in &config.role_adapters {
                        println!("{} -> {}", role, adapter);
                    }
                }
            }
        },
        Commands::ListBackends => {
            for backend in ctx.inference_service.available_backends() {
                println!(
                    "{}  {}  capabilities: {:?}",
                    backend.id, backend.name, backend.capabilities
                );
            }
        }
        Commands::ListModels { backend } => {
            if let Some(backend_id) = backend {
                match ctx.inference_service.list_models(Some(&backend_id)).await {
                    Ok(models) => {
                        if models.is_empty() {
                            println!("No models available");
                        } else {
                            for m in models {
                                println!("{}  {}", m.id, m.name);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to list models: {}", e);
                        std::process::exit(1);
                    }
                }
            } else {
                match model_manager.list_models() {
                    Ok(models) => {
                        if models.is_empty() {
                            println!(
                                "No models configured. Add entries to ~/.config/crytex/manifest.toml"
                            );
                        } else {
                            for m in models {
                                let status = match m.status {
                                    crytex_core::services::ModelStatus::Available => {
                                        "available".to_string()
                                    }
                                    crytex_core::services::ModelStatus::Downloaded => {
                                        "downloaded".to_string()
                                    }
                                    crytex_core::services::ModelStatus::Downloading(p) => {
                                        format!("downloading {:.0}%", p * 100.0)
                                    }
                                    crytex_core::services::ModelStatus::Error(ref e) => {
                                        format!("error: {}", e)
                                    }
                                };
                                println!("{}  {}  [{}]", m.id, m.name, status);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to list managed models: {}", e);
                        std::process::exit(1);
                    }
                }
            }
        }
        Commands::DownloadModel {
            id,
            activate,
            backend_id,
        } => {
            let mut rx = ctx.event_service.subscribe();
            let progress_id = id.clone();
            let progress_handle = tokio::spawn(async move {
                while let Ok(event) = rx.recv().await {
                    if let Event::ModelDownloadProgress { model_id, progress } = event
                        && model_id == progress_id
                    {
                        print!("\rDownloading {}: {:.0}%", model_id, progress * 100.0);
                        let _ = std::io::Write::flush(&mut std::io::stdout());
                        if (progress - 1.0).abs() < f32::EPSILON {
                            break;
                        }
                    }
                }
            });

            let result = model_manager.download_model(&id).await;
            progress_handle.await.ok();
            println!();

            match result {
                Ok(model) => {
                    println!("Downloaded model {} to {:?}", model.id, model.local_path);
                    if activate {
                        let recommendation = model_manager
                            .recommend_config(&model.id)
                            .unwrap_or_else(|e| {
                                eprintln!("Failed to recommend config: {}", e);
                                std::process::exit(1);
                            });
                        let backend_config = build_downloaded_model_backend_config(
                            &backend_id,
                            &model,
                            &recommendation,
                        )
                        .unwrap_or_else(|e| {
                            eprintln!("Failed to build backend config: {}", e);
                            std::process::exit(1);
                        });
                        let mut config = ctx.config.clone();
                        config
                            .inference
                            .backends
                            .retain(|backend| backend.id != backend_config.id);
                        config.inference.default_backend = Some(backend_config.id.clone());
                        config.inference.backends.push(backend_config);
                        if let Err(e) = config.save() {
                            eprintln!("Failed to save activated backend config: {}", e);
                            std::process::exit(1);
                        }
                        println!(
                            "Activated model {} as backend {}. Use probe-model --id {} --backend {}.",
                            model.id, backend_id, model.id, backend_id
                        );
                    }
                }
                Err(e) => {
                    eprintln!("Failed to download model: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Commands::AddModel { .. } => {
            unreachable!("add-model is handled before AppContext initialization")
        }
        Commands::ProveHfModel { .. } => {
            unreachable!("prove-hf-model is handled before AppContext initialization")
        }
        Commands::ShowModel { id } => {
            let model = model_manager.get_model(&id).unwrap_or_else(|e| {
                eprintln!("Failed to get model: {}", e);
                std::process::exit(1);
            });
            let recommendation = model_manager.recommend_config(&id).unwrap_or_else(|e| {
                eprintln!("Failed to recommend config: {}", e);
                std::process::exit(1);
            });
            println!("ID:          {}", model.id);
            println!("Name:        {}", model.name);
            println!("Repo:        {}", model.repo.as_deref().unwrap_or("-"));
            println!("Filename:    {}", model.filename.as_deref().unwrap_or("-"));
            println!(
                "Quantization: {}",
                model
                    .quantization
                    .map(|q| q.as_str().to_string())
                    .unwrap_or_else(|| "-".to_string())
            );
            println!("Backend:     {:?}", model.preferred_backend);
            println!(
                "Local path:  {}",
                model
                    .local_path
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "not downloaded".to_string())
            );
            println!("Status:      {:?}", model.status);
            println!("Recommended:");
            println!("  backend:       {:?}", recommendation.backend);
            println!("  quantization:  {}", recommendation.quantization.as_str());
            println!("  gpu_layers:    {:?}", recommendation.gpu_layers);
            println!("  context_size:  {}", recommendation.context_size);
        }
        Commands::RecommendModel { id } => {
            let recommendation = model_manager.recommend_config(&id).unwrap_or_else(|e| {
                eprintln!("Failed to recommend config: {}", e);
                std::process::exit(1);
            });
            let json = unwrap_or_exit!(
                serde_json::to_string_pretty(&recommendation),
                "Failed to serialize recommendation"
            );
            println!("{}", json);
        }
        Commands::ResolveHfGguf { .. } => {
            unreachable!("resolve-hf-gguf is handled before AppContext initialization")
        }
        Commands::ProbeModel { .. } => {
            unreachable!("probe-model is handled before AppContext initialization")
        }
        Commands::ProbeRuntimeMatrix { .. } => {
            unreachable!("probe-runtime-matrix is handled before AppContext initialization")
        }
        Commands::SwitchBackend { id } => {
            if ctx
                .inference_service
                .available_backends()
                .iter()
                .any(|b| b.id == id)
            {
                let mut config = ctx.config.clone();
                config.inference.default_backend = Some(id.clone());
                if let Err(e) = config.save() {
                    eprintln!("Failed to save config: {}", e);
                    std::process::exit(1);
                }
                println!("Default backend switched to {}. Restart to apply.", id);
            } else {
                eprintln!("Backend {} not found", id);
                std::process::exit(1);
            }
        }
        Commands::AddBackend {
            id,
            kind,
            model,
            url,
            api_key,
            header,
            gpu_layers,
            context_size,
        } => {
            let kind = parse_backend_kind(&kind).unwrap_or_else(|e| {
                eprintln!("{}", e);
                std::process::exit(1);
            });
            let headers = parse_headers(&header).unwrap_or_else(|e| {
                eprintln!("{}", e);
                std::process::exit(1);
            });
            let backend_config = BackendConfig {
                id: id.clone(),
                kind,
                model,
                url,
                api_key,
                headers,
                timeout_seconds: None,
                context_size,
                gpu_layers,
                supports_lora: false,
            };
            let mut config = ctx.config.clone();
            config.inference.backends.retain(|b| b.id != id);
            config.inference.backends.push(backend_config);
            if let Err(e) = config.save() {
                eprintln!("Failed to save config: {}", e);
                std::process::exit(1);
            }
            println!("Backend {} added. Use switch-backend to select it.", id);
        }
        Commands::Run => {
            info!("=== Crytex Kernel Run ===");
            info!("Registered agents: {:?}", ctx.agent_service.list().await);
            info!(
                "Inference backends: {:?}",
                ctx.inference_service.available_backends()
            );

            let architect = require_or_exit!(
                ctx.agent_service.find("architect").await,
                "architect agent must be registered"
            );
            let orchestrator: Arc<dyn Orchestrator> = Arc::new(
                OrchestratorImpl::new(ctx.task_service.clone())
                    .with_planning_agent(architect)
                    .with_inference(ctx.inference_service.clone())
                    .with_tools(ctx.tool_service.clone())
                    .with_workflow_repository(workflow_repository.clone())
                    .with_workflow_executor(Arc::new(
                        AgentWorkflowNodeExecutor::new(
                            ctx.agent_service.clone(),
                            ctx.inference_service.clone(),
                            ctx.tool_service.clone(),
                        )
                        .with_lora_router(ctx.lora_router.clone()),
                    )),
            );
            let scheduler = Arc::new(SchedulerImpl::new(ctx.task_service.clone()));
            let worker_pool = Arc::new(WorkerPool::new(4));

            let critic_council = CriticCouncil::new(
                ctx.agent_service.clone(),
                ctx.task_service.clone(),
                ctx.inference_service.clone(),
                ctx.tool_service.clone(),
                ctx.audit_service.clone(),
            );

            let (_ide_bridge, _ide_service) =
                start_ide_bridge(ctx.event_service.clone(), ctx.persistence.clone()).await;
            info!("IDE bridge started");

            let mut _watcher_shutdowns = Vec::new();
            if ctx.config.indexing.incremental_enabled {
                match ctx.project_service.list().await {
                    Ok(projects) => {
                        for project in projects {
                            let root = PathBuf::from(&project.root_path);
                            if !root.exists() {
                                warn!(%project.id, %project.root_path, "skipping watcher for missing project root");
                                continue;
                            }
                            let indexer = create_project_indexer(
                                embedder.clone(),
                                vector_store.clone(),
                                sparse_embedder.clone(),
                            );
                            let watcher = ProjectWatcher::new(indexer, ctx.event_service.clone())
                                .with_debounce(ctx.config.indexing.debounce_ms);
                            let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
                            _watcher_shutdowns.push(shutdown_tx);
                            let _handle =
                                tokio::spawn(watcher.watch(project.id.clone(), root, shutdown_rx));
                            info!(%project.id, %project.root_path, "incremental file watcher started");
                        }
                    }
                    Err(err) => {
                        warn!(%err, "failed to list projects for incremental indexing");
                    }
                }
            }

            if let Err(e) = ctx.task_service.load_all_tasks().await {
                error!("Failed to load tasks from storage: {}", e);
                std::process::exit(1);
            }

            let event_handle = tokio::spawn({
                let event_service = ctx.event_service.clone();
                let orchestrator = orchestrator.clone();
                let task_service = ctx.task_service.clone();
                async move {
                    let mut rx = event_service.subscribe();
                    while let Ok(event) = rx.recv().await {
                        match event {
                            Event::TaskCreated {
                                task_id,
                                project_id,
                            } => {
                                info!("Task {} created in project {}", task_id, project_id);
                                if let Ok(Some(task)) = task_service.get(&task_id).await
                                    && task.kind == "codegen"
                                    && task.parent_id.is_none()
                                    && let Err(e) = orchestrator.orchestrate(&task).await
                                {
                                    error!("Orchestration failed for {}: {}", task_id, e);
                                }
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
                            Event::AgentThinking {
                                task_id,
                                agent,
                                message,
                            } => {
                                info!("Agent {} thinking on task {}: {}", agent, task_id, message);
                            }
                            _ => {}
                        }
                    }
                }
            });

            let runner = tokio::spawn({
                let worker_pool = worker_pool.clone();
                let scheduler = scheduler.clone();
                let handler = Arc::new(AgentTaskHandler {
                    task_service: ctx.task_service.clone(),
                    agent_service: ctx.agent_service.clone(),
                    inference: inference.clone(),
                    tool_service: ctx.tool_service.clone(),
                    critic_council: Some(critic_council.clone()),
                    metrics_service: ctx.metrics_service.clone(),
                    code_graph: code_graph.clone(),
                    lora_router: ctx.lora_router.clone(),
                });
                async move {
                    let _ = worker_pool.run(scheduler, handler).await;
                }
            });

            let metrics_monitor = tokio::spawn({
                let metrics_service = ctx.metrics_service.clone();
                let alert_service = ctx.alert_service.clone();
                let event_service = ctx.event_service.clone();
                async move {
                    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(60));
                    loop {
                        interval.tick().await;
                        match metrics_service.snapshot().await {
                            Ok(snapshot) => {
                                let _ = alert_service.check(&snapshot).await;
                                event_service.publish(Event::MetricsSnapshot {
                                    snapshot: snapshot.clone(),
                                });
                            }
                            Err(e) => warn!("Failed to collect metrics: {}", e),
                        }
                    }
                }
            });

            tokio::select! {
                _ = event_handle => {},
                _ = runner => {},
                _ = metrics_monitor => {},
            }
        }
        Commands::Index { project_id, path } => {
            let sparse_embedder = create_sparse_embedder(&ctx.config);
            let indexer =
                create_project_indexer(embedder.clone(), vector_store.clone(), sparse_embedder);
            let stats = unwrap_or_exit!(
                indexer.index(&project_id, &path).await,
                "Failed to index project"
            );
            println!(
                "Indexed {} files, {} chunks",
                stats.files_indexed, stats.chunks_indexed
            );
        }
        Commands::WatchMetrics { interval_secs } => {
            let mut rx = ctx.event_service.subscribe();
            let interval = tokio::time::Duration::from_secs(interval_secs);
            let start = tokio::time::Instant::now();
            loop {
                match tokio::time::timeout(interval, rx.recv()).await {
                    Ok(Ok(Event::MetricsSnapshot { snapshot })) => {
                        println!(
                            "{}",
                            serde_json::to_string(&snapshot).unwrap_or_else(|_| "{}".to_string())
                        );
                    }
                    Ok(Ok(_)) => {}
                    Ok(Err(_)) | Err(_) => {
                        if start.elapsed() >= interval {
                            break;
                        }
                    }
                }
            }
        }
        Commands::State { project, json } => {
            let state = export_project_state(
                ctx.project_service.clone(),
                ctx.task_service.clone(),
                ctx.audit_service.clone(),
                ctx.persistence.clone(),
                ctx.metrics_service.clone(),
                &project,
            )
            .await
            .unwrap_or_else(|e| {
                eprintln!("Failed to export project state: {e}");
                std::process::exit(1);
            });

            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&state).unwrap_or_else(|_| "{}".to_string())
                );
            } else {
                println!("Project: {} ({})", state.project.name, state.project.id);
                println!("Tasks: {}", state.tasks.len());
                println!("Recent logs: {}", state.recent_logs.len());
                println!("Latest snapshot: {:?}", state.latest_snapshot.map(|s| s.id));
            }
        }
        Commands::Bench { command } => match command {
            BenchCommands::Run {
                name,
                golden_set,
                kind,
                project,
                agent,
                lora,
                prompt,
                backend,
                scorer,
                concurrency,
            } => {
                let project = unwrap_or_exit!(
                    ctx.project_service.get(&project).await,
                    "Failed to load project"
                );
                let project = require_or_exit!(project, "Project not found");

                let scorer: Arc<dyn crytex_bench::Scorer> = match scorer.as_str() {
                    "exact" => Arc::new(ExactMatchScorer),
                    "schema" => Arc::new(JsonSchemaScorer),
                    "sandbox" => Arc::new(SandboxTestScorer::new(sandbox_service.clone())),
                    "llm-judge" => {
                        let model = ctx
                            .config
                            .inference
                            .default_backend_config()
                            .map(|b| b.model.clone())
                            .unwrap_or_default();
                        Arc::new(LlmJudgeScorer::new(
                            ctx.inference_service.clone(),
                            model,
                            backend.clone(),
                        ))
                    }
                    other => {
                        eprintln!("Unknown scorer: {}", other);
                        std::process::exit(1);
                    }
                };

                let runner: Arc<dyn crytex_bench::BenchmarkRunner> =
                    Arc::new(AgentBenchmarkRunner::new(
                        project.id.clone(),
                        kind,
                        ctx.task_service.clone(),
                        ctx.agent_service.clone(),
                        ctx.inference_service.clone(),
                        ctx.tool_service.clone(),
                    ));

                let variant = BenchmarkVariant {
                    name: agent.clone().unwrap_or_else(|| "default".into()),
                    agent_role: agent,
                    lora_adapter_id: lora,
                    prompt_version_id: prompt,
                    backend_id: backend,
                };

                let request = BenchmarkRunRequest {
                    name,
                    golden_set_path: golden_set,
                    variant,
                    scorer,
                    runner,
                    max_concurrency: concurrency,
                    project_id: Some(project.id),
                };

                let run =
                    unwrap_or_exit!(benchmark_harness.run(request).await, "Benchmark run failed");
                println!(
                    "Run {}: {} cases, pass_rate={:.2}",
                    run.summary.id, run.summary.total_cases, run.summary.pass_rate
                );
            }
            BenchCommands::List { limit } => {
                let runs =
                    unwrap_or_exit!(benchmark_repo.list_runs(limit).await, "Failed to list runs");
                if runs.is_empty() {
                    println!("No benchmark runs");
                } else {
                    for r in runs {
                        println!(
                            "{}  {}  {}/{}  {:.2}",
                            r.id, r.name, r.pass_count, r.total_cases, r.pass_rate
                        );
                    }
                }
            }
            BenchCommands::Show { id, json } => {
                let run = unwrap_or_exit!(benchmark_repo.get_run(&id).await, "Failed to load run");
                let run = require_or_exit!(run, "Run not found");
                if json {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&run).unwrap_or_else(|_| "{}".into())
                    );
                } else {
                    println!("Run {}: {}", run.summary.id, run.summary.name);
                    println!("  pass_rate: {:.2}", run.summary.pass_rate);
                    println!(
                        "  cases: {}/{}",
                        run.summary.pass_count, run.summary.total_cases
                    );
                    println!("  mean_latency_ms: {:.0}", run.summary.mean_latency_ms);
                }
            }
            BenchCommands::Compare {
                baseline,
                challenger,
                alpha,
            } => {
                let report = unwrap_or_exit!(
                    ABTest::new(baseline, challenger)
                        .with_significance(alpha)
                        .compare(benchmark_repo.as_ref())
                        .await,
                    "A/B test failed"
                );
                println!("Baseline pass rate:  {:.2}", report.baseline.pass_rate);
                println!("Challenger pass rate: {:.2}", report.challenger.pass_rate);
                println!("Delta: {:.2}", report.delta_pass_rate);
                println!("McNemar p-value: {:.4}", report.mc_nemar_p_value);
                println!("Winner: {:?}", report.winner);
            }
        },
        Commands::ABTest { command } => match command {
            ABTestCommands::Run {
                baseline,
                challenger,
                alpha,
            } => {
                let report = unwrap_or_exit!(
                    ABTest::new(baseline, challenger)
                        .with_significance(alpha)
                        .compare(benchmark_repo.as_ref())
                        .await,
                    "A/B test failed"
                );
                println!("Baseline pass rate:  {:.2}", report.baseline.pass_rate);
                println!("Challenger pass rate: {:.2}", report.challenger.pass_rate);
                println!("Delta: {:.2}", report.delta_pass_rate);
                println!("McNemar p-value: {:.4}", report.mc_nemar_p_value);
                println!("Winner: {:?}", report.winner);
                ctx.event_service
                    .publish(crytex_core::bus::Event::ABTestCompleted {
                        report_id: Ulid::new().to_string(),
                        baseline_run_id: report.baseline.id.clone(),
                        challenger_run_id: report.challenger.id.clone(),
                        winner: format!("{:?}", report.winner),
                    });
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_model_command_builds_hf_gguf_manifest_entry() {
        let entry = build_manifest_entry(
            "hf-tinyllama-chat-q2-gguf".into(),
            Some("HF TinyLlama Chat Q2 GGUF".into()),
            Some("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF".into()),
            Some("tinyllama-1.1b-chat-v1.0.Q2_K.gguf".into()),
            Some("Q2_K".into()),
            "mistralrs".into(),
            Some(1.1),
        )
        .unwrap();

        assert_eq!(entry.id.as_deref(), Some("hf-tinyllama-chat-q2-gguf"));
        assert_eq!(
            entry.repo.as_deref(),
            Some("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
        );
        assert_eq!(
            entry.filename.as_deref(),
            Some("tinyllama-1.1b-chat-v1.0.Q2_K.gguf")
        );
        assert_eq!(entry.quantization.as_deref(), Some("Q2_K"));
        assert_eq!(entry.backend.as_deref(), Some("mistralrs"));
        assert_eq!(entry.params_b, Some(1.1));
    }

    #[test]
    fn prove_hf_model_manifest_entry_uses_resolved_gguf_when_filename_is_omitted() {
        let resolution = crytex_core::services::HfGgufResolution {
            selected: crytex_core::services::HfGgufVariant {
                repo: "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF".into(),
                filename: "tinyllama-1.1b-chat-v1.0.Q2_K.gguf".into(),
                quantization: crytex_core::services::Quantization::Q2K,
            },
            variants: Vec::new(),
            recommendation: crytex_core::services::RecommendedConfig {
                backend: BackendKind::MistralRs,
                quantization: crytex_core::services::Quantization::Q2K,
                gpu_layers: None,
                context_size: 4096,
            },
        };

        let entry = build_hf_proof_manifest_entry(
            "hf-tinyllama-chat-q2-gguf".into(),
            None,
            "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF".into(),
            None,
            None,
            Some(1.1),
            Some(&resolution),
        )
        .unwrap();

        assert_eq!(
            entry.filename.as_deref(),
            Some("tinyllama-1.1b-chat-v1.0.Q2_K.gguf")
        );
        assert_eq!(entry.quantization.as_deref(), Some("Q2_K"));
        assert_eq!(entry.backend.as_deref(), Some("mistralrs"));
    }

    #[test]
    fn downloaded_hf_model_builds_runtime_backend_config_from_recommendation() {
        let model_path = PathBuf::from("B:/crytex-data/models/tiny/model.gguf");
        let model = crytex_core::services::ManagedModel {
            id: "hf-tiny".into(),
            name: "HF Tiny".into(),
            repo: Some("owner/repo".into()),
            filename: Some("model.gguf".into()),
            local_path: Some(model_path.clone()),
            quantization: Some(crytex_core::services::Quantization::Q2K),
            preferred_backend: BackendKind::MistralRs,
            params_b: Some(1.1),
            status: crytex_core::services::ModelStatus::Downloaded,
        };
        let recommendation = crytex_core::services::RecommendedConfig {
            backend: BackendKind::MistralRs,
            quantization: crytex_core::services::Quantization::Q2K,
            gpu_layers: Some(12),
            context_size: 2048,
        };

        let backend =
            build_downloaded_model_backend_config("hf-runtime", &model, &recommendation).unwrap();

        assert_eq!(backend.id, "hf-runtime");
        assert_eq!(backend.kind, BackendKind::MistralRs);
        assert_eq!(backend.model, model_path.display().to_string());
        assert_eq!(backend.context_size, Some(2048));
        assert_eq!(backend.gpu_layers, Some(12));
    }

    #[test]
    fn hf_model_proof_report_includes_download_activation_and_generation_evidence() {
        let model_path = PathBuf::from("B:/crytex-data/models/tiny/model.gguf");
        let model = crytex_core::services::ManagedModel {
            id: "hf-tiny".into(),
            name: "HF Tiny".into(),
            repo: Some("owner/repo".into()),
            filename: Some("model.gguf".into()),
            local_path: Some(model_path.clone()),
            quantization: Some(crytex_core::services::Quantization::Q2K),
            preferred_backend: BackendKind::MistralRs,
            params_b: Some(1.1),
            status: crytex_core::services::ModelStatus::Downloaded,
        };
        let recommendation = crytex_core::services::RecommendedConfig {
            backend: BackendKind::MistralRs,
            quantization: crytex_core::services::Quantization::Q2K,
            gpu_layers: Some(12),
            context_size: 2048,
        };
        let runtime_probe = crytex_core::services::ModelRuntimeProbeReport {
            trace_id: "trace-hf-proof".into(),
            model_id: "hf-tiny".into(),
            backend_id: Some("local-hf-proof".into()),
            backend_capability: None,
            compatibility: crytex_core::services::ModelCompatibilityPlanner::plan(
                &model,
                &crytex_core::services::DeviceKind::Cpu,
                &RuntimeFeatureSet {
                    cuda_available: false,
                    metal_available: false,
                    gdn_cuda_available: false,
                    cuda_unquantized_moe_fallback_available: false,
                },
            ),
            stages: Vec::new(),
            generated_preview: Some("ok".into()),
            passed: true,
        };

        let report = build_hf_model_proof_report(
            "local-hf-proof".into(),
            &model,
            recommendation,
            runtime_probe,
        );

        assert!(report.passed);
        assert_eq!(report.trace_id, "trace-hf-proof");
        assert_eq!(report.model_id, "hf-tiny");
        assert_eq!(report.repo.as_deref(), Some("owner/repo"));
        assert_eq!(report.filename.as_deref(), Some("model.gguf"));
        assert_eq!(
            report.local_path.as_deref(),
            Some(model_path.to_str().unwrap())
        );
        assert_eq!(report.backend_id, "local-hf-proof");
        assert_eq!(
            report.runtime_probe.generated_preview.as_deref(),
            Some("ok")
        );
    }

    #[test]
    fn hf_runtime_placement_marks_mistral_cuda_none_gpu_layers_as_auto_mapping() {
        let model = crytex_core::services::ManagedModel {
            id: "hf-tiny".into(),
            name: "HF Tiny".into(),
            repo: Some("owner/repo".into()),
            filename: Some("model.gguf".into()),
            local_path: Some(PathBuf::from("B:/crytex-data/models/tiny/model.gguf")),
            quantization: Some(crytex_core::services::Quantization::Q2K),
            preferred_backend: BackendKind::MistralRs,
            params_b: Some(1.1),
            status: crytex_core::services::ModelStatus::Downloaded,
        };
        let recommendation = crytex_core::services::RecommendedConfig {
            backend: BackendKind::MistralRs,
            quantization: crytex_core::services::Quantization::Q2K,
            gpu_layers: None,
            context_size: 4096,
        };
        let runtime_probe = crytex_core::services::ModelRuntimeProbeReport {
            trace_id: "trace-hf-proof".into(),
            model_id: "hf-tiny".into(),
            backend_id: Some("local-hf-proof".into()),
            backend_capability: None,
            compatibility: crytex_core::services::ModelCompatibilityPlan {
                format: crytex_core::services::ModelFormat::Gguf,
                features: vec![crytex_core::services::ModelFeature::Dense],
                strategy: crytex_core::services::ExecutionStrategy::CudaFused,
                status: crytex_core::services::CompatibilityStatus::Ready,
                actions: vec!["use CudaFused execution strategy".into()],
                warnings: Vec::new(),
                blockers: Vec::new(),
            },
            stages: Vec::new(),
            generated_preview: Some("ok".into()),
            passed: true,
        };

        let report = build_hf_model_proof_report(
            "local-hf-proof".into(),
            &model,
            recommendation,
            runtime_probe,
        );

        assert_eq!(report.runtime_placement.kind, "cuda_auto_device_mapping");
        assert_eq!(report.runtime_placement.gpu_layers, None);
        assert_eq!(report.runtime_placement.compatibility_strategy, "CudaFused");
        assert!(
            report
                .runtime_placement
                .evidence
                .contains("automatic device mapping")
        );
    }

    #[test]
    fn hf_runtime_placement_marks_mistral_cuda_999_gpu_layers_as_full_offload() {
        let model = crytex_core::services::ManagedModel {
            id: "hf-tiny".into(),
            name: "HF Tiny".into(),
            repo: Some("owner/repo".into()),
            filename: Some("model.gguf".into()),
            local_path: Some(PathBuf::from("B:/crytex-data/models/tiny/model.gguf")),
            quantization: Some(crytex_core::services::Quantization::Q2K),
            preferred_backend: BackendKind::MistralRs,
            params_b: Some(1.1),
            status: crytex_core::services::ModelStatus::Downloaded,
        };
        let recommendation = crytex_core::services::RecommendedConfig {
            backend: BackendKind::MistralRs,
            quantization: crytex_core::services::Quantization::Q2K,
            gpu_layers: Some(999),
            context_size: 4096,
        };
        let runtime_probe = crytex_core::services::ModelRuntimeProbeReport {
            trace_id: "trace-hf-proof".into(),
            model_id: "hf-tiny".into(),
            backend_id: Some("local-hf-proof".into()),
            backend_capability: None,
            compatibility: crytex_core::services::ModelCompatibilityPlan {
                format: crytex_core::services::ModelFormat::Gguf,
                features: vec![crytex_core::services::ModelFeature::Dense],
                strategy: crytex_core::services::ExecutionStrategy::CudaFused,
                status: crytex_core::services::CompatibilityStatus::Ready,
                actions: vec!["use CudaFused execution strategy".into()],
                warnings: Vec::new(),
                blockers: Vec::new(),
            },
            stages: Vec::new(),
            generated_preview: Some("ok".into()),
            passed: true,
        };

        let report = build_hf_model_proof_report(
            "local-hf-proof".into(),
            &model,
            recommendation,
            runtime_probe,
        );

        assert_eq!(report.runtime_placement.kind, "cuda_full_offload");
        assert_eq!(report.runtime_placement.gpu_layers, Some(999));
        assert!(
            report
                .runtime_placement
                .evidence
                .contains("all supported GGUF layers")
        );
    }
}
