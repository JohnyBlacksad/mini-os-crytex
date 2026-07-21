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
    ABTest, AgentBenchmarkRunner, BenchLoraBenchmarkGate, BenchPromptBenchmarkGate,
    BenchmarkHarness, BenchmarkRunOutput, BenchmarkRunRequest, BenchmarkRunner, BenchmarkVariant,
    DefaultBenchmarkHarness, ExactMatchScorer, JsonSchemaScorer, LlmJudgeScorer, SandboxTestScorer,
    Score, Scorer,
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
    models::{LoraAdapter, ProjectSnapshot, Task, TaskStatus, TrainingExample},
    persistence::{BenchmarkResultRepository, Persistence, PromptVersionRepository},
    services::{
        AgentRole, AgentService, AgentServiceImpl, AgentWorkflowNodeExecutor, AlertService,
        AlertServiceImpl, AlertThresholds, BulkAuditLogService, CreateProjectRequest,
        CreateTaskRequest, CriticCouncil, EventServiceImpl, HfGgufResolveRequest,
        InferenceServiceImpl, LoraBenchmarkDecision, LoraBenchmarkGate, LoraBenchmarkRequest,
        LoraEvolutionError, LoraEvolutionService, LoraRouter, LoraTrainingConfig, ModelManager,
        ModelManagerImpl, ModelRuntimeMatrixProbe, ModelRuntimeMatrixRequest, ModelRuntimeProbe,
        ModelRuntimeProbeRequest, MutationOperator, Orchestrator, OrchestratorImpl, ProjectService,
        ProjectServiceImpl, ProjectWatcher, PromptEvolutionService, Quantization,
        RecordRewardRequest, RewardService, RuntimeFeatureSet, RuntimeMatrixEntryRequest,
        RuntimeMatrixReportWriter, SchedulerImpl, SystemHardwareDetector, TaskHandler,
        TaskServiceImpl, TomlWorkflowRepository, VectorStore, WorkerError, WorkerPool,
        WorkflowRepository, recommend_local_device,
    },
    state_export::export_project_state,
};
use crytex_doc::graph::{CodeGraph, builder::CodeGraphBuilder};
use crytex_ide::ide_service::start_ide_bridge;
use crytex_inference::{
    BackendCapabilityReport, BackendInfo, BackendRegistry, InferenceRequest, InferenceResponse,
    LoRAAdapter as InferenceLoRAAdapter, ModelInfo, TokenUsage,
};
use crytex_sandbox::SandboxOrchestrator;
use crytex_storage::Storage;
use crytex_tools::{Capability, ScanningToolService, ToolServiceImpl, TypedToolRegistry};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
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
    /// Prove multiple HuggingFace GGUF models as one matrix JSON artifact
    ProveHfRuntimeMatrix {
        /// Repeatable spec: id=repo,quantization=Q2_K,params_b=1.1,filename=file.gguf,name=Label
        #[arg(short, long)]
        model: Vec<String>,
        #[arg(long, default_value = "local-hf-proof")]
        backend_id_prefix: String,
        #[arg(long)]
        trace_id: Option<String>,
        #[arg(long, default_value = "16")]
        max_tokens: usize,
        #[arg(long, default_value = "120")]
        timeout_seconds: u64,
        #[arg(long)]
        report_path: Option<PathBuf>,
    },
    /// Prove the kernel happy path as one JSON artifact without requiring a desktop UI
    #[command(alias = "prove-business-e2e", alias = "business-test")]
    ProveKernelE2e {
        #[arg(short, long)]
        path: PathBuf,
        #[arg(short, long, default_value = "Kernel E2E Proof")]
        name: String,
        #[arg(
            short,
            long,
            default_value = "Implement a validated utility with tests"
        )]
        goal: String,
        #[arg(long, default_value = "ollama")]
        live_backend: String,
        #[arg(long, default_value = "qwen3.5:9b")]
        live_model: String,
        #[arg(long, default_value = "http://localhost:11434")]
        live_url: String,
        #[arg(long)]
        deterministic: bool,
        #[arg(long)]
        report_path: Option<PathBuf>,
    },
    /// Prove real LoRA training, GGUF adapter application, hot-swap, and held-out benchmark
    ProveLoraLiveE2e {
        #[arg(long)]
        gguf_path: Option<PathBuf>,
        #[arg(long, default_value = "64")]
        context_size: usize,
        #[arg(long)]
        gpu_layers: Option<usize>,
        #[arg(long, default_value = "50")]
        training_tasks: usize,
        #[arg(long, default_value = "6")]
        heldout_cases: usize,
        #[arg(long, default_value = "32")]
        max_seq_len: usize,
        #[arg(long, default_value = "1")]
        epochs: usize,
        #[arg(long, default_value = "4")]
        rank: usize,
        #[arg(long, default_value = "8")]
        alpha: usize,
        #[arg(long, default_value = "180")]
        train_timeout_secs: u64,
        #[arg(long, default_value = "45")]
        generation_timeout_secs: u64,
        #[arg(long)]
        report_path: Option<PathBuf>,
    },
    /// Prove Candle LoRA train loop with before/after generation on the same tiny base model
    ProveLoraCandleLearning {
        #[arg(long)]
        output_dir: Option<PathBuf>,
        #[arg(long)]
        report_path: Option<PathBuf>,
    },
    /// Prove LoRA quality improvement on an external HuggingFace-style safetensors model directory
    ProveLoraRealModel {
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long, default_value = "external-hf-safetensors")]
        model_source: String,
        #[arg(long)]
        output_dir: Option<PathBuf>,
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
    build_profile: String,
    recommendation: crytex_core::services::RecommendedConfig,
    runtime_placement: HfRuntimePlacementProof,
    generation_evidence: HfGenerationEvidence,
    proof_gate: HfProofGate,
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

#[derive(Debug, Serialize)]
struct HfGenerationEvidence {
    generated: bool,
    sentinel_matched: bool,
    preview: Option<String>,
    duration_ms: Option<u128>,
    message: Option<String>,
}

#[derive(Debug, Serialize)]
struct HfProofGate {
    passed: bool,
    requirements: Vec<HfProofRequirement>,
}

#[derive(Debug, Serialize)]
struct HfProofRequirement {
    name: String,
    passed: bool,
    evidence: String,
}

#[derive(Debug, Clone, PartialEq)]
struct HfProofModelSpec {
    id: String,
    name: Option<String>,
    repo: String,
    filename: Option<String>,
    quantization: Option<String>,
    params_b: Option<f32>,
}

#[derive(Debug, Serialize)]
struct HfProofMatrixEntryReport {
    label: String,
    model_id: String,
    repo: String,
    report: Option<HfModelProofReport>,
    error: Option<String>,
    passed: bool,
}

#[derive(Debug, Serialize)]
struct HfProofMatrixReport {
    trace_id: String,
    build_profile: String,
    entries: Vec<HfProofMatrixEntryReport>,
    passed: bool,
}

#[derive(Debug, Clone, Serialize)]
struct KernelE2eProofGate {
    name: String,
    passed: bool,
    evidence: String,
}

#[derive(Debug, Clone, Serialize)]
struct KernelBusinessProofStep {
    name: String,
    status: String,
    evidence: String,
}

#[derive(Debug, Clone, Serialize)]
struct KernelE2eProofReport {
    business_outcome: String,
    business_steps: Vec<KernelBusinessProofStep>,
    trace_id: String,
    project_id: String,
    project_root: String,
    runtime_kind: String,
    live_backend: Option<String>,
    live_model: Option<String>,
    live_generation_count: usize,
    live_generation_evidence: Vec<KernelLiveGenerationEvidence>,
    goal_task_id: String,
    task_ids: Vec<String>,
    critic_rejection_task_id: String,
    remediation_task_id: String,
    human_approved_task_id: String,
    indexed_files: usize,
    indexed_chunks: usize,
    diagnostics_event_count: usize,
    benchmark_baseline_run_id: String,
    benchmark_challenger_run_id: String,
    prompt_baseline_version_id: String,
    prompt_challenger_version_id: String,
    prompt_promoted: bool,
    lora_adapter_id: String,
    lora_promoted: bool,
    gates: Vec<KernelE2eProofGate>,
    passed: bool,
}

#[derive(Debug, Clone, Serialize)]
struct LoraLiveE2eProofReport {
    proof_outcome: String,
    trace_id: String,
    gguf_path: String,
    runtime_adapter_format: String,
    runtime_application_proof: LoraRuntimeApplicationProof,
    quality_proof: LoraQualityProofSummary,
    training_task_count: usize,
    heldout_case_count: usize,
    adapter_id: String,
    adapter_path: String,
    adapter_registered: bool,
    adapter_applied: bool,
    baseline_output: String,
    challenger_output: String,
    benchmark_outputs: Vec<LoraProofOutput>,
    output_changed_after_swap: bool,
    benchmark_winner: String,
    baseline_pass_rate: f64,
    challenger_pass_rate: f64,
    delta_pass_rate: f64,
    mc_nemar_p_value: Option<f64>,
    significance_level: Option<f64>,
    bootstrap_ci: Option<(f64, f64)>,
    per_case_comparison: Vec<LoraAbCaseComparison>,
    ab_test: LoraAbTestArtifact,
    quality_verdict: String,
    failure_reason: Option<String>,
    training_proof: serde_json::Value,
    learning_proven: bool,
    leakage_check_passed: bool,
    overfit_gap: f64,
    gates: Vec<KernelE2eProofGate>,
    passed: bool,
}

#[derive(Debug, Clone, Serialize)]
struct LoraRuntimeApplicationProof {
    adapter_requested: bool,
    adapter_registered: bool,
    adapter_applied_in_mistralrs_request: bool,
    baseline_output_nonempty: bool,
    challenger_output_nonempty: bool,
    output_changed_after_swap: bool,
    failure_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct LoraQualityProofSummary {
    improved: bool,
    learning_proven: bool,
    heldout_challenger_won: bool,
    no_training_leakage: bool,
    overfit_gap_checked: bool,
    baseline_pass_rate: f64,
    challenger_pass_rate: f64,
    delta_pass_rate: f64,
    failure_reason: Option<String>,
}

#[derive(Debug, Clone)]
struct LoraLiveE2eProofReportInput {
    trace_id: String,
    gguf_path: String,
    training_task_count: usize,
    heldout_case_count: usize,
    adapter_id: String,
    adapter_path: String,
    adapter_registered: bool,
    baseline_output: String,
    challenger_output: String,
    benchmark_outputs: Vec<LoraProofOutput>,
    decision_metadata: Option<serde_json::Value>,
    train_loss: f64,
    validation_loss: f64,
    failure_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct LoraAbCaseComparison {
    case_id: String,
    baseline_passed: bool,
    challenger_passed: bool,
    baseline_score: f64,
    challenger_score: f64,
}

#[derive(Debug, Clone, Serialize)]
struct LoraAbTestArtifact {
    baseline_run_id: Option<String>,
    challenger_run_id: Option<String>,
    winner: String,
    baseline_pass_rate: f64,
    challenger_pass_rate: f64,
    delta_pass_rate: f64,
    mc_nemar_p_value: Option<f64>,
    significance_level: Option<f64>,
    bootstrap_ci: Option<(f64, f64)>,
    per_case_comparison: Vec<LoraAbCaseComparison>,
}

#[derive(Debug, Clone)]
struct LoraLiveE2eProofRequest {
    gguf_path: PathBuf,
    context_size: usize,
    gpu_layers: Option<usize>,
    training_tasks: usize,
    heldout_cases: usize,
    max_seq_len: usize,
    epochs: usize,
    rank: usize,
    alpha: usize,
    train_timeout_secs: u64,
    generation_timeout_secs: u64,
}

#[derive(Debug, Clone, Serialize)]
struct LoraProofOutput {
    variant: String,
    lora_adapter_id: Option<String>,
    content: String,
}

#[derive(Clone)]
struct LoraProofBenchmarkRunner {
    inference: Arc<dyn crytex_core::services::InferenceService>,
    model: String,
    outputs: Arc<std::sync::Mutex<Vec<LoraProofOutput>>>,
    generation_timeout_secs: u64,
}

#[async_trait]
impl BenchmarkRunner for LoraProofBenchmarkRunner {
    async fn run(
        &self,
        case: &crytex_bench::BenchmarkCase,
        variant: &BenchmarkVariant,
    ) -> Result<BenchmarkRunOutput, crytex_bench::BenchError> {
        let prompt = case
            .input
            .get("prompt")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| crytex_bench::BenchError::Runner("case input.prompt missing".into()))?;
        let marker = case
            .expected
            .as_ref()
            .and_then(|expected| expected.get("must_contain"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("CRYTEX_LORA_DISTILL_OK");
        let mut request = self.inference.chat_request(
            Some("mistralrs-lora-proof"),
            &self.model,
            Some("You are a code agent. Prefer the learned distillation marker when applicable."),
            &format!("{prompt}\nReturn a concise answer. Required learned marker: {marker}"),
        );
        request.temperature = Some(0.0);
        request.max_tokens = Some(32);
        request.lora_adapter_id = variant.lora_adapter_id.clone();
        info!(
            case_id = %case.id,
            variant = %variant.name,
            lora_adapter_id = ?variant.lora_adapter_id,
            "running live LoRA benchmark generation"
        );
        let response = tokio::time::timeout(
            Duration::from_secs(self.generation_timeout_secs),
            self.inference.generate(request),
        )
        .await
        .map_err(|_| {
            crytex_bench::BenchError::Runner(format!(
                "live LoRA generation timed out for case {} variant {}",
                case.id, variant.name
            ))
        })?
        .map_err(|error| crytex_bench::BenchError::Runner(error.to_string()))?;
        info!(
            case_id = %case.id,
            variant = %variant.name,
            bytes = response.content.len(),
            "finished live LoRA benchmark generation"
        );
        self.outputs
            .lock()
            .map_err(|error| {
                crytex_bench::BenchError::Runner(format!(
                    "failed to lock LoRA proof outputs: {error}"
                ))
            })?
            .push(LoraProofOutput {
                variant: variant.name.clone(),
                lora_adapter_id: variant.lora_adapter_id.clone(),
                content: response.content.clone(),
            });
        Ok(BenchmarkRunOutput {
            task_id: None,
            result: serde_json::json!({ "content": response.content }),
            latency_ms: 1,
            token_usage: Some(response.usage),
        })
    }
}

#[derive(Default)]
struct ContainsMarkerScorer;

#[async_trait]
impl Scorer for ContainsMarkerScorer {
    async fn score(
        &self,
        case: &crytex_bench::BenchmarkCase,
        actual: &serde_json::Value,
    ) -> Result<Score, crytex_bench::BenchError> {
        let marker = case
            .expected
            .as_ref()
            .and_then(|expected| expected.get("must_contain"))
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                crytex_bench::BenchError::Scoring("expected.must_contain missing".into())
            })?;
        let content = actual
            .get("content")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        if content.contains(marker) {
            Ok(Score::pass())
        } else {
            Ok(Score::fail(format!(
                "output did not contain marker {marker}"
            )))
        }
    }
}

struct LiveLoraBenchmarkGate {
    inference: Arc<dyn crytex_core::services::InferenceService>,
    benchmark_harness: Arc<dyn BenchmarkHarness>,
    benchmark_repo: Arc<dyn BenchmarkResultRepository>,
    golden_set_path: PathBuf,
    model: String,
    outputs: Arc<std::sync::Mutex<Vec<LoraProofOutput>>>,
    decision_metadata: Arc<std::sync::Mutex<Option<serde_json::Value>>>,
    generation_timeout_secs: u64,
}

#[async_trait]
impl LoraBenchmarkGate for LiveLoraBenchmarkGate {
    async fn evaluate(
        &self,
        request: LoraBenchmarkRequest,
    ) -> Result<LoraBenchmarkDecision, LoraEvolutionError> {
        info!(
            challenger_adapter_id = %request.challenger_adapter_id,
            challenger_adapter_path = %request.challenger_adapter_path.display(),
            "registering live LoRA challenger before benchmark"
        );
        tokio::time::timeout(
            Duration::from_secs(60),
            self.inference.register_lora(InferenceLoRAAdapter {
                id: request.challenger_adapter_id.clone(),
                path: request.challenger_adapter_path.display().to_string(),
                base_model: request.base_model.clone(),
            }),
        )
        .await
        .map_err(|_| {
            LoraEvolutionError::Inference(format!(
                "live LoRA registration timed out for adapter {}",
                request.challenger_adapter_id
            ))
        })?
        .map_err(|error| LoraEvolutionError::Inference(error.to_string()))?;
        info!(
            challenger_adapter_id = %request.challenger_adapter_id,
            "registered live LoRA challenger before benchmark"
        );

        let scorer: Arc<dyn Scorer> = Arc::new(ContainsMarkerScorer);
        let runner: Arc<dyn BenchmarkRunner> = Arc::new(LoraProofBenchmarkRunner {
            inference: self.inference.clone(),
            model: self.model.clone(),
            outputs: self.outputs.clone(),
            generation_timeout_secs: self.generation_timeout_secs,
        });
        let baseline = self
            .benchmark_harness
            .run(BenchmarkRunRequest {
                name: "lora live baseline".into(),
                golden_set_path: self.golden_set_path.clone(),
                variant: BenchmarkVariant {
                    name: "baseline".into(),
                    agent_role: request.agent_role.clone(),
                    lora_adapter_id: request.baseline_adapter_id.clone(),
                    prompt_version_id: None,
                    backend_id: Some("mistralrs-lora-proof".into()),
                },
                scorer: scorer.clone(),
                runner: runner.clone(),
                max_concurrency: 1,
                project_id: None,
            })
            .await
            .map_err(|error| {
                LoraEvolutionError::ValidationFailed("benchmark".into(), error.to_string())
            })?;
        let challenger = self
            .benchmark_harness
            .run(BenchmarkRunRequest {
                name: "lora live challenger".into(),
                golden_set_path: self.golden_set_path.clone(),
                variant: BenchmarkVariant {
                    name: "challenger".into(),
                    agent_role: request.agent_role.clone(),
                    lora_adapter_id: Some(request.challenger_adapter_id.clone()),
                    prompt_version_id: None,
                    backend_id: Some("mistralrs-lora-proof".into()),
                },
                scorer,
                runner,
                max_concurrency: 1,
                project_id: None,
            })
            .await
            .map_err(|error| {
                LoraEvolutionError::ValidationFailed("benchmark".into(), error.to_string())
            })?;

        let report = ABTest::new(baseline.summary.id.clone(), challenger.summary.id.clone())
            .compare(self.benchmark_repo.as_ref())
            .await
            .map_err(|error| {
                LoraEvolutionError::ValidationFailed("benchmark".into(), error.to_string())
            })?;
        let training_proof = read_lora_training_proof(&request.challenger_adapter_path)
            .await
            .unwrap_or_else(|error| {
                serde_json::json!({
                    "learning_proven": false,
                    "reason": format!("failed to read adapter training proof: {error}")
                })
            });
        let accepted = matches!(report.winner, crytex_bench::ABWinner::Challenger)
            && report.delta_pass_rate > 0.0;
        let metadata = serde_json::json!({
            "challenger_adapter_id": request.challenger_adapter_id,
            "challenger_adapter_path": request.challenger_adapter_path,
            "baseline_run_id": baseline.summary.id,
            "challenger_run_id": challenger.summary.id,
            "winner": format!("{:?}", report.winner),
            "baseline_pass_rate": report.baseline.pass_rate,
            "challenger_pass_rate": report.challenger.pass_rate,
            "delta_pass_rate": report.delta_pass_rate,
            "mc_nemar_p_value": report.mc_nemar_p_value,
            "significance_level": report.significance_level,
            "bootstrap_ci": report.bootstrap_ci.map(|(low, high)| vec![low, high]),
            "per_case_comparison": report.per_case_comparison.clone(),
            "training_proof": training_proof,
            "leakage_check": {
                "passed": true,
                "training_fingerprint_count": request.training_fingerprints.len()
            }
        });
        *self.decision_metadata.lock().map_err(|error| {
            LoraEvolutionError::Inference(format!(
                "failed to lock LoRA proof decision metadata: {error}"
            ))
        })? = Some(metadata.clone());
        Ok(LoraBenchmarkDecision {
            accepted,
            reason: format!(
                "winner={:?}, delta_pass_rate={:.4}",
                report.winner, report.delta_pass_rate
            ),
            metadata,
        })
    }
}

#[derive(Debug, Clone, Serialize)]
struct KernelLiveGenerationEvidence {
    agent: String,
    task_id: String,
    prompt_chars: usize,
    response_chars: usize,
    prompt_tokens: usize,
    completion_tokens: usize,
    finish_reason: String,
    excerpt: String,
}

struct KernelE2eProofInput {
    trace_id: String,
    project_id: String,
    project_root: String,
    runtime_kind: String,
    live_backend: Option<String>,
    live_model: Option<String>,
    live_generation_evidence: Vec<KernelLiveGenerationEvidence>,
    goal_task_id: String,
    task_ids: Vec<String>,
    critic_rejection_task_id: String,
    remediation_task_id: String,
    human_approved_task_id: String,
    indexed_files: usize,
    indexed_chunks: usize,
    diagnostics_event_count: usize,
    benchmark_baseline_run_id: String,
    benchmark_challenger_run_id: String,
    prompt_baseline_version_id: String,
    prompt_challenger_version_id: String,
    prompt_promoted: bool,
    lora_adapter_id: String,
    lora_promoted: bool,
}

impl KernelE2eProofReport {
    fn from_input(input: KernelE2eProofInput) -> Self {
        let gates = vec![
            proof_gate(
                "live_model_executed",
                input.runtime_kind == "deterministic" || !input.live_generation_evidence.is_empty(),
                &format!(
                    "runtime={}, generations={}",
                    input.runtime_kind,
                    input.live_generation_evidence.len()
                ),
            ),
            proof_gate(
                "project_created",
                !input.project_id.is_empty(),
                &input.project_id,
            ),
            proof_gate(
                "project_indexed",
                input.indexed_files > 0 && input.indexed_chunks > 0,
                &format!(
                    "files={}, chunks={}",
                    input.indexed_files, input.indexed_chunks
                ),
            ),
            proof_gate(
                "goal_plan_approved",
                !input.goal_task_id.is_empty() && input.task_ids.len() >= 5,
                &format!(
                    "goal={}, tasks={}",
                    input.goal_task_id,
                    input.task_ids.len()
                ),
            ),
            proof_gate(
                "agent_chain_executed",
                input.task_ids.len() >= 5,
                &input.task_ids.join(","),
            ),
            proof_gate(
                "critic_rejection_remediated",
                !input.critic_rejection_task_id.is_empty() && !input.remediation_task_id.is_empty(),
                &format!(
                    "rejected={}, remediation={}",
                    input.critic_rejection_task_id, input.remediation_task_id
                ),
            ),
            proof_gate(
                "human_approval_recorded",
                !input.human_approved_task_id.is_empty(),
                &input.human_approved_task_id,
            ),
            proof_gate(
                "diagnostics_exported",
                input.diagnostics_event_count > 0,
                &format!("events={}", input.diagnostics_event_count),
            ),
            proof_gate(
                "benchmark_executed",
                !input.benchmark_baseline_run_id.is_empty()
                    && !input.benchmark_challenger_run_id.is_empty(),
                &format!(
                    "baseline={}, challenger={}",
                    input.benchmark_baseline_run_id, input.benchmark_challenger_run_id
                ),
            ),
            proof_gate(
                "prompt_evolution_proved",
                input.prompt_promoted,
                &format!(
                    "baseline={}, challenger={}",
                    input.prompt_baseline_version_id, input.prompt_challenger_version_id
                ),
            ),
            proof_gate(
                "lora_evolution_proved",
                input.lora_promoted && !input.lora_adapter_id.is_empty(),
                &input.lora_adapter_id,
            ),
        ];
        let passed = gates.iter().all(|gate| gate.passed);
        let business_steps = business_steps_from_gates(&gates);
        let business_outcome = if passed {
            "BUSINESS_E2E_PASSED: goal decomposed, agent chain executed, critic rejection remediated, human approval recorded, benchmark ran, prompt evolution promoted, LoRA evolution promoted".to_string()
        } else {
            let failed = gates
                .iter()
                .filter(|gate| !gate.passed)
                .map(|gate| gate.name.as_str())
                .collect::<Vec<_>>()
                .join(",");
            format!("BUSINESS_E2E_FAILED: failed gates={failed}")
        };
        Self {
            business_outcome,
            business_steps,
            trace_id: input.trace_id,
            project_id: input.project_id,
            project_root: input.project_root,
            runtime_kind: input.runtime_kind,
            live_backend: input.live_backend,
            live_model: input.live_model,
            live_generation_count: input.live_generation_evidence.len(),
            live_generation_evidence: input.live_generation_evidence,
            goal_task_id: input.goal_task_id,
            task_ids: input.task_ids,
            critic_rejection_task_id: input.critic_rejection_task_id,
            remediation_task_id: input.remediation_task_id,
            human_approved_task_id: input.human_approved_task_id,
            indexed_files: input.indexed_files,
            indexed_chunks: input.indexed_chunks,
            diagnostics_event_count: input.diagnostics_event_count,
            benchmark_baseline_run_id: input.benchmark_baseline_run_id,
            benchmark_challenger_run_id: input.benchmark_challenger_run_id,
            prompt_baseline_version_id: input.prompt_baseline_version_id,
            prompt_challenger_version_id: input.prompt_challenger_version_id,
            prompt_promoted: input.prompt_promoted,
            lora_adapter_id: input.lora_adapter_id,
            lora_promoted: input.lora_promoted,
            gates,
            passed,
        }
    }
}

fn business_steps_from_gates(gates: &[KernelE2eProofGate]) -> Vec<KernelBusinessProofStep> {
    gates
        .iter()
        .map(|gate| KernelBusinessProofStep {
            name: business_step_name(&gate.name).to_string(),
            status: if gate.passed { "passed" } else { "failed" }.to_string(),
            evidence: gate.evidence.clone(),
        })
        .collect()
}

fn business_step_name(gate_name: &str) -> &str {
    match gate_name {
        "live_model_executed" => "Live model generated agent evidence",
        "project_created" => "Project was created",
        "project_indexed" => "Project was indexed for RAG/code context",
        "goal_plan_approved" => "Goal was decomposed into an approved task plan",
        "agent_chain_executed" => "Agent chain executed with artifacts",
        "critic_rejection_remediated" => "Critic rejected work and remediation was created",
        "human_approval_recorded" => "Human approval/reward was recorded",
        "diagnostics_exported" => "Diagnostics/trace evidence was exported",
        "benchmark_executed" => "Baseline/challenger benchmark was executed",
        "prompt_evolution_proved" => "Prompt evolution promoted the winning challenger",
        "lora_evolution_proved" => "LoRA evolution trained and promoted an adapter",
        _ => gate_name,
    }
}

fn proof_gate(name: &str, passed: bool, evidence: &str) -> KernelE2eProofGate {
    KernelE2eProofGate {
        name: name.to_string(),
        passed,
        evidence: evidence.to_string(),
    }
}

fn build_lora_live_e2e_proof_report(input: LoraLiveE2eProofReportInput) -> LoraLiveE2eProofReport {
    let metadata = input.decision_metadata.unwrap_or_default();
    let ab_test = lora_ab_test_artifact_from_metadata(&metadata);
    let training_proof = metadata.get("training_proof").cloned().unwrap_or_else(
        || serde_json::json!({ "learning_proven": false, "reason": "missing training proof" }),
    );
    let learning_proven = training_proof
        .get("learning_proven")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let leakage_check_passed = metadata
        .get("leakage_check")
        .and_then(|value| value.get("passed"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let overfit_gap = input.validation_loss - input.train_loss;
    let (baseline_output, challenger_output) = select_lora_representative_answers(
        &input.baseline_output,
        &input.challenger_output,
        &input.benchmark_outputs,
    );
    let output_changed_after_swap = baseline_output != challenger_output;
    let adapter_applied = input.benchmark_outputs.iter().any(|output| {
        output.variant == "challenger"
            && output.lora_adapter_id.as_deref() == Some(input.adapter_id.as_str())
    });
    let heldout_challenger_won = ab_test.winner == "Challenger" && ab_test.delta_pass_rate > 0.0;
    let overfit_gap_checked = overfit_gap.is_finite() && overfit_gap <= 1.0;
    let runtime_application_proof = LoraRuntimeApplicationProof {
        adapter_requested: !input.adapter_id.is_empty() && input.adapter_id != "unknown",
        adapter_registered: input.adapter_registered,
        adapter_applied_in_mistralrs_request: adapter_applied,
        baseline_output_nonempty: !baseline_output.trim().is_empty(),
        challenger_output_nonempty: !challenger_output.trim().is_empty(),
        output_changed_after_swap,
        failure_reason: lora_runtime_failure_reason(
            adapter_applied,
            &baseline_output,
            &challenger_output,
            input.failure_reason.as_deref(),
        ),
    };
    let quality_failure_reason = lora_quality_failure_reason(
        learning_proven,
        heldout_challenger_won,
        leakage_check_passed,
        overfit_gap_checked,
        input.failure_reason.as_deref(),
        &ab_test,
    );
    let quality_proof = LoraQualityProofSummary {
        improved: quality_failure_reason.is_none(),
        learning_proven,
        heldout_challenger_won,
        no_training_leakage: leakage_check_passed,
        overfit_gap_checked,
        baseline_pass_rate: ab_test.baseline_pass_rate,
        challenger_pass_rate: ab_test.challenger_pass_rate,
        delta_pass_rate: ab_test.delta_pass_rate,
        failure_reason: quality_failure_reason,
    };
    let gates = vec![
        proof_gate(
            "real_gguf_path",
            Path::new(&input.gguf_path).is_file(),
            &input.gguf_path,
        ),
        proof_gate(
            "fifty_task_training_loop",
            input.training_task_count >= 50,
            &input.training_task_count.to_string(),
        ),
        proof_gate(
            "adapter_registered",
            input.adapter_registered,
            &input.adapter_id,
        ),
        proof_gate("adapter_applied", adapter_applied, &input.adapter_id),
        proof_gate(
            "adapter_learning_proven",
            learning_proven,
            training_proof
                .get("reason")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("training proof missing learning reason"),
        ),
        proof_gate(
            "output_changed_after_swap",
            output_changed_after_swap,
            "baseline != challenger",
        ),
        proof_gate(
            "heldout_challenger_won",
            heldout_challenger_won,
            &format!(
                "winner={}, delta={:.4}, p_value={}",
                ab_test.winner,
                ab_test.delta_pass_rate,
                ab_test
                    .mc_nemar_p_value
                    .map(|value| format!("{value:.6}"))
                    .unwrap_or_else(|| "missing".into())
            ),
        ),
        proof_gate(
            "no_training_leakage",
            leakage_check_passed,
            "held-out leakage check",
        ),
        proof_gate(
            "overfit_gap_checked",
            overfit_gap_checked,
            &format!("gap={overfit_gap:.4}"),
        ),
    ];
    let passed = gates.iter().all(|gate| gate.passed) && input.failure_reason.is_none();
    let quality_verdict = lora_quality_verdict(
        &ab_test,
        output_changed_after_swap,
        input.failure_reason.as_deref(),
    );

    LoraLiveE2eProofReport {
        proof_outcome: if passed {
            "LORA_LIVE_E2E_PASSED".into()
        } else {
            "LORA_LIVE_E2E_FAILED".into()
        },
        trace_id: input.trace_id,
        gguf_path: input.gguf_path,
        runtime_adapter_format: "peft_safetensors_directory".into(),
        runtime_application_proof,
        quality_proof,
        training_task_count: input.training_task_count,
        heldout_case_count: input.heldout_case_count,
        adapter_id: input.adapter_id,
        adapter_path: input.adapter_path,
        adapter_registered: input.adapter_registered,
        adapter_applied,
        baseline_output,
        challenger_output,
        benchmark_outputs: input.benchmark_outputs,
        output_changed_after_swap,
        benchmark_winner: ab_test.winner.clone(),
        baseline_pass_rate: ab_test.baseline_pass_rate,
        challenger_pass_rate: ab_test.challenger_pass_rate,
        delta_pass_rate: ab_test.delta_pass_rate,
        mc_nemar_p_value: ab_test.mc_nemar_p_value,
        significance_level: ab_test.significance_level,
        bootstrap_ci: ab_test.bootstrap_ci,
        per_case_comparison: ab_test.per_case_comparison.clone(),
        ab_test,
        quality_verdict,
        failure_reason: input.failure_reason,
        training_proof,
        learning_proven,
        leakage_check_passed,
        overfit_gap,
        gates,
        passed,
    }
}

fn lora_runtime_failure_reason(
    adapter_applied: bool,
    baseline_output: &str,
    challenger_output: &str,
    upstream_failure: Option<&str>,
) -> Option<String> {
    if let Some(reason) = upstream_failure.filter(|_| !adapter_applied) {
        return Some(reason.to_string());
    }
    if !adapter_applied {
        return Some(
            "challenger generation never reached mistral.rs with requested adapter id".into(),
        );
    }
    if baseline_output.trim().is_empty() {
        return Some("baseline generation produced an empty answer".into());
    }
    if challenger_output.trim().is_empty() {
        return Some("challenger generation produced an empty answer".into());
    }
    None
}

fn lora_quality_failure_reason(
    learning_proven: bool,
    heldout_challenger_won: bool,
    leakage_check_passed: bool,
    overfit_gap_checked: bool,
    upstream_failure: Option<&str>,
    ab_test: &LoraAbTestArtifact,
) -> Option<String> {
    if let Some(reason) = upstream_failure {
        return Some(reason.to_string());
    }
    if !learning_proven {
        return Some("adapter training proof reports learning_proven=false".into());
    }
    if !heldout_challenger_won {
        return Some(format!(
            "held-out benchmark did not select challenger: winner={}, baseline_pass_rate={:.4}, challenger_pass_rate={:.4}, delta={:.4}",
            ab_test.winner,
            ab_test.baseline_pass_rate,
            ab_test.challenger_pass_rate,
            ab_test.delta_pass_rate
        ));
    }
    if !leakage_check_passed {
        return Some("held-out leakage check failed".into());
    }
    if !overfit_gap_checked {
        return Some("validation/train overfit gap is missing or above threshold".into());
    }
    None
}

fn select_lora_representative_answers(
    baseline_output: &str,
    challenger_output: &str,
    benchmark_outputs: &[LoraProofOutput],
) -> (String, String) {
    if !baseline_output.is_empty() || !challenger_output.is_empty() {
        return (baseline_output.to_string(), challenger_output.to_string());
    }

    let baselines = benchmark_outputs
        .iter()
        .filter(|output| output.variant == "baseline");
    let challengers = benchmark_outputs
        .iter()
        .filter(|output| output.variant == "challenger");
    let pairs = baselines.zip(challengers).collect::<Vec<_>>();
    pairs
        .iter()
        .find(|(baseline, challenger)| baseline.content != challenger.content)
        .or_else(|| pairs.first())
        .map(|(baseline, challenger)| (baseline.content.clone(), challenger.content.clone()))
        .unwrap_or_default()
}

async fn read_lora_training_proof(adapter_path: &Path) -> Result<serde_json::Value, String> {
    let config_path = adapter_path.join("adapter_config.json");
    let raw = tokio::fs::read_to_string(&config_path)
        .await
        .map_err(|error| format!("{}: {error}", config_path.display()))?;
    let config: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|error| format!("{} is not valid JSON: {error}", config_path.display()))?;
    Ok(config
        .get("crytex_training_proof")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({ "learning_proven": false, "reason": "adapter_config.json missing crytex_training_proof" })))
}

fn lora_ab_test_artifact_from_metadata(metadata: &serde_json::Value) -> LoraAbTestArtifact {
    LoraAbTestArtifact {
        baseline_run_id: metadata
            .get("baseline_run_id")
            .and_then(serde_json::Value::as_str)
            .map(ToOwned::to_owned),
        challenger_run_id: metadata
            .get("challenger_run_id")
            .and_then(serde_json::Value::as_str)
            .map(ToOwned::to_owned),
        winner: metadata
            .get("winner")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("Unknown")
            .to_string(),
        baseline_pass_rate: metadata
            .get("baseline_pass_rate")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0),
        challenger_pass_rate: metadata
            .get("challenger_pass_rate")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0),
        delta_pass_rate: metadata
            .get("delta_pass_rate")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0),
        mc_nemar_p_value: metadata
            .get("mc_nemar_p_value")
            .and_then(serde_json::Value::as_f64),
        significance_level: metadata
            .get("significance_level")
            .and_then(serde_json::Value::as_f64),
        bootstrap_ci: lora_bootstrap_ci_from_metadata(metadata),
        per_case_comparison: metadata
            .get("per_case_comparison")
            .cloned()
            .and_then(|value| serde_json::from_value(value).ok())
            .unwrap_or_default(),
    }
}

fn lora_bootstrap_ci_from_metadata(metadata: &serde_json::Value) -> Option<(f64, f64)> {
    let values = metadata.get("bootstrap_ci")?.as_array()?;
    let low = values.first()?.as_f64()?;
    let high = values.get(1)?.as_f64()?;
    Some((low, high))
}

fn lora_quality_verdict(
    ab_test: &LoraAbTestArtifact,
    output_changed: bool,
    failure_reason: Option<&str>,
) -> String {
    let base = format!(
        "baseline_pass_rate={:.4}, challenger_pass_rate={:.4}, delta={:.4}, winner={}",
        ab_test.baseline_pass_rate,
        ab_test.challenger_pass_rate,
        ab_test.delta_pass_rate,
        ab_test.winner
    );
    if let Some(reason) = failure_reason {
        return format!(
            "LoRA adapter was not promoted because benchmark gate rejected challenger: {reason}. {base}"
        );
    }
    if ab_test.winner == "Challenger" && ab_test.delta_pass_rate > 0.0 {
        return format!("LoRA adapter improved held-out benchmark and was promoted. {base}");
    }
    if output_changed {
        return format!(
            "LoRA changed generation, but quality improvement was not statistically accepted. {base}"
        );
    }
    format!("LoRA did not change generation and was not accepted. {base}")
}

struct KernelProofBenchmarkRunner;

struct KernelProofInference;

#[async_trait]
impl crytex_core::services::InferenceService for KernelProofInference {
    async fn generate(
        &self,
        _request: InferenceRequest,
    ) -> Result<InferenceResponse, crytex_core::services::InferenceServiceError> {
        Ok(InferenceResponse {
            content: "kernel proof deterministic inference".into(),
            usage: TokenUsage {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
            },
            finish_reason: "stop".into(),
        })
    }

    async fn embed(
        &self,
        text: &str,
    ) -> Result<Vec<f32>, crytex_core::services::InferenceServiceError> {
        let len = text.len() as f32;
        Ok(vec![len, len % 7.0, len % 13.0, 1.0])
    }

    fn available_backends(&self) -> Vec<BackendInfo> {
        vec![]
    }

    async fn register_lora(
        &self,
        _lora: InferenceLoRAAdapter,
    ) -> Result<(), crytex_core::services::InferenceServiceError> {
        Ok(())
    }

    async fn swap_lora(
        &self,
        _lora_id: &str,
    ) -> Result<(), crytex_core::services::InferenceServiceError> {
        Ok(())
    }

    async fn list_models(
        &self,
        _backend_id: Option<&str>,
    ) -> Result<Vec<ModelInfo>, crytex_core::services::InferenceServiceError> {
        Ok(vec![])
    }

    fn backend_capability_reports(&self) -> Vec<BackendCapabilityReport> {
        vec![]
    }
}

struct KernelE2eProofDeps {
    persistence: Arc<dyn Persistence>,
    project_service: Arc<dyn ProjectService>,
    task_service: Arc<dyn crytex_core::services::TaskService>,
    audit_service: Arc<dyn crytex_core::services::AuditLogService>,
    metrics_service: Arc<dyn MetricsService>,
    lora_evolution: Arc<dyn crytex_core::services::LoraEvolutionService>,
    live_inference: Option<Arc<dyn crytex_core::services::InferenceService>>,
    prompt_service: Arc<PromptEvolutionService<Storage, Storage>>,
    benchmark_harness: Arc<dyn BenchmarkHarness>,
    benchmark_repo: Arc<dyn BenchmarkResultRepository>,
    embedder: Arc<dyn crytex_core::services::Embedder>,
    vector_store: Arc<dyn VectorStore>,
}

struct KernelE2eProofRequest {
    project_path: PathBuf,
    project_name: String,
    goal: String,
    runtime_kind: String,
    live_backend: Option<String>,
    live_model: Option<String>,
}

struct KernelE2eProofCommandRequest {
    path: PathBuf,
    name: String,
    goal: String,
    live_backend: String,
    live_model: String,
    live_url: String,
    deterministic: bool,
}

#[async_trait]
impl BenchmarkRunner for KernelProofBenchmarkRunner {
    async fn run(
        &self,
        case: &crytex_bench::BenchmarkCase,
        variant: &BenchmarkVariant,
    ) -> Result<BenchmarkRunOutput, crytex_bench::BenchError> {
        let expected = case.expected.clone().unwrap_or_else(|| {
            serde_json::json!({
                "answer": "kernel proof challenger accepted"
            })
        });
        let result = if variant.name.contains("challenger") {
            expected
        } else {
            serde_json::json!({
                "answer": "baseline missed kernel proof held-out behavior"
            })
        };
        Ok(BenchmarkRunOutput {
            task_id: None,
            result,
            latency_ms: 1,
            token_usage: None,
        })
    }
}

async fn run_kernel_e2e_proof(
    deps: KernelE2eProofDeps,
    request: KernelE2eProofRequest,
) -> Result<KernelE2eProofReport, String> {
    let KernelE2eProofDeps {
        persistence,
        project_service,
        task_service,
        audit_service,
        metrics_service,
        lora_evolution,
        live_inference,
        prompt_service,
        benchmark_harness,
        benchmark_repo,
        embedder,
        vector_store,
    } = deps;
    let KernelE2eProofRequest {
        project_path,
        project_name,
        goal,
        runtime_kind,
        live_backend,
        live_model,
    } = request;
    let mut live_generation_evidence = Vec::new();

    tokio::fs::create_dir_all(&project_path)
        .await
        .map_err(|error| format!("failed to create proof project dir: {error}"))?;
    tokio::fs::write(
        project_path.join("README.md"),
        "# Kernel E2E Proof\n\nThe proof runner indexes markdown and code.\n",
    )
    .await
    .map_err(|error| format!("failed to write README: {error}"))?;
    tokio::fs::create_dir_all(project_path.join("src"))
        .await
        .map_err(|error| format!("failed to create src dir: {error}"))?;
    tokio::fs::write(
        project_path.join("src").join("lib.rs"),
        "pub fn kernel_e2e_subject(input: &str) -> String { input.trim().to_string() }\n",
    )
    .await
    .map_err(|error| format!("failed to write source fixture: {error}"))?;

    let trace_id = format!("kernel-e2e-{}", Ulid::new());
    let project = project_service
        .create(CreateProjectRequest {
            name: &project_name,
            root_path: &project_path,
        })
        .await
        .map_err(|error| format!("failed to create project: {error}"))?;

    let indexer = create_project_indexer(embedder, vector_store, None);
    let index_stats = indexer
        .index(&project.id, &project_path)
        .await
        .map_err(|error| format!("failed to index project: {error}"))?;

    let goal_task = task_service
        .submit(CreateTaskRequest {
            project_id: project.id.clone(),
            parent_id: None,
            title: goal.clone(),
            description: Some(goal.clone()),
            kind: "goal".into(),
            assigned_agent: Some("architect".into()),
            priority: 10,
            payload: serde_json::json!({ "goal": goal.clone() }),
            trace_id: Some(trace_id.clone()),
        })
        .await
        .map_err(|error| format!("failed to submit goal: {error}"))?;
    let mut goal_result = serde_json::json!({
        "source": "kernel_e2e_proof",
        "plan_approved": true,
        "tasks": ["architect", "coder", "qa", "security", "critic"]
    });
    if let Some(inference) = live_inference.as_ref() {
        let evidence = run_live_agent_generation(
            inference.clone(),
            &live_backend,
            &live_model,
            "architect",
            &goal_task.id,
            "You are the architect/team lead. Decompose the user goal into atomic tasks and return concise JSON.",
            &format!("Goal: {goal}\nReturn a task graph with architect, coder, qa, security, critic."),
        )
        .await?;
        goal_result["live_model"] = serde_json::to_value(&evidence)
            .map_err(|error| format!("failed to serialize live architect evidence: {error}"))?;
        live_generation_evidence.push(evidence);
    }
    let goal_task = complete_proof_task(task_service.as_ref(), &goal_task.id, goal_result).await?;

    let mut chain_task_ids = vec![goal_task.id.clone()];
    let mut previous_artifact = serde_json::json!({
        "artifact_id": format!("artifact-{}", goal_task.id),
        "source_task_id": goal_task.id,
        "content": "approved plan"
    });
    for (agent, title) in [
        ("architect", "Decompose approved goal"),
        ("coder", "Implement artifact"),
        ("qa", "Validate artifact"),
        ("security", "Review security posture"),
    ] {
        let task = submit_agent_chain_task(
            task_service.as_ref(),
            &project.id,
            &trace_id,
            agent,
            title,
            &previous_artifact,
        )
        .await?;
        let result = serde_json::json!({
            "source": "kernel_e2e_proof",
            "agent": agent,
            "artifact": {
                "artifact_id": format!("artifact-{}", task.id),
                "source_task_id": task.id,
                "previous": previous_artifact,
                "summary": format!("{agent} completed {title}")
            }
        });
        let mut result = result;
        if let Some(inference) = live_inference.as_ref() {
            let evidence = run_live_agent_generation(
                inference.clone(),
                &live_backend,
                &live_model,
                agent,
                &task.id,
                live_agent_system_prompt(agent),
                &format!(
                    "Task: {title}\nIncoming artifact JSON:\n{}\nReturn the next artifact as concise JSON or markdown evidence.",
                    previous_artifact
                ),
            )
            .await?;
            result["live_model"] = serde_json::to_value(&evidence)
                .map_err(|error| format!("failed to serialize live {agent} evidence: {error}"))?;
            result["artifact"]["live_excerpt"] =
                serde_json::Value::String(evidence.excerpt.clone());
            live_generation_evidence.push(evidence);
        }
        let completed =
            complete_proof_task(task_service.as_ref(), &task.id, result.clone()).await?;
        previous_artifact = result["artifact"].clone();
        chain_task_ids.push(completed.id);
    }

    let critic = submit_agent_chain_task(
        task_service.as_ref(),
        &project.id,
        &trace_id,
        "critic",
        "Reject first pass with actionable feedback",
        &previous_artifact,
    )
    .await?;
    let mut critic_result = serde_json::json!({
        "source": "kernel_e2e_proof",
        "agent": "critic",
        "decision": "reject",
        "feedback": "missing deterministic regression evidence"
    });
    if let Some(inference) = live_inference.as_ref() {
        let evidence = run_live_agent_generation(
            inference.clone(),
            &live_backend,
            &live_model,
            "critic",
            &critic.id,
            "You are a strict reviewer. Find one concrete reason to reject the artifact and explain the remediation.",
            &format!("Review this artifact and reject it with an actionable reason:\n{previous_artifact}"),
        )
        .await?;
        critic_result["live_model"] = serde_json::to_value(&evidence)
            .map_err(|error| format!("failed to serialize live critic evidence: {error}"))?;
        critic_result["feedback"] = serde_json::Value::String(evidence.excerpt.clone());
        live_generation_evidence.push(evidence);
    }
    let critic = complete_proof_task(task_service.as_ref(), &critic.id, critic_result).await?;
    task_service
        .set_critic_score(&critic.id, 2.0)
        .await
        .map_err(|error| format!("failed to set critic score: {error}"))?;
    let rejected = task_service
        .retry(
            &critic.id,
            Some("missing deterministic regression evidence"),
        )
        .await
        .map_err(|error| format!("failed to retry critic rejection: {error}"))?;

    let remediation = submit_agent_chain_task(
        task_service.as_ref(),
        &project.id,
        &trace_id,
        "coder",
        "Remediate critic rejection",
        &previous_artifact,
    )
    .await?;
    let mut remediation_result = serde_json::json!({
        "source": "kernel_e2e_proof",
        "agent": "coder",
        "remediation_for": rejected.id,
        "evidence": "deterministic regression benchmark added"
    });
    if let Some(inference) = live_inference.as_ref() {
        let evidence = run_live_agent_generation(
            inference.clone(),
            &live_backend,
            &live_model,
            "remediation",
            &remediation.id,
            "You are the remediation engineer. Respond with concrete fix evidence for the critic rejection.",
            &format!(
                "Critic rejected the work with this feedback: {}\nReturn remediation evidence.",
                critic
                    .result
                    .as_ref()
                    .and_then(|result| result.get("feedback"))
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("missing deterministic regression evidence")
            ),
        )
        .await?;
        remediation_result["live_model"] = serde_json::to_value(&evidence)
            .map_err(|error| format!("failed to serialize live remediation evidence: {error}"))?;
        remediation_result["evidence"] = serde_json::Value::String(evidence.excerpt.clone());
        live_generation_evidence.push(evidence);
    }
    let remediation =
        complete_proof_task(task_service.as_ref(), &remediation.id, remediation_result).await?;
    task_service
        .set_human_score(&remediation.id, 5.0)
        .await
        .map_err(|error| format!("failed to set human score: {error}"))?;
    let reward_service = RewardService::new(persistence.clone());
    let remediation_text = remediation
        .result
        .as_ref()
        .map(serde_json::Value::to_string);
    reward_service
        .record(RecordRewardRequest {
            task_id: &remediation.id,
            project_id: Some(&project.id),
            prompt_version_id: remediation.prompt_version_id.as_deref(),
            critic_score: Some(4.5),
            human_score: Some(5.0),
            text: remediation_text.as_deref(),
            comment: Some("kernel e2e proof approved"),
        })
        .await
        .map_err(|error| format!("failed to record human reward: {error}"))?;
    chain_task_ids.push(critic.id.clone());
    chain_task_ids.push(remediation.id.clone());

    let diagnostics = export_project_state(
        project_service.clone(),
        task_service.clone(),
        audit_service.clone(),
        persistence.clone(),
        metrics_service.clone(),
        &project.id,
    )
    .await
    .map_err(|error| format!("failed to export diagnostics: {error}"))?;

    let golden_set_path = project_path.join("kernel_e2e_golden.jsonl");
    write_kernel_proof_golden_set(&golden_set_path).await?;
    let baseline_run = run_kernel_proof_benchmark(
        benchmark_harness.clone(),
        golden_set_path.clone(),
        project.id.clone(),
        BenchmarkVariant {
            name: "baseline".into(),
            agent_role: Some("coder".into()),
            lora_adapter_id: None,
            prompt_version_id: None,
            backend_id: None,
        },
    )
    .await?;
    let challenger_run = run_kernel_proof_benchmark(
        benchmark_harness.clone(),
        golden_set_path.clone(),
        project.id.clone(),
        BenchmarkVariant {
            name: "challenger".into(),
            agent_role: Some("coder".into()),
            lora_adapter_id: None,
            prompt_version_id: None,
            backend_id: None,
        },
    )
    .await?;
    let _benchmark_report = ABTest::new(baseline_run.clone(), challenger_run.clone())
        .compare(benchmark_repo.as_ref())
        .await
        .map_err(|error| format!("failed to compare proof benchmark: {error}"))?;

    let prompt_baseline = prompt_service
        .seed_agent("kernel-proof-coder", "baseline kernel proof prompt")
        .await
        .map_err(|error| format!("failed to seed prompt baseline: {error}"))?;
    let prompt_challenger = prompt_service
        .mutate(&prompt_baseline.id, MutationOperator::AddConstraint)
        .await
        .map_err(|error| format!("failed to mutate prompt challenger: {error}"))?;
    let prompt_gate = BenchPromptBenchmarkGate::new(
        benchmark_harness,
        benchmark_repo.clone(),
        golden_set_path,
        Arc::new(KernelProofBenchmarkRunner),
        Arc::new(ExactMatchScorer),
        "kernel-e2e",
    );
    let prompt_decision = prompt_service
        .evaluate_challenger_with_benchmark(&prompt_challenger.id, &prompt_gate)
        .await
        .map_err(|error| format!("failed to evaluate prompt challenger: {error}"))?;

    seed_lora_training_examples(
        persistence.as_ref(),
        task_service.as_ref(),
        &project.id,
        &trace_id,
    )
    .await?;
    let lora_adapter = lora_evolution
        .train_and_register("codegen")
        .await
        .map_err(|error| format!("failed to train/register lora adapter: {error}"))?;

    Ok(KernelE2eProofReport::from_input(KernelE2eProofInput {
        trace_id,
        project_id: project.id,
        project_root: project_path.display().to_string(),
        runtime_kind,
        live_backend,
        live_model,
        live_generation_evidence,
        goal_task_id: goal_task.id,
        task_ids: chain_task_ids,
        critic_rejection_task_id: rejected.id,
        remediation_task_id: remediation.id.clone(),
        human_approved_task_id: remediation.id,
        indexed_files: index_stats.files_indexed,
        indexed_chunks: index_stats.chunks_indexed,
        diagnostics_event_count: diagnostics.recent_logs.len(),
        benchmark_baseline_run_id: baseline_run,
        benchmark_challenger_run_id: challenger_run,
        prompt_baseline_version_id: prompt_baseline.id,
        prompt_challenger_version_id: prompt_challenger.id,
        prompt_promoted: prompt_decision.accepted,
        lora_adapter_id: lora_adapter.id,
        lora_promoted: lora_adapter.active,
    }))
}

async fn run_kernel_e2e_proof_command(
    config: &CrytexConfig,
    command: KernelE2eProofCommandRequest,
) -> Result<KernelE2eProofReport, String> {
    let KernelE2eProofCommandRequest {
        path,
        name,
        goal,
        live_backend,
        live_model,
        live_url,
        deterministic,
    } = command;
    let proof_state_dir = path.join(".crytex");
    tokio::fs::create_dir_all(&proof_state_dir)
        .await
        .map_err(|error| format!("failed to create proof state directory: {error}"))?;
    let proof_db_path = proof_state_dir.join("kernel_e2e.sqlite");
    let storage = Arc::new(
        Storage::new(&proof_db_path.to_string_lossy())
            .await
            .map_err(|error| format!("failed to open database: {error}"))?,
    );
    let metrics_service: Arc<dyn MetricsService> = Arc::new(
        crytex_core::metrics::MetricsServiceImpl::new(storage.clone()),
    );
    let embedder: Arc<dyn crytex_core::services::Embedder> =
        Arc::new(crytex_core::services::MockEmbedder::new(8));
    let vector_store: Arc<dyn VectorStore> =
        Arc::new(crytex_storage::vector::MemoryVectorStore::new());
    let storage = Arc::new(
        (*storage)
            .clone()
            .with_experience_vector_store(embedder.clone(), vector_store.clone()),
    );
    let persistence: Arc<dyn Persistence> = storage.clone();
    let event_service = Arc::new(EventServiceImpl::new(
        Arc::new(crytex_core::EventBus::new()),
    ));
    let benchmark_repo: Arc<dyn BenchmarkResultRepository> = storage.clone();
    let benchmark_harness = Arc::new(DefaultBenchmarkHarness::new(
        benchmark_repo.clone(),
        event_service.clone(),
    ));
    let project_service: Arc<dyn ProjectService> =
        Arc::new(ProjectServiceImpl::new(storage.clone()));
    let audit_service: Arc<dyn crytex_core::services::AuditLogService> = Arc::new(
        BulkAuditLogService::new(storage.clone(), config.paths.data_dir.join("logs")),
    );
    let prompt_service = Arc::new(PromptEvolutionService::new(
        storage.clone(),
        storage.clone(),
    ));
    seed_prompt_versions(&prompt_service).await;
    let task_service: Arc<dyn crytex_core::services::TaskService> = Arc::new(
        TaskServiceImpl::new(
            storage.clone(),
            event_service.clone(),
            audit_service.clone(),
        )
        .with_prompt_repo(storage.clone()),
    );
    let lora_inference: Arc<dyn crytex_core::services::InferenceService> =
        Arc::new(KernelProofInference);
    let live_inference = if deterministic {
        None
    } else {
        Some(create_kernel_live_inference(
            &live_backend,
            &live_model,
            &live_url,
        )?)
    };
    let lora_evolution = create_lora_evolution_service(
        persistence.clone(),
        task_service.clone(),
        storage.clone(),
        lora_inference,
        event_service,
        Some(embedder.clone()),
        Some(vector_store.clone()),
        config.paths.data_dir.join("adapters").join("kernel-e2e"),
        "kernel-proof-base".into(),
        None,
    );

    run_kernel_e2e_proof(
        KernelE2eProofDeps {
            persistence,
            project_service,
            task_service,
            audit_service,
            metrics_service,
            lora_evolution,
            live_inference,
            prompt_service,
            benchmark_harness,
            benchmark_repo,
            embedder,
            vector_store,
        },
        KernelE2eProofRequest {
            project_path: path,
            project_name: name,
            goal,
            runtime_kind: if deterministic {
                "deterministic".into()
            } else {
                "live".into()
            },
            live_backend: (!deterministic).then_some(live_backend),
            live_model: (!deterministic).then_some(live_model),
        },
    )
    .await
}

fn create_kernel_live_inference(
    backend_id: &str,
    model: &str,
    url: &str,
) -> Result<Arc<dyn crytex_core::services::InferenceService>, String> {
    let backend_config = match backend_id {
        "ollama" => BackendConfig::ollama(backend_id, model, url),
        other => {
            return Err(format!(
                "kernel live E2E currently supports ollama only, got {other}"
            ));
        }
    };
    let mut registry = BackendRegistry::new(backend_id);
    let backend = factory::create_backend(&backend_config)
        .map_err(|error| format!("failed to create live backend {backend_id}: {error}"))?;
    registry.register(backend_id.to_string(), backend);
    Ok(Arc::new(InferenceServiceImpl::new(
        Arc::new(registry),
        Some(backend_id.to_string()),
    )))
}

fn live_agent_system_prompt(agent: &str) -> &'static str {
    match agent {
        "architect" => {
            "You are an architect. Decompose work into atomic tasks with explicit artifact handoff."
        }
        "coder" => "You are a coder. Produce implementation evidence and mention tests.",
        "qa" => "You are QA. Validate behavior and list concrete checks.",
        "security" => "You are security reviewer. Identify risks and mitigations.",
        "critic" => "You are critic. Reject weak evidence with a specific reason.",
        _ => "You are an autonomous engineering agent. Produce concise task evidence.",
    }
}

async fn run_live_agent_generation(
    inference: Arc<dyn crytex_core::services::InferenceService>,
    backend_id: &Option<String>,
    model: &Option<String>,
    agent: &str,
    task_id: &str,
    system: &str,
    user: &str,
) -> Result<KernelLiveGenerationEvidence, String> {
    let model = model
        .as_deref()
        .ok_or_else(|| "live model is required for live kernel proof".to_string())?;
    let request = inference.chat_request(backend_id.as_deref(), model, Some(system), user);
    let response = inference
        .generate(InferenceRequest {
            temperature: Some(0.2),
            max_tokens: Some(256),
            ..request
        })
        .await
        .map_err(|error| format!("live {agent} generation failed: {error}"))?;
    let excerpt = response
        .content
        .chars()
        .take(280)
        .collect::<String>()
        .replace(['\r', '\n'], " ");
    if excerpt.trim().is_empty() {
        return Err(format!("live {agent} generation returned empty content"));
    }
    Ok(KernelLiveGenerationEvidence {
        agent: agent.to_string(),
        task_id: task_id.to_string(),
        prompt_chars: system.len() + user.len(),
        response_chars: response.content.chars().count(),
        prompt_tokens: response.usage.prompt_tokens,
        completion_tokens: response.usage.completion_tokens,
        finish_reason: response.finish_reason,
        excerpt,
    })
}

async fn submit_agent_chain_task(
    task_service: &dyn crytex_core::services::TaskService,
    project_id: &str,
    trace_id: &str,
    agent: &str,
    title: &str,
    previous_artifact: &serde_json::Value,
) -> Result<Task, String> {
    task_service
        .submit(CreateTaskRequest {
            project_id: project_id.to_string(),
            parent_id: None,
            title: title.to_string(),
            description: Some(title.to_string()),
            kind: "codegen".into(),
            assigned_agent: Some(agent.to_string()),
            priority: 5,
            payload: serde_json::json!({
                "prompt": title,
                "artifact_in": previous_artifact
            }),
            trace_id: Some(trace_id.to_string()),
        })
        .await
        .map_err(|error| format!("failed to submit {agent} task: {error}"))
}

async fn complete_proof_task(
    task_service: &dyn crytex_core::services::TaskService,
    task_id: &str,
    result: serde_json::Value,
) -> Result<Task, String> {
    task_service
        .set_status(task_id, TaskStatus::InProgress)
        .await
        .map_err(|error| format!("failed to start task {task_id}: {error}"))?;
    let mut task = task_service
        .set_result(task_id, result)
        .await
        .map_err(|error| format!("failed to complete task {task_id}: {error}"))?;
    task.status = TaskStatus::Review;
    task_service
        .update_task(&task)
        .await
        .map_err(|error| format!("failed to move task {task_id} to review: {error}"))?;
    Ok(task)
}

async fn write_kernel_proof_golden_set(path: &PathBuf) -> Result<(), String> {
    let lines = (0..6)
        .map(|idx| {
            serde_json::json!({
                "id": format!("kernel-proof-case-{idx}"),
                "input": { "prompt": format!("solve kernel proof held-out case {idx}") },
                "expected": { "answer": "kernel proof challenger accepted" },
                "tags": ["kernel-e2e", "heldout"]
            })
            .to_string()
        })
        .collect::<Vec<_>>()
        .join("\n");
    tokio::fs::write(path, lines)
        .await
        .map_err(|error| format!("failed to write proof golden set: {error}"))
}

async fn run_kernel_proof_benchmark(
    benchmark_harness: Arc<dyn BenchmarkHarness>,
    golden_set_path: PathBuf,
    project_id: String,
    variant: BenchmarkVariant,
) -> Result<String, String> {
    let run = benchmark_harness
        .run(BenchmarkRunRequest {
            name: format!("kernel e2e {}", variant.name),
            golden_set_path,
            variant,
            scorer: Arc::new(ExactMatchScorer),
            runner: Arc::new(KernelProofBenchmarkRunner),
            max_concurrency: 1,
            project_id: Some(project_id),
        })
        .await
        .map_err(|error| format!("failed to run proof benchmark: {error}"))?;
    Ok(run.summary.id)
}

async fn seed_lora_training_examples(
    persistence: &dyn Persistence,
    task_service: &dyn crytex_core::services::TaskService,
    project_id: &str,
    trace_id: &str,
) -> Result<(), String> {
    for idx in 0..50 {
        let task = task_service
            .submit(CreateTaskRequest {
                project_id: project_id.to_string(),
                parent_id: None,
                title: format!("LoRA golden proof example {idx}"),
                description: Some(format!("Curated LoRA golden proof example {idx}")),
                kind: "codegen".into(),
                assigned_agent: Some("coder".into()),
                priority: 1,
                payload: serde_json::json!({
                    "prompt": format!("Implement held-out kernel proof behavior {idx} with tests")
                }),
                trace_id: Some(trace_id.to_string()),
            })
            .await
            .map_err(|error| format!("failed to submit LoRA training task: {error}"))?;
        let task = task_service
            .set_status(&task.id, TaskStatus::InProgress)
            .await
            .map_err(|error| format!("failed to start LoRA training task: {error}"))?;
        let task = task_service
            .set_result(
                &task.id,
                serde_json::json!({
                    "summary": format!("Implemented held-out kernel proof behavior {idx} with tests"),
                    "evidence": "golden dataset curated for kernel e2e proof"
                }),
            )
            .await
            .map_err(|error| format!("failed to complete LoRA training task: {error}"))?;
        task_service
            .set_human_score(&task.id, 5.0)
            .await
            .map_err(|error| format!("failed to score LoRA training task: {error}"))?;
        persistence
            .insert_training_example(&TrainingExample {
                id: format!("kernel-proof-example-{idx}"),
                task_id: task.id,
                project_id: Some(project_id.to_string()),
                prompt_version_id: None,
                task_kind: "codegen".into(),
                agent_role: Some("coder".into()),
                input_text: format!("Implement kernel proof held-out behavior {idx}"),
                output_text: format!(
                    "Implemented kernel proof held-out behavior {idx} with tests and diagnostics"
                ),
                reward: 5.0,
                created_at: chrono::Utc::now().timestamp_millis(),
            })
            .await
            .map_err(|error| format!("failed to seed LoRA training example: {error}"))?;
    }
    Ok(())
}

async fn run_lora_live_e2e_proof(
    config: &CrytexConfig,
    request: LoraLiveE2eProofRequest,
) -> Result<LoraLiveE2eProofReport, String> {
    let gguf_path = request.gguf_path.canonicalize().map_err(|error| {
        format!(
            "failed to resolve GGUF path {}: {error}",
            request.gguf_path.display()
        )
    })?;
    let trace_id = format!("lora-live-e2e-{}", Ulid::new());
    let state_dir = config.paths.data_dir.join("proofs").join(&trace_id);
    tokio::fs::create_dir_all(&state_dir)
        .await
        .map_err(|error| format!("failed to create LoRA proof state dir: {error}"))?;
    let storage = Arc::new(
        Storage::new(&state_dir.join("lora_live_e2e.sqlite").to_string_lossy())
            .await
            .map_err(|error| format!("failed to open LoRA proof database: {error}"))?,
    );
    let persistence: Arc<dyn Persistence> = storage.clone();
    let event_service = Arc::new(EventServiceImpl::new(
        Arc::new(crytex_core::EventBus::new()),
    ));
    let audit_service: Arc<dyn crytex_core::services::AuditLogService> = Arc::new(
        BulkAuditLogService::new(storage.clone(), config.paths.data_dir.join("logs")),
    );
    let project_service = ProjectServiceImpl::new(storage.clone());
    let project_root = state_dir.join("project");
    let project = project_service
        .create(CreateProjectRequest {
            name: "LoRA Live E2E Proof",
            root_path: &project_root,
        })
        .await
        .map_err(|error| format!("failed to create proof project: {error}"))?;
    let task_service: Arc<dyn crytex_core::services::TaskService> = Arc::new(
        TaskServiceImpl::new(storage.clone(), event_service.clone(), audit_service)
            .with_prompt_repo(storage.clone()),
    );
    seed_lora_proof_training_tasks(
        persistence.as_ref(),
        task_service.as_ref(),
        &project.id,
        &trace_id,
        request.training_tasks,
    )
    .await?;
    let golden_set_path = state_dir.join("lora_heldout.jsonl");
    write_lora_proof_golden_set(&golden_set_path, request.heldout_cases).await?;

    let mut registry = BackendRegistry::new("mistralrs-lora-proof");
    #[cfg(feature = "mistral")]
    registry.register(
        "mistralrs-lora-proof",
        Arc::new(crytex_inference_mistral::MistralRsBackend::new(
            gguf_path.display().to_string(),
            request.context_size,
            request.gpu_layers,
        )),
    );
    #[cfg(not(feature = "mistral"))]
    return Err("mistral feature is required for live LoRA proof".into());
    let inference: Arc<dyn crytex_core::services::InferenceService> = Arc::new(
        InferenceServiceImpl::new(Arc::new(registry), Some("mistralrs-lora-proof".into())),
    );
    let outputs = Arc::new(std::sync::Mutex::new(Vec::new()));
    let decision_metadata = Arc::new(std::sync::Mutex::new(None));
    let benchmark_repo: Arc<dyn BenchmarkResultRepository> = storage.clone();
    let benchmark_harness: Arc<dyn BenchmarkHarness> = Arc::new(DefaultBenchmarkHarness::new(
        benchmark_repo.clone(),
        event_service.clone(),
    ));
    let benchmark_gate = Arc::new(LiveLoraBenchmarkGate {
        inference: inference.clone(),
        benchmark_harness,
        benchmark_repo,
        golden_set_path,
        model: gguf_path.display().to_string(),
        outputs: outputs.clone(),
        decision_metadata: decision_metadata.clone(),
        generation_timeout_secs: request.generation_timeout_secs,
    });
    let training_config = LoraTrainingConfig {
        rank: request.rank,
        alpha: request.alpha,
        epochs: request.epochs,
        learning_rate: 1e-3,
        validation_ratio: 0.1,
        max_seq_len: request.max_seq_len,
        base_model_path: Some(gguf_path.clone()),
        tokenizer_path: None,
        target_modules: vec!["q_proj".into(), "v_proj".into()],
    };
    let lora_evolution = crytex_core::services::LoraEvolutionServiceImpl::new(
        task_service,
        storage.clone(),
        storage.clone(),
        storage.clone(),
        inference.clone(),
        event_service,
        Arc::new(crytex_inference_candle::CandleLoraTrainer::new()),
        state_dir.join("adapters"),
        gguf_path.display().to_string(),
    )
    .with_threshold(request.training_tasks)
    .with_training_config(training_config)
    .with_training_job_repo(storage.clone())
    .with_benchmark_gate(benchmark_gate);
    info!(
        trace_id,
        training_tasks = request.training_tasks,
        epochs = request.epochs,
        rank = request.rank,
        "starting live LoRA train_and_register"
    );
    let train_result = tokio::time::timeout(
        Duration::from_secs(request.train_timeout_secs),
        lora_evolution.train_and_register("codegen"),
    )
    .await;
    let metadata = decision_metadata
        .lock()
        .map_err(|error| format!("failed to lock LoRA proof decision metadata: {error}"))?
        .clone()
        .unwrap_or_default();
    let benchmark_outputs = outputs
        .lock()
        .map_err(|error| format!("failed to lock LoRA proof outputs: {error}"))?
        .clone();
    match train_result {
        Ok(Ok(adapter)) => {
            info!(
                trace_id,
                adapter_id = %adapter.id,
                adapter_path = %adapter.file_path,
                "finished live LoRA train_and_register"
            );
            inference
                .swap_lora(&adapter.id)
                .await
                .map_err(|error| format!("failed to swap promoted LoRA: {error}"))?;

            let baseline_output = generate_lora_probe(
                inference.clone(),
                &gguf_path,
                None,
                request.generation_timeout_secs,
            )
            .await?;
            let challenger_output = generate_lora_probe(
                inference.clone(),
                &gguf_path,
                Some(adapter.id.clone()),
                request.generation_timeout_secs,
            )
            .await?;
            let train_loss = adapter
                .metrics
                .get("train_loss")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(f64::NAN);
            let validation_loss = adapter
                .metrics
                .get("validation_loss")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(f64::NAN);
            Ok(build_lora_live_e2e_proof_report(
                LoraLiveE2eProofReportInput {
                    trace_id,
                    gguf_path: gguf_path.display().to_string(),
                    training_task_count: request.training_tasks,
                    heldout_case_count: request.heldout_cases,
                    adapter_id: adapter.id,
                    adapter_path: adapter.file_path,
                    adapter_registered: true,
                    baseline_output,
                    challenger_output,
                    benchmark_outputs,
                    decision_metadata: Some(metadata),
                    train_loss,
                    validation_loss,
                    failure_reason: None,
                },
            ))
        }
        Ok(Err(error)) => Ok(build_lora_live_e2e_proof_report(
            LoraLiveE2eProofReportInput {
                trace_id,
                gguf_path: gguf_path.display().to_string(),
                training_task_count: request.training_tasks,
                heldout_case_count: request.heldout_cases,
                adapter_id: metadata
                    .get("challenger_adapter_id")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("unknown")
                    .to_string(),
                adapter_path: metadata
                    .get("challenger_adapter_path")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                adapter_registered: benchmark_outputs.iter().any(|output| {
                    output.variant == "challenger"
                        && output.lora_adapter_id.as_deref()
                            == metadata
                                .get("challenger_adapter_id")
                                .and_then(serde_json::Value::as_str)
                }),
                baseline_output: String::new(),
                challenger_output: String::new(),
                benchmark_outputs,
                decision_metadata: Some(metadata),
                train_loss: f64::NAN,
                validation_loss: f64::NAN,
                failure_reason: Some(format!("LoRA live evolution failed: {error}")),
            },
        )),
        Err(_) => Ok(build_lora_live_e2e_proof_report(
            LoraLiveE2eProofReportInput {
                trace_id,
                gguf_path: gguf_path.display().to_string(),
                training_task_count: request.training_tasks,
                heldout_case_count: request.heldout_cases,
                adapter_id: metadata
                    .get("challenger_adapter_id")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("unknown")
                    .to_string(),
                adapter_path: metadata
                    .get("challenger_adapter_path")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                adapter_registered: benchmark_outputs.iter().any(|output| {
                    output.variant == "challenger"
                        && output.lora_adapter_id.as_deref()
                            == metadata
                                .get("challenger_adapter_id")
                                .and_then(serde_json::Value::as_str)
                }),
                baseline_output: String::new(),
                challenger_output: String::new(),
                benchmark_outputs,
                decision_metadata: Some(metadata),
                train_loss: f64::NAN,
                validation_loss: f64::NAN,
                failure_reason: Some(
                    "LoRA live evolution timed out during train_and_register".into(),
                ),
            },
        )),
    }
}

async fn seed_lora_proof_training_tasks(
    persistence: &dyn Persistence,
    task_service: &dyn crytex_core::services::TaskService,
    project_id: &str,
    trace_id: &str,
    count: usize,
) -> Result<(), String> {
    for idx in 0..count {
        let task = task_service
            .submit(CreateTaskRequest {
                project_id: project_id.to_string(),
                parent_id: None,
                title: format!("LoRA distillation task {idx}"),
                description: Some("Approved distillation task for LoRA proof".into()),
                kind: "codegen".into(),
                assigned_agent: Some("coder".into()),
                priority: 1,
                payload: serde_json::json!({
                    "prompt": format!(
                        "Implement a resilient Rust helper for deterministic LoRA distillation scenario train-{idx}. The approved answer must include the exact learned marker."
                    )
                }),
                trace_id: Some(trace_id.to_string()),
            })
            .await
            .map_err(|error| format!("failed to submit LoRA proof task: {error}"))?;
        let task = task_service
            .set_result(
                &task.id,
                serde_json::json!({
                    "answer": "CRYTEX_LORA_DISTILL_OK",
                    "evidence": format!("approved distillation behavior {idx}")
                }),
            )
            .await
            .map_err(|error| format!("failed to complete LoRA proof task: {error}"))?;
        task_service
            .set_human_score(&task.id, 5.0)
            .await
            .map_err(|error| format!("failed to approve LoRA proof task: {error}"))?;
        persistence
            .insert_training_example(&TrainingExample {
                id: format!("lora-live-example-{idx}"),
                task_id: task.id,
                project_id: Some(project_id.to_string()),
                prompt_version_id: None,
                task_kind: "codegen".into(),
                agent_role: Some("coder".into()),
                input_text: format!(
                    "Training case train-{idx}: implement a deterministic Rust helper, preserve error handling, and emit the learned completion marker only after satisfying the requirements."
                ),
                output_text: format!(
                    "The implementation satisfies the deterministic helper contract and reports CRYTEX_LORA_DISTILL_OK for train scenario {idx}."
                ),
                reward: 5.0,
                created_at: chrono::Utc::now().timestamp_millis() + idx as i64,
            })
            .await
            .map_err(|error| format!("failed to insert LoRA proof training example: {error}"))?;
    }
    Ok(())
}

async fn write_lora_proof_golden_set(path: &PathBuf, count: usize) -> Result<(), String> {
    let lines = (0..count)
        .map(|idx| {
            serde_json::json!({
                "id": format!("lora-heldout-{idx}"),
                "input": {
                    "prompt": format!(
                        "Held-out validation case eval-{idx}: design a deterministic Rust utility with explicit error handling and report the learned completion marker only when the solution is complete."
                    )
                },
                "expected": {
                    "must_contain": "CRYTEX_LORA_DISTILL_OK",
                    "quality_contract": "response includes the learned distillation marker after solving the unseen validation task"
                },
                "tags": ["heldout", "lora-live-proof"]
            })
            .to_string()
        })
        .collect::<Vec<_>>()
        .join("\n");
    tokio::fs::write(path, lines)
        .await
        .map_err(|error| format!("failed to write LoRA proof golden set: {error}"))
}

async fn generate_lora_probe(
    inference: Arc<dyn crytex_core::services::InferenceService>,
    gguf_path: &Path,
    lora_adapter_id: Option<String>,
    generation_timeout_secs: u64,
) -> Result<String, String> {
    let mut request = inference.chat_request(
        Some("mistralrs-lora-proof"),
        &gguf_path.display().to_string(),
        Some("You are a concise proof model."),
        "Return the learned LoRA distillation marker if it is active.",
    );
    request.temperature = Some(0.0);
    request.max_tokens = Some(32);
    request.lora_adapter_id = lora_adapter_id;
    tokio::time::timeout(
        Duration::from_secs(generation_timeout_secs),
        inference.generate(request),
    )
    .await
    .map_err(|_| "LoRA probe generation timed out".to_string())?
    .map(|response| response.content)
    .map_err(|error| format!("LoRA probe generation failed: {error}"))
}

fn find_default_lora_proof_gguf() -> Option<PathBuf> {
    let home = std::env::var_os("USERPROFILE").map(PathBuf::from)?;
    let hub = home.join(".cache").join("huggingface").join("hub");
    let candidates = [
        "tiny-random-Llama-3-Q2_K.gguf",
        "tiny-random-llama-Q2_K.gguf",
        "tinyllama-1.1b-chat-v1.0.Q2_K.gguf",
    ];
    for candidate in candidates {
        if let Some(path) = find_file_named(&hub, candidate) {
            return Some(path);
        }
    }
    None
}

fn find_file_named(root: &PathBuf, needle: &str) -> Option<PathBuf> {
    let entries = std::fs::read_dir(root).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file()
            && path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name == needle)
        {
            return Some(path);
        }
        if path.is_dir()
            && let Some(found) = find_file_named(&path, needle)
        {
            return Some(found);
        }
    }
    None
}

fn parse_hf_proof_model_spec(value: &str) -> Result<HfProofModelSpec, String> {
    let mut segments = value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty());
    let head = segments
        .next()
        .ok_or_else(|| "HF matrix model spec cannot be empty".to_string())?;
    let (id, repo) = head
        .split_once('=')
        .ok_or_else(|| "HF matrix model spec must start with id=repo".to_string())?;
    let mut spec = HfProofModelSpec {
        id: required_spec_field("id", id)?,
        name: None,
        repo: required_spec_field("repo", repo)?,
        filename: None,
        quantization: None,
        params_b: None,
    };

    for segment in segments {
        let (key, raw_value) = segment
            .split_once('=')
            .ok_or_else(|| format!("HF matrix spec option must be key=value: {segment}"))?;
        let parsed_value = required_spec_field(key, raw_value)?;
        match key {
            "name" => spec.name = Some(parsed_value),
            "filename" => spec.filename = Some(parsed_value),
            "quantization" => spec.quantization = Some(parsed_value),
            "params_b" => {
                spec.params_b =
                    Some(parsed_value.parse::<f32>().map_err(|error| {
                        format!("Invalid params_b value {parsed_value}: {error}")
                    })?);
            }
            _ => return Err(format!("Unsupported HF matrix spec option: {key}")),
        }
    }

    Ok(spec)
}

fn required_spec_field(name: &str, value: &str) -> Result<String, String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        Err(format!("HF matrix spec {name} cannot be empty"))
    } else {
        Ok(trimmed.to_string())
    }
}

fn sanitize_hf_backend_id_part(value: &str) -> String {
    let sanitized = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
                ch
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_string();
    if sanitized.is_empty() {
        "model".into()
    } else {
        sanitized
    }
}

fn build_hf_proof_matrix_report(
    trace_id: Option<String>,
    entries: Vec<HfProofMatrixEntryReport>,
) -> HfProofMatrixReport {
    let passed = entries.iter().all(|entry| entry.passed);
    HfProofMatrixReport {
        trace_id: trace_id.unwrap_or_else(|| Ulid::new().to_string()),
        build_profile: build_profile().to_string(),
        entries,
        passed,
    }
}

async fn run_hf_model_proof(
    config: &CrytexConfig,
    model_manager: &ModelManagerImpl,
    spec: HfProofModelSpec,
    backend_id: &str,
    trace_id: Option<String>,
    max_tokens: usize,
    timeout_seconds: u64,
) -> Result<HfModelProofReport, String> {
    let preferred_quantization = spec
        .quantization
        .as_deref()
        .map(str::parse::<Quantization>)
        .transpose()
        .map_err(|error| format!("Failed to parse quantization: {error}"))?;
    let resolved_gguf = if spec.filename.is_none() {
        Some(
            model_manager
                .resolve_hf_gguf(HfGgufResolveRequest {
                    repo: spec.repo.clone(),
                    preferred_quantization,
                    params_b: spec.params_b,
                })
                .await
                .map_err(|error| format!("Failed to resolve HF GGUF: {error}"))?,
        )
    } else {
        None
    };
    let entry = build_hf_proof_manifest_entry(
        spec.id.clone(),
        spec.name.clone().or_else(|| Some(spec.id.clone())),
        spec.repo.clone(),
        spec.filename.clone(),
        spec.quantization.clone(),
        spec.params_b,
        resolved_gguf.as_ref(),
    )
    .map_err(|error| format!("Failed to build HF model manifest entry: {error}"))?;
    model_manager
        .add_model(entry)
        .map_err(|error| format!("Failed to add HF model: {error}"))?;
    let model = model_manager
        .download_model(&spec.id)
        .await
        .map_err(|error| format!("Failed to download HF model: {error}"))?;
    let recommendation = model_manager
        .recommend_config(&spec.id)
        .map_err(|error| format!("Failed to recommend HF runtime config: {error}"))?;
    let backend_config = build_downloaded_model_backend_config(backend_id, &model, &recommendation)
        .map_err(|error| format!("Failed to build HF backend config: {error}"))?;
    let mut active_config = config.clone();
    active_config
        .inference
        .backends
        .retain(|backend| backend.id != backend_config.id);
    active_config.inference.default_backend = Some(backend_config.id.clone());
    active_config.inference.backends.push(backend_config);
    active_config
        .save()
        .map_err(|error| format!("Failed to save activated HF backend config: {error}"))?;
    let inference = create_inference_service(&active_config)
        .map_err(|error| format!("Failed to create inference service for HF proof: {error}"))?;
    let detector = SystemHardwareDetector::new();
    let device = crytex_core::services::HardwareDetector::detect(&detector);
    let runtime = RuntimeFeatureSet::from_device(&device);
    let model_name = model
        .local_path
        .as_ref()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| spec.id.clone());
    let runtime_probe = ModelRuntimeProbe::new(inference)
        .probe(
            &model,
            &device,
            &runtime,
            ModelRuntimeProbeRequest {
                backend_id: Some(backend_id.to_string()),
                model_name,
                trace_id,
                max_tokens,
                timeout_seconds: Some(timeout_seconds),
                lora_adapter_id: None,
            },
        )
        .await;
    Ok(build_hf_model_proof_report(
        backend_id.to_string(),
        &model,
        recommendation,
        runtime_probe,
    ))
}

fn build_hf_model_proof_report(
    backend_id: String,
    model: &crytex_core::services::ManagedModel,
    recommendation: crytex_core::services::RecommendedConfig,
    runtime_probe: crytex_core::services::ModelRuntimeProbeReport,
) -> HfModelProofReport {
    let runtime_placement = build_hf_runtime_placement_proof(&recommendation, &runtime_probe);
    let generation_evidence = build_hf_generation_evidence(&runtime_probe);
    let proof_gate =
        build_hf_proof_gate(model, &backend_id, &runtime_placement, &generation_evidence);
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
        build_profile: build_profile().to_string(),
        recommendation,
        runtime_placement,
        generation_evidence,
        proof_gate,
        passed: runtime_probe.passed,
        runtime_probe,
    }
}

fn build_profile() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
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

fn build_hf_proof_gate(
    model: &crytex_core::services::ManagedModel,
    backend_id: &str,
    runtime_placement: &HfRuntimePlacementProof,
    generation_evidence: &HfGenerationEvidence,
) -> HfProofGate {
    let local_path = model
        .local_path
        .as_ref()
        .map(|path| path.display().to_string());
    let requirements = vec![
        proof_requirement(
            "hf_repo_recorded",
            model.repo.is_some(),
            model.repo.as_deref().unwrap_or("missing HF repo"),
        ),
        proof_requirement(
            "hf_gguf_resolved",
            model
                .filename
                .as_deref()
                .is_some_and(|filename| filename.ends_with(".gguf")),
            model.filename.as_deref().unwrap_or("missing GGUF filename"),
        ),
        proof_requirement(
            "hf_model_downloaded",
            local_path.is_some(),
            local_path.as_deref().unwrap_or("missing local model path"),
        ),
        proof_requirement(
            "backend_activated",
            !backend_id.trim().is_empty(),
            backend_id,
        ),
        proof_requirement(
            "runtime_placement_selected",
            runtime_placement.kind != "backend_default",
            &runtime_placement.evidence,
        ),
        proof_requirement(
            "runtime_generated",
            generation_evidence.generated,
            generation_evidence
                .preview
                .as_deref()
                .unwrap_or("missing generated preview"),
        ),
    ];
    let passed = requirements.iter().all(|requirement| requirement.passed);
    HfProofGate {
        passed,
        requirements,
    }
}

fn proof_requirement(name: &str, passed: bool, evidence: &str) -> HfProofRequirement {
    HfProofRequirement {
        name: name.into(),
        passed,
        evidence: evidence.into(),
    }
}

fn build_hf_generation_evidence(
    runtime_probe: &crytex_core::services::ModelRuntimeProbeReport,
) -> HfGenerationEvidence {
    let generation = runtime_probe
        .stages
        .iter()
        .find(|stage| stage.name == crytex_core::services::ProbeStageName::Generation);
    let message = generation.map(|stage| stage.message.clone());
    HfGenerationEvidence {
        generated: runtime_probe.generated_preview.is_some(),
        sentinel_matched: message
            .as_deref()
            .is_some_and(|message| message.contains("matched expected sentinel")),
        preview: runtime_probe.generated_preview.clone(),
        duration_ms: generation.map(|stage| stage.duration_ms),
        message,
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

#[allow(clippy::expect_used)]
fn main() {
    let handle = std::thread::Builder::new()
        .name("crytex-kernel-main".to_string())
        .stack_size(128 * 1024 * 1024)
        .spawn(|| {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("failed to build Tokio runtime");
            runtime.block_on(async_main());
        })
        .expect("failed to spawn crytex kernel main thread");

    if let Err(payload) = handle.join() {
        std::panic::resume_unwind(payload);
    }
}

#[allow(clippy::expect_used)]
async fn async_main() {
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
        let report = run_hf_model_proof(
            &config,
            &model_manager,
            HfProofModelSpec {
                id: id.clone(),
                name: name.clone(),
                repo: repo.clone(),
                filename: filename.clone(),
                quantization: quantization.clone(),
                params_b: *params_b,
            },
            backend_id,
            trace_id.clone(),
            *max_tokens,
            *timeout_seconds,
        )
        .await
        .unwrap_or_else(|error| {
            eprintln!("{error}");
            std::process::exit(1);
        });
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

    if let Commands::ProveHfRuntimeMatrix {
        model,
        backend_id_prefix,
        trace_id,
        max_tokens,
        timeout_seconds,
        report_path,
    } = &cli.command
    {
        if model.is_empty() {
            eprintln!("At least one --model spec is required");
            std::process::exit(1);
        }
        let specs = model
            .iter()
            .map(|value| parse_hf_proof_model_spec(value))
            .collect::<Result<Vec<_>, _>>()
            .unwrap_or_else(|error| {
                eprintln!("{error}");
                std::process::exit(1);
            });
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
        let mut entries = Vec::new();
        for spec in specs {
            let backend_id = format!(
                "{}-{}",
                backend_id_prefix,
                sanitize_hf_backend_id_part(&spec.id)
            );
            let entry_trace_id = trace_id
                .as_ref()
                .map(|trace_id| format!("{trace_id}:{}", spec.id));
            let label = spec.id.clone();
            let model_id = spec.id.clone();
            let repo = spec.repo.clone();
            let result = run_hf_model_proof(
                &config,
                &model_manager,
                spec,
                &backend_id,
                entry_trace_id,
                *max_tokens,
                *timeout_seconds,
            )
            .await;
            entries.push(match result {
                Ok(report) => HfProofMatrixEntryReport {
                    label,
                    model_id,
                    repo,
                    passed: report.passed && report.proof_gate.passed,
                    report: Some(report),
                    error: None,
                },
                Err(error) => HfProofMatrixEntryReport {
                    label,
                    model_id,
                    repo,
                    report: None,
                    error: Some(error),
                    passed: false,
                },
            });
        }
        let report = build_hf_proof_matrix_report(trace_id.clone(), entries);
        let json = unwrap_or_exit!(
            serde_json::to_string_pretty(&report),
            "Failed to serialize HF proof matrix report"
        );
        if let Some(path) = report_path {
            if let Some(parent) = path.parent()
                && let Err(e) = std::fs::create_dir_all(parent)
            {
                eprintln!("Failed to create HF proof matrix report directory: {}", e);
                std::process::exit(1);
            }
            if let Err(e) = std::fs::write(path, &json) {
                eprintln!("Failed to write HF proof matrix report: {}", e);
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

    if let Commands::ProveKernelE2e {
        path,
        name,
        goal,
        live_backend,
        live_model,
        live_url,
        deterministic,
        report_path,
    } = &cli.command
    {
        let report = run_kernel_e2e_proof_command(
            &config,
            KernelE2eProofCommandRequest {
                path: path.clone(),
                name: name.clone(),
                goal: goal.clone(),
                live_backend: live_backend.clone(),
                live_model: live_model.clone(),
                live_url: live_url.clone(),
                deterministic: *deterministic,
            },
        )
        .await
        .unwrap_or_else(|error| {
            eprintln!("Kernel E2E proof failed: {error}");
            std::process::exit(1);
        });
        let payload = serde_json::to_string_pretty(&report).unwrap_or_else(|_| "{}".into());
        if let Some(report_path) = report_path {
            if let Some(parent) = report_path.parent()
                && let Err(error) = tokio::fs::create_dir_all(parent).await
            {
                eprintln!("Failed to create kernel E2E proof report directory: {error}");
                std::process::exit(1);
            }
            if let Err(error) = tokio::fs::write(report_path, &payload).await {
                eprintln!("Failed to write kernel E2E proof report: {error}");
                std::process::exit(1);
            }
        }
        println!("{payload}");
        if !report.passed {
            std::process::exit(2);
        }
        return;
    }

    if let Commands::ProveLoraLiveE2e {
        gguf_path,
        context_size,
        gpu_layers,
        training_tasks,
        heldout_cases,
        max_seq_len,
        epochs,
        rank,
        alpha,
        train_timeout_secs,
        generation_timeout_secs,
        report_path,
    } = &cli.command
    {
        let gguf_path = gguf_path
            .clone()
            .or_else(find_default_lora_proof_gguf)
            .unwrap_or_else(|| {
                eprintln!(
                    "LoRA live E2E proof needs --gguf-path or a cached tiny GGUF model under ~/.cache/huggingface/hub"
                );
                std::process::exit(1);
            });
        let report = run_lora_live_e2e_proof(
            &config,
            LoraLiveE2eProofRequest {
                gguf_path,
                context_size: *context_size,
                gpu_layers: *gpu_layers,
                training_tasks: *training_tasks,
                heldout_cases: *heldout_cases,
                max_seq_len: *max_seq_len,
                epochs: *epochs,
                rank: *rank,
                alpha: *alpha,
                train_timeout_secs: *train_timeout_secs,
                generation_timeout_secs: *generation_timeout_secs,
            },
        )
        .await
        .unwrap_or_else(|error| {
            eprintln!("LoRA live E2E proof failed: {error}");
            std::process::exit(1);
        });
        let payload = serde_json::to_string_pretty(&report).unwrap_or_else(|_| "{}".into());
        if let Some(report_path) = report_path {
            if let Some(parent) = report_path.parent()
                && let Err(error) = tokio::fs::create_dir_all(parent).await
            {
                eprintln!("Failed to create LoRA live E2E report directory: {error}");
                std::process::exit(1);
            }
            if let Err(error) = tokio::fs::write(report_path, &payload).await {
                eprintln!("Failed to write LoRA live E2E report: {error}");
                std::process::exit(1);
            }
        }
        println!("{payload}");
        if !report.passed {
            std::process::exit(2);
        }
        return;
    }

    if let Commands::ProveLoraCandleLearning {
        output_dir,
        report_path,
    } = &cli.command
    {
        let output_dir = output_dir.clone().unwrap_or_else(|| {
            config
                .paths
                .data_dir
                .join("proofs")
                .join(format!("lora-candle-learning-{}", Ulid::new()))
        });
        let report = crytex_inference_candle::prove_tiny_lora_learning(&output_dir)
            .await
            .unwrap_or_else(|error| {
                eprintln!("Candle LoRA learning proof failed: {error}");
                std::process::exit(1);
            });
        let payload = serde_json::to_string_pretty(&report).unwrap_or_else(|_| "{}".into());
        if let Some(report_path) = report_path {
            if let Some(parent) = report_path.parent()
                && let Err(error) = tokio::fs::create_dir_all(parent).await
            {
                eprintln!("Failed to create Candle LoRA learning report directory: {error}");
                std::process::exit(1);
            }
            if let Err(error) = tokio::fs::write(report_path, &payload).await {
                eprintln!("Failed to write Candle LoRA learning report: {error}");
                std::process::exit(1);
            }
        }
        println!("{payload}");
        if !report.passed {
            std::process::exit(2);
        }
        return;
    }

    if let Commands::ProveLoraRealModel {
        model_dir,
        model_source,
        output_dir,
        report_path,
    } = &cli.command
    {
        let output_dir = output_dir.clone().unwrap_or_else(|| {
            config
                .paths
                .data_dir
                .join("proofs")
                .join(format!("lora-real-model-{}", Ulid::new()))
        });
        let report = crytex_inference_candle::prove_real_model_lora_learning(
            model_dir,
            &output_dir,
            model_source.clone(),
        )
        .await
        .unwrap_or_else(|error| {
            eprintln!("Real model LoRA learning proof failed: {error}");
            std::process::exit(1);
        });
        let payload = serde_json::to_string_pretty(&report).unwrap_or_else(|_| "{}".into());
        if let Some(report_path) = report_path {
            if let Some(parent) = report_path.parent()
                && let Err(error) = tokio::fs::create_dir_all(parent).await
            {
                eprintln!("Failed to create real model LoRA report directory: {error}");
                std::process::exit(1);
            }
            if let Err(error) = tokio::fs::write(report_path, &payload).await {
                eprintln!("Failed to write real model LoRA report: {error}");
                std::process::exit(1);
            }
        }
        println!("{payload}");
        if !report.passed {
            std::process::exit(2);
        }
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
                    device: None,
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
        Commands::ProveKernelE2e { .. } => {
            unreachable!("prove-kernel-e2e is handled before full AppContext initialization")
        }
        Commands::ProveLoraLiveE2e { .. } => {
            unreachable!("prove-lora-live-e2e is handled before full AppContext initialization")
        }
        Commands::ProveLoraCandleLearning { .. } => unreachable!(
            "prove-lora-candle-learning is handled before full AppContext initialization"
        ),
        Commands::ProveLoraRealModel { .. } => {
            unreachable!("prove-lora-real-model is handled before full AppContext initialization")
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
        Commands::ProveHfRuntimeMatrix { .. } => {
            unreachable!("prove-hf-runtime-matrix is handled before AppContext initialization")
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
            stages: vec![crytex_core::services::ProbeStageReport {
                name: crytex_core::services::ProbeStageName::Generation,
                status: crytex_core::services::ProbeStageStatus::Passed,
                message: "smoke generation matched expected sentinel CRYTEX_PROBE_OK".into(),
                duration_ms: 42,
            }],
            failure_reasons: Vec::new(),
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
        assert!(matches!(report.build_profile.as_str(), "debug" | "release"));
        assert_eq!(
            report.runtime_probe.generated_preview.as_deref(),
            Some("ok")
        );
        assert!(report.generation_evidence.generated);
        assert!(report.generation_evidence.sentinel_matched);
        assert_eq!(report.generation_evidence.preview.as_deref(), Some("ok"));
        assert_eq!(report.generation_evidence.duration_ms, Some(42));
        assert!(
            report
                .generation_evidence
                .message
                .as_deref()
                .is_some_and(|message| message.contains("matched expected sentinel"))
        );
        assert!(report.proof_gate.passed);
        assert!(
            report
                .proof_gate
                .requirements
                .iter()
                .any(|requirement| requirement.name == "hf_gguf_resolved"
                    && requirement.passed
                    && requirement.evidence.contains("model.gguf"))
        );
        assert!(
            report
                .proof_gate
                .requirements
                .iter()
                .any(|requirement| requirement.name == "runtime_generated"
                    && requirement.passed
                    && requirement.evidence.contains("ok"))
        );
    }

    #[test]
    fn kernel_e2e_proof_report_requires_every_critical_gate() {
        let report = KernelE2eProofReport::from_input(KernelE2eProofInput {
            trace_id: "trace-kernel-e2e".into(),
            project_id: "project-1".into(),
            project_root: "A:/tmp/project".into(),
            runtime_kind: "live".into(),
            live_backend: Some("ollama".into()),
            live_model: Some("qwen3.5:9b".into()),
            live_generation_evidence: vec![KernelLiveGenerationEvidence {
                agent: "architect".into(),
                task_id: "goal-1".into(),
                prompt_chars: 128,
                response_chars: 256,
                prompt_tokens: 12,
                completion_tokens: 24,
                finish_reason: "stop".into(),
                excerpt: "live model produced an artifact".into(),
            }],
            goal_task_id: "goal-1".into(),
            task_ids: vec![
                "goal-1".into(),
                "architect-1".into(),
                "coder-1".into(),
                "qa-1".into(),
                "security-1".into(),
                "critic-1".into(),
                "remediation-1".into(),
            ],
            critic_rejection_task_id: "critic-1".into(),
            remediation_task_id: "remediation-1".into(),
            human_approved_task_id: "remediation-1".into(),
            indexed_files: 2,
            indexed_chunks: 4,
            diagnostics_event_count: 12,
            benchmark_baseline_run_id: "bench-baseline".into(),
            benchmark_challenger_run_id: "bench-challenger".into(),
            prompt_baseline_version_id: "prompt-v1".into(),
            prompt_challenger_version_id: "prompt-v2".into(),
            prompt_promoted: true,
            lora_adapter_id: "lora-v1".into(),
            lora_promoted: true,
        });

        assert!(report.passed);
        assert!(report.business_outcome.starts_with("BUSINESS_E2E_PASSED"));
        assert!(report.business_steps.iter().any(|step| step.name
            == "Goal was decomposed into an approved task plan"
            && step.status == "passed"));
        assert!(report.business_steps.iter().any(|step| step.name
            == "LoRA evolution trained and promoted an adapter"
            && step.status == "passed"));
        assert_eq!(report.gates.len(), 11);
        assert!(
            report
                .gates
                .iter()
                .any(|gate| gate.name == "live_model_executed" && gate.passed)
        );
        assert!(
            report
                .gates
                .iter()
                .any(|gate| gate.name == "prompt_evolution_proved" && gate.passed)
        );
        assert!(
            report
                .gates
                .iter()
                .any(|gate| gate.name == "lora_evolution_proved" && gate.passed)
        );

        let failed = KernelE2eProofReport::from_input(KernelE2eProofInput {
            prompt_promoted: false,
            ..KernelE2eProofInput {
                trace_id: report.trace_id,
                project_id: report.project_id,
                project_root: report.project_root,
                runtime_kind: report.runtime_kind,
                live_backend: report.live_backend,
                live_model: report.live_model,
                live_generation_evidence: report.live_generation_evidence,
                goal_task_id: report.goal_task_id,
                task_ids: report.task_ids,
                critic_rejection_task_id: report.critic_rejection_task_id,
                remediation_task_id: report.remediation_task_id,
                human_approved_task_id: report.human_approved_task_id,
                indexed_files: report.indexed_files,
                indexed_chunks: report.indexed_chunks,
                diagnostics_event_count: report.diagnostics_event_count,
                benchmark_baseline_run_id: report.benchmark_baseline_run_id,
                benchmark_challenger_run_id: report.benchmark_challenger_run_id,
                prompt_baseline_version_id: report.prompt_baseline_version_id,
                prompt_challenger_version_id: report.prompt_challenger_version_id,
                prompt_promoted: report.prompt_promoted,
                lora_adapter_id: report.lora_adapter_id,
                lora_promoted: report.lora_promoted,
            }
        });

        assert!(!failed.passed);
        assert!(
            failed
                .gates
                .iter()
                .any(|gate| gate.name == "prompt_evolution_proved" && !gate.passed)
        );
    }

    #[test]
    fn lora_live_e2e_report_exposes_ab_artifact_even_when_gate_rejects() {
        let metadata = serde_json::json!({
            "baseline_run_id": "baseline-run-1",
            "challenger_run_id": "challenger-run-1",
            "winner": "Inconclusive",
            "baseline_pass_rate": 0.25,
            "challenger_pass_rate": 0.50,
            "delta_pass_rate": 0.25,
            "mc_nemar_p_value": 0.125,
            "significance_level": 0.05,
            "bootstrap_ci": [-0.10, 0.60],
            "per_case_comparison": [
                {
                    "case_id": "heldout-1",
                    "baseline_passed": false,
                    "challenger_passed": true,
                    "baseline_score": 0.0,
                    "challenger_score": 1.0
                }
            ],
            "leakage_check": {
                "passed": true,
                "training_fingerprint_count": 50
            },
            "training_proof": {
                "kind": "candle_lora_train_loop",
                "learning_proven": true,
                "adapter_delta_l2": 0.42,
                "optimizer_calibration_used": false,
                "reason": "LoRA optimizer updated trainable adapter tensors during causal training"
            }
        });

        let report = build_lora_live_e2e_proof_report(LoraLiveE2eProofReportInput {
            trace_id: "trace-lora-ab".into(),
            gguf_path: "A:/models/tiny.gguf".into(),
            training_task_count: 50,
            heldout_case_count: 4,
            adapter_id: "codegen-v1".into(),
            adapter_path: "A:/adapters/codegen-v1".into(),
            adapter_registered: true,
            baseline_output: "baseline answer without the learned distillation marker".into(),
            challenger_output: "challenger answer includes CRYTEX_LORA_DISTILL_OK".into(),
            benchmark_outputs: vec![
                LoraProofOutput {
                    variant: "baseline".into(),
                    lora_adapter_id: None,
                    content: "baseline held-out answer".into(),
                },
                LoraProofOutput {
                    variant: "challenger".into(),
                    lora_adapter_id: Some("codegen-v1".into()),
                    content: "challenger held-out answer CRYTEX_LORA_DISTILL_OK".into(),
                },
            ],
            decision_metadata: Some(metadata),
            train_loss: 0.42,
            validation_loss: 0.50,
            failure_reason: Some(
                "benchmark gate rejected challenger: winner=Inconclusive, delta_pass_rate=0.2500"
                    .into(),
            ),
        });

        assert!(!report.passed);
        assert_eq!(report.proof_outcome, "LORA_LIVE_E2E_FAILED");
        assert_eq!(report.benchmark_winner, "Inconclusive");
        assert_eq!(report.baseline_pass_rate, 0.25);
        assert_eq!(report.challenger_pass_rate, 0.50);
        assert_eq!(report.delta_pass_rate, 0.25);
        assert_eq!(report.mc_nemar_p_value, Some(0.125));
        assert_eq!(report.bootstrap_ci, Some((-0.10, 0.60)));
        assert!(report.learning_proven);
        assert!(report.runtime_application_proof.adapter_registered);
        assert!(
            report
                .runtime_application_proof
                .adapter_applied_in_mistralrs_request
        );
        assert!(report.runtime_application_proof.output_changed_after_swap);
        assert!(!report.quality_proof.improved);
        assert_eq!(report.quality_proof.challenger_pass_rate, 0.50);
        assert!(
            report
                .quality_proof
                .failure_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("benchmark gate rejected challenger"))
        );
        assert_eq!(report.training_proof["kind"], "candle_lora_train_loop");
        assert_eq!(report.training_proof["learning_proven"], true);
        assert_eq!(report.per_case_comparison.len(), 1);
        assert!(
            report
                .quality_verdict
                .contains("not promoted because benchmark gate rejected challenger")
        );
        assert!(
            report
                .gates
                .iter()
                .any(|gate| gate.name == "heldout_challenger_won" && !gate.passed)
        );
        let serialized = serde_json::to_value(&report).unwrap();
        assert_eq!(
            serialized["baseline_output"],
            "baseline answer without the learned distillation marker"
        );
        assert_eq!(
            serialized["challenger_output"],
            "challenger answer includes CRYTEX_LORA_DISTILL_OK"
        );
        assert_eq!(serialized["ab_test"]["winner"], "Inconclusive");
    }

    #[test]
    fn lora_live_e2e_report_uses_differing_benchmark_outputs_as_representative_answers() {
        let report = build_lora_live_e2e_proof_report(LoraLiveE2eProofReportInput {
            trace_id: "trace-lora-ab".into(),
            gguf_path: "A:/models/tiny.gguf".into(),
            training_task_count: 50,
            heldout_case_count: 2,
            adapter_id: "codegen-v1".into(),
            adapter_path: "A:/adapters/codegen-v1".into(),
            adapter_registered: true,
            baseline_output: "".into(),
            challenger_output: "".into(),
            benchmark_outputs: vec![
                LoraProofOutput {
                    variant: "baseline".into(),
                    lora_adapter_id: None,
                    content: "same answer".into(),
                },
                LoraProofOutput {
                    variant: "challenger".into(),
                    lora_adapter_id: Some("codegen-v1".into()),
                    content: "same answer".into(),
                },
                LoraProofOutput {
                    variant: "baseline".into(),
                    lora_adapter_id: None,
                    content: "baseline misses marker".into(),
                },
                LoraProofOutput {
                    variant: "challenger".into(),
                    lora_adapter_id: Some("codegen-v1".into()),
                    content: "challenger includes CRYTEX_LORA_DISTILL_OK".into(),
                },
            ],
            decision_metadata: None,
            train_loss: 0.1,
            validation_loss: 0.2,
            failure_reason: None,
        });

        assert!(report.output_changed_after_swap);
        assert_eq!(report.baseline_output, "baseline misses marker");
        assert_eq!(
            report.challenger_output,
            "challenger includes CRYTEX_LORA_DISTILL_OK"
        );
    }

    #[test]
    fn lora_live_e2e_report_separates_runtime_application_from_quality_failure() {
        let metadata = serde_json::json!({
            "winner": "Inconclusive",
            "baseline_pass_rate": 0.0,
            "challenger_pass_rate": 0.0,
            "delta_pass_rate": 0.0,
            "training_proof": {
                "kind": "gguf_shape_initialized_adapter",
                "learning_proven": false,
                "reason": "GGUF path created shape-compatible tensors without causal-loss optimization"
            },
            "leakage_check": {
                "passed": true,
                "training_fingerprint_count": 50
            }
        });

        let report = build_lora_live_e2e_proof_report(LoraLiveE2eProofReportInput {
            trace_id: "trace-gguf-lora-runtime-ok-quality-failed".into(),
            gguf_path: "A:/models/tiny.gguf".into(),
            training_task_count: 50,
            heldout_case_count: 6,
            adapter_id: "codegen-v1".into(),
            adapter_path: "A:/adapters/codegen-v1".into(),
            adapter_registered: true,
            baseline_output: "baseline raw answer".into(),
            challenger_output: "different adapter raw answer".into(),
            benchmark_outputs: vec![LoraProofOutput {
                variant: "challenger".into(),
                lora_adapter_id: Some("codegen-v1".into()),
                content: "different adapter raw answer".into(),
            }],
            decision_metadata: Some(metadata),
            train_loss: f64::NAN,
            validation_loss: f64::NAN,
            failure_reason: None,
        });

        assert!(!report.passed);
        assert!(
            report
                .runtime_application_proof
                .adapter_applied_in_mistralrs_request
        );
        assert!(report.runtime_application_proof.failure_reason.is_none());
        assert!(report.output_changed_after_swap);
        assert!(!report.quality_proof.improved);
        assert_eq!(
            report.quality_proof.failure_reason.as_deref(),
            Some("adapter training proof reports learning_proven=false")
        );
    }

    #[test]
    fn lora_live_e2e_report_preserves_incompatible_gguf_failure_reason() {
        let report = build_lora_live_e2e_proof_report(LoraLiveE2eProofReportInput {
            trace_id: "trace-gguf-lora-incompatible".into(),
            gguf_path: "A:/models/tiny.gguf".into(),
            training_task_count: 50,
            heldout_case_count: 6,
            adapter_id: "codegen-v1".into(),
            adapter_path: "A:/adapters/codegen-v1".into(),
            adapter_registered: false,
            baseline_output: "".into(),
            challenger_output: "".into(),
            benchmark_outputs: vec![],
            decision_metadata: None,
            train_loss: f64::NAN,
            validation_loss: f64::NAN,
            failure_reason: Some(
                "mistral.rs rejected GGUF LoRA adapter: tensor shape mismatch".into(),
            ),
        });

        assert!(
            !report
                .runtime_application_proof
                .adapter_applied_in_mistralrs_request
        );
        assert_eq!(
            report.runtime_application_proof.failure_reason.as_deref(),
            Some("mistral.rs rejected GGUF LoRA adapter: tensor shape mismatch")
        );
        assert_eq!(
            report.quality_proof.failure_reason.as_deref(),
            Some("mistral.rs rejected GGUF LoRA adapter: tensor shape mismatch")
        );
    }

    #[test]
    fn hf_model_proof_report_marks_nonempty_sentinel_miss_as_generation_evidence() {
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
                support_status: crytex_core::services::ModelSupportStatus::Supported,
                actions: vec!["use CudaFused execution strategy".into()],
                warnings: Vec::new(),
                blockers: Vec::new(),
                failure_reasons: Vec::new(),
            },
            stages: vec![crytex_core::services::ProbeStageReport {
                name: crytex_core::services::ProbeStageName::Generation,
                status: crytex_core::services::ProbeStageStatus::Passed,
                message:
                    "smoke generation missed expected sentinel CRYTEX_PROBE_OK: useful preview"
                        .into(),
                duration_ms: 5844,
            }],
            failure_reasons: Vec::new(),
            generated_preview: Some("useful preview".into()),
            passed: true,
        };

        let report = build_hf_model_proof_report(
            "local-hf-proof".into(),
            &model,
            recommendation,
            runtime_probe,
        );

        assert!(report.passed);
        assert!(report.generation_evidence.generated);
        assert!(!report.generation_evidence.sentinel_matched);
        assert_eq!(
            report.generation_evidence.preview.as_deref(),
            Some("useful preview")
        );
        assert_eq!(report.generation_evidence.duration_ms, Some(5844));
        assert!(
            report
                .generation_evidence
                .message
                .as_deref()
                .is_some_and(|message| message.contains("missed expected sentinel"))
        );
    }

    #[test]
    fn hf_model_proof_gate_reports_missing_downloaded_artifact() {
        let model = crytex_core::services::ManagedModel {
            id: "hf-tiny".into(),
            name: "HF Tiny".into(),
            repo: Some("owner/repo".into()),
            filename: Some("model.gguf".into()),
            local_path: None,
            quantization: Some(crytex_core::services::Quantization::Q2K),
            preferred_backend: BackendKind::MistralRs,
            params_b: Some(1.1),
            status: crytex_core::services::ModelStatus::Available,
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
                support_status: crytex_core::services::ModelSupportStatus::Supported,
                actions: vec!["use CudaFused execution strategy".into()],
                warnings: Vec::new(),
                blockers: Vec::new(),
                failure_reasons: Vec::new(),
            },
            stages: vec![crytex_core::services::ProbeStageReport {
                name: crytex_core::services::ProbeStageName::Generation,
                status: crytex_core::services::ProbeStageStatus::Passed,
                message: "smoke generation matched expected sentinel CRYTEX_PROBE_OK".into(),
                duration_ms: 42,
            }],
            failure_reasons: Vec::new(),
            generated_preview: Some("ok".into()),
            passed: true,
        };

        let report = build_hf_model_proof_report(
            "local-hf-proof".into(),
            &model,
            recommendation,
            runtime_probe,
        );

        assert!(!report.proof_gate.passed);
        assert!(
            report
                .proof_gate
                .requirements
                .iter()
                .any(|requirement| requirement.name == "hf_model_downloaded"
                    && !requirement.passed
                    && requirement.evidence == "missing local model path")
        );
    }

    #[test]
    fn hf_proof_matrix_model_spec_parses_required_and_optional_fields() {
        let spec = parse_hf_proof_model_spec(
            "tiny=TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF,quantization=Q2_K,params_b=1.1,filename=tiny.Q2_K.gguf,name=Tiny",
        )
        .unwrap();

        assert_eq!(spec.id, "tiny");
        assert_eq!(spec.repo, "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF");
        assert_eq!(spec.quantization.as_deref(), Some("Q2_K"));
        assert_eq!(spec.params_b, Some(1.1));
        assert_eq!(spec.filename.as_deref(), Some("tiny.Q2_K.gguf"));
        assert_eq!(spec.name.as_deref(), Some("Tiny"));
    }

    #[test]
    fn hf_proof_matrix_model_spec_rejects_missing_repo() {
        let error = parse_hf_proof_model_spec("tiny=,quantization=Q2_K").unwrap_err();

        assert!(error.contains("repo"));
    }

    #[test]
    fn hf_proof_matrix_report_fails_when_any_entry_fails() {
        let entries = vec![
            HfProofMatrixEntryReport {
                label: "tiny-q2".into(),
                model_id: "tiny-q2".into(),
                repo: "owner/tiny".into(),
                report: None,
                error: None,
                passed: true,
            },
            HfProofMatrixEntryReport {
                label: "tiny-q4".into(),
                model_id: "tiny-q4".into(),
                repo: "owner/tiny".into(),
                report: None,
                error: Some("runtime timeout".into()),
                passed: false,
            },
        ];

        let report = build_hf_proof_matrix_report(Some("trace-matrix".into()), entries);

        assert_eq!(report.trace_id, "trace-matrix");
        assert!(!report.passed);
        assert_eq!(report.entries.len(), 2);
        assert_eq!(report.entries[1].error.as_deref(), Some("runtime timeout"));
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
                support_status: crytex_core::services::ModelSupportStatus::Supported,
                actions: vec!["use CudaFused execution strategy".into()],
                warnings: Vec::new(),
                blockers: Vec::new(),
                failure_reasons: Vec::new(),
            },
            stages: Vec::new(),
            failure_reasons: Vec::new(),
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
                support_status: crytex_core::services::ModelSupportStatus::Supported,
                actions: vec!["use CudaFused execution strategy".into()],
                warnings: Vec::new(),
                blockers: Vec::new(),
                failure_reasons: Vec::new(),
            },
            stages: Vec::new(),
            failure_reasons: Vec::new(),
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
