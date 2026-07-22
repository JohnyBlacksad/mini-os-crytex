use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "crytex-kernel")]
#[command(about = "Crytex autonomous coding kernel")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
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
    /// Manage configured, downloaded, active, and probed models
    Models {
        #[command(subcommand)]
        command: ModelCommands,
    },
    /// Runtime diagnostics and preflight probes
    Diag {
        #[command(subcommand)]
        command: DiagCommands,
    },
    /// Inspect and prove sandbox isolation policy
    Sandbox {
        #[command(subcommand)]
        command: SandboxCommands,
    },
    /// Prove security controls against malicious project inputs
    Security {
        #[command(subcommand)]
        command: SecurityCommands,
    },
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
    #[command(
        alias = "prove-business-e2e",
        alias = "business-test",
        alias = "canonical-backend-acceptance"
    )]
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
    /// Run the canonical backend acceptance harness and emit one JSON proof artifact
    BackendAcceptance {
        #[arg(long)]
        full: bool,
        #[arg(long)]
        json: bool,
        #[arg(long)]
        deterministic: bool,
        #[arg(long, value_enum, default_value_t = AcceptanceRuntimeMode::Deterministic)]
        runtime: AcceptanceRuntimeMode,
        #[arg(long)]
        path: Option<PathBuf>,
        #[arg(long, default_value = "Backend Acceptance")]
        name: String,
        #[arg(long, default_value = "Prove Crytex backend CLI acceptance path")]
        goal: String,
        #[arg(long, default_value = "qwen3.5:9b")]
        live_model: String,
        #[arg(long, default_value = "http://localhost:11434")]
        live_url: String,
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
    /// Prove the full LoRA evolution loop from approved/rejected tasks to promote and rollback
    ProveLoraEvolutionLoop {
        #[arg(long)]
        gguf_path: Option<PathBuf>,
        #[arg(long, default_value = "64")]
        context_size: usize,
        #[arg(long)]
        gpu_layers: Option<usize>,
        #[arg(long, default_value = "50")]
        approved_tasks: usize,
        #[arg(long, default_value = "10")]
        rejected_tasks: usize,
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
        #[arg(long, default_value = "0.10")]
        min_improvement_delta: f64,
        #[arg(long, default_value = "1.5")]
        max_overfit_gap: f64,
        #[arg(long, default_value = "180")]
        train_timeout_secs: u64,
        #[arg(long, default_value = "45")]
        generation_timeout_secs: u64,
        #[arg(long)]
        report_path: Option<PathBuf>,
    },
    /// Prove LoRA active-adapter hot-swap without reloading the already loaded GGUF model
    ProveLoraHotSwap {
        #[arg(long)]
        gguf_path: Option<PathBuf>,
        #[arg(long)]
        adapter_a_path: PathBuf,
        #[arg(long)]
        adapter_b_path: PathBuf,
        #[arg(long, default_value = "adapter-a")]
        adapter_a_id: String,
        #[arg(long, default_value = "adapter-b")]
        adapter_b_id: String,
        #[arg(long, default_value = "64")]
        context_size: usize,
        #[arg(long)]
        gpu_layers: Option<usize>,
        #[arg(long, default_value = "8")]
        max_tokens: usize,
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
    /// Prove a stable LoRA quality gate acceptance artifact with corpus/leakage/overfit decisions
    ProveLoraRealQualityGate {
        #[arg(long)]
        model_dir: Option<PathBuf>,
        #[arg(long, default_value = "stable-candle-quality-gate")]
        model_source: String,
        #[arg(long)]
        output_dir: Option<PathBuf>,
        #[arg(long, default_value = "0.0001")]
        min_heldout_score_delta: f64,
        #[arg(long, default_value = "1.5")]
        max_overfit_gap: f64,
        #[arg(long)]
        report_path: Option<PathBuf>,
    },
    /// Prove role-based LoRA adapter routing through real agent swarm sessions
    ProveAgentSwarmLoraRouting {
        #[arg(long, default_value = "coder-lora-v1")]
        coder_adapter_id: String,
        #[arg(long, default_value = "critic-lora-v1")]
        critic_adapter_id: String,
        #[arg(long)]
        report_path: Option<PathBuf>,
    },
    /// Prove orchestrator atomic decomposition, dependencies, criteria, and remediation gates
    ProveOrchestratorQualityGate {
        #[arg(long)]
        report_path: Option<PathBuf>,
    },
    /// Prove mixed-project RAG indexing, hybrid retrieval, rerank, and context evidence
    ProveRagFull {
        #[arg(long)]
        report_path: Option<PathBuf>,
    },
    /// Prove backend Kanban projection, history, and task movement diagnostics
    ProveKanbanProjection {
        #[arg(long)]
        report_path: Option<PathBuf>,
    },
    /// Prove token headroom, shared context, CCR offload, and required-fact preservation
    ProveTokenEconomy {
        #[arg(long, default_value = "ollama")]
        backend: String,
        #[arg(long, default_value = "qwen3.5:9b")]
        model: String,
        #[arg(long, default_value = "32768")]
        context_window: usize,
        #[arg(long, default_value = "512")]
        expected_completion_tokens: usize,
        #[arg(long)]
        report_path: Option<PathBuf>,
    },
    /// Prove role quality contracts, role smoke fixtures, critic feedback, and role LoRA swaps
    ProveRoleQualityContracts {
        #[arg(long)]
        report_path: Option<PathBuf>,
    },
    /// Prove autonomous prompt evolution with challenger, regression gate, diagnostics, and rollback
    ProvePromptEvolution {
        #[arg(long)]
        report_path: Option<PathBuf>,
    },
    /// Prove LoRA positive/negative dataset construction, filtering, balancing, and leakage checks
    ProveLoraDataset {
        #[arg(long)]
        report_path: Option<PathBuf>,
    },
    /// Prove objective-aware LoRA training contracts, metadata, state, and artifact validation
    ProveLoraTrainingObjectives {
        #[arg(long)]
        report_path: Option<PathBuf>,
    },
    /// Prove LoRA promotion only after quality, safety, runtime, and rollback gates pass
    ProveLoraQualityGate {
        #[arg(long)]
        report_path: Option<PathBuf>,
    },
    /// Prove autonomous evolution policy routes failures to the right module
    ProveEvolutionPolicy {
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
    /// Show, watch, and inspect backend Kanban task projections
    Kanban {
        #[command(subcommand)]
        command: KanbanCommands,
    },
    /// Search or prove the project RAG brain.
    Rag {
        #[command(subcommand)]
        command: RagCommands,
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
    /// Manage prompt versions through benchmark-gated evolution
    Prompts {
        #[command(subcommand)]
        command: PromptCommands,
    },
    /// Run autonomous evolution policy routing
    Evolution {
        #[command(subcommand)]
        command: EvolutionCommands,
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
pub enum LoraCommands {
    /// Build, inspect, and summarize role-specific LoRA datasets
    Dataset {
        #[command(subcommand)]
        command: LoraDatasetCommands,
    },
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
    /// Train a new adapter for a task kind or role objective
    Train {
        kind: String,
        #[arg(long, value_enum, default_value_t = LoraObjectiveArg::Sft)]
        objective: LoraObjectiveArg,
        #[arg(long)]
        role: Option<String>,
    },
    /// Bind an adapter to a canonical agent role
    SelectRole { role: String, adapter: String },
    /// List role -> adapter bindings
    ListRoles,
}

#[derive(Subcommand)]
pub enum LoraDatasetCommands {
    /// Build or refresh a role-specific positive/negative dataset.
    Build {
        role: String,
        #[arg(long)]
        preference: bool,
        #[arg(long)]
        json: bool,
    },
    /// Inspect role-specific dataset rows and diagnostics.
    Inspect {
        role: String,
        #[arg(long)]
        json: bool,
    },
    /// Show dataset stats, balancing, leakage, and low-information filtering.
    Stats {
        role: String,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum LoraObjectiveArg {
    Sft,
    Dpo,
    Orpo,
    Kto,
}

#[derive(Subcommand)]
pub enum PromptCommands {
    /// Show active/challenger prompt status for an agent.
    Status {
        #[arg(short, long)]
        agent: String,
        #[arg(long)]
        json: bool,
    },
    /// Create an inactive challenger prompt from the active baseline.
    Propose {
        #[arg(short, long)]
        agent: String,
        #[arg(short, long, value_enum, default_value_t = PromptMutationOperatorArg::Rephrase)]
        operator: PromptMutationOperatorArg,
        #[arg(long)]
        json: bool,
    },
    /// Run the benchmark gate for a challenger. A regression suite is mandatory.
    Benchmark {
        #[arg(short, long)]
        agent: String,
        #[arg(long)]
        challenger: String,
        #[arg(long)]
        regression_suite: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Activate a prompt only when it already has an accepted benchmark decision.
    Promote {
        #[arg(short, long)]
        agent: String,
        #[arg(long)]
        version: String,
        #[arg(long)]
        json: bool,
    },
    /// Roll back an agent to an earlier prompt version.
    Rollback {
        #[arg(short, long)]
        agent: String,
        #[arg(long)]
        to: String,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum PromptMutationOperatorArg {
    Rephrase,
    AddConstraint,
    InjectExample,
    ChangeTone,
}

#[derive(Subcommand)]
pub enum EvolutionCommands {
    /// Attribute failures and route improvements.
    Run {
        #[arg(long)]
        all_roles: bool,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
pub enum ModelCommands {
    /// List backend inventory or managed model registry.
    List {
        #[arg(short, long)]
        backend: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Add or update a managed HuggingFace/local model entry.
    Add {
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
    /// Download a managed model from its configured source.
    Download {
        #[arg(short, long)]
        id: String,
        #[arg(long)]
        activate: bool,
        #[arg(long, default_value = "local-hf")]
        backend_id: String,
    },
    /// Activate a downloaded model as an inference backend.
    Activate {
        #[arg(short, long)]
        id: String,
        #[arg(long, default_value = "local")]
        backend_id: String,
    },
    /// Prove metadata, compatibility, and smoke generation for a managed model.
    Prove {
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
        #[arg(long)]
        report_path: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
pub enum DiagCommands {
    /// Probe and explain backend/model support matrix.
    ProbeRuntimeMatrix {
        #[arg(long)]
        json: bool,
        #[arg(long)]
        report_path: Option<PathBuf>,
    },
    /// Prove storage migrations, backup/export/import, resume, locks, and adapter recovery.
    StorageRecovery {
        #[arg(long)]
        json: bool,
        #[arg(long)]
        report_path: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
pub enum SandboxCommands {
    /// Report Docker/WASI/host sandbox backend availability and isolation posture.
    Doctor {
        #[arg(long)]
        json: bool,
    },
    /// Prove sandbox permissions, path isolation, and audit coverage.
    Prove {
        #[arg(long)]
        json: bool,
        #[arg(long)]
        report_path: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
pub enum SecurityCommands {
    /// Prove malicious RAG and tool-use inputs are blocked and routed to learning signals.
    Prove {
        #[arg(long)]
        malicious_rag_fixture: bool,
        #[arg(long)]
        json: bool,
        #[arg(long)]
        report_path: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
pub enum BenchCommands {
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
pub enum ABTestCommands {
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

#[derive(Subcommand)]
pub enum KanbanCommands {
    /// Show the canonical backend Kanban projection.
    Show {
        #[arg(long)]
        project_id: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Stream backend task movement diagnostics as they happen.
    Watch {
        #[arg(long)]
        project_id: Option<String>,
        #[arg(long)]
        json: bool,
        #[arg(long, default_value = "30")]
        duration_seconds: u64,
    },
    /// Show Kanban movement history for a run.
    History {
        #[arg(long)]
        project_id: Option<String>,
        #[arg(long, default_value = "latest")]
        run: String,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
pub enum RagCommands {
    /// Search indexed project context and explain dense/sparse/fusion/rerank/selection decisions.
    Search {
        query: String,
        #[arg(short, long)]
        project_id: String,
        #[arg(long)]
        path: Option<PathBuf>,
        #[arg(long)]
        rerank: bool,
        #[arg(long)]
        explain: bool,
        #[arg(long)]
        json: bool,
        #[arg(long)]
        diagnostics_path: Option<PathBuf>,
        #[arg(long, default_value = "8")]
        top_k: usize,
        #[arg(long, default_value = "2048")]
        token_budget: usize,
    },
    /// Build a mixed fixture and prove end-to-end RAG retrieval evidence.
    Prove {
        #[arg(long, default_value = "mixed-docs-code")]
        fixture: String,
        #[arg(long)]
        report_path: Option<PathBuf>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum AcceptanceRuntimeMode {
    Deterministic,
    Ollama,
    Mistral,
}

impl AcceptanceRuntimeMode {
    pub fn backend_id(self) -> &'static str {
        match self {
            Self::Deterministic => "deterministic",
            Self::Ollama => "ollama",
            Self::Mistral => "mistral",
        }
    }

    pub fn is_deterministic(self, explicit_deterministic: bool) -> bool {
        explicit_deterministic || self == Self::Deterministic
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn backend_acceptance_command_parses_full_json_deterministic_contract() {
        let cli = Cli::parse_from([
            "crytex-kernel",
            "backend-acceptance",
            "--full",
            "--json",
            "--deterministic",
        ]);

        let Commands::BackendAcceptance {
            full,
            json,
            deterministic,
            runtime,
            ..
        } = cli.command
        else {
            panic!("expected backend acceptance command");
        };

        assert!(full);
        assert!(json);
        assert!(deterministic);
        assert_eq!(runtime, AcceptanceRuntimeMode::Deterministic);
    }

    #[test]
    fn backend_acceptance_help_documents_runtime_profiles() {
        let mut command = Cli::command();
        let help = command
            .find_subcommand_mut("backend-acceptance")
            .expect("backend-acceptance command exists")
            .render_long_help()
            .to_string();

        assert!(help.contains("--full"));
        assert!(help.contains("--json"));
        assert!(help.contains("--runtime"));
        assert!(help.contains("deterministic"));
        assert!(help.contains("ollama"));
        assert!(help.contains("mistral"));
    }

    #[test]
    fn rag_search_command_parses_explainable_json_contract() {
        let cli = Cli::parse_from([
            "crytex-kernel",
            "rag",
            "search",
            "where is retry policy documented?",
            "--project-id",
            "proj-1",
            "--rerank",
            "--explain",
            "--json",
        ]);

        let Commands::Rag {
            command:
                RagCommands::Search {
                    query,
                    project_id,
                    rerank,
                    explain,
                    json,
                    ..
                },
        } = cli.command
        else {
            panic!("expected rag search command");
        };

        assert_eq!(query, "where is retry policy documented?");
        assert_eq!(project_id, "proj-1");
        assert!(rerank);
        assert!(explain);
        assert!(json);
    }

    #[test]
    fn rag_prove_command_parses_mixed_docs_code_fixture() {
        let cli = Cli::parse_from([
            "crytex-kernel",
            "rag",
            "prove",
            "--fixture",
            "mixed-docs-code",
        ]);

        let Commands::Rag {
            command: RagCommands::Prove { fixture, .. },
        } = cli.command
        else {
            panic!("expected rag prove command");
        };

        assert_eq!(fixture, "mixed-docs-code");
    }

    #[test]
    fn kanban_show_watch_and_history_parse_json_contract() {
        let show = Cli::parse_from(["crytex-kernel", "kanban", "show", "--json"]);
        assert!(matches!(
            show.command,
            Commands::Kanban {
                command: KanbanCommands::Show { json: true, .. }
            }
        ));

        let watch = Cli::parse_from([
            "crytex-kernel",
            "kanban",
            "watch",
            "--project-id",
            "project-1",
            "--json",
            "--duration-seconds",
            "1",
        ]);
        assert!(matches!(
            watch.command,
            Commands::Kanban {
                command: KanbanCommands::Watch {
                    json: true,
                    duration_seconds: 1,
                    ..
                }
            }
        ));

        let history = Cli::parse_from([
            "crytex-kernel",
            "kanban",
            "history",
            "--run",
            "latest",
            "--json",
        ]);
        assert!(matches!(
            history.command,
            Commands::Kanban {
                command: KanbanCommands::History {
                    run,
                    json: true,
                    ..
                }
            } if run == "latest"
        ));
    }

    #[test]
    fn token_economy_proof_command_parses_backend_model_and_report_path() {
        let cli = Cli::parse_from([
            "crytex-kernel",
            "prove-token-economy",
            "--backend",
            "ollama",
            "--model",
            "qwen3.5:9b",
            "--context-window",
            "32768",
            "--report-path",
            "reports/token-economy.json",
        ]);

        let Commands::ProveTokenEconomy {
            backend,
            model,
            context_window,
            report_path,
            ..
        } = cli.command
        else {
            panic!("expected token economy proof command");
        };

        assert_eq!(backend, "ollama");
        assert_eq!(model, "qwen3.5:9b");
        assert_eq!(context_window, 32768);
        assert_eq!(
            report_path,
            Some(PathBuf::from("reports/token-economy.json"))
        );
    }

    #[test]
    fn kanban_projection_proof_command_parses_report_path() {
        let cli = Cli::parse_from([
            "crytex-kernel",
            "prove-kanban-projection",
            "--report-path",
            "reports/kanban-p5.json",
        ]);

        let Commands::ProveKanbanProjection { report_path } = cli.command else {
            panic!("expected kanban projection proof command");
        };

        assert_eq!(report_path, Some(PathBuf::from("reports/kanban-p5.json")));
    }

    #[test]
    fn role_quality_contracts_proof_command_parses_report_path() {
        let cli = Cli::parse_from([
            "crytex-kernel",
            "prove-role-quality-contracts",
            "--report-path",
            "reports/role-quality-p6.json",
        ]);

        let Commands::ProveRoleQualityContracts { report_path } = cli.command else {
            panic!("expected role quality contracts proof command");
        };

        assert_eq!(
            report_path,
            Some(PathBuf::from("reports/role-quality-p6.json"))
        );
    }

    #[test]
    fn prompts_group_parses_status_propose_benchmark_promote_and_rollback() {
        let status = Cli::parse_from([
            "crytex-kernel",
            "prompts",
            "status",
            "--agent",
            "coder",
            "--json",
        ]);
        assert!(matches!(
            status.command,
            Commands::Prompts {
                command: PromptCommands::Status { json: true, .. }
            }
        ));

        let propose = Cli::parse_from([
            "crytex-kernel",
            "prompts",
            "propose",
            "--agent",
            "coder",
            "--operator",
            "inject-example",
            "--json",
        ]);
        assert!(matches!(
            propose.command,
            Commands::Prompts {
                command: PromptCommands::Propose {
                    operator: PromptMutationOperatorArg::InjectExample,
                    json: true,
                    ..
                }
            }
        ));

        let benchmark = Cli::parse_from([
            "crytex-kernel",
            "prompts",
            "benchmark",
            "--agent",
            "coder",
            "--challenger",
            "prompt-v2",
            "--regression-suite",
            "fixtures/prompt-regression.jsonl",
            "--json",
        ]);
        assert!(matches!(
            benchmark.command,
            Commands::Prompts {
                command: PromptCommands::Benchmark {
                    challenger,
                    regression_suite: Some(_),
                    json: true,
                    ..
                }
            } if challenger == "prompt-v2"
        ));

        let promote = Cli::parse_from([
            "crytex-kernel",
            "prompts",
            "promote",
            "--agent",
            "coder",
            "--version",
            "prompt-v2",
            "--json",
        ]);
        assert!(matches!(
            promote.command,
            Commands::Prompts {
                command: PromptCommands::Promote {
                    version,
                    json: true,
                    ..
                }
            } if version == "prompt-v2"
        ));

        let rollback = Cli::parse_from([
            "crytex-kernel",
            "prompts",
            "rollback",
            "--agent",
            "coder",
            "--to",
            "prompt-v1",
            "--json",
        ]);
        assert!(matches!(
            rollback.command,
            Commands::Prompts {
                command: PromptCommands::Rollback { to, json: true, .. }
            } if to == "prompt-v1"
        ));
    }

    #[test]
    fn prompt_evolution_proof_command_parses_report_path() {
        let cli = Cli::parse_from([
            "crytex-kernel",
            "prove-prompt-evolution",
            "--report-path",
            "reports/prompt-evolution-p7.json",
        ]);

        let Commands::ProvePromptEvolution { report_path } = cli.command else {
            panic!("expected prompt evolution proof command");
        };

        assert_eq!(
            report_path,
            Some(PathBuf::from("reports/prompt-evolution-p7.json"))
        );
    }

    #[test]
    fn lora_dataset_commands_parse_build_inspect_and_stats() {
        let build = Cli::parse_from([
            "crytex-kernel",
            "lora",
            "dataset",
            "build",
            "coder-python",
            "--preference",
            "--json",
        ]);
        assert!(matches!(
            build.command,
            Commands::Lora {
                command: LoraCommands::Dataset {
                    command: LoraDatasetCommands::Build {
                        role,
                        preference: true,
                        json: true
                    }
                }
            } if role == "coder-python"
        ));

        let inspect = Cli::parse_from([
            "crytex-kernel",
            "lora",
            "dataset",
            "inspect",
            "qa",
            "--json",
        ]);
        assert!(matches!(
            inspect.command,
            Commands::Lora {
                command: LoraCommands::Dataset {
                    command: LoraDatasetCommands::Inspect { role, json: true }
                }
            } if role == "qa"
        ));

        let stats = Cli::parse_from([
            "crytex-kernel",
            "lora",
            "dataset",
            "stats",
            "orchestrator",
            "--json",
        ]);
        assert!(matches!(
            stats.command,
            Commands::Lora {
                command: LoraCommands::Dataset {
                    command: LoraDatasetCommands::Stats { role, json: true }
                }
            } if role == "orchestrator"
        ));
    }

    #[test]
    fn lora_dataset_proof_command_parses_report_path() {
        let cli = Cli::parse_from([
            "crytex-kernel",
            "prove-lora-dataset",
            "--report-path",
            "reports/lora-dataset-p8.json",
        ]);

        let Commands::ProveLoraDataset { report_path } = cli.command else {
            panic!("expected lora dataset proof command");
        };

        assert_eq!(
            report_path,
            Some(PathBuf::from("reports/lora-dataset-p8.json"))
        );
    }

    #[test]
    fn lora_train_parses_typed_objective_and_role() {
        let cli = Cli::parse_from([
            "crytex-kernel",
            "lora",
            "train",
            "coder-python",
            "--objective",
            "dpo",
            "--role",
            "coder-python",
        ]);

        assert!(matches!(
            cli.command,
            Commands::Lora {
                command: LoraCommands::Train {
                    kind,
                    objective: LoraObjectiveArg::Dpo,
                    role: Some(role),
                }
            } if kind == "coder-python" && role == "coder-python"
        ));
    }

    #[test]
    fn lora_training_objectives_proof_command_parses_report_path() {
        let cli = Cli::parse_from([
            "crytex-kernel",
            "prove-lora-training-objectives",
            "--report-path",
            "reports/lora-training-objectives-p9.json",
        ]);

        let Commands::ProveLoraTrainingObjectives { report_path } = cli.command else {
            panic!("expected lora training objectives proof command");
        };

        assert_eq!(
            report_path,
            Some(PathBuf::from("reports/lora-training-objectives-p9.json"))
        );
    }

    #[test]
    fn lora_quality_gate_proof_command_parses_report_path() {
        let cli = Cli::parse_from([
            "crytex-kernel",
            "prove-lora-quality-gate",
            "--report-path",
            "reports/lora-quality-gate-p10.json",
        ]);

        let Commands::ProveLoraQualityGate { report_path } = cli.command else {
            panic!("expected lora quality gate proof command");
        };

        assert_eq!(
            report_path,
            Some(PathBuf::from("reports/lora-quality-gate-p10.json"))
        );
    }

    #[test]
    fn evolution_run_parses_all_roles_json_contract() {
        let cli = Cli::parse_from(["crytex-kernel", "evolution", "run", "--all-roles", "--json"]);

        assert!(matches!(
            cli.command,
            Commands::Evolution {
                command: EvolutionCommands::Run {
                    all_roles: true,
                    json: true
                }
            }
        ));
    }

    #[test]
    fn evolution_policy_proof_command_parses_report_path() {
        let cli = Cli::parse_from([
            "crytex-kernel",
            "prove-evolution-policy",
            "--report-path",
            "reports/evolution-policy-p11.json",
        ]);

        let Commands::ProveEvolutionPolicy { report_path } = cli.command else {
            panic!("expected evolution policy proof command");
        };

        assert_eq!(
            report_path,
            Some(PathBuf::from("reports/evolution-policy-p11.json"))
        );
    }

    #[test]
    fn models_group_parses_list_add_download_activate_and_prove_contract() {
        let list = Cli::parse_from(["crytex-kernel", "models", "list", "--json"]);
        assert!(matches!(
            list.command,
            Commands::Models {
                command: ModelCommands::List {
                    backend: None,
                    json: true
                }
            }
        ));

        let add = Cli::parse_from([
            "crytex-kernel",
            "models",
            "add",
            "--id",
            "qwen-local",
            "--repo",
            "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
            "--backend",
            "mistralrs",
        ]);
        assert!(matches!(
            add.command,
            Commands::Models {
                command: ModelCommands::Add {
                    id,
                    repo: Some(_),
                    backend,
                    ..
                }
            } if id == "qwen-local" && backend == "mistralrs"
        ));

        let download =
            Cli::parse_from(["crytex-kernel", "models", "download", "--id", "qwen-local"]);
        assert!(matches!(
            download.command,
            Commands::Models {
                command: ModelCommands::Download { id, activate: false, .. }
            } if id == "qwen-local"
        ));

        let activate = Cli::parse_from([
            "crytex-kernel",
            "models",
            "activate",
            "--id",
            "qwen-local",
            "--backend-id",
            "local",
        ]);
        assert!(matches!(
            activate.command,
            Commands::Models {
                command: ModelCommands::Activate { id, backend_id }
            } if id == "qwen-local" && backend_id == "local"
        ));

        let prove = Cli::parse_from([
            "crytex-kernel",
            "models",
            "prove",
            "--id",
            "qwen-local",
            "--report-path",
            "reports/model-runtime-p12.json",
        ]);
        assert!(matches!(
            prove.command,
            Commands::Models {
                command: ModelCommands::Prove {
                    id,
                    report_path: Some(_),
                    ..
                }
            } if id == "qwen-local"
        ));
    }

    #[test]
    fn diag_group_parses_probe_runtime_matrix_contract() {
        let cli = Cli::parse_from([
            "crytex-kernel",
            "diag",
            "probe-runtime-matrix",
            "--json",
            "--report-path",
            "reports/runtime-matrix-p12-proof.json",
        ]);

        assert!(matches!(
            cli.command,
            Commands::Diag {
                command: DiagCommands::ProbeRuntimeMatrix {
                    json: true,
                    report_path: Some(_)
                }
            }
        ));
    }

    #[test]
    fn diag_group_parses_storage_recovery_contract() {
        let cli = Cli::parse_from([
            "crytex-kernel",
            "diag",
            "storage-recovery",
            "--json",
            "--report-path",
            "reports/storage-recovery-p14-proof.json",
        ]);

        assert!(matches!(
            cli.command,
            Commands::Diag {
                command: DiagCommands::StorageRecovery {
                    json: true,
                    report_path: Some(_)
                }
            }
        ));
    }

    #[test]
    fn sandbox_group_parses_doctor_and_prove_contract() {
        let doctor = Cli::parse_from(["crytex-kernel", "sandbox", "doctor", "--json"]);
        assert!(matches!(
            doctor.command,
            Commands::Sandbox {
                command: SandboxCommands::Doctor { json: true }
            }
        ));

        let prove = Cli::parse_from([
            "crytex-kernel",
            "sandbox",
            "prove",
            "--json",
            "--report-path",
            "reports/sandbox-security-p13-proof.json",
        ]);
        assert!(matches!(
            prove.command,
            Commands::Sandbox {
                command: SandboxCommands::Prove {
                    json: true,
                    report_path: Some(_)
                }
            }
        ));
    }

    #[test]
    fn security_prove_parses_malicious_rag_fixture_contract() {
        let cli = Cli::parse_from([
            "crytex-kernel",
            "security",
            "prove",
            "--malicious-rag-fixture",
            "--json",
            "--report-path",
            "reports/security-p13-proof.json",
        ]);

        assert!(matches!(
            cli.command,
            Commands::Security {
                command: SecurityCommands::Prove {
                    malicious_rag_fixture: true,
                    json: true,
                    report_path: Some(_)
                }
            }
        ));
    }
}
