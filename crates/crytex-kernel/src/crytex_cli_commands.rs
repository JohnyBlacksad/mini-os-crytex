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
pub enum LoraCommands {
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
}
