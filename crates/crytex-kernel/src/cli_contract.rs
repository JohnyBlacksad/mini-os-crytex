use clap::{Parser, Subcommand, ValueEnum};

/// Stable production CLI contract for the Crytex backend product.
///
/// The current kernel still contains legacy command handlers while this contract
/// is the source of truth for the product-facing command surface. Keeping it in
/// a small module makes the surface testable without booting inference, storage,
/// model runtimes, or CUDA.
#[derive(Debug, Parser)]
#[command(name = "crytex")]
#[command(about = "Autonomous self-improving agentic CLI for project work")]
#[command(version)]
pub struct ProductCli {
    /// Emit stable JSON to stdout. Human progress and logs go to stderr.
    #[arg(long, global = true)]
    pub json: bool,
    /// Correlate commands, diagnostics, traces, and reports.
    #[arg(long, global = true)]
    pub trace_id: Option<String>,
    /// Project id or path for project-scoped commands.
    #[arg(long, global = true)]
    pub project: Option<String>,
    #[command(subcommand)]
    pub command: ProductCommand,
}

#[derive(Debug, Subcommand)]
pub enum ProductCommand {
    /// Validate config, storage, runtimes, tools, sandbox, RAG, and evolution gates.
    Doctor(DoctorArgs),
    /// Create, open, list, inspect, and reopen projects.
    Project {
        #[command(subcommand)]
        command: ProjectCommand,
    },
    /// Index and inspect project knowledge.
    Index {
        #[command(subcommand)]
        command: IndexCommand,
    },
    /// Search and prove retrieval-augmented context with reranking evidence.
    Rag {
        #[command(subcommand)]
        command: RagCommand,
    },
    /// Plan token headroom, shared context, CCR offload, and quality gates.
    TokenEconomy {
        #[command(subcommand)]
        command: TokenEconomyCommand,
    },
    /// Submit and inspect user goals.
    Goal {
        #[command(subcommand)]
        command: GoalCommand,
    },
    /// Inspect, approve, and reject generated plans.
    Plan {
        #[command(subcommand)]
        command: PlanCommand,
    },
    /// Show the canonical task graph projection.
    Kanban {
        #[command(subcommand)]
        command: KanbanCommand,
    },
    /// Start, resume, cancel, and inspect autonomous runs.
    Run {
        #[command(subcommand)]
        command: RunCommand,
    },
    /// Inspect review gates and record automated or policy decisions.
    Review {
        #[command(subcommand)]
        command: ReviewCommand,
    },
    /// Export diagnostics, traces, events, and proof reports.
    Diag {
        #[command(subcommand)]
        command: DiagCommand,
    },
    /// Manage inference models and runtime compatibility.
    Models {
        #[command(subcommand)]
        command: ModelsCommand,
    },
    /// Manage prompt versions and prompt evolution.
    Prompts {
        #[command(subcommand)]
        command: PromptsCommand,
    },
    /// Manage role-specific LoRA datasets, training, benchmarks, and promotion.
    Lora {
        #[command(subcommand)]
        command: LoraCommand,
    },
    /// Run autonomous prompt/LoRA improvement loops.
    Evolution {
        #[command(subcommand)]
        command: EvolutionCommand,
    },
    /// Run and compare benchmark suites.
    Bench {
        #[command(subcommand)]
        command: BenchCommand,
    },
    /// Validate sandbox backends and policy boundaries.
    Sandbox {
        #[command(subcommand)]
        command: SandboxCommand,
    },
    /// Run canonical backend product acceptance.
    BackendAcceptance(BackendAcceptanceArgs),
    /// Run explicit proof-only gates and matrix probes.
    Prove {
        #[command(subcommand)]
        command: ProveCommand,
    },
}

#[derive(Debug, Clone, clap::Args)]
pub struct DoctorArgs {
    #[arg(long)]
    pub strict: bool,
    #[arg(long)]
    pub require_gpu: bool,
}

#[derive(Debug, Clone, clap::Args)]
pub struct BackendAcceptanceArgs {
    #[arg(long)]
    pub full: bool,
    #[arg(long)]
    pub deterministic: bool,
    #[arg(long)]
    pub include_runtime: Option<BackendKindArg>,
    #[arg(long)]
    pub report_path: Option<std::path::PathBuf>,
}

#[derive(Debug, Subcommand)]
pub enum ProjectCommand {
    Open {
        path: std::path::PathBuf,
    },
    Create {
        #[arg(long)]
        name: String,
        #[arg(long)]
        path: std::path::PathBuf,
    },
    List,
    Status,
    Reopen {
        id: String,
    },
}

#[derive(Debug, Subcommand)]
pub enum IndexCommand {
    Run { path: Option<std::path::PathBuf> },
    Status,
    Rebuild,
}

#[derive(Debug, Subcommand)]
pub enum RagCommand {
    Search {
        query: String,
        #[arg(long)]
        rerank: bool,
        #[arg(long)]
        explain: bool,
    },
    Prove {
        #[arg(long)]
        fixture: Option<std::path::PathBuf>,
    },
}

#[derive(Debug, Subcommand)]
pub enum TokenEconomyCommand {
    Plan {
        #[arg(long, value_enum)]
        backend: BackendKindArg,
        #[arg(long)]
        model: String,
        #[arg(long)]
        prompt_tokens: usize,
        #[arg(long)]
        completion_tokens: usize,
    },
    SharedContext {
        #[command(subcommand)]
        command: TokenEconomySharedContextCommand,
    },
}

#[derive(Debug, Subcommand)]
pub enum TokenEconomySharedContextCommand {
    Stats,
}

#[derive(Debug, Subcommand)]
pub enum GoalCommand {
    Submit { goal: String },
    Status { id: Option<String> },
    List,
}

#[derive(Debug, Subcommand)]
pub enum PlanCommand {
    Show {
        goal: Option<String>,
    },
    Approve {
        id: Option<String>,
    },
    Reject {
        id: Option<String>,
        #[arg(long)]
        comment: String,
    },
}

#[derive(Debug, Subcommand)]
pub enum KanbanCommand {
    Show {
        #[arg(long, value_enum)]
        status: Option<TaskStatusArg>,
        #[arg(long, value_enum)]
        role: Option<AgentRoleArg>,
    },
    Watch {
        #[arg(long, value_enum)]
        status: Option<TaskStatusArg>,
        #[arg(long, value_enum)]
        role: Option<AgentRoleArg>,
    },
    History {
        #[arg(long)]
        run: Option<String>,
        #[arg(long, value_enum)]
        status: Option<TaskStatusArg>,
    },
}

#[derive(Debug, Subcommand)]
pub enum RunCommand {
    Start { goal: Option<String> },
    Status { id: Option<String> },
    Resume { id: Option<String> },
    Cancel { id: String },
}

#[derive(Debug, Subcommand)]
pub enum ReviewCommand {
    Show {
        task: Option<String>,
    },
    Approve {
        task: String,
        score: Option<f64>,
    },
    Reject {
        task: String,
        #[arg(long)]
        comment: String,
        #[arg(long, value_enum)]
        failure_type: Option<FailureTypeArg>,
        #[arg(long)]
        score: Option<f64>,
    },
}

#[derive(Debug, Subcommand)]
pub enum DiagCommand {
    Export {
        #[arg(long, default_value = "latest")]
        run: String,
        #[arg(long)]
        out: Option<std::path::PathBuf>,
    },
    RuntimeMatrix {
        #[arg(long)]
        model: Option<String>,
    },
}

#[derive(Debug, Subcommand)]
pub enum ModelsCommand {
    List,
    Add {
        id: String,
        #[arg(long)]
        repo: Option<String>,
        #[arg(long)]
        filename: Option<String>,
        #[arg(long, value_enum)]
        backend: Option<BackendKindArg>,
    },
    Download {
        id: String,
    },
    Activate {
        id: String,
    },
    Prove {
        id: String,
    },
}

#[derive(Debug, Subcommand)]
pub enum PromptsCommand {
    Status { role: Option<AgentRoleArg> },
    Propose { role: AgentRoleArg },
    Benchmark { role: AgentRoleArg },
    Promote { role: AgentRoleArg, version: String },
    Rollback { role: AgentRoleArg },
}

#[derive(Debug, Subcommand)]
pub enum LoraCommand {
    Status {
        role: Option<AgentRoleArg>,
    },
    Dataset {
        #[command(subcommand)]
        command: LoraDatasetCommand,
    },
    Train {
        role: AgentRoleArg,
        #[arg(long, value_enum)]
        objective: TrainingObjectiveArg,
    },
    Benchmark {
        role: AgentRoleArg,
        #[arg(long)]
        include_negative: bool,
    },
    Promote {
        role: AgentRoleArg,
        adapter: String,
    },
    Rollback {
        role: AgentRoleArg,
    },
    ProveLive {
        role: AgentRoleArg,
    },
}

#[derive(Debug, Subcommand)]
pub enum LoraDatasetCommand {
    Build {
        role: AgentRoleArg,
        #[arg(long)]
        preference: bool,
    },
    Inspect {
        role: AgentRoleArg,
    },
    Stats {
        role: Option<AgentRoleArg>,
    },
}

#[derive(Debug, Subcommand)]
pub enum EvolutionCommand {
    Status,
    Run {
        role: Option<AgentRoleArg>,
        #[arg(long)]
        all_roles: bool,
        #[arg(long)]
        dry_run: bool,
    },
}

#[derive(Debug, Subcommand)]
pub enum BenchCommand {
    Run {
        name: String,
        golden_set: std::path::PathBuf,
        #[arg(long)]
        role: Option<AgentRoleArg>,
    },
    Compare {
        baseline: String,
        challenger: String,
    },
    Show {
        id: String,
    },
}

#[derive(Debug, Subcommand)]
pub enum SandboxCommand {
    Doctor,
    Prove,
}

#[derive(Debug, Subcommand)]
pub enum ProveCommand {
    KernelE2e(BackendAcceptanceArgs),
    HfModel { id: String, repo: String },
    HfRuntimeMatrix,
    RagFull,
    TokenEconomy,
    OrchestratorQuality,
    AgentSwarmLoraRouting,
    LoraLiveE2e { role: Option<AgentRoleArg> },
    LoraEvolutionLoop { role: Option<AgentRoleArg> },
    LoraHotSwap,
    LoraCandleLearning,
    LoraRealModel,
    LoraRealQualityGate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum AgentRoleArg {
    Orchestrator,
    Architect,
    CoderPython,
    CoderRust,
    CoderTypescript,
    Analyst,
    Researcher,
    Qa,
    Security,
    Critic,
    Summarizer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum BackendKindArg {
    Ollama,
    Mistral,
    Onnx,
    OpenAiCompatible,
    Anthropic,
    Custom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum TrainingObjectiveArg {
    Sft,
    Preference,
    Dpo,
    Orpo,
    Kto,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum TaskStatusArg {
    Backlog,
    Ready,
    InProgress,
    Review,
    Remediation,
    Done,
    Failed,
    Blocked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum FailureTypeArg {
    MissingTests,
    UnsafeCode,
    WrongApi,
    HallucinatedFile,
    WeakAnalysis,
    IncompleteCritique,
    BadDecomposition,
    PromptInjection,
    ContextMiss,
    TokenBudgetExceeded,
}

#[cfg(test)]
fn product_help() -> Result<String, String> {
    use clap::CommandFactory;

    let mut command = ProductCli::command();
    let mut output = Vec::new();
    command
        .write_long_help(&mut output)
        .map_err(|error| error.to_string())?;
    String::from_utf8(output).map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn product_cli_help_contains_all_top_level_commands() {
        let help = render_product_help();
        for command in [
            "doctor",
            "project",
            "index",
            "rag",
            "token-economy",
            "goal",
            "plan",
            "kanban",
            "run",
            "review",
            "diag",
            "models",
            "prompts",
            "lora",
            "evolution",
            "bench",
            "sandbox",
            "backend-acceptance",
            "prove",
        ] {
            assert!(
                help.contains(command),
                "product help should mention `{command}`:\n{help}"
            );
        }
    }

    #[test]
    fn proof_only_commands_are_nested_under_prove() {
        let mut command = ProductCli::command();
        let prove = command
            .find_subcommand_mut("prove")
            .expect("prove subcommand exists");
        let prove_help = prove.render_long_help().to_string();
        for command in [
            "kernel-e2e",
            "hf-model",
            "hf-runtime-matrix",
            "rag-full",
            "token-economy",
            "orchestrator-quality",
            "agent-swarm-lora-routing",
            "lora-live-e2e",
            "lora-evolution-loop",
            "lora-hot-swap",
            "lora-candle-learning",
            "lora-real-model",
            "lora-real-quality-gate",
        ] {
            assert!(
                prove_help.contains(command),
                "prove help should mention `{command}`:\n{prove_help}"
            );
        }
    }

    #[test]
    fn typed_enums_are_visible_in_help() {
        let mut command = ProductCli::command();
        let prompts = command
            .find_subcommand_mut("prompts")
            .expect("prompts subcommand exists");
        let prompt_help = prompts
            .find_subcommand_mut("status")
            .expect("prompts status subcommand exists")
            .render_long_help()
            .to_string();
        assert!(prompt_help.contains("coder-python"));
        assert!(prompt_help.contains("critic"));

        let mut command = ProductCli::command();
        let lora = command
            .find_subcommand_mut("lora")
            .expect("lora subcommand exists");
        let lora_help = lora
            .find_subcommand_mut("train")
            .expect("lora train subcommand exists")
            .render_long_help()
            .to_string();
        assert!(lora_help.contains("preference"));
        assert!(lora_help.contains("dpo"));

        let mut command = ProductCli::command();
        let kanban = command
            .find_subcommand_mut("kanban")
            .expect("kanban subcommand exists");
        let kanban_help = kanban
            .find_subcommand_mut("show")
            .expect("kanban show subcommand exists")
            .render_long_help()
            .to_string();
        assert!(kanban_help.contains("in-progress"));
        assert!(kanban_help.contains("remediation"));

        let mut command = ProductCli::command();
        let review = command
            .find_subcommand_mut("review")
            .expect("review subcommand exists");
        let review_help = review
            .find_subcommand_mut("reject")
            .expect("review reject subcommand exists")
            .render_long_help()
            .to_string();
        assert!(review_help.contains("missing-tests"));
        assert!(review_help.contains("bad-decomposition"));
    }

    #[test]
    fn all_command_help_snapshots_render_and_match_documented_paths() {
        let snapshots = collect_help_snapshots(ProductCli::command(), vec!["crytex".into()]);
        assert!(
            snapshots.len() >= 58,
            "expected every product command and subcommand help to render"
        );

        let documented = include_str!("../../../docs/CLI.md");
        for (path, help) in snapshots {
            assert!(!help.trim().is_empty(), "help for `{path}` should render");
            assert!(
                documented.contains(&path),
                "CLI reference should document `{path}`"
            );
        }
    }

    #[test]
    fn global_json_trace_and_project_flags_are_documented() {
        let help = render_product_help();
        assert!(help.contains("--json"));
        assert!(help.contains("--trace-id"));
        assert!(help.contains("--project"));
    }

    #[test]
    fn readme_and_cli_reference_document_the_product_contract() {
        let readme = include_str!("../../../README.md");
        let cli = include_str!("../../../docs/CLI.md");
        for command in [
            "doctor",
            "project",
            "index",
            "rag",
            "token-economy",
            "goal",
            "plan",
            "kanban",
            "run",
            "review",
            "diag",
            "models",
            "prompts",
            "lora",
            "evolution",
            "bench",
            "sandbox",
            "backend-acceptance",
            "prove",
        ] {
            assert!(
                readme.contains(command),
                "README should mention `{command}`"
            );
            assert!(
                cli.contains(command),
                "CLI reference should mention `{command}`"
            );
        }
        for rule in ["stdout", "stderr", "Exit codes", "--json"] {
            assert!(cli.contains(rule), "CLI reference should document `{rule}`");
        }
    }

    fn render_product_help() -> String {
        match product_help() {
            Ok(help) => help,
            Err(error) => panic!("product help should render: {error}"),
        }
    }

    fn collect_help_snapshots(
        mut command: clap::Command,
        path: Vec<String>,
    ) -> Vec<(String, String)> {
        let mut snapshots = vec![(path.join(" "), command.render_long_help().to_string())];
        let subcommand_names = command
            .get_subcommands()
            .map(|subcommand| subcommand.get_name().to_string())
            .filter(|name| name != "help")
            .collect::<Vec<_>>();

        for name in subcommand_names {
            let Some(subcommand) = command.find_subcommand_mut(&name) else {
                panic!("subcommand `{name}` should exist while collecting help snapshots");
            };
            let mut subcommand_path = path.clone();
            subcommand_path.push(name);
            snapshots.extend(collect_help_snapshots(subcommand.clone(), subcommand_path));
        }

        snapshots
    }
}
