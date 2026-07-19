use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TaskStatus {
    Backlog,
    Pending,
    InProgress,
    Review,
    Completed,
    Failed,
    Cancelled,
}

impl TaskStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            TaskStatus::Backlog => "backlog",
            TaskStatus::Pending => "pending",
            TaskStatus::InProgress => "in_progress",
            TaskStatus::Review => "review",
            TaskStatus::Completed => "completed",
            TaskStatus::Failed => "failed",
            TaskStatus::Cancelled => "cancelled",
        }
    }

    /// A terminal status cannot be resumed without explicit reset.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            TaskStatus::Completed | TaskStatus::Failed | TaskStatus::Cancelled
        )
    }
}

impl std::fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for TaskStatus {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "backlog" => Ok(TaskStatus::Backlog),
            "pending" => Ok(TaskStatus::Pending),
            "in_progress" => Ok(TaskStatus::InProgress),
            "review" => Ok(TaskStatus::Review),
            "completed" => Ok(TaskStatus::Completed),
            "failed" => Ok(TaskStatus::Failed),
            "cancelled" => Ok(TaskStatus::Cancelled),
            other => Err(format!("unknown task status: {other}")),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Project {
    pub id: String,
    pub name: String,
    pub root_path: String,
    pub created_at: i64,
    pub updated_at: i64,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: String,
    pub project_id: String,
    pub parent_id: Option<String>,
    pub title: String,
    pub description: Option<String>,
    pub kind: String,
    pub status: TaskStatus,
    pub assigned_agent: Option<String>,
    pub priority: i32,
    pub created_at: i64,
    pub started_at: Option<i64>,
    pub finished_at: Option<i64>,
    pub payload: serde_json::Value,
    pub result: Option<serde_json::Value>,
    pub iteration_count: u32,
    pub priority_score: f64,
    pub critic_score: Option<f64>,
    pub human_score: Option<f64>,
    pub prompt_version_id: Option<String>,
    pub lora_adapter_id: Option<String>,
    pub trace_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskDependency {
    pub task_id: String,
    pub depends_on: String,
    pub dep_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    pub id: String,
    pub task_id: String,
    pub file_path: String,
    pub file_type: String,
    pub commit_sha: Option<String>,
    pub created_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraAdapter {
    pub id: String,
    pub project_id: Option<String>,
    pub name: String,
    pub file_path: String,
    pub base_model: String,
    pub task_kind: Option<String>,
    /// Optional agent role this adapter was trained for (e.g. "coder", "architect").
    pub agent_role: Option<String>,
    pub metrics: serde_json::Value,
    pub created_at: i64,
    pub active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PromptVersion {
    pub id: String,
    pub agent: String,
    pub project_id: Option<String>,
    pub system_prompt: String,
    pub fitness: Option<f64>,
    pub parent_id: Option<String>,
    #[serde(default)]
    pub metrics: serde_json::Value,
    pub created_at: i64,
    pub active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    pub id: String,
    pub task_id: String,
    pub project_id: Option<String>,
    pub prompt_version_id: Option<String>,
    pub text: Option<String>,
    pub critic_score: Option<f64>,
    pub human_score: Option<f64>,
    pub reward: f64,
    pub comment: Option<String>,
    pub created_at: i64,
}

/// A curated supervised-fine-tuning pair produced from an approved task.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainingExample {
    pub id: String,
    pub task_id: String,
    pub project_id: Option<String>,
    pub prompt_version_id: Option<String>,
    pub task_kind: String,
    /// Optional agent role associated with this example.
    pub agent_role: Option<String>,
    pub input_text: String,
    pub output_text: String,
    pub reward: f64,
    pub created_at: i64,
}

/// A single fact stored in the session memory bank.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryEntry {
    pub id: String,
    pub project_id: Option<String>,
    pub session_id: Option<String>,
    pub kind: String,
    pub text: String,
    pub metadata: serde_json::Value,
    pub created_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AuditLogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

impl AuditLogLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            AuditLogLevel::Debug => "debug",
            AuditLogLevel::Info => "info",
            AuditLogLevel::Warn => "warn",
            AuditLogLevel::Error => "error",
        }
    }
}

impl std::fmt::Display for AuditLogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectSnapshot {
    pub id: String,
    pub project_id: String,
    pub name: String,
    pub state_json: serde_json::Value,
    pub created_at: i64,
}

/// Status of a LoRA adapter training job.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TrainingJobStatus {
    Pending,
    Running,
    Succeeded,
    Failed,
    RolledBack,
}

impl TrainingJobStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            TrainingJobStatus::Pending => "pending",
            TrainingJobStatus::Running => "running",
            TrainingJobStatus::Succeeded => "succeeded",
            TrainingJobStatus::Failed => "failed",
            TrainingJobStatus::RolledBack => "rolled_back",
        }
    }
}

impl std::fmt::Display for TrainingJobStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for TrainingJobStatus {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "pending" => Ok(TrainingJobStatus::Pending),
            "running" => Ok(TrainingJobStatus::Running),
            "succeeded" => Ok(TrainingJobStatus::Succeeded),
            "failed" => Ok(TrainingJobStatus::Failed),
            "rolled_back" => Ok(TrainingJobStatus::RolledBack),
            other => Err(format!("unknown training job status: {other}")),
        }
    }
}

/// A tracked LoRA adapter training run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingJob {
    pub id: String,
    pub task_kind: String,
    pub status: TrainingJobStatus,
    pub started_at: i64,
    pub finished_at: Option<i64>,
    pub adapter_id: Option<String>,
    pub metrics: serde_json::Value,
    pub error_message: Option<String>,
}

/// A card shown in the Kanban UI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KanbanTaskCard {
    pub id: String,
    pub title: String,
    pub kind: String,
    pub status: TaskStatus,
    pub priority: i32,
    pub assigned_agent: Option<String>,
}

/// A single column in the Kanban board.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KanbanColumn {
    pub status: TaskStatus,
    pub title: String,
    pub tasks: Vec<KanbanTaskCard>,
}

/// Full Kanban board state for a project.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KanbanState {
    pub project_id: String,
    pub columns: Vec<KanbanColumn>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentLog {
    pub id: String,
    pub project_id: Option<String>,
    pub task_id: Option<String>,
    pub agent: String,
    pub action: String,
    pub message: Option<String>,
    pub level: String,
    pub timestamp: i64,
    pub metadata: serde_json::Value,
}

/// A single curated example in a benchmark golden set.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BenchmarkCase {
    pub id: String,
    pub input: serde_json::Value,
    #[serde(default)]
    pub expected: Option<serde_json::Value>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

/// A variant (configuration) being benchmarked.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct BenchmarkVariant {
    pub name: String,
    #[serde(default)]
    pub agent_role: Option<String>,
    #[serde(default)]
    pub lora_adapter_id: Option<String>,
    #[serde(default)]
    pub prompt_version_id: Option<String>,
    #[serde(default)]
    pub backend_id: Option<String>,
}

/// A scored outcome for a single benchmark case.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BenchmarkResult {
    pub id: String,
    pub run_id: String,
    pub case_id: String,
    pub case_input: serde_json::Value,
    pub expected: Option<serde_json::Value>,
    pub actual: serde_json::Value,
    pub passed: bool,
    pub score_value: f64,
    pub latency_ms: u64,
    pub token_usage: Option<crytex_inference::TokenUsage>,
    pub explanation: Option<String>,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

/// Aggregate summary of a benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BenchmarkRunSummary {
    pub id: String,
    pub name: String,
    pub golden_set_path: std::path::PathBuf,
    pub variant_name: String,
    pub pass_count: usize,
    pub fail_count: usize,
    pub total_cases: usize,
    pub pass_rate: f64,
    pub mean_latency_ms: f64,
    pub total_tokens: usize,
}

/// A full benchmark run including all per-case results.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BenchmarkRun {
    #[serde(flatten)]
    pub summary: BenchmarkRunSummary,
    pub project_id: Option<String>,
    pub variant: BenchmarkVariant,
    pub scorer_kind: String,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub finished_at: Option<chrono::DateTime<chrono::Utc>>,
    pub results: Vec<BenchmarkResult>,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

impl BenchmarkRun {
    pub fn summary(&self) -> &BenchmarkRunSummary {
        &self.summary
    }
}
