//! Collects approved tasks as golden examples and evolves per-domain LoRA adapters.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use crytex_inference::LoRAAdapter as InferenceLoRAAdapter;
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::{BTreeMap, HashMap, HashSet};
use thiserror::Error;
use ulid::Ulid;

use crate::bus::Event;
use crate::models::{
    Experience, LoraAdapter, Task, TaskStatus, TrainingExample, TrainingJob, TrainingJobStatus,
};
use crate::persistence::{
    ExperienceRepository, LoraAdapterRepository, PersistenceError, PromptVersionRepository,
    TrainingExampleRepository, TrainingJobRepository,
};
use crate::services::LoraMetrics;
use crate::services::{
    AdapterMetadata, AgentRole, Embedder, EventService, InferenceService, LoraTrainer,
    LoraTrainingConfig, LoraTrainingError, LoraTrainingObjective, LoraTrainingResult,
    RewardService, TaskError, TaskService, validate_objective_examples,
    vector_store::{VectorPoint, VectorStore},
};

/// Errors returned by [`LoraEvolutionService`].
#[derive(Debug, Error)]
pub enum LoraEvolutionError {
    #[error("task service error: {0}")]
    Task(#[from] TaskError),
    #[error("persistence error: {0}")]
    Persistence(#[from] PersistenceError),
    #[error("training error: {0}")]
    Training(#[from] LoraTrainingError),
    #[error("inference error: {0}")]
    Inference(String),
    #[error("task {0} is not a valid golden example")]
    InvalidGoldenExample(String),
    #[error("validation failed for kind {0}: {1}")]
    ValidationFailed(String, String),
    #[error("adapter indexing error: {0}")]
    Index(String),
}

const LORA_ADAPTER_COLLECTION: &str = "lora_adapters";
const DEFAULT_MAX_ADAPTER_BYTES: u64 = 512 * 1024 * 1024;
const DEFAULT_MAX_TRAIN_VALIDATION_LOSS_GAP: f64 = 1.0;
const MIN_EXAMPLE_TEXT_CHARS: usize = 4;

/// Input passed to a benchmark gate before a freshly trained LoRA adapter is promoted.
#[derive(Debug, Clone, PartialEq)]
pub struct LoraBenchmarkRequest {
    pub task_kind: String,
    pub agent_role: Option<String>,
    pub baseline_adapter_id: Option<String>,
    pub challenger_adapter_id: String,
    pub challenger_adapter_path: PathBuf,
    pub base_model: String,
    pub challenger_metrics: serde_json::Value,
    pub validation_reward: f64,
    pub training_fingerprints: Vec<String>,
}

/// Promotion decision returned by a LoRA benchmark gate.
#[derive(Debug, Clone, PartialEq)]
pub struct LoraBenchmarkDecision {
    pub accepted: bool,
    pub reason: String,
    pub metadata: serde_json::Value,
    pub quality_gates: Vec<LoraQualityGateResult>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoraQualityGateName {
    PositiveBenchmark,
    NegativeBenchmark,
    RegressionBenchmark,
    SafetyBenchmark,
    RuntimeApplication,
    OutputChanged,
}

impl LoraQualityGateName {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::PositiveBenchmark => "positive_benchmark",
            Self::NegativeBenchmark => "negative_benchmark",
            Self::RegressionBenchmark => "regression_benchmark",
            Self::SafetyBenchmark => "safety_benchmark",
            Self::RuntimeApplication => "runtime_application",
            Self::OutputChanged => "output_changed",
        }
    }

    fn required() -> [Self; 6] {
        [
            Self::PositiveBenchmark,
            Self::NegativeBenchmark,
            Self::RegressionBenchmark,
            Self::SafetyBenchmark,
            Self::RuntimeApplication,
            Self::OutputChanged,
        ]
    }
}

impl std::fmt::Display for LoraQualityGateName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoraQualityGateResult {
    pub name: LoraQualityGateName,
    pub passed: bool,
    pub evidence: String,
}

/// Leakage diagnostics for a role dataset.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoraDatasetLeakageReport {
    pub duplicate_task_count: usize,
    pub duplicate_output_count: usize,
}

/// Low-information filtering diagnostics for a role dataset.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoraDatasetLowInformationReport {
    pub filtered_count: usize,
    pub example_ids: Vec<String>,
}

/// Failure-type balancing plan for a role dataset.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoraDatasetBalancingReport {
    pub target_per_failure_type: BTreeMap<String, usize>,
    pub selected_example_ids: Vec<String>,
}

impl LoraDatasetBalancingReport {
    pub fn failure_type_target_count(&self, failure_type: &str) -> Option<usize> {
        self.target_per_failure_type.get(failure_type).copied()
    }
}

/// Inspectable dataset report used by CLI and proofs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoraDatasetReport {
    pub role: String,
    pub total_examples: usize,
    pub positive_examples: usize,
    pub negative_examples: usize,
    pub preference_pairs: usize,
    pub failure_type_counts: BTreeMap<String, usize>,
    pub leakage: LoraDatasetLeakageReport,
    pub low_information: LoraDatasetLowInformationReport,
    pub balancing: LoraDatasetBalancingReport,
}

impl LoraDatasetReport {
    pub fn empty(role: &str) -> Self {
        Self {
            role: role.to_string(),
            total_examples: 0,
            positive_examples: 0,
            negative_examples: 0,
            preference_pairs: 0,
            failure_type_counts: BTreeMap::new(),
            leakage: LoraDatasetLeakageReport {
                duplicate_task_count: 0,
                duplicate_output_count: 0,
            },
            low_information: LoraDatasetLowInformationReport {
                filtered_count: 0,
                example_ids: Vec::new(),
            },
            balancing: LoraDatasetBalancingReport {
                target_per_failure_type: BTreeMap::new(),
                selected_example_ids: Vec::new(),
            },
        }
    }
}

/// Pure dataset inspector used by services, CLI proofs, and tests.
pub struct LoraDatasetInspector;

impl LoraDatasetInspector {
    pub fn report(role: &str, examples: Vec<TrainingExample>) -> LoraDatasetReport {
        let failure_type_counts = failure_type_counts(&examples);
        let low_information = low_information_report(&examples);
        let leakage = leakage_report(&examples);
        let balancing = balancing_report(&examples, &failure_type_counts);
        let positive_examples = examples
            .iter()
            .filter(|example| example.accepted_output.is_some() || example.reward > 0.0)
            .count();
        let negative_examples = examples
            .iter()
            .filter(|example| example.rejected_output.is_some() || example.reward <= 0.0)
            .count();
        let preference_pairs = examples
            .iter()
            .filter(|example| {
                example.accepted_output.is_some() && example.rejected_output.is_some()
            })
            .count();

        LoraDatasetReport {
            role: role.to_string(),
            total_examples: examples.len(),
            positive_examples,
            negative_examples,
            preference_pairs,
            failure_type_counts,
            leakage,
            low_information,
            balancing,
        }
    }
}

impl LoraBenchmarkDecision {
    pub fn accept(reason: impl Into<String>) -> Self {
        Self {
            accepted: true,
            reason: reason.into(),
            metadata: serde_json::Value::Null,
            quality_gates: Vec::new(),
        }
    }

    pub fn reject(reason: impl Into<String>) -> Self {
        Self {
            accepted: false,
            reason: reason.into(),
            metadata: serde_json::Value::Null,
            quality_gates: Vec::new(),
        }
    }

    pub fn accept_with_quality_gates(
        reason: impl Into<String>,
        quality_gates: Vec<LoraQualityGateResult>,
        metadata: serde_json::Value,
    ) -> Self {
        Self {
            accepted: true,
            reason: reason.into(),
            metadata,
            quality_gates,
        }
    }

    pub fn missing_required_quality_gates(&self) -> Vec<LoraQualityGateName> {
        LoraQualityGateName::required()
            .into_iter()
            .filter(|required| {
                !self
                    .quality_gates
                    .iter()
                    .any(|gate| gate.name == *required && gate.passed)
            })
            .collect()
    }

    pub fn all_required_quality_gates_passed(&self) -> bool {
        self.missing_required_quality_gates().is_empty()
    }
}

pub fn lora_quality_gate(
    name: LoraQualityGateName,
    passed: bool,
    evidence: impl Into<String>,
) -> LoraQualityGateResult {
    LoraQualityGateResult {
        name,
        passed,
        evidence: evidence.into(),
    }
}

pub fn passed_lora_quality_gates(evidence: impl AsRef<str>) -> Vec<LoraQualityGateResult> {
    let evidence = evidence.as_ref();
    LoraQualityGateName::required()
        .into_iter()
        .map(|name| lora_quality_gate(name, true, evidence))
        .collect()
}

/// Compares a newly trained LoRA adapter against the current baseline.
#[async_trait]
pub trait LoraBenchmarkGate: Send + Sync {
    async fn evaluate(
        &self,
        request: LoraBenchmarkRequest,
    ) -> Result<LoraBenchmarkDecision, LoraEvolutionError>;
}

/// Evolves LoRA adapters from human-approved task outcomes.
#[async_trait]
pub trait LoraEvolutionService: Send + Sync {
    /// Store a curated `(input, output, reward)` example from an approved task.
    async fn collect_golden_example(&self, task_id: &str) -> Result<(), LoraEvolutionError>;

    /// Store a low-reward counter-example from a rejected task.
    async fn collect_counter_example(&self, task_id: &str) -> Result<(), LoraEvolutionError>;

    /// Return `true` when enough golden examples exist to train a new adapter.
    async fn should_train(&self, task_kind: &str) -> Result<bool, LoraEvolutionError>;

    /// Train and register a new adapter for the given task kind.
    async fn train_and_register(&self, task_kind: &str) -> Result<LoraAdapter, LoraEvolutionError>;

    /// Return `true` when enough golden examples exist for the given role.
    async fn should_train_for_role(&self, role: AgentRole) -> Result<bool, LoraEvolutionError>;

    /// Train and register a new adapter specialized for the given role.
    async fn train_and_register_for_role(
        &self,
        role: AgentRole,
    ) -> Result<LoraAdapter, LoraEvolutionError>;

    /// Train and register a new role adapter with an explicit training objective.
    async fn train_and_register_for_role_objective(
        &self,
        role: AgentRole,
        objective: LoraTrainingObjective,
    ) -> Result<LoraAdapter, LoraEvolutionError> {
        let _ = role;
        Err(LoraEvolutionError::Training(
            LoraTrainingError::UnsupportedObjective {
                backend: "lora-evolution-service".into(),
                objective,
            },
        ))
    }

    /// Select the best registered adapter for a task.
    async fn select_lora(
        &self,
        task: &Task,
        _project_id: &str,
    ) -> Result<Option<String>, LoraEvolutionError>;

    /// Select the best registered adapter for a role.
    async fn select_lora_by_role(
        &self,
        role: AgentRole,
        _project_id: &str,
    ) -> Result<Option<String>, LoraEvolutionError>;

    /// Build or refresh a role-specific LoRA preference dataset.
    async fn build_dataset_for_role(
        &self,
        role: &str,
    ) -> Result<LoraDatasetReport, LoraEvolutionError> {
        Ok(LoraDatasetReport::empty(role))
    }

    /// Inspect a role-specific LoRA dataset.
    async fn inspect_dataset_for_role(
        &self,
        role: &str,
    ) -> Result<LoraDatasetReport, LoraEvolutionError> {
        Ok(LoraDatasetReport::empty(role))
    }

    /// Return role-specific dataset statistics.
    async fn dataset_stats_for_role(
        &self,
        role: &str,
    ) -> Result<LoraDatasetReport, LoraEvolutionError> {
        Ok(LoraDatasetReport::empty(role))
    }
}

/// Default implementation of [`LoraEvolutionService`].
pub struct LoraEvolutionServiceImpl {
    task_service: Arc<dyn TaskService>,
    prompt_version_repo: Arc<dyn PromptVersionRepository>,
    training_example_repo: Arc<dyn TrainingExampleRepository>,
    lora_adapter_repo: Arc<dyn LoraAdapterRepository>,
    inference_service: Arc<dyn InferenceService>,
    event_service: Arc<dyn EventService>,
    trainer: Arc<dyn LoraTrainer>,
    embedder: Option<Arc<dyn Embedder>>,
    vector_store: Option<Arc<dyn VectorStore>>,
    adapters_dir: PathBuf,
    base_model: String,
    threshold: usize,
    validation_reward_threshold: f64,
    validation_loss_threshold: f64,
    min_human_score: f64,
    experience_repo: Option<Arc<dyn ExperienceRepository>>,
    training_job_repo: Option<Arc<dyn TrainingJobRepository>>,
    benchmark_gate: Option<Arc<dyn LoraBenchmarkGate>>,
    max_adapter_bytes: u64,
    max_train_validation_loss_gap: f64,
    training_config: LoraTrainingConfig,
}

impl LoraEvolutionServiceImpl {
    /// Create a new evolution service.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        task_service: Arc<dyn TaskService>,
        prompt_version_repo: Arc<dyn PromptVersionRepository>,
        training_example_repo: Arc<dyn TrainingExampleRepository>,
        lora_adapter_repo: Arc<dyn LoraAdapterRepository>,
        inference_service: Arc<dyn InferenceService>,
        event_service: Arc<dyn EventService>,
        trainer: Arc<dyn LoraTrainer>,
        adapters_dir: PathBuf,
        base_model: String,
    ) -> Self {
        Self {
            task_service,
            prompt_version_repo,
            training_example_repo,
            lora_adapter_repo,
            inference_service,
            event_service,
            trainer,
            embedder: None,
            vector_store: None,
            adapters_dir,
            base_model,
            threshold: 50,
            validation_reward_threshold: 3.0,
            validation_loss_threshold: 0.5,
            min_human_score: 4.0,
            experience_repo: None,
            training_job_repo: None,
            benchmark_gate: None,
            max_adapter_bytes: DEFAULT_MAX_ADAPTER_BYTES,
            max_train_validation_loss_gap: DEFAULT_MAX_TRAIN_VALIDATION_LOSS_GAP,
            training_config: LoraTrainingConfig::default(),
        }
    }

    /// Enable semantic indexing of trained adapters for later vector-search fallback.
    pub fn with_vector_index(
        mut self,
        embedder: Arc<dyn Embedder>,
        vector_store: Arc<dyn VectorStore>,
    ) -> Self {
        self.embedder = Some(embedder);
        self.vector_store = Some(vector_store);
        self
    }

    /// Override the minimum number of golden examples required before training.
    pub fn with_threshold(mut self, threshold: usize) -> Self {
        self.threshold = threshold;
        self
    }

    /// Override the minimum average reward required to accept a trained adapter.
    pub fn with_validation_reward_threshold(mut self, threshold: f64) -> Self {
        self.validation_reward_threshold = threshold;
        self
    }

    /// Override the maximum validation loss allowed to accept a trained adapter.
    pub fn with_validation_loss_threshold(mut self, threshold: f64) -> Self {
        self.validation_loss_threshold = threshold;
        self
    }

    /// Attach an experience repository so golden/counter examples are also indexed
    /// for semantic search.
    pub fn with_experience_repo(mut self, repo: Arc<dyn ExperienceRepository>) -> Self {
        self.experience_repo = Some(repo);
        self
    }

    /// Attach a training-job repository so train runs are tracked.
    pub fn with_training_job_repo(mut self, repo: Arc<dyn TrainingJobRepository>) -> Self {
        self.training_job_repo = Some(repo);
        self
    }

    /// Attach a benchmark gate that must accept a new adapter before promotion.
    pub fn with_benchmark_gate(mut self, gate: Arc<dyn LoraBenchmarkGate>) -> Self {
        self.benchmark_gate = Some(gate);
        self
    }

    /// Override the maximum accepted LoRA artifact size.
    pub fn with_max_adapter_bytes(mut self, bytes: u64) -> Self {
        self.max_adapter_bytes = bytes;
        self
    }

    /// Override the maximum accepted validation-loss minus train-loss gap.
    pub fn with_max_train_validation_loss_gap(mut self, gap: f64) -> Self {
        self.max_train_validation_loss_gap = gap;
        self
    }

    /// Override the minimum human score that makes an approved task a golden example.
    pub fn with_min_human_score(mut self, score: f64) -> Self {
        self.min_human_score = score;
        self
    }

    /// Override LoRA training hyper-parameters for this service.
    pub fn with_training_config(mut self, config: LoraTrainingConfig) -> Self {
        self.training_config = config;
        self
    }

    fn build_input_text(task: &Task, system_prompt: Option<&str>) -> String {
        let mut text = String::new();
        if let Some(system) = system_prompt {
            text.push_str(system);
            text.push('\n');
            text.push('\n');
        }
        text.push_str("Task: ");
        text.push_str(&task.title);
        text.push('\n');
        if let Some(description) = &task.description {
            text.push_str("Description: ");
            text.push_str(description);
            text.push('\n');
        }
        text
    }

    fn build_output_text(task: &Task) -> String {
        task.result
            .as_ref()
            .map(|r| r.to_string())
            .unwrap_or_default()
    }

    async fn index_adapter(
        &self,
        adapter: &LoraAdapter,
        train_examples: &[TrainingExample],
    ) -> Result<(), String> {
        let (embedder, vector_store) = match (&self.embedder, &self.vector_store) {
            (Some(e), Some(v)) => (e, v),
            _ => return Ok(()),
        };

        let sample_inputs: Vec<_> = train_examples
            .iter()
            .take(3)
            .map(|e| e.input_text.as_str())
            .collect();
        let text = format!(
            "{}\n{}",
            adapter.task_kind.as_deref().unwrap_or(""),
            sample_inputs.join("\n---\n")
        );

        let dim = embedder.dimension().await.map_err(|e| e.to_string())?;
        let vector = embedder.embed(&text).await.map_err(|e| e.to_string())?;
        vector_store
            .create_collection(LORA_ADAPTER_COLLECTION, dim)
            .await
            .map_err(|e| e.to_string())?;
        vector_store
            .upsert(
                LORA_ADAPTER_COLLECTION,
                vec![VectorPoint {
                    id: adapter.id.clone(),
                    vector,
                    payload: serde_json::json!({ "adapter_id": adapter.id }),
                }],
            )
            .await
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    fn next_adapter_id(&self, task_kind: &str, existing: &[LoraAdapter]) -> String {
        let max_version = existing
            .iter()
            .filter_map(|a| {
                a.name
                    .strip_prefix(&format!("{task_kind}-v"))
                    .and_then(|s| s.parse::<usize>().ok())
            })
            .max()
            .unwrap_or(0);
        format!("{task_kind}-v{}", max_version + 1)
    }

    fn validate_training_examples(
        &self,
        examples: &[TrainingExample],
        key: &str,
    ) -> Result<(), LoraEvolutionError> {
        let mut task_ids = HashSet::with_capacity(examples.len());
        if let Some(example) = examples
            .iter()
            .find(|example| !task_ids.insert(example.task_id.as_str()))
        {
            return Err(LoraEvolutionError::ValidationFailed(
                key.to_string(),
                format!(
                    "golden dataset contains duplicate task_id {}; train/validation split would leak task output",
                    example.task_id
                ),
            ));
        }

        if let Some(example) = examples.iter().find(|example| {
            self.is_golden_training_example(example)
                && (example.input_text.trim().chars().count() < MIN_EXAMPLE_TEXT_CHARS
                    || example.output_text.trim().chars().count() < MIN_EXAMPLE_TEXT_CHARS)
        }) {
            return Err(LoraEvolutionError::ValidationFailed(
                key.to_string(),
                format!(
                    "golden dataset contains low-information example {}",
                    example.id
                ),
            ));
        }

        Ok(())
    }

    fn is_golden_training_example(&self, example: &TrainingExample) -> bool {
        example.reward >= self.validation_reward_threshold
    }

    fn training_fingerprints(examples: &[TrainingExample]) -> Vec<String> {
        examples
            .iter()
            .map(|example| format!("{} {}", example.input_text, example.output_text))
            .collect()
    }

    async fn validate_adapter_artifact(
        &self,
        result: &LoraTrainingResult,
        key: &str,
    ) -> Result<(), LoraEvolutionError> {
        let metadata = Self::read_metadata(&result.adapter_path, key).await?;
        if !metadata.is_dir() {
            return Err(LoraEvolutionError::ValidationFailed(
                key.to_string(),
                format!(
                    "adapter artifact must be a directory containing adapter_config.json and adapter_model.safetensors: {}",
                    result.adapter_path.display()
                ),
            ));
        }

        let config_path = result.adapter_path.join("adapter_config.json");
        let weights_path = result.adapter_path.join("adapter_model.safetensors");
        let metadata_path = result.adapter_path.join("adapter_metadata.json");
        let config = tokio::fs::read_to_string(&config_path).await.map_err(|e| {
            LoraEvolutionError::ValidationFailed(
                key.to_string(),
                format!(
                    "adapter_config.json is unreadable at {}: {e}",
                    config_path.display()
                ),
            )
        })?;
        let config: serde_json::Value = serde_json::from_str(&config).map_err(|e| {
            LoraEvolutionError::ValidationFailed(
                key.to_string(),
                format!(
                    "adapter_config.json is not valid JSON at {}: {e}",
                    config_path.display()
                ),
            )
        })?;
        if config
            .get("peft_type")
            .and_then(|value| value.as_str())
            .is_none_or(|peft_type| !peft_type.eq_ignore_ascii_case("LORA"))
        {
            return Err(LoraEvolutionError::ValidationFailed(
                key.to_string(),
                "adapter_config.json must declare peft_type=LORA".to_string(),
            ));
        }

        let weights_metadata = Self::read_metadata(&weights_path, key).await?;
        if !weights_metadata.is_file() || weights_metadata.len() == 0 {
            return Err(LoraEvolutionError::ValidationFailed(
                key.to_string(),
                format!(
                    "adapter_model.safetensors must be a non-empty file at {}",
                    weights_path.display()
                ),
            ));
        }

        let metadata = tokio::fs::read_to_string(&metadata_path)
            .await
            .map_err(|e| {
                LoraEvolutionError::ValidationFailed(
                    key.to_string(),
                    format!(
                        "adapter_metadata.json is unreadable at {}: {e}",
                        metadata_path.display()
                    ),
                )
            })?;
        let metadata: AdapterMetadata = serde_json::from_str(&metadata).map_err(|e| {
            LoraEvolutionError::ValidationFailed(
                key.to_string(),
                format!(
                    "adapter_metadata.json is not valid adapter metadata at {}: {e}",
                    metadata_path.display()
                ),
            )
        })?;
        if metadata.dataset_hash.trim().is_empty() {
            return Err(LoraEvolutionError::ValidationFailed(
                key.to_string(),
                "adapter_metadata.json must contain dataset_hash".to_string(),
            ));
        }

        let artifact_bytes = weights_metadata.len();
        if artifact_bytes > self.max_adapter_bytes {
            return Err(LoraEvolutionError::ValidationFailed(
                key.to_string(),
                format!(
                    "adapter artifact is too large: {} bytes exceeds {} bytes",
                    artifact_bytes, self.max_adapter_bytes
                ),
            ));
        }

        Ok(())
    }

    async fn read_metadata(
        path: &Path,
        key: &str,
    ) -> Result<std::fs::Metadata, LoraEvolutionError> {
        tokio::fs::metadata(path).await.map_err(|e| {
            LoraEvolutionError::ValidationFailed(
                key.to_string(),
                format!(
                    "adapter artifact metadata is unreadable at {}: {e}",
                    path.display()
                ),
            )
        })
    }

    async fn remove_adapter_artifact(path: &Path) {
        match tokio::fs::metadata(path).await {
            Ok(metadata) if metadata.is_dir() => {
                let _ = tokio::fs::remove_dir_all(path).await;
            }
            Ok(_) => {
                let _ = tokio::fs::remove_file(path).await;
            }
            Err(_) => {}
        }
    }

    fn validate_overfit_gap(
        &self,
        metrics: &LoraMetrics,
        key: &str,
    ) -> Result<(), LoraEvolutionError> {
        let gap = metrics.validation_loss - metrics.train_loss;
        if gap > self.max_train_validation_loss_gap {
            return Err(LoraEvolutionError::ValidationFailed(
                key.to_string(),
                format!(
                    "overfit risk: validation/train loss gap {:.4} exceeds {:.4}",
                    gap, self.max_train_validation_loss_gap
                ),
            ));
        }

        Ok(())
    }

    async fn role_dataset_examples(
        &self,
        role: &str,
    ) -> Result<Vec<TrainingExample>, LoraEvolutionError> {
        Ok(self
            .training_example_repo
            .list_training_examples_by_role(role)
            .await?)
    }

    async fn insert_role_dataset_pairs(&self, role: &str) -> Result<(), LoraEvolutionError> {
        let existing_task_ids = self
            .role_dataset_examples(role)
            .await?
            .into_iter()
            .map(|example| example.task_id)
            .collect::<HashSet<_>>();
        let tasks = self.task_service.load_all_tasks().await?;
        let by_id = tasks
            .iter()
            .map(|task| (task.id.as_str(), task))
            .collect::<HashMap<_, _>>();

        for accepted in tasks.iter().filter(|task| {
            task.status == TaskStatus::Completed
                && task.human_score.unwrap_or_default() >= self.min_human_score
                && task_role(task).as_deref() == Some(role)
                && !existing_task_ids.contains(&task.id)
        }) {
            let rejected = accepted
                .payload
                .get("remediation_for")
                .and_then(serde_json::Value::as_str)
                .and_then(|id| by_id.get(id).copied());
            let example = self
                .build_training_example_from_task(accepted, rejected, role)
                .await?;
            self.training_example_repo
                .insert_training_example(&example)
                .await?;
        }
        Ok(())
    }

    async fn build_training_example_from_task(
        &self,
        task: &Task,
        rejected: Option<&Task>,
        role: &str,
    ) -> Result<TrainingExample, LoraEvolutionError> {
        let system_prompt = if let Some(version_id) = task.prompt_version_id.as_deref() {
            self.prompt_version_repo
                .get_prompt_version(version_id)
                .await?
                .map(|v| v.system_prompt)
        } else {
            None
        };
        let accepted_output = Self::build_output_text(task);
        let rejected_output = rejected.map(Self::build_output_text);
        let critic_feedback = rejected.and_then(task_critic_feedback);
        let failure_type = rejected
            .and_then(task_failure_type)
            .or_else(|| task_failure_type(task));

        Ok(TrainingExample {
            id: Ulid::new().to_string(),
            task_id: task.id.clone(),
            project_id: Some(task.project_id.clone()),
            prompt_version_id: task.prompt_version_id.clone(),
            task_kind: task.kind.clone(),
            agent_role: Some(role.to_string()),
            model_id: task_model_id(task),
            rag_evidence_ids: task_rag_evidence_ids(task),
            input_text: Self::build_input_text(task, system_prompt.as_deref()),
            output_text: accepted_output.clone(),
            accepted_output: Some(accepted_output),
            rejected_output,
            critic_feedback,
            failure_type,
            reward: RewardService::compute(task.critic_score, task.human_score),
            created_at: Utc::now().timestamp_millis(),
        })
    }

    fn build_dataset_report(role: &str, examples: Vec<TrainingExample>) -> LoraDatasetReport {
        LoraDatasetInspector::report(role, examples)
    }
}

impl LoraEvolutionServiceImpl {
    async fn collect_example(
        &self,
        task_id: &str,
        counter: bool,
    ) -> Result<(), LoraEvolutionError> {
        let task = self
            .task_service
            .get(task_id)
            .await?
            .ok_or_else(|| LoraEvolutionError::InvalidGoldenExample(task_id.to_string()))?;

        let valid_status = if counter {
            matches!(task.status, TaskStatus::Completed | TaskStatus::Pending)
        } else {
            task.status == TaskStatus::Completed
        };
        if !valid_status {
            return Err(LoraEvolutionError::InvalidGoldenExample(task.id));
        }

        let human_score = task.human_score.unwrap_or(0.0);
        if !counter && human_score < self.min_human_score {
            return Err(LoraEvolutionError::InvalidGoldenExample(task.id));
        }

        let system_prompt = if let Some(version_id) = task.prompt_version_id.as_deref() {
            self.prompt_version_repo
                .get_prompt_version(version_id)
                .await?
                .map(|v| v.system_prompt)
        } else {
            None
        };

        let reward = if counter {
            0.0
        } else {
            RewardService::compute(task.critic_score, task.human_score)
        };
        let output_text = Self::build_output_text(&task);
        let example = TrainingExample {
            id: Ulid::new().to_string(),
            task_id: task.id.clone(),
            project_id: Some(task.project_id.clone()),
            prompt_version_id: task.prompt_version_id.clone(),
            task_kind: task.kind.clone(),
            agent_role: AgentRole::from_agent(
                task.assigned_agent.as_deref().unwrap_or(task.kind.as_str()),
            )
            .map(|r| r.as_str().to_string()),
            model_id: task_model_id(&task),
            rag_evidence_ids: task_rag_evidence_ids(&task),
            input_text: Self::build_input_text(&task, system_prompt.as_deref()),
            output_text: if counter {
                String::new()
            } else {
                output_text.clone()
            },
            accepted_output: (!counter).then_some(output_text.clone()),
            rejected_output: counter.then_some(output_text),
            critic_feedback: task_critic_feedback(&task),
            failure_type: task_failure_type(&task),
            reward,
            created_at: Utc::now().timestamp_millis(),
        };

        self.training_example_repo
            .insert_training_example(&example)
            .await?;

        if let Some(repo) = &self.experience_repo {
            let experience = Experience {
                id: Ulid::new().to_string(),
                task_id: example.task_id.clone(),
                project_id: example.project_id.clone(),
                prompt_version_id: example.prompt_version_id.clone(),
                text: Some(format!(
                    "{}\n---\n{}",
                    example.input_text, example.output_text
                )),
                critic_score: task.critic_score,
                human_score: task.human_score,
                reward: example.reward,
                comment: Some(if counter {
                    "counter-example".into()
                } else {
                    "golden example".into()
                }),
                created_at: example.created_at,
            };
            repo.insert_experience(&experience).await?;
        }

        Ok(())
    }

    async fn train_and_register_with_examples(
        &self,
        mut examples: Vec<TrainingExample>,
        key: &str,
        agent_role: Option<String>,
        objective: LoraTrainingObjective,
    ) -> Result<LoraAdapter, LoraEvolutionError> {
        examples.sort_by_key(|e| e.created_at);
        self.validate_training_examples(&examples, key)?;
        let training_examples = examples
            .iter()
            .filter(|example| {
                self.is_golden_training_example(example)
                    || (objective.requires_preference_pairs()
                        && example.accepted_output.is_some()
                        && example.rejected_output.is_some())
            })
            .cloned()
            .collect::<Vec<_>>();
        validate_objective_examples(&objective, &training_examples)?;
        if !self.trainer.supports_objective(&objective) {
            return Err(LoraEvolutionError::Training(
                LoraTrainingError::UnsupportedObjective {
                    backend: self.trainer.backend_name().into(),
                    objective,
                },
            ));
        }
        if training_examples.len() < self.threshold {
            return Err(LoraEvolutionError::ValidationFailed(
                key.to_string(),
                format!(
                    "golden dataset contains only {} usable examples below training threshold {}",
                    training_examples.len(),
                    self.threshold
                ),
            ));
        }

        let validation_count = ((training_examples.len() as f64
            * self.training_config.validation_ratio)
            .ceil() as usize)
            .max(1)
            .min(training_examples.len().saturating_sub(1));
        let split_index = training_examples.len() - validation_count;
        let (train_examples, validation_examples) = training_examples.split_at(split_index);

        let job_id = Ulid::new().to_string();
        let job_repo = self.training_job_repo.clone();
        if let Some(repo) = &job_repo {
            let job = TrainingJob {
                id: job_id.clone(),
                task_kind: key.to_string(),
                status: TrainingJobStatus::Queued,
                started_at: Utc::now().timestamp_millis(),
                finished_at: None,
                adapter_id: None,
                metrics: serde_json::json!({
                    "objective": objective.as_str(),
                    "backend": self.trainer.backend_name(),
                }),
                error_message: None,
            };
            repo.insert_training_job(&job).await?;
            let mut running = job.clone();
            running.status = TrainingJobStatus::Running;
            repo.update_training_job(&running).await?;
        }

        let output_dir = self.adapters_dir.join(key);
        let mut training_config = self.training_config.clone();
        training_config.objective = objective.clone();
        training_config.role = agent_role.clone();
        training_config.base_model_id = Some(self.base_model.clone());
        let result = match self
            .trainer
            .train(train_examples.to_vec(), training_config, &output_dir)
            .await
        {
            Ok(r) => r,
            Err(e) => {
                if let Some(repo) = &job_repo {
                    let job = TrainingJob {
                        id: job_id,
                        task_kind: key.to_string(),
                        status: TrainingJobStatus::Failed,
                        started_at: Utc::now().timestamp_millis(),
                        finished_at: Some(Utc::now().timestamp_millis()),
                        adapter_id: None,
                        metrics: serde_json::Value::Null,
                        error_message: Some(e.to_string()),
                    };
                    let _ = repo.update_training_job(&job).await;
                }
                return Err(LoraEvolutionError::Training(e));
            }
        };
        if let Err(e) = self.validate_adapter_artifact(&result, key).await {
            Self::remove_adapter_artifact(&result.adapter_path).await;
            if let Some(repo) = &job_repo {
                let job = TrainingJob {
                    id: job_id.clone(),
                    task_kind: key.to_string(),
                    status: TrainingJobStatus::RolledBack,
                    started_at: Utc::now().timestamp_millis(),
                    finished_at: Some(Utc::now().timestamp_millis()),
                    adapter_id: None,
                    metrics: serde_json::to_value(&result.metrics).unwrap_or_default(),
                    error_message: Some(e.to_string()),
                };
                let _ = repo.update_training_job(&job).await;
            }
            return Err(e);
        }
        if let Err(e) = self.validate_overfit_gap(&result.metrics, key) {
            Self::remove_adapter_artifact(&result.adapter_path).await;
            if let Some(repo) = &job_repo {
                let job = TrainingJob {
                    id: job_id.clone(),
                    task_kind: key.to_string(),
                    status: TrainingJobStatus::RolledBack,
                    started_at: Utc::now().timestamp_millis(),
                    finished_at: Some(Utc::now().timestamp_millis()),
                    adapter_id: None,
                    metrics: serde_json::to_value(&result.metrics).unwrap_or_default(),
                    error_message: Some(e.to_string()),
                };
                let _ = repo.update_training_job(&job).await;
            }
            return Err(e);
        }

        let validation_reward = validation_examples.iter().map(|e| e.reward).sum::<f64>()
            / validation_examples.len().max(1) as f64;
        let validation_loss = result.metrics.validation_loss;

        if validation_reward < self.validation_reward_threshold
            || validation_loss > self.validation_loss_threshold
        {
            let reason = if validation_reward < self.validation_reward_threshold {
                format!(
                    "validation reward {:.2} below threshold {:.2}",
                    validation_reward, self.validation_reward_threshold
                )
            } else {
                format!(
                    "validation loss {:.2} above threshold {:.2}",
                    validation_loss, self.validation_loss_threshold
                )
            };

            // Roll back the failed adapter artifact.
            Self::remove_adapter_artifact(&result.adapter_path).await;

            if let Some(repo) = &job_repo {
                let job = TrainingJob {
                    id: job_id,
                    task_kind: key.to_string(),
                    status: TrainingJobStatus::RolledBack,
                    started_at: Utc::now().timestamp_millis(),
                    finished_at: Some(Utc::now().timestamp_millis()),
                    adapter_id: None,
                    metrics: serde_json::to_value(&result.metrics).unwrap_or_default(),
                    error_message: Some(reason.clone()),
                };
                let _ = repo.update_training_job(&job).await;
            }

            return Err(LoraEvolutionError::ValidationFailed(
                key.to_string(),
                reason,
            ));
        }

        let existing = if agent_role.is_some() {
            self.lora_adapter_repo
                .list_lora_adapters_by_role(key)
                .await?
        } else {
            self.lora_adapter_repo
                .list_lora_adapters_by_kind(key)
                .await?
        };
        let adapter_id = self.next_adapter_id(key, &existing);
        let adapter_path = result.adapter_path.to_string_lossy().to_string();
        let baseline_adapter_id = existing
            .iter()
            .find(|adapter| adapter.active)
            .or_else(|| existing.first())
            .map(|adapter| adapter.id.clone());
        let mut metrics = serde_json::to_value(&result.metrics).unwrap_or_default();
        let adapter_metadata = AdapterMetadata {
            role: result.metadata.role.clone().or(agent_role.clone()),
            base_model: if result.metadata.base_model == "unknown" {
                self.base_model.clone()
            } else {
                result.metadata.base_model.clone()
            },
            objective: result.metadata.objective.clone(),
            dataset_hash: result.metadata.dataset_hash.clone(),
        };
        if let Some(metrics_object) = metrics.as_object_mut() {
            metrics_object.insert(
                "adapter_metadata".into(),
                serde_json::to_value(&adapter_metadata).unwrap_or_default(),
            );
        }

        if let Some(gate) = &self.benchmark_gate {
            let decision = gate
                .evaluate(LoraBenchmarkRequest {
                    task_kind: key.to_string(),
                    agent_role: agent_role.clone(),
                    baseline_adapter_id,
                    challenger_adapter_id: adapter_id.clone(),
                    challenger_adapter_path: result.adapter_path.clone(),
                    base_model: self.base_model.clone(),
                    challenger_metrics: metrics.clone(),
                    validation_reward,
                    training_fingerprints: Self::training_fingerprints(&examples),
                })
                .await?;
            let gate_metadata = serde_json::json!({
                "accepted": decision.accepted,
                "reason": decision.reason.clone(),
                "metadata": decision.metadata.clone(),
                "quality_gates": decision.quality_gates.clone(),
            });
            let mut benchmark_gate = gate_metadata.clone();
            if let (Some(target), Some(source)) = (
                benchmark_gate.as_object_mut(),
                gate_metadata["metadata"].as_object(),
            ) {
                for (key, value) in source {
                    target.insert(key.clone(), value.clone());
                }
            }
            if let Some(metrics_object) = metrics.as_object_mut() {
                metrics_object.insert("benchmark_gate".into(), benchmark_gate);
            }

            if !decision.accepted || !decision.all_required_quality_gates_passed() {
                let missing = decision
                    .missing_required_quality_gates()
                    .into_iter()
                    .map(|gate| gate.as_str().to_string())
                    .collect::<Vec<_>>();
                let reason = if missing.is_empty() {
                    format!("benchmark gate rejected challenger: {}", decision.reason)
                } else {
                    format!(
                        "benchmark gate rejected challenger: {}; missing or failed quality gates: {}",
                        decision.reason,
                        missing.join(",")
                    )
                };
                Self::remove_adapter_artifact(&result.adapter_path).await;

                if let Some(repo) = &job_repo {
                    let job = TrainingJob {
                        id: job_id.clone(),
                        task_kind: key.to_string(),
                        status: TrainingJobStatus::RolledBack,
                        started_at: Utc::now().timestamp_millis(),
                        finished_at: Some(Utc::now().timestamp_millis()),
                        adapter_id: None,
                        metrics: metrics.clone(),
                        error_message: Some(reason.clone()),
                    };
                    let _ = repo.update_training_job(&job).await;
                }
                self.publish_evolution_observed(
                    "lora_evolution_rejected",
                    key,
                    &examples,
                    &job_id,
                    Some(&adapter_id),
                    &metrics,
                )
                .await;

                return Err(LoraEvolutionError::ValidationFailed(
                    key.to_string(),
                    reason,
                ));
            }
        }

        let adapter = LoraAdapter {
            id: adapter_id.clone(),
            project_id: None,
            name: adapter_id.clone(),
            file_path: adapter_path.clone(),
            base_model: self.base_model.clone(),
            task_kind: Some(key.to_string()),
            agent_role: agent_role.clone(),
            metrics,
            created_at: Utc::now().timestamp_millis(),
            active: true,
        };

        for previous in existing.iter().filter(|adapter| adapter.active) {
            self.lora_adapter_repo
                .set_lora_adapter_active(&previous.id, false)
                .await?;
        }
        self.lora_adapter_repo.insert_lora_adapter(&adapter).await?;

        self.inference_service
            .register_lora(InferenceLoRAAdapter {
                id: adapter_id.clone(),
                path: adapter_path,
                base_model: self.base_model.clone(),
            })
            .await
            .map_err(|e| LoraEvolutionError::Inference(e.to_string()))?;

        self.index_adapter(&adapter, train_examples)
            .await
            .map_err(|e| LoraEvolutionError::Index(e.to_string()))?;

        if let Some(repo) = &job_repo {
            let job = TrainingJob {
                id: job_id.clone(),
                task_kind: key.to_string(),
                status: TrainingJobStatus::Promoted,
                started_at: Utc::now().timestamp_millis(),
                finished_at: Some(Utc::now().timestamp_millis()),
                adapter_id: Some(adapter_id.clone()),
                metrics: adapter.metrics.clone(),
                error_message: None,
            };
            let _ = repo.update_training_job(&job).await;
        }

        self.publish_evolution_observed(
            "lora_evolution_promoted",
            key,
            &examples,
            &job_id,
            Some(&adapter_id),
            &adapter.metrics,
        )
        .await;

        self.event_service.publish(Event::LoraSwapped {
            project_id: String::new(),
            lora_id: adapter_id.clone(),
        });

        Ok(adapter)
    }

    async fn publish_evolution_observed(
        &self,
        action: &str,
        task_kind: &str,
        examples: &[TrainingExample],
        training_job_id: &str,
        adapter_id: Option<&str>,
        metrics: &serde_json::Value,
    ) {
        let project_id = examples
            .iter()
            .find_map(|example| example.project_id.clone())
            .unwrap_or_default();
        let triggering_task_id = examples.last().map(|example| example.task_id.clone());
        let triggering_task = match triggering_task_id.as_deref() {
            Some(task_id) => self.task_service.get(task_id).await.ok().flatten(),
            None => None,
        };
        let trace_id = triggering_task
            .as_ref()
            .map(|task| task.trace_id.clone())
            .unwrap_or_default();
        let run_id = triggering_task
            .as_ref()
            .and_then(|task| task.result.as_ref())
            .and_then(|result| result.get("run_id"))
            .and_then(serde_json::Value::as_str)
            .map(ToOwned::to_owned);
        let mut metadata = serde_json::json!({
            "training_job_id": training_job_id,
            "task_kind": task_kind,
            "adapter_id": adapter_id,
            "trace_id": trace_id,
            "run_id": run_id,
            "triggering_task_id": triggering_task_id,
            "training_example_count": examples.len(),
            "metrics": metrics,
        });
        if let Some(benchmark_gate) = metrics.get("benchmark_gate")
            && let Some(object) = metadata.as_object_mut()
        {
            object.insert("benchmark_gate".into(), benchmark_gate.clone());
        }

        self.event_service.publish(Event::RunObserved {
            project_id,
            task_id: triggering_task_id,
            trace_id,
            action: action.to_string(),
            metadata,
        });
    }
}

#[async_trait]
impl LoraEvolutionService for LoraEvolutionServiceImpl {
    async fn collect_golden_example(&self, task_id: &str) -> Result<(), LoraEvolutionError> {
        self.collect_example(task_id, false).await
    }

    async fn collect_counter_example(&self, task_id: &str) -> Result<(), LoraEvolutionError> {
        self.collect_example(task_id, true).await
    }

    async fn should_train(&self, task_kind: &str) -> Result<bool, LoraEvolutionError> {
        let count = self
            .training_example_repo
            .list_training_examples_by_kind(task_kind)
            .await?
            .iter()
            .filter(|example| self.is_golden_training_example(example))
            .count();
        Ok(count >= self.threshold)
    }

    async fn should_train_for_role(&self, role: AgentRole) -> Result<bool, LoraEvolutionError> {
        let count = self
            .training_example_repo
            .list_training_examples_by_role(role.as_str())
            .await?
            .iter()
            .filter(|example| self.is_golden_training_example(example))
            .count();
        Ok(count >= self.threshold)
    }

    async fn train_and_register(&self, task_kind: &str) -> Result<LoraAdapter, LoraEvolutionError> {
        if !self.should_train(task_kind).await? {
            return Err(LoraEvolutionError::ValidationFailed(
                task_kind.to_string(),
                "not enough golden examples".to_string(),
            ));
        }

        let examples = self
            .training_example_repo
            .list_training_examples_by_kind(task_kind)
            .await?;
        self.train_and_register_with_examples(examples, task_kind, None, LoraTrainingObjective::Sft)
            .await
    }

    async fn train_and_register_for_role(
        &self,
        role: AgentRole,
    ) -> Result<LoraAdapter, LoraEvolutionError> {
        let role_str = role.as_str();
        if !self.should_train_for_role(role).await? {
            return Err(LoraEvolutionError::ValidationFailed(
                role_str.to_string(),
                "not enough golden examples".to_string(),
            ));
        }

        let examples = self
            .training_example_repo
            .list_training_examples_by_role(role_str)
            .await?;
        self.train_and_register_with_examples(
            examples,
            role_str,
            Some(role_str.to_string()),
            LoraTrainingObjective::Sft,
        )
        .await
    }

    async fn train_and_register_for_role_objective(
        &self,
        role: AgentRole,
        objective: LoraTrainingObjective,
    ) -> Result<LoraAdapter, LoraEvolutionError> {
        let role_str = role.as_str();
        let examples = self
            .training_example_repo
            .list_training_examples_by_role(role_str)
            .await?;
        self.train_and_register_with_examples(
            examples,
            role_str,
            Some(role_str.to_string()),
            objective,
        )
        .await
    }

    async fn select_lora(
        &self,
        task: &Task,
        project_id: &str,
    ) -> Result<Option<String>, LoraEvolutionError> {
        if let Some(role) = task
            .assigned_agent
            .as_deref()
            .and_then(AgentRole::from_agent)
            && let Some(id) = self.select_lora_by_role(role, project_id).await?
        {
            return Ok(Some(id));
        }

        let adapters = self
            .lora_adapter_repo
            .list_lora_adapters_by_kind(&task.kind)
            .await?;
        // Prefer an active adapter, otherwise the newest one.
        let active = adapters.iter().find(|a| a.active).cloned();
        let chosen = active.or_else(|| adapters.into_iter().next());
        Ok(chosen.map(|a| a.id))
    }

    async fn select_lora_by_role(
        &self,
        role: AgentRole,
        _project_id: &str,
    ) -> Result<Option<String>, LoraEvolutionError> {
        let adapters = self
            .lora_adapter_repo
            .list_lora_adapters_by_role(role.as_str())
            .await?;
        // Prefer an active adapter, otherwise the newest one.
        let active = adapters.iter().find(|a| a.active).cloned();
        let chosen = active.or_else(|| adapters.into_iter().next());
        Ok(chosen.map(|a| a.id))
    }

    async fn build_dataset_for_role(
        &self,
        role: &str,
    ) -> Result<LoraDatasetReport, LoraEvolutionError> {
        self.insert_role_dataset_pairs(role).await?;
        self.dataset_stats_for_role(role).await
    }

    async fn inspect_dataset_for_role(
        &self,
        role: &str,
    ) -> Result<LoraDatasetReport, LoraEvolutionError> {
        self.dataset_stats_for_role(role).await
    }

    async fn dataset_stats_for_role(
        &self,
        role: &str,
    ) -> Result<LoraDatasetReport, LoraEvolutionError> {
        let examples = self.role_dataset_examples(role).await?;
        Ok(Self::build_dataset_report(role, examples))
    }
}

fn task_role(task: &Task) -> Option<String> {
    task.assigned_agent
        .as_deref()
        .and_then(AgentRole::from_agent)
        .map(|role| role.as_str().to_string())
        .or_else(|| task.assigned_agent.clone())
}

fn task_model_id(task: &Task) -> Option<String> {
    task.payload
        .get("model_id")
        .and_then(serde_json::Value::as_str)
        .map(ToString::to_string)
}

fn task_rag_evidence_ids(task: &Task) -> Vec<String> {
    task.payload
        .get("rag_evidence_ids")
        .and_then(serde_json::Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(serde_json::Value::as_str)
                .map(ToString::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn task_critic_feedback(task: &Task) -> Option<String> {
    task.payload
        .get("critic_feedback")
        .and_then(serde_json::Value::as_str)
        .map(ToString::to_string)
}

fn task_failure_type(task: &Task) -> Option<String> {
    task.payload
        .get("failure_type")
        .and_then(serde_json::Value::as_str)
        .map(ToString::to_string)
}

fn failure_type_counts(examples: &[TrainingExample]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for failure_type in examples
        .iter()
        .filter_map(|example| example.failure_type.as_deref())
    {
        *counts.entry(failure_type.to_string()).or_insert(0) += 1;
    }
    counts
}

fn low_information_report(examples: &[TrainingExample]) -> LoraDatasetLowInformationReport {
    let example_ids = examples
        .iter()
        .filter(|example| {
            example.input_text.trim().chars().count() < MIN_EXAMPLE_TEXT_CHARS
                || example
                    .accepted_output
                    .as_deref()
                    .unwrap_or(&example.output_text)
                    .trim()
                    .chars()
                    .count()
                    < MIN_EXAMPLE_TEXT_CHARS
                || example
                    .rejected_output
                    .as_deref()
                    .is_some_and(|text| text.trim().chars().count() < MIN_EXAMPLE_TEXT_CHARS)
        })
        .map(|example| example.id.clone())
        .collect::<Vec<_>>();
    LoraDatasetLowInformationReport {
        filtered_count: example_ids.len(),
        example_ids,
    }
}

fn leakage_report(examples: &[TrainingExample]) -> LoraDatasetLeakageReport {
    let duplicate_task_count =
        duplicate_count(examples.iter().map(|example| example.task_id.as_str()));
    let duplicate_output_count = duplicate_count(examples.iter().map(|example| {
        example
            .accepted_output
            .as_deref()
            .unwrap_or(&example.output_text)
    }));
    LoraDatasetLeakageReport {
        duplicate_task_count,
        duplicate_output_count,
    }
}

fn duplicate_count<'a>(items: impl Iterator<Item = &'a str>) -> usize {
    let mut seen = HashSet::new();
    let mut duplicates = 0;
    for item in items {
        if !seen.insert(item) {
            duplicates += 1;
        }
    }
    duplicates
}

fn balancing_report(
    examples: &[TrainingExample],
    counts: &BTreeMap<String, usize>,
) -> LoraDatasetBalancingReport {
    let target = counts.values().copied().min().unwrap_or(0).min(1);
    let target_per_failure_type = counts
        .keys()
        .map(|failure_type| (failure_type.clone(), target))
        .collect::<BTreeMap<_, _>>();
    let mut selected_by_failure_type: HashMap<&str, usize> = HashMap::new();
    let selected_example_ids = examples
        .iter()
        .filter(|example| {
            example.failure_type.as_deref().is_some_and(|failure_type| {
                let count = selected_by_failure_type.entry(failure_type).or_insert(0);
                if *count < target {
                    *count += 1;
                    true
                } else {
                    false
                }
            })
        })
        .map(|example| example.id.clone())
        .collect();
    LoraDatasetBalancingReport {
        target_per_failure_type,
        selected_example_ids,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{PromptVersion, TaskStatus};
    use crate::persistence::{
        ExperienceRepository, LoraAdapterRepository, PersistenceError, PromptVersionRepository,
        TrainingExampleRepository, TrainingJobRepository,
    };
    use crate::services::{
        CreateTaskRequest, EventService, InferenceService, InferenceServiceError, LoraMetrics,
        LoraTrainer, LoraTrainingConfig, LoraTrainingResult, MockEmbedder, TaskError, TaskService,
        vector_store::{SearchOptions, VectorPoint, VectorStore, VectorStoreError},
    };
    use async_trait::async_trait;
    use crytex_inference::{
        BackendInfo, InferenceRequest, InferenceResponse, LoRAAdapter, ModelInfo,
    };
    use serde_json::json;
    use std::collections::HashMap;
    use std::path::Path;
    use std::sync::Mutex;

    fn make_task(
        project_id: &str,
        kind: &str,
        status: TaskStatus,
        human_score: Option<f64>,
    ) -> Task {
        Task {
            id: "t1".into(),
            project_id: project_id.into(),
            parent_id: None,
            title: "title".into(),
            description: Some("desc".into()),
            kind: kind.into(),
            status,
            assigned_agent: None,
            priority: 0,
            created_at: 0,
            started_at: None,
            finished_at: None,
            payload: json!({}),
            result: Some(json!("fn x() {}")),
            iteration_count: 0,
            priority_score: 0.0,
            critic_score: Some(4.0),
            human_score,
            prompt_version_id: Some("pv1".into()),
            lora_adapter_id: None,
            trace_id: "trace".into(),
        }
    }

    struct DummyTaskService {
        tasks: Mutex<HashMap<String, Task>>,
    }

    impl DummyTaskService {
        fn with_task(task: Task) -> Self {
            Self::with_tasks(vec![task])
        }

        fn with_tasks(tasks: Vec<Task>) -> Self {
            let mut task_map = HashMap::new();
            for task in tasks {
                task_map.insert(task.id.clone(), task);
            }
            Self {
                tasks: Mutex::new(task_map),
            }
        }
    }

    #[async_trait]
    impl TaskService for DummyTaskService {
        async fn submit(&self, _request: CreateTaskRequest) -> Result<Task, TaskError> {
            unimplemented!()
        }
        async fn add_dependency(
            &self,
            _dep: crate::models::TaskDependency,
        ) -> Result<(), TaskError> {
            unimplemented!()
        }
        async fn get(&self, id: &str) -> Result<Option<Task>, TaskError> {
            Ok(self.tasks.lock().unwrap().get(id).cloned())
        }
        async fn list_by_project(&self, _project_id: &str) -> Result<Vec<Task>, TaskError> {
            unimplemented!()
        }
        async fn list_ready(&self) -> Result<Vec<Task>, TaskError> {
            unimplemented!()
        }
        async fn set_status(&self, _id: &str, _status: TaskStatus) -> Result<Task, TaskError> {
            unimplemented!()
        }
        async fn cancel(&self, _id: &str) -> Result<Task, TaskError> {
            unimplemented!()
        }
        async fn set_result(
            &self,
            _id: &str,
            _result: serde_json::Value,
        ) -> Result<Task, TaskError> {
            unimplemented!()
        }
        async fn set_critic_score(&self, _id: &str, _score: f64) -> Result<Task, TaskError> {
            unimplemented!()
        }
        async fn set_human_score(&self, _id: &str, _score: f64) -> Result<Task, TaskError> {
            unimplemented!()
        }
        async fn retry(&self, _id: &str, _feedback: Option<&str>) -> Result<Task, TaskError> {
            unimplemented!()
        }
        async fn load_all_tasks(&self) -> Result<Vec<Task>, TaskError> {
            Ok(self.tasks.lock().unwrap().values().cloned().collect())
        }
        async fn update_task(&self, _task: &Task) -> Result<(), TaskError> {
            unimplemented!()
        }
    }

    struct DummyPromptVersionRepo {
        versions: Mutex<HashMap<String, PromptVersion>>,
    }

    impl Default for DummyPromptVersionRepo {
        fn default() -> Self {
            let mut versions = HashMap::new();
            versions.insert(
                "pv1".into(),
                PromptVersion {
                    id: "pv1".into(),
                    agent: "coder".into(),
                    project_id: None,
                    system_prompt: "You are a coder.".into(),
                    fitness: None,
                    parent_id: None,
                    metrics: serde_json::Value::Null,
                    created_at: 0,
                    active: true,
                },
            );
            Self {
                versions: Mutex::new(versions),
            }
        }
    }

    #[async_trait]
    impl PromptVersionRepository for DummyPromptVersionRepo {
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
            id: &str,
        ) -> Result<Option<PromptVersion>, PersistenceError> {
            Ok(self.versions.lock().unwrap().get(id).cloned())
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

    #[derive(Default)]
    struct InMemoryTrainingExampleRepo {
        examples: Mutex<Vec<TrainingExample>>,
    }

    #[async_trait]
    impl TrainingExampleRepository for InMemoryTrainingExampleRepo {
        async fn insert_training_example(
            &self,
            example: &TrainingExample,
        ) -> Result<(), PersistenceError> {
            self.examples.lock().unwrap().push(example.clone());
            Ok(())
        }
        async fn list_training_examples_by_kind(
            &self,
            task_kind: &str,
        ) -> Result<Vec<TrainingExample>, PersistenceError> {
            Ok(self
                .examples
                .lock()
                .unwrap()
                .iter()
                .filter(|e| e.task_kind == task_kind)
                .cloned()
                .collect())
        }
        async fn count_training_examples_by_kind(
            &self,
            task_kind: &str,
        ) -> Result<usize, PersistenceError> {
            Ok(self
                .examples
                .lock()
                .unwrap()
                .iter()
                .filter(|e| e.task_kind == task_kind)
                .count())
        }
        async fn list_training_examples_by_project(
            &self,
            _project_id: &str,
        ) -> Result<Vec<TrainingExample>, PersistenceError> {
            Ok(vec![])
        }
        async fn list_training_examples_by_role(
            &self,
            agent_role: &str,
        ) -> Result<Vec<TrainingExample>, PersistenceError> {
            Ok(self
                .examples
                .lock()
                .unwrap()
                .iter()
                .filter(|e| e.agent_role.as_deref() == Some(agent_role))
                .cloned()
                .collect())
        }
        async fn count_training_examples_by_role(
            &self,
            agent_role: &str,
        ) -> Result<usize, PersistenceError> {
            Ok(self
                .examples
                .lock()
                .unwrap()
                .iter()
                .filter(|e| e.agent_role.as_deref() == Some(agent_role))
                .count())
        }
    }

    #[derive(Default)]
    struct InMemoryExperienceRepo {
        experiences: Mutex<Vec<Experience>>,
    }

    #[async_trait]
    impl ExperienceRepository for InMemoryExperienceRepo {
        async fn insert_experience(&self, exp: &Experience) -> Result<(), PersistenceError> {
            self.experiences.lock().unwrap().push(exp.clone());
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

    #[derive(Default)]
    struct InMemoryLoraAdapterRepo {
        adapters: Mutex<Vec<LoraAdapter>>,
    }

    #[async_trait]
    impl LoraAdapterRepository for InMemoryLoraAdapterRepo {
        async fn insert_lora_adapter(&self, adapter: &LoraAdapter) -> Result<(), PersistenceError> {
            self.adapters.lock().unwrap().push(adapter.clone());
            Ok(())
        }
        async fn get_lora_adapter(
            &self,
            id: &str,
        ) -> Result<Option<LoraAdapter>, PersistenceError> {
            Ok(self
                .adapters
                .lock()
                .unwrap()
                .iter()
                .find(|a| a.id == id)
                .cloned())
        }
        async fn list_lora_adapters_by_kind(
            &self,
            task_kind: &str,
        ) -> Result<Vec<LoraAdapter>, PersistenceError> {
            Ok(self
                .adapters
                .lock()
                .unwrap()
                .iter()
                .filter(|a| a.task_kind.as_deref() == Some(task_kind))
                .cloned()
                .collect())
        }
        async fn list_lora_adapters_by_project(
            &self,
            _project_id: &str,
        ) -> Result<Vec<LoraAdapter>, PersistenceError> {
            Ok(vec![])
        }
        async fn list_lora_adapters_by_role(
            &self,
            agent_role: &str,
        ) -> Result<Vec<LoraAdapter>, PersistenceError> {
            Ok(self
                .adapters
                .lock()
                .unwrap()
                .iter()
                .filter(|a| a.agent_role.as_deref() == Some(agent_role))
                .cloned()
                .collect())
        }
        async fn set_lora_adapter_active(
            &self,
            id: &str,
            active: bool,
        ) -> Result<(), PersistenceError> {
            if let Some(adapter) = self
                .adapters
                .lock()
                .unwrap()
                .iter_mut()
                .find(|adapter| adapter.id == id)
            {
                adapter.active = active;
            }
            Ok(())
        }
    }

    struct RejectingBenchmarkGate;

    #[async_trait]
    impl LoraBenchmarkGate for RejectingBenchmarkGate {
        async fn evaluate(
            &self,
            _request: LoraBenchmarkRequest,
        ) -> Result<LoraBenchmarkDecision, LoraEvolutionError> {
            Ok(LoraBenchmarkDecision::reject(
                "baseline kept: challenger regressed benchmark pass rate",
            ))
        }
    }

    #[derive(Default)]
    struct RecordingBenchmarkGate {
        requests: Mutex<Vec<LoraBenchmarkRequest>>,
    }

    #[async_trait]
    impl LoraBenchmarkGate for RecordingBenchmarkGate {
        async fn evaluate(
            &self,
            request: LoraBenchmarkRequest,
        ) -> Result<LoraBenchmarkDecision, LoraEvolutionError> {
            self.requests.lock().unwrap().push(request);
            Ok(LoraBenchmarkDecision::accept_with_quality_gates(
                "challenger improved all quality gates",
                passed_lora_quality_gates("deterministic test gate passed"),
                serde_json::json!({
                    "positive_benchmark": {"passed": true},
                    "negative_benchmark": {"passed": true},
                    "regression_benchmark": {"passed": true},
                    "safety_benchmark": {"passed": true},
                    "runtime_application": {"passed": true},
                    "output_changed": {"passed": true}
                }),
            ))
        }
    }

    struct IncompleteAcceptingBenchmarkGate;

    #[async_trait]
    impl LoraBenchmarkGate for IncompleteAcceptingBenchmarkGate {
        async fn evaluate(
            &self,
            _request: LoraBenchmarkRequest,
        ) -> Result<LoraBenchmarkDecision, LoraEvolutionError> {
            Ok(LoraBenchmarkDecision {
                accepted: true,
                reason: "legacy accepted flag without P10 evidence".into(),
                metadata: serde_json::json!({ "legacy": true }),
                quality_gates: vec![lora_quality_gate(
                    LoraQualityGateName::PositiveBenchmark,
                    true,
                    "only positive benchmark passed",
                )],
            })
        }
    }

    #[derive(Default)]
    struct DummyInferenceService {
        registered: Mutex<Vec<String>>,
    }

    #[async_trait]
    impl InferenceService for DummyInferenceService {
        async fn generate(
            &self,
            _request: InferenceRequest,
        ) -> Result<InferenceResponse, InferenceServiceError> {
            unimplemented!()
        }
        async fn embed(&self, _text: &str) -> Result<Vec<f32>, InferenceServiceError> {
            unimplemented!()
        }
        fn available_backends(&self) -> Vec<BackendInfo> {
            vec![]
        }
        async fn register_lora(&self, lora: LoRAAdapter) -> Result<(), InferenceServiceError> {
            self.registered.lock().unwrap().push(lora.id);
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

    #[derive(Default)]
    struct InMemoryTrainingJobRepo {
        jobs: Mutex<Vec<TrainingJob>>,
    }

    #[async_trait]
    impl TrainingJobRepository for InMemoryTrainingJobRepo {
        async fn insert_training_job(&self, job: &TrainingJob) -> Result<(), PersistenceError> {
            self.jobs.lock().unwrap().push(job.clone());
            Ok(())
        }
        async fn update_training_job(&self, job: &TrainingJob) -> Result<(), PersistenceError> {
            let mut jobs = self.jobs.lock().unwrap();
            if let Some(existing) = jobs.iter_mut().find(|j| j.id == job.id) {
                *existing = job.clone();
            } else {
                jobs.push(job.clone());
            }
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
            task_kind: &str,
        ) -> Result<Vec<TrainingJob>, PersistenceError> {
            Ok(self
                .jobs
                .lock()
                .unwrap()
                .iter()
                .filter(|j| j.task_kind == task_kind)
                .cloned()
                .collect())
        }
    }

    #[derive(Default)]
    struct DummyEventService {
        events: Mutex<Vec<Event>>,
    }

    #[async_trait]
    impl EventService for DummyEventService {
        fn publish(&self, event: Event) {
            self.events.lock().unwrap().push(event);
        }
        fn subscribe(&self) -> tokio::sync::broadcast::Receiver<Event> {
            let (tx, _rx) = tokio::sync::broadcast::channel(1);
            tx.subscribe()
        }
        async fn start_handler(&self, _handler: Arc<dyn crate::services::EventHandler>) {}
    }

    struct MockTrainer {
        adapter_bytes: usize,
        train_loss: f64,
        validation_loss: f64,
        single_file_layout: bool,
    }

    impl MockTrainer {
        fn new() -> Self {
            Self {
                adapter_bytes: 5,
                train_loss: 0.1,
                validation_loss: 0.2,
                single_file_layout: false,
            }
        }

        fn with_adapter_bytes(mut self, bytes: usize) -> Self {
            self.adapter_bytes = bytes;
            self
        }

        fn with_losses(mut self, train_loss: f64, validation_loss: f64) -> Self {
            self.train_loss = train_loss;
            self.validation_loss = validation_loss;
            self
        }

        fn with_single_file_layout(mut self) -> Self {
            self.single_file_layout = true;
            self
        }
    }

    #[async_trait]
    impl LoraTrainer for MockTrainer {
        fn backend_name(&self) -> &'static str {
            "mock"
        }

        fn supports_objective(&self, _objective: &LoraTrainingObjective) -> bool {
            true
        }

        async fn train(
            &self,
            examples: Vec<TrainingExample>,
            config: LoraTrainingConfig,
            output_dir: &Path,
        ) -> Result<LoraTrainingResult, LoraTrainingError> {
            validate_objective_examples(&config.objective, &examples)?;
            tokio::fs::create_dir_all(output_dir).await?;
            let average_reward =
                examples.iter().map(|e| e.reward).sum::<f64>() / examples.len() as f64;
            let metadata = AdapterMetadata::from_examples(&config, &examples);
            let adapter_path = if self.single_file_layout {
                let adapter_path = output_dir.join("adapter.safetensors");
                tokio::fs::write(&adapter_path, vec![b'x'; self.adapter_bytes]).await?;
                adapter_path
            } else {
                let adapter_path = output_dir.join("adapter");
                tokio::fs::create_dir_all(&adapter_path).await?;
                tokio::fs::write(
                    adapter_path.join("adapter_config.json"),
                    serde_json::json!({
                        "peft_type": "LORA",
                        "base_model_name_or_path": "mistral-7b",
                        "r": 16,
                        "lora_alpha": 32,
                        "target_modules": ["q_proj", "v_proj"]
                    })
                    .to_string(),
                )
                .await?;
                tokio::fs::write(
                    adapter_path.join("adapter_model.safetensors"),
                    vec![b'x'; self.adapter_bytes],
                )
                .await?;
                tokio::fs::write(
                    adapter_path.join("adapter_metadata.json"),
                    serde_json::to_vec_pretty(&metadata)
                        .map_err(|error| LoraTrainingError::Backend(error.to_string()))?,
                )
                .await?;
                adapter_path
            };
            Ok(LoraTrainingResult {
                adapter_id: "mock-adapter".into(),
                adapter_path,
                metrics: LoraMetrics {
                    train_loss: self.train_loss,
                    validation_loss: self.validation_loss,
                    average_reward,
                },
                metadata,
            })
        }
    }

    struct RecordingTrainer {
        inner: MockTrainer,
        trained_examples: Mutex<Vec<TrainingExample>>,
        configs: Mutex<Vec<LoraTrainingConfig>>,
    }

    impl RecordingTrainer {
        fn new(inner: MockTrainer) -> Self {
            Self {
                inner,
                trained_examples: Mutex::new(Vec::new()),
                configs: Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait]
    impl LoraTrainer for RecordingTrainer {
        fn backend_name(&self) -> &'static str {
            self.inner.backend_name()
        }

        fn supports_objective(&self, objective: &LoraTrainingObjective) -> bool {
            self.inner.supports_objective(objective)
        }

        async fn train(
            &self,
            examples: Vec<TrainingExample>,
            config: LoraTrainingConfig,
            output_dir: &Path,
        ) -> Result<LoraTrainingResult, LoraTrainingError> {
            *self.trained_examples.lock().unwrap() = examples.clone();
            self.configs.lock().unwrap().push(config.clone());
            self.inner.train(examples, config, output_dir).await
        }
    }

    fn evolution_service(
        task: Task,
        examples: Vec<TrainingExample>,
        adapters: Vec<LoraAdapter>,
    ) -> (
        LoraEvolutionServiceImpl,
        Arc<InMemoryTrainingExampleRepo>,
        Arc<InMemoryLoraAdapterRepo>,
        Arc<DummyInferenceService>,
        Arc<DummyEventService>,
    ) {
        let task_service = Arc::new(DummyTaskService::with_task(task));
        let prompt_repo: Arc<dyn PromptVersionRepository> =
            Arc::new(DummyPromptVersionRepo::default());
        let example_repo = Arc::new(InMemoryTrainingExampleRepo::default());
        for e in examples {
            example_repo.examples.lock().unwrap().push(e);
        }
        let adapter_repo = Arc::new(InMemoryLoraAdapterRepo::default());
        for a in adapters {
            adapter_repo.adapters.lock().unwrap().push(a);
        }
        let inference = Arc::new(DummyInferenceService::default());
        let events = Arc::new(DummyEventService::default());
        let trainer: Arc<dyn LoraTrainer> = Arc::new(MockTrainer::new());

        let service = LoraEvolutionServiceImpl::new(
            task_service,
            prompt_repo,
            example_repo.clone(),
            adapter_repo.clone(),
            inference.clone(),
            events.clone(),
            trainer,
            std::env::temp_dir().join(format!("crytex-test-adapters-{}", Ulid::new())),
            "mistral-7b".into(),
        )
        .with_threshold(2)
        .with_validation_reward_threshold(3.0)
        .with_min_human_score(4.0);

        (service, example_repo, adapter_repo, inference, events)
    }

    fn evolution_service_with_tasks(
        tasks: Vec<Task>,
        examples: Vec<TrainingExample>,
        adapters: Vec<LoraAdapter>,
    ) -> (
        LoraEvolutionServiceImpl,
        Arc<InMemoryTrainingExampleRepo>,
        Arc<InMemoryLoraAdapterRepo>,
        Arc<DummyInferenceService>,
        Arc<DummyEventService>,
    ) {
        let first_task = tasks
            .first()
            .cloned()
            .unwrap_or_else(|| make_task("p1", "codegen", TaskStatus::Completed, Some(5.0)));
        let (mut service, example_repo, adapter_repo, inference, events) =
            evolution_service(first_task, examples, adapters);
        service.task_service = Arc::new(DummyTaskService::with_tasks(tasks));
        (service, example_repo, adapter_repo, inference, events)
    }

    fn evolution_service_with_trainer(
        task: Task,
        examples: Vec<TrainingExample>,
        trainer: Arc<dyn LoraTrainer>,
    ) -> (
        LoraEvolutionServiceImpl,
        Arc<InMemoryTrainingExampleRepo>,
        Arc<InMemoryLoraAdapterRepo>,
        Arc<DummyInferenceService>,
    ) {
        let task_service = Arc::new(DummyTaskService::with_task(task));
        let prompt_repo: Arc<dyn PromptVersionRepository> =
            Arc::new(DummyPromptVersionRepo::default());
        let example_repo = Arc::new(InMemoryTrainingExampleRepo::default());
        for e in examples {
            example_repo.examples.lock().unwrap().push(e);
        }
        let adapter_repo = Arc::new(InMemoryLoraAdapterRepo::default());
        let inference = Arc::new(DummyInferenceService::default());
        let events = Arc::new(DummyEventService::default());

        let service = LoraEvolutionServiceImpl::new(
            task_service,
            prompt_repo,
            example_repo.clone(),
            adapter_repo.clone(),
            inference.clone(),
            events,
            trainer,
            std::env::temp_dir().join(format!("crytex-test-adapters-{}", Ulid::new())),
            "mistral-7b".into(),
        )
        .with_threshold(2)
        .with_validation_reward_threshold(3.0)
        .with_min_human_score(4.0);

        (service, example_repo, adapter_repo, inference)
    }

    fn example(kind: &str, reward: f64, created_at: i64) -> TrainingExample {
        TrainingExample {
            id: Ulid::new().to_string(),
            task_id: format!("task-{created_at}"),
            project_id: Some("p1".into()),
            prompt_version_id: Some("pv1".into()),
            task_kind: kind.into(),
            agent_role: None,
            model_id: None,
            rag_evidence_ids: Vec::new(),
            input_text: "input".into(),
            output_text: "output".into(),
            accepted_output: Some("output".into()),
            rejected_output: None,
            critic_feedback: None,
            failure_type: None,
            reward,
            created_at,
        }
    }

    type CollectionMap = HashMap<String, (usize, HashMap<String, VectorPoint>)>;

    #[derive(Default)]
    struct TestVectorStore {
        collections: Mutex<CollectionMap>,
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }

    #[async_trait]
    impl VectorStore for TestVectorStore {
        async fn create_collection(
            &self,
            collection: &str,
            dim: usize,
        ) -> Result<(), VectorStoreError> {
            self.collections
                .lock()
                .unwrap()
                .entry(collection.to_string())
                .or_insert((dim, HashMap::new()));
            Ok(())
        }
        async fn delete_collection(&self, collection: &str) -> Result<(), VectorStoreError> {
            self.collections.lock().unwrap().remove(collection);
            Ok(())
        }
        async fn upsert(
            &self,
            collection: &str,
            points: Vec<VectorPoint>,
        ) -> Result<(), VectorStoreError> {
            let mut collections = self.collections.lock().unwrap();
            let entry = collections.get_mut(collection).ok_or_else(|| {
                VectorStoreError::Collection(format!("collection {} does not exist", collection))
            })?;
            for point in points {
                if point.vector.len() != entry.0 {
                    return Err(VectorStoreError::DimensionMismatch {
                        expected: entry.0,
                        actual: point.vector.len(),
                    });
                }
                entry.1.insert(point.id.clone(), point);
            }
            Ok(())
        }
        async fn search(
            &self,
            collection: &str,
            vector: &[f32],
            options: SearchOptions,
        ) -> Result<Vec<crate::services::vector_store::SearchResult>, VectorStoreError> {
            let collections = self.collections.lock().unwrap();
            let entry = collections.get(collection).ok_or_else(|| {
                VectorStoreError::Collection(format!("collection {} does not exist", collection))
            })?;
            let mut results: Vec<_> = entry
                .1
                .values()
                .map(|point| crate::services::vector_store::SearchResult {
                    id: point.id.clone(),
                    score: cosine_similarity(vector, &point.vector),
                    payload: point.payload.clone(),
                })
                .filter(|result| {
                    options
                        .score_threshold
                        .is_none_or(|threshold| result.score >= threshold)
                })
                .collect();
            results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            results.truncate(options.limit.max(1));
            Ok(results)
        }
    }

    #[tokio::test]
    async fn approve_creates_golden_example_with_high_reward() {
        let task = make_task("p1", "codegen", TaskStatus::Completed, Some(5.0));
        let (service, example_repo, _, _, _) = evolution_service(task, vec![], vec![]);

        service.collect_golden_example("t1").await.unwrap();

        let examples = example_repo
            .list_training_examples_by_kind("codegen")
            .await
            .unwrap();
        assert_eq!(examples.len(), 1);
        assert!((examples[0].reward - 4.4).abs() < 0.001);
    }

    #[tokio::test]
    async fn low_human_score_is_rejected_as_golden_example() {
        let task = make_task("p1", "codegen", TaskStatus::Completed, Some(3.0));
        let (service, example_repo, _, _, _) = evolution_service(task, vec![], vec![]);

        assert!(service.collect_golden_example("t1").await.is_err());
        assert!(
            example_repo
                .list_training_examples_by_kind("codegen")
                .await
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn rejected_retry_task_creates_counter_example() {
        let task = make_task("p1", "codegen", TaskStatus::Pending, Some(0.0));
        let (service, example_repo, _, _, _) = evolution_service(task, vec![], vec![]);

        service.collect_counter_example("t1").await.unwrap();

        let examples = example_repo
            .list_training_examples_by_kind("codegen")
            .await
            .unwrap();
        assert_eq!(examples.len(), 1);
        assert_eq!(examples[0].reward, 0.0);
    }

    #[tokio::test]
    async fn should_train_returns_true_after_50_examples() {
        let task = make_task("p1", "codegen", TaskStatus::Completed, Some(5.0));
        let examples: Vec<_> = (0..50).map(|i| example("codegen", 4.5, i as i64)).collect();
        let (service, _, _, _, _) = evolution_service(task, examples, vec![]);

        assert!(service.should_train("codegen").await.unwrap());
        assert!(!service.should_train("architecture").await.unwrap());
    }

    #[tokio::test]
    async fn validation_rejects_dataset_below_usable_golden_threshold() {
        let task = make_task("p1", "codegen", TaskStatus::Completed, Some(5.0));
        // Low-reward rows are counter examples; they must not count as usable SFT targets.
        let examples = vec![
            example("codegen", 5.0, 0),
            example("codegen", 1.0, 1),
            example("codegen", 1.0, 2),
        ];
        let (service, _, _, _, _) = evolution_service(task, examples, vec![]);

        let result = service.train_and_register("codegen").await;
        assert!(matches!(
            result,
            Err(LoraEvolutionError::ValidationFailed(_, _))
        ));
    }

    #[tokio::test]
    async fn train_and_register_creates_lora_adapter_record() {
        let task = make_task("p1", "codegen", TaskStatus::Completed, Some(5.0));
        let examples = vec![
            example("codegen", 5.0, 0),
            example("codegen", 5.0, 1),
            example("codegen", 5.0, 2),
        ];
        let (service, _, adapter_repo, inference, events) =
            evolution_service(task, examples, vec![]);

        let adapter = service.train_and_register("codegen").await.unwrap();
        assert_eq!(adapter.task_kind, Some("codegen".into()));

        let registered = inference.registered.lock().unwrap().clone();
        assert!(registered.contains(&adapter.id));

        let stored = adapter_repo
            .list_lora_adapters_by_kind("codegen")
            .await
            .unwrap();
        assert_eq!(stored.len(), 1);

        let event_ids: Vec<_> = events
            .events
            .lock()
            .unwrap()
            .iter()
            .filter_map(|e| match e {
                Event::LoraSwapped { lora_id, .. } => Some(lora_id.clone()),
                _ => None,
            })
            .collect();
        assert!(event_ids.contains(&adapter.id));
    }

    #[tokio::test]
    async fn train_and_register_indexes_adapter_in_vector_store() {
        let task = make_task("p1", "codegen", TaskStatus::Completed, Some(5.0));
        let examples = vec![
            example("codegen", 5.0, 0),
            example("codegen", 5.0, 1),
            example("codegen", 5.0, 2),
        ];
        let embedder = Arc::new(MockEmbedder::new(16));
        let vector_store: Arc<dyn VectorStore> = Arc::new(TestVectorStore::default());

        let (service, _, _, _, _) = evolution_service(task, examples, vec![]);
        let service = service.with_vector_index(embedder.clone(), vector_store.clone());

        let adapter = service.train_and_register("codegen").await.unwrap();

        let query = embedder.embed("codegen input").await.unwrap();
        let results = vector_store
            .search(
                LORA_ADAPTER_COLLECTION,
                &query,
                SearchOptions {
                    limit: 10,
                    filter: None,
                    score_threshold: None,
                },
            )
            .await
            .unwrap();
        assert!(results.iter().any(|result| {
            result.payload.get("adapter_id").and_then(|v| v.as_str()) == Some(adapter.id.as_str())
        }));
    }

    #[tokio::test]
    async fn select_lora_uses_active_adapter_for_kind() {
        let task = make_task("p1", "codegen", TaskStatus::Pending, None);
        let adapters = vec![
            LoraAdapter {
                id: "codegen-v1".into(),
                project_id: None,
                name: "codegen-v1".into(),
                file_path: "/tmp/a.safetensors".into(),
                base_model: "mistral-7b".into(),
                task_kind: Some("codegen".into()),
                agent_role: None,
                metrics: json!({}),
                created_at: 1,
                active: false,
            },
            LoraAdapter {
                id: "codegen-v2".into(),
                project_id: None,
                name: "codegen-v2".into(),
                file_path: "/tmp/b.safetensors".into(),
                base_model: "mistral-7b".into(),
                task_kind: Some("codegen".into()),
                agent_role: None,
                metrics: json!({}),
                created_at: 2,
                active: true,
            },
        ];
        let (service, _, _, _, _) = evolution_service(task, vec![], adapters);

        let selected = service
            .select_lora(&make_task("p1", "codegen", TaskStatus::Pending, None), "p1")
            .await
            .unwrap();
        assert_eq!(selected, Some("codegen-v2".into()));
    }

    #[tokio::test]
    async fn should_train_for_role_is_true_when_threshold_met() {
        let task = make_task("p1", "codegen", TaskStatus::Completed, Some(5.0));
        let examples = vec![
            example_with_role("coder", 5.0, 0),
            example_with_role("coder", 5.0, 1),
        ];
        let (service, _, _, _, _) = evolution_service(task, examples, vec![]);

        assert!(
            service
                .should_train_for_role(AgentRole::Coder)
                .await
                .unwrap()
        );
    }

    #[tokio::test]
    async fn select_lora_prefers_role_adapter() {
        let mut task = make_task("p1", "codegen", TaskStatus::Pending, None);
        task.assigned_agent = Some("coder".into());
        let adapters = vec![
            LoraAdapter {
                id: "codegen-v1".into(),
                project_id: None,
                name: "codegen-v1".into(),
                file_path: "/tmp/a.safetensors".into(),
                base_model: "mistral-7b".into(),
                task_kind: Some("codegen".into()),
                agent_role: None,
                metrics: json!({}),
                created_at: 1,
                active: true,
            },
            LoraAdapter {
                id: "coder-v1".into(),
                project_id: None,
                name: "coder-v1".into(),
                file_path: "/tmp/b.safetensors".into(),
                base_model: "mistral-7b".into(),
                task_kind: Some("coder".into()),
                agent_role: Some("coder".into()),
                metrics: json!({}),
                created_at: 2,
                active: true,
            },
        ];
        let (service, _, _, _, _) = evolution_service(task.clone(), vec![], adapters);

        let selected = service.select_lora(&task, "p1").await.unwrap();
        assert_eq!(selected, Some("coder-v1".into()));
    }

    #[tokio::test]
    async fn select_lora_by_role_prefers_active_adapter() {
        let adapters = vec![
            LoraAdapter {
                id: "coder-v1".into(),
                project_id: None,
                name: "coder-v1".into(),
                file_path: "/tmp/a.safetensors".into(),
                base_model: "mistral-7b".into(),
                task_kind: Some("coder".into()),
                agent_role: Some("coder".into()),
                metrics: json!({}),
                created_at: 1,
                active: false,
            },
            LoraAdapter {
                id: "coder-v2".into(),
                project_id: None,
                name: "coder-v2".into(),
                file_path: "/tmp/b.safetensors".into(),
                base_model: "mistral-7b".into(),
                task_kind: Some("coder".into()),
                agent_role: Some("coder".into()),
                metrics: json!({}),
                created_at: 2,
                active: true,
            },
        ];
        let (service, _, _, _, _) = evolution_service(
            make_task("p1", "codegen", TaskStatus::Pending, None),
            vec![],
            adapters,
        );

        let selected = service
            .select_lora_by_role(AgentRole::Coder, "p1")
            .await
            .unwrap();
        assert_eq!(selected, Some("coder-v2".into()));
    }

    fn example_with_role(agent_role: &str, reward: f64, created_at: i64) -> TrainingExample {
        TrainingExample {
            id: Ulid::new().to_string(),
            task_id: format!("task-{agent_role}-{created_at}"),
            project_id: Some("p1".into()),
            prompt_version_id: Some("pv1".into()),
            task_kind: "codegen".into(),
            agent_role: Some(agent_role.into()),
            model_id: None,
            rag_evidence_ids: Vec::new(),
            input_text: "input".into(),
            output_text: "output".into(),
            accepted_output: Some("output".into()),
            rejected_output: None,
            critic_feedback: None,
            failure_type: None,
            reward,
            created_at,
        }
    }

    #[tokio::test]
    async fn reject_creates_counter_example_with_zero_reward() {
        let task = make_task("p1", "codegen", TaskStatus::Completed, Some(1.0));
        let (service, example_repo, _, _, _) = evolution_service(task, vec![], vec![]);

        service.collect_counter_example("t1").await.unwrap();

        let examples = example_repo
            .list_training_examples_by_kind("codegen")
            .await
            .unwrap();
        assert_eq!(examples.len(), 1);
        assert_eq!(examples[0].reward, 0.0);
    }

    #[tokio::test]
    async fn should_train_ignores_counter_examples_below_golden_threshold() {
        let task = make_task("p1", "codegen", TaskStatus::Completed, Some(5.0));
        let examples = vec![
            example("codegen", 5.0, 0),
            example("codegen", 0.0, 1),
            example("codegen", 0.0, 2),
        ];
        let (service, _, _, _, _) = evolution_service(task, examples, vec![]);

        assert!(!service.should_train("codegen").await.unwrap());
    }

    #[tokio::test]
    async fn train_and_register_does_not_train_on_counter_examples_as_targets() {
        let task = make_task("p1", "codegen", TaskStatus::Completed, Some(5.0));
        let examples = vec![
            example("codegen", 5.0, 0),
            example("codegen", 5.0, 1),
            TrainingExample {
                input_text: "bad rejected input".into(),
                output_text: "bad rejected output must not become target".into(),
                reward: 0.0,
                ..example("codegen", 0.0, 2)
            },
        ];
        let trainer = Arc::new(RecordingTrainer::new(MockTrainer::new()));
        let (service, _, _, _) = evolution_service_with_trainer(task, examples, trainer.clone());

        service.train_and_register("codegen").await.unwrap();

        let trained = trainer.trained_examples.lock().unwrap().clone();
        assert_eq!(trained.len(), 1);
        assert!(trained.iter().all(|example| example.reward >= 3.0));
        assert!(
            !trained
                .iter()
                .any(|example| example.output_text.contains("bad rejected output"))
        );
    }

    #[tokio::test]
    async fn remediation_builds_chosen_rejected_preference_pair_with_feedback() {
        let mut accepted = make_task("p1", "codegen", TaskStatus::Completed, Some(5.0));
        accepted.id = "accepted-fix".into();
        accepted.assigned_agent = Some("coder-python".into());
        accepted.prompt_version_id = Some("prompt-v1".into());
        accepted.payload = json!({
            "model_id": "qwen3.5:9b",
            "rag_evidence_ids": ["chunk-a", "chunk-b"],
            "remediation_for": "rejected-bug"
        });
        accepted.result =
            Some(json!({"summary": "fixed parser", "files_changed": ["src/parser.py"]}));

        let mut rejected = make_task("p1", "codegen", TaskStatus::Completed, Some(1.0));
        rejected.id = "rejected-bug".into();
        rejected.assigned_agent = Some("coder-python".into());
        rejected.payload = json!({
            "failure_type": "missing-tests",
            "critic_feedback": "missing pytest coverage for bad csv rows"
        });
        rejected.result = Some(json!({"summary": "patched parser without tests"}));

        let (service, example_repo, _, _, _) =
            evolution_service_with_tasks(vec![accepted, rejected], vec![], vec![]);

        let report = service
            .build_dataset_for_role("coder-python")
            .await
            .unwrap();
        let examples = example_repo
            .list_training_examples_by_role("coder-python")
            .await
            .unwrap();

        assert_eq!(report.role, "coder-python");
        assert_eq!(report.preference_pairs, 1);
        assert_eq!(examples.len(), 1);
        assert_eq!(
            examples[0].accepted_output.as_deref(),
            Some("{\"files_changed\":[\"src/parser.py\"],\"summary\":\"fixed parser\"}")
        );
        assert_eq!(
            examples[0].rejected_output.as_deref(),
            Some("{\"summary\":\"patched parser without tests\"}")
        );
        assert_eq!(
            examples[0].critic_feedback.as_deref(),
            Some("missing pytest coverage for bad csv rows")
        );
        assert_eq!(examples[0].failure_type.as_deref(), Some("missing-tests"));
        assert_eq!(examples[0].model_id.as_deref(), Some("qwen3.5:9b"));
        assert_eq!(examples[0].rag_evidence_ids, vec!["chunk-a", "chunk-b"]);
    }

    #[tokio::test]
    async fn preference_negative_side_is_not_trained_as_sft_target() {
        let task = make_task("p1", "codegen", TaskStatus::Completed, Some(5.0));
        let examples = vec![
            TrainingExample {
                reward: 5.0,
                accepted_output: Some("chosen good answer".into()),
                rejected_output: Some("rejected bad answer".into()),
                output_text: "chosen good answer".into(),
                failure_type: Some("wrong-api".into()),
                agent_role: Some("coder-python".into()),
                ..example("codegen", 5.0, 0)
            },
            TrainingExample {
                reward: 5.0,
                accepted_output: Some("chosen second answer".into()),
                rejected_output: Some("rejected second answer".into()),
                output_text: "chosen second answer".into(),
                failure_type: Some("missing-tests".into()),
                agent_role: Some("coder-python".into()),
                ..example("codegen", 5.0, 1)
            },
            TrainingExample {
                reward: 5.0,
                accepted_output: Some("chosen third answer".into()),
                rejected_output: Some("rejected third answer".into()),
                output_text: "chosen third answer".into(),
                failure_type: Some("missing-tests".into()),
                agent_role: Some("coder-python".into()),
                ..example("codegen", 5.0, 2)
            },
        ];
        let trainer = Arc::new(RecordingTrainer::new(MockTrainer::new()));
        let (service, _, _, _) = evolution_service_with_trainer(task, examples, trainer.clone());

        service
            .train_and_register_for_role(AgentRole::CoderPython)
            .await
            .unwrap();

        let trained = trainer.trained_examples.lock().unwrap().clone();
        assert!(trained.iter().all(|example| {
            !example.output_text.contains("rejected")
                && example
                    .rejected_output
                    .as_deref()
                    .is_some_and(|text| text.contains("rejected"))
        }));
    }

    #[tokio::test]
    async fn dataset_stats_report_balancing_leakage_and_low_information() {
        let task = make_task("p1", "codegen", TaskStatus::Completed, Some(5.0));
        let examples = vec![
            TrainingExample {
                agent_role: Some("coder-python".into()),
                failure_type: Some("missing-tests".into()),
                accepted_output: Some("same leaked text".into()),
                rejected_output: Some("bad output with enough info".into()),
                output_text: "same leaked text".into(),
                ..example("codegen", 5.0, 0)
            },
            TrainingExample {
                agent_role: Some("coder-python".into()),
                failure_type: Some("missing-tests".into()),
                accepted_output: Some("same leaked text".into()),
                rejected_output: Some("x".into()),
                output_text: "same leaked text".into(),
                ..example("codegen", 5.0, 1)
            },
            TrainingExample {
                agent_role: Some("qa".into()),
                failure_type: Some("weak-analysis".into()),
                accepted_output: Some("qa accepted output".into()),
                rejected_output: Some("qa rejected output".into()),
                output_text: "qa accepted output".into(),
                ..example("qa", 5.0, 2)
            },
        ];
        let (service, _, _, _, _) = evolution_service(task, examples, vec![]);

        let report = service
            .dataset_stats_for_role("coder-python")
            .await
            .unwrap();

        assert_eq!(report.role, "coder-python");
        assert_eq!(report.total_examples, 2);
        assert_eq!(report.failure_type_counts["missing-tests"], 2);
        assert_eq!(report.leakage.duplicate_output_count, 1);
        assert_eq!(report.low_information.filtered_count, 1);
        assert_eq!(
            report.balancing.failure_type_target_count("missing-tests"),
            Some(1)
        );
    }

    #[tokio::test]
    async fn train_and_register_passes_custom_training_config_to_trainer() {
        let task = make_task("p1", "codegen", TaskStatus::Completed, Some(5.0));
        let base_model_path = std::env::temp_dir().join(format!("base-{}.gguf", Ulid::new()));
        let examples = vec![
            example("codegen", 5.0, 0),
            example("codegen", 5.0, 1),
            example("codegen", 5.0, 2),
        ];
        let trainer = Arc::new(RecordingTrainer::new(MockTrainer::new()));
        let (service, _, _, _) = evolution_service_with_trainer(task, examples, trainer.clone());
        let service = service.with_training_config(LoraTrainingConfig {
            rank: 4,
            alpha: 8,
            epochs: 1,
            learning_rate: 1e-3,
            validation_ratio: 0.25,
            max_seq_len: 32,
            base_model_path: Some(base_model_path.clone()),
            tokenizer_path: None,
            target_modules: vec!["q_proj".into(), "v_proj".into()],
            ..Default::default()
        });

        service.train_and_register("codegen").await.unwrap();

        let configs = trainer.configs.lock().unwrap();
        assert_eq!(configs.len(), 1);
        assert_eq!(configs[0].base_model_path.as_ref(), Some(&base_model_path));
        assert_eq!(configs[0].rank, 4);
        assert_eq!(configs[0].max_seq_len, 32);
    }

    #[tokio::test]
    async fn golden_example_is_written_to_experience_repository() {
        let task = make_task("p1", "codegen", TaskStatus::Completed, Some(5.0));
        let (service, _, _, _, _) = evolution_service(task, vec![], vec![]);
        let experience_repo = Arc::new(InMemoryExperienceRepo::default());
        let service = service.with_experience_repo(experience_repo.clone());

        service.collect_golden_example("t1").await.unwrap();

        let experiences = experience_repo.experiences.lock().unwrap().clone();
        assert_eq!(experiences.len(), 1);
        assert!(
            experiences[0]
                .text
                .as_deref()
                .unwrap()
                .contains("Task: title")
        );
        assert!((experiences[0].reward - 4.4).abs() < 0.001);
        assert_eq!(experiences[0].comment.as_deref(), Some("golden example"));
    }

    #[tokio::test]
    async fn validation_rejects_adapter_above_loss_threshold() {
        let task = make_task("p1", "codegen", TaskStatus::Completed, Some(5.0));
        // High rewards pass the reward gate; a very low loss threshold fails on loss.
        let examples = vec![
            example("codegen", 5.0, 0),
            example("codegen", 5.0, 1),
            example("codegen", 5.0, 2),
        ];
        let (service, _, _, _, _) = evolution_service(task, examples, vec![]);
        let service = service.with_validation_loss_threshold(0.1);

        let result = service.train_and_register("codegen").await;
        assert!(matches!(
            result,
            Err(LoraEvolutionError::ValidationFailed(_, _))
        ));
    }

    #[tokio::test]
    async fn train_and_register_creates_training_job_record() {
        let task = make_task("p1", "codegen", TaskStatus::Completed, Some(5.0));
        let examples = vec![
            example("codegen", 5.0, 0),
            example("codegen", 5.0, 1),
            example("codegen", 5.0, 2),
        ];
        let (service, _, _, _, _) = evolution_service(task, examples, vec![]);
        let job_repo = Arc::new(InMemoryTrainingJobRepo::default());
        let service = service.with_training_job_repo(job_repo.clone());

        let adapter = service.train_and_register("codegen").await.unwrap();

        let jobs = job_repo
            .list_training_jobs_by_kind("codegen")
            .await
            .unwrap();
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].status, TrainingJobStatus::Promoted);
        assert_eq!(jobs[0].adapter_id, Some(adapter.id));
    }

    #[tokio::test]
    async fn benchmark_gate_rejects_lora_challenger_without_registering_adapter() {
        let task = make_task("p1", "codegen", TaskStatus::Completed, Some(5.0));
        let examples = vec![
            example("codegen", 5.0, 0),
            example("codegen", 5.0, 1),
            TrainingExample {
                task_id: "t1".into(),
                ..example("codegen", 5.0, 2)
            },
        ];
        let existing_adapter = LoraAdapter {
            id: "codegen-v1".into(),
            project_id: None,
            name: "codegen-v1".into(),
            file_path: "/tmp/baseline.safetensors".into(),
            base_model: "mistral-7b".into(),
            task_kind: Some("codegen".into()),
            agent_role: None,
            metrics: json!({ "benchmark": { "winner": "baseline" } }),
            created_at: 1,
            active: true,
        };
        let (service, _, adapter_repo, inference, events) =
            evolution_service(task, examples, vec![existing_adapter]);
        let job_repo = Arc::new(InMemoryTrainingJobRepo::default());
        let service = service
            .with_training_job_repo(job_repo.clone())
            .with_benchmark_gate(Arc::new(RejectingBenchmarkGate));

        let result = service.train_and_register("codegen").await;

        assert!(matches!(
            result,
            Err(LoraEvolutionError::ValidationFailed(kind, reason))
                if kind == "codegen" && reason.contains("benchmark")
        ));
        assert!(inference.registered.lock().unwrap().is_empty());

        let adapters = adapter_repo
            .list_lora_adapters_by_kind("codegen")
            .await
            .unwrap();
        assert_eq!(adapters.len(), 1);
        assert_eq!(adapters[0].id, "codegen-v1");
        assert!(adapters[0].active);

        let jobs = job_repo
            .list_training_jobs_by_kind("codegen")
            .await
            .unwrap();
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].status, TrainingJobStatus::RolledBack);
        assert_eq!(
            jobs[0].metrics["benchmark_gate"]["accepted"],
            serde_json::Value::Bool(false)
        );
        assert!(
            jobs[0]
                .metrics
                .get("benchmark_gate")
                .and_then(|gate| gate.get("reason"))
                .and_then(|reason| reason.as_str())
                .is_some_and(|reason| reason.contains("regressed benchmark"))
        );
        assert!(
            jobs[0]
                .error_message
                .as_deref()
                .unwrap()
                .contains("benchmark")
        );
        let emitted = events.events.lock().unwrap().clone();
        let rejected = emitted
            .iter()
            .find_map(|event| match event {
                Event::RunObserved {
                    action, metadata, ..
                } if action == "lora_evolution_rejected" => Some(metadata),
                _ => None,
            })
            .expect("rejection event should be emitted");
        assert_eq!(rejected["training_job_id"], jobs[0].id);
        assert_eq!(rejected["adapter_id"], "codegen-v2");
        assert_eq!(rejected["trace_id"], "trace");
        assert_eq!(rejected["benchmark_gate"]["accepted"], false);
        assert!(emitted.iter().any(|event| matches!(
            event,
            Event::RunObserved {
                action,
                trace_id,
                ..
            } if action == "lora_evolution_rejected" && trace_id == "trace"
        )));
    }

    #[tokio::test]
    async fn benchmark_gate_rejects_accepted_decision_without_all_quality_gates() {
        let task = make_task("p1", "codegen", TaskStatus::Completed, Some(5.0));
        let examples = vec![
            example("codegen", 5.0, 0),
            example("codegen", 5.0, 1),
            example("codegen", 5.0, 2),
        ];
        let (service, _, adapter_repo, inference, _) = evolution_service(task, examples, vec![]);
        let service = service.with_benchmark_gate(Arc::new(IncompleteAcceptingBenchmarkGate));

        let error = service.train_and_register("codegen").await.unwrap_err();

        assert!(matches!(
            error,
            LoraEvolutionError::ValidationFailed(_, reason)
                if reason.contains("missing or failed quality gates")
                    && reason.contains("negative_benchmark")
                    && reason.contains("runtime_application")
        ));
        assert!(adapter_repo.adapters.lock().unwrap().is_empty());
        assert!(inference.registered.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn benchmark_gate_accepts_lora_challenger_and_promotes_it() {
        let task = make_task("p1", "codegen", TaskStatus::Completed, Some(5.0));
        let examples = vec![
            example("codegen", 5.0, 0),
            example("codegen", 5.0, 1),
            TrainingExample {
                task_id: "t1".into(),
                ..example("codegen", 5.0, 2)
            },
        ];
        let existing_adapter = LoraAdapter {
            id: "codegen-v1".into(),
            project_id: None,
            name: "codegen-v1".into(),
            file_path: "/tmp/baseline.safetensors".into(),
            base_model: "mistral-7b".into(),
            task_kind: Some("codegen".into()),
            agent_role: None,
            metrics: json!({}),
            created_at: 1,
            active: true,
        };
        let (service, _, adapter_repo, inference, events) =
            evolution_service(task, examples, vec![existing_adapter]);
        let gate = Arc::new(RecordingBenchmarkGate::default());
        let service = service.with_benchmark_gate(gate.clone());

        let adapter = service.train_and_register("codegen").await.unwrap();

        assert_eq!(adapter.id, "codegen-v2");
        assert_eq!(
            inference.registered.lock().unwrap().as_slice(),
            ["codegen-v2"]
        );

        let requests = gate.requests.lock().unwrap().clone();
        assert_eq!(requests.len(), 1);
        assert_eq!(
            requests[0].baseline_adapter_id.as_deref(),
            Some("codegen-v1")
        );
        assert_eq!(requests[0].challenger_adapter_id, "codegen-v2");
        assert_eq!(requests[0].base_model, "mistral-7b");
        assert_eq!(requests[0].training_fingerprints.len(), 3);
        assert!(
            requests[0]
                .training_fingerprints
                .iter()
                .any(|fingerprint| fingerprint.contains("input") && fingerprint.contains("output"))
        );
        assert!(requests[0].challenger_adapter_path.ends_with("adapter"));
        assert!(
            requests[0]
                .challenger_adapter_path
                .join("adapter_config.json")
                .exists()
        );
        assert!(
            requests[0]
                .challenger_adapter_path
                .join("adapter_model.safetensors")
                .exists()
        );

        let adapters = adapter_repo
            .list_lora_adapters_by_kind("codegen")
            .await
            .unwrap();
        let baseline = adapters
            .iter()
            .find(|adapter| adapter.id == "codegen-v1")
            .unwrap();
        let challenger = adapters
            .iter()
            .find(|adapter| adapter.id == "codegen-v2")
            .unwrap();
        assert!(!baseline.active);
        assert!(challenger.active);

        let emitted = events.events.lock().unwrap().clone();
        let promoted = emitted
            .iter()
            .find_map(|event| match event {
                Event::RunObserved {
                    action, metadata, ..
                } if action == "lora_evolution_promoted" => Some(metadata),
                _ => None,
            })
            .expect("promotion event should be emitted");
        assert_eq!(promoted["adapter_id"], "codegen-v2");
        assert_eq!(promoted["trace_id"], "trace");
        assert_eq!(promoted["benchmark_gate"]["accepted"], true);
        assert!(emitted.iter().any(|event| matches!(
            event,
            Event::RunObserved {
                action,
                trace_id,
                ..
            } if action == "lora_evolution_promoted" && trace_id == "trace"
        )));
    }

    #[tokio::test]
    async fn training_rejects_golden_dataset_with_empty_outputs() {
        let task = make_task("p1", "codegen", TaskStatus::Completed, Some(5.0));
        let examples = vec![
            TrainingExample {
                output_text: "   ".into(),
                ..example("codegen", 5.0, 0)
            },
            example("codegen", 5.0, 1),
            example("codegen", 5.0, 2),
        ];
        let (service, _, adapter_repo, inference) = evolution_service_with_trainer(
            task,
            examples,
            Arc::new(MockTrainer::new().with_single_file_layout()),
        );

        let result = service.train_and_register("codegen").await;

        assert!(matches!(
            result,
            Err(LoraEvolutionError::ValidationFailed(kind, reason))
                if kind == "codegen" && reason.contains("golden dataset")
        ));
        assert!(
            adapter_repo
                .list_lora_adapters_by_kind("codegen")
                .await
                .unwrap()
                .is_empty()
        );
        assert!(inference.registered.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn training_rejects_duplicate_task_ids_before_train_validation_split() {
        let task = make_task("p1", "codegen", TaskStatus::Completed, Some(5.0));
        let examples = vec![
            TrainingExample {
                task_id: "same-task".into(),
                ..example("codegen", 5.0, 0)
            },
            TrainingExample {
                task_id: "same-task".into(),
                ..example("codegen", 5.0, 1)
            },
            example("codegen", 5.0, 2),
        ];
        let (service, _, adapter_repo, inference) =
            evolution_service_with_trainer(task, examples, Arc::new(MockTrainer::new()));

        let result = service.train_and_register("codegen").await;

        assert!(matches!(
            result,
            Err(LoraEvolutionError::ValidationFailed(kind, reason))
                if kind == "codegen" && reason.contains("duplicate task_id")
        ));
        assert!(
            adapter_repo
                .list_lora_adapters_by_kind("codegen")
                .await
                .unwrap()
                .is_empty()
        );
        assert!(inference.registered.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn training_rejects_oversized_lora_artifact_before_registering() {
        let task = make_task("p1", "codegen", TaskStatus::Completed, Some(5.0));
        let examples = vec![
            example("codegen", 5.0, 0),
            example("codegen", 5.0, 1),
            example("codegen", 5.0, 2),
        ];
        let (service, _, adapter_repo, inference) = evolution_service_with_trainer(
            task,
            examples,
            Arc::new(MockTrainer::new().with_adapter_bytes(128)),
        );
        let service = service.with_max_adapter_bytes(16);

        let result = service.train_and_register("codegen").await;

        assert!(matches!(
            result,
            Err(LoraEvolutionError::ValidationFailed(kind, reason))
                if kind == "codegen" && reason.contains("adapter artifact")
        ));
        assert!(
            adapter_repo
                .list_lora_adapters_by_kind("codegen")
                .await
                .unwrap()
                .is_empty()
        );
        assert!(inference.registered.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn training_rejects_single_file_lora_artifact_before_registering() {
        let task = make_task("p1", "codegen", TaskStatus::Completed, Some(5.0));
        let examples = vec![
            example("codegen", 5.0, 0),
            example("codegen", 5.0, 1),
            example("codegen", 5.0, 2),
        ];
        let (service, _, adapter_repo, inference) = evolution_service_with_trainer(
            task,
            examples,
            Arc::new(MockTrainer::new().with_single_file_layout()),
        );

        let result = service.train_and_register("codegen").await;

        assert!(matches!(
            result,
            Err(LoraEvolutionError::ValidationFailed(kind, reason))
                if kind == "codegen" && reason.contains("adapter_config.json")
        ));
        assert!(
            adapter_repo
                .list_lora_adapters_by_kind("codegen")
                .await
                .unwrap()
                .is_empty()
        );
        assert!(inference.registered.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn training_rejects_overfit_loss_gap_before_promotion() {
        let task = make_task("p1", "codegen", TaskStatus::Completed, Some(5.0));
        let examples = vec![
            example("codegen", 5.0, 0),
            example("codegen", 5.0, 1),
            example("codegen", 5.0, 2),
        ];
        let (service, _, adapter_repo, inference) = evolution_service_with_trainer(
            task,
            examples,
            Arc::new(MockTrainer::new().with_losses(0.01, 0.49)),
        );
        let service = service.with_max_train_validation_loss_gap(0.1);

        let result = service.train_and_register("codegen").await;

        assert!(matches!(
            result,
            Err(LoraEvolutionError::ValidationFailed(kind, reason))
                if kind == "codegen" && reason.contains("overfit")
        ));
        assert!(
            adapter_repo
                .list_lora_adapters_by_kind("codegen")
                .await
                .unwrap()
                .is_empty()
        );
        assert!(inference.registered.lock().unwrap().is_empty());
    }
}
