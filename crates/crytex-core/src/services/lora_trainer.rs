//! Abstraction for training LoRA adapters from curated [`TrainingExample`]s.
//!
//! The trait is backend-agnostic: `crytex-inference-mistral` will provide a
//! mock implementation for integration tests, while a real trainer can be
//! plugged in later without changing `LoraEvolutionService`.

use crate::models::TrainingExample;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Objective used to optimize a role-specific LoRA adapter.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "kebab-case")]
pub enum LoraTrainingObjective {
    /// Supervised fine-tuning against accepted outputs.
    #[default]
    Sft,
    /// Direct Preference Optimization over chosen/rejected pairs.
    Dpo,
    /// Odds Ratio Preference Optimization over chosen/rejected pairs.
    Orpo,
    /// KTO-style utility optimization from positive/negative feedback.
    Kto,
}

impl LoraTrainingObjective {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Sft => "sft",
            Self::Dpo => "dpo",
            Self::Orpo => "orpo",
            Self::Kto => "kto",
        }
    }

    pub fn requires_preference_pairs(&self) -> bool {
        matches!(self, Self::Dpo | Self::Orpo)
    }
}

impl std::fmt::Display for LoraTrainingObjective {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for LoraTrainingObjective {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "sft" => Ok(Self::Sft),
            "dpo" => Ok(Self::Dpo),
            "orpo" => Ok(Self::Orpo),
            "kto" => Ok(Self::Kto),
            other => Err(format!("unknown LoRA training objective: {other}")),
        }
    }
}

/// Errors that can occur while training a LoRA adapter.
#[derive(Debug, thiserror::Error)]
pub enum LoraTrainingError {
    #[error("not enough examples: got {0}, need at least {1}")]
    NotEnoughExamples(usize, usize),
    #[error("validation failed: {0}")]
    ValidationFailed(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("adapter serialization error: {0}")]
    AdapterSerialization(String),
    #[error("training objective {objective} is unsupported by backend {backend}")]
    UnsupportedObjective {
        backend: String,
        objective: LoraTrainingObjective,
    },
    #[error("training backend error: {0}")]
    Backend(String),
}

fn default_max_seq_len() -> usize {
    128
}

/// Hyper-parameters for a LoRA training run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraTrainingConfig {
    #[serde(default)]
    pub objective: LoraTrainingObjective,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub base_model_id: Option<String>,
    pub rank: usize,
    pub alpha: usize,
    pub target_modules: Vec<String>,
    pub epochs: usize,
    pub learning_rate: f64,
    pub validation_ratio: f64,
    /// Optional path to a directory containing base model weights.
    /// When `None`, the Candle backend trains a tiny built-in transformer.
    #[serde(default)]
    pub base_model_path: Option<PathBuf>,
    /// Optional path to a Hugging Face `tokenizer.json` file.
    /// When `None`, the Candle backend falls back to byte-level tokenization.
    #[serde(default)]
    pub tokenizer_path: Option<PathBuf>,
    /// Maximum sequence length used during training.
    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,
}

impl Default for LoraTrainingConfig {
    fn default() -> Self {
        Self {
            objective: LoraTrainingObjective::Sft,
            role: None,
            base_model_id: None,
            rank: 16,
            alpha: 32,
            target_modules: vec!["q_proj".into(), "v_proj".into()],
            epochs: 3,
            learning_rate: 1e-4,
            validation_ratio: 0.1,
            base_model_path: None,
            tokenizer_path: None,
            max_seq_len: default_max_seq_len(),
        }
    }
}

/// Metrics produced by a training run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LoraMetrics {
    pub train_loss: f64,
    pub validation_loss: f64,
    pub average_reward: f64,
}

/// The artifact produced by a successful training run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LoraTrainingResult {
    pub adapter_id: String,
    pub adapter_path: PathBuf,
    pub metrics: LoraMetrics,
    #[serde(default)]
    pub metadata: AdapterMetadata,
}

/// Metadata written next to each adapter and persisted in adapter metrics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AdapterMetadata {
    pub role: Option<String>,
    pub base_model: String,
    pub objective: LoraTrainingObjective,
    pub dataset_hash: String,
}

impl Default for AdapterMetadata {
    fn default() -> Self {
        Self {
            role: None,
            base_model: "unknown".into(),
            objective: LoraTrainingObjective::Sft,
            dataset_hash: String::new(),
        }
    }
}

impl AdapterMetadata {
    pub fn from_examples(config: &LoraTrainingConfig, examples: &[TrainingExample]) -> Self {
        Self {
            role: config.role.clone().or_else(|| {
                examples
                    .iter()
                    .find_map(|example| example.agent_role.clone())
            }),
            base_model: config
                .base_model_id
                .clone()
                .or_else(|| {
                    config
                        .base_model_path
                        .as_ref()
                        .map(|path| path.to_string_lossy().to_string())
                })
                .unwrap_or_else(|| "unknown".into()),
            objective: config.objective.clone(),
            dataset_hash: dataset_hash(examples),
        }
    }
}

pub fn dataset_hash(examples: &[TrainingExample]) -> String {
    let mut rows = examples
        .iter()
        .map(|example| {
            format!(
                "{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}",
                example.id,
                example.input_text,
                example
                    .accepted_output
                    .as_deref()
                    .unwrap_or(&example.output_text),
                example.rejected_output.as_deref().unwrap_or(""),
                example.failure_type.as_deref().unwrap_or("")
            )
        })
        .collect::<Vec<_>>();
    rows.sort();
    let hash = rows
        .join("\u{1e}")
        .bytes()
        .fold(0xcbf29ce484222325u64, |acc, byte| {
            (acc ^ byte as u64).wrapping_mul(0x100000001b3)
        });
    format!("fnv1a64:{hash:016x}")
}

pub fn validate_objective_examples(
    objective: &LoraTrainingObjective,
    examples: &[TrainingExample],
) -> Result<(), LoraTrainingError> {
    if objective.requires_preference_pairs()
        && examples.iter().any(|example| {
            example
                .accepted_output
                .as_deref()
                .unwrap_or("")
                .trim()
                .is_empty()
                || example
                    .rejected_output
                    .as_deref()
                    .unwrap_or("")
                    .trim()
                    .is_empty()
        })
    {
        return Err(LoraTrainingError::ValidationFailed(format!(
            "{objective} requires every training example to contain accepted_output and rejected_output"
        )));
    }

    Ok(())
}

/// Backend-independent adapter artifact validation.
pub struct AdapterArtifactValidator;

impl AdapterArtifactValidator {
    pub fn validate_dir(path: &Path) -> Result<(), LoraTrainingError> {
        if !path.is_dir() {
            return Err(LoraTrainingError::ValidationFailed(format!(
                "adapter artifact must be a directory: {}",
                path.display()
            )));
        }
        let config = path.join("adapter_config.json");
        let weights = path.join("adapter_model.safetensors");
        let metadata = path.join("adapter_metadata.json");
        [config.as_path(), weights.as_path(), metadata.as_path()]
            .iter()
            .try_for_each(|required| {
                required.exists().then_some(()).ok_or_else(|| {
                    LoraTrainingError::ValidationFailed(format!(
                        "adapter artifact is missing {}",
                        required.display()
                    ))
                })
            })
    }
}

/// Backend-independent interface for training LoRA adapters.
#[async_trait]
pub trait LoraTrainer: Send + Sync {
    fn backend_name(&self) -> &'static str;

    fn supports_objective(&self, objective: &LoraTrainingObjective) -> bool {
        objective == &LoraTrainingObjective::Sft
    }

    /// Train an adapter from curated examples.
    ///
    /// The trainer is responsible for splitting a hold-out validation set,
    /// writing the resulting adapter file to `output_dir`, and returning
    /// metrics.
    async fn train(
        &self,
        examples: Vec<TrainingExample>,
        config: LoraTrainingConfig,
        output_dir: &Path,
    ) -> Result<LoraTrainingResult, LoraTrainingError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use ulid::Ulid;

    struct SftOnlyTrainer;

    #[async_trait]
    impl LoraTrainer for SftOnlyTrainer {
        fn backend_name(&self) -> &'static str {
            "sft-only"
        }

        async fn train(
            &self,
            examples: Vec<TrainingExample>,
            config: LoraTrainingConfig,
            _output_dir: &Path,
        ) -> Result<LoraTrainingResult, LoraTrainingError> {
            validate_objective_examples(&config.objective, &examples)?;
            self.supports_objective(&config.objective)
                .then_some(())
                .ok_or_else(|| LoraTrainingError::UnsupportedObjective {
                    backend: self.backend_name().into(),
                    objective: config.objective,
                })?;
            unreachable!("red test only exercises validation and unsupported paths")
        }
    }

    fn example(accepted: Option<&str>, rejected: Option<&str>) -> TrainingExample {
        TrainingExample {
            id: Ulid::new().to_string(),
            task_id: Ulid::new().to_string(),
            project_id: Some("p1".into()),
            prompt_version_id: Some("pv1".into()),
            task_kind: "codegen".into(),
            agent_role: Some("coder-python".into()),
            model_id: Some("mistral-7b".into()),
            rag_evidence_ids: vec!["rag-1".into()],
            input_text: "write python".into(),
            output_text: accepted.unwrap_or("").into(),
            accepted_output: accepted.map(str::to_string),
            rejected_output: rejected.map(str::to_string),
            critic_feedback: Some("missing error handling".into()),
            failure_type: Some("code-quality".into()),
            reward: 5.0,
            created_at: 1,
        }
    }

    #[tokio::test]
    async fn should_return_typed_unsupported_for_unimplemented_objective() {
        let error = SftOnlyTrainer
            .train(
                vec![example(Some("good"), Some("bad"))],
                LoraTrainingConfig {
                    objective: LoraTrainingObjective::Dpo,
                    ..Default::default()
                },
                Path::new("."),
            )
            .await
            .unwrap_err();

        assert!(matches!(
            error,
            LoraTrainingError::UnsupportedObjective { backend, objective }
                if backend == "sft-only" && objective == LoraTrainingObjective::Dpo
        ));
    }

    #[test]
    fn should_reject_preference_objective_without_chosen_rejected_pairs() {
        let error = validate_objective_examples(
            &LoraTrainingObjective::Orpo,
            &[example(Some("good"), None)],
        )
        .unwrap_err();

        assert!(matches!(error, LoraTrainingError::ValidationFailed(_)));
    }

    #[test]
    fn should_create_stable_dataset_hash_from_role_examples() {
        let first = vec![example(Some("good"), Some("bad"))];
        let mut second = first.clone();
        second.reverse();

        assert_eq!(dataset_hash(&first), dataset_hash(&second));
        assert!(dataset_hash(&first).starts_with("fnv1a64:"));
    }
}
