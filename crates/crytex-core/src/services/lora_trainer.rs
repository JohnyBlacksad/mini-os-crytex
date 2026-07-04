//! Abstraction for training LoRA adapters from curated [`TrainingExample`]s.
//!
//! The trait is backend-agnostic: `crytex-inference-mistral` will provide a
//! mock implementation for integration tests, while a real trainer can be
//! plugged in later without changing `LoraEvolutionService`.

use crate::models::TrainingExample;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

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
    #[error("training backend error: {0}")]
    Backend(String),
}

fn default_max_seq_len() -> usize {
    128
}

/// Hyper-parameters for a LoRA training run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraTrainingConfig {
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
}

/// Backend-independent interface for training LoRA adapters.
#[async_trait]
pub trait LoraTrainer: Send + Sync {
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
