//! LoRA training implementations for the mistral.rs backend.
//!
//! Currently this module provides a mock trainer that writes a minimal
//! `.safetensors` adapter file and returns deterministic metrics.  It exists
//! so the rest of the system (collection, triggering, routing, and selection)
//! can be built and tested before a real fine-tuning backend is integrated.

use async_trait::async_trait;
use crytex_core::models::TrainingExample;
use crytex_core::services::{
    LoraMetrics, LoraTrainer, LoraTrainingConfig, LoraTrainingError, LoraTrainingResult,
};
use safetensors::tensor::{TensorView, serialize_to_file};
use std::collections::HashMap;
use std::path::Path;
use ulid::Ulid;

/// Mock trainer for integration tests.
///
/// Writes a minimal (empty tensor map) `.safetensors` file to `output_dir`
/// and returns metrics derived from the input examples.
pub struct MockLoraTrainer;

impl MockLoraTrainer {
    /// Create a new mock trainer.
    pub fn new() -> Self {
        Self
    }
}

impl Default for MockLoraTrainer {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LoraTrainer for MockLoraTrainer {
    async fn train(
        &self,
        examples: Vec<TrainingExample>,
        _config: LoraTrainingConfig,
        output_dir: &Path,
    ) -> Result<LoraTrainingResult, LoraTrainingError> {
        if examples.is_empty() {
            return Err(LoraTrainingError::NotEnoughExamples(0, 1));
        }

        tokio::fs::create_dir_all(output_dir).await?;

        let average_reward = examples.iter().map(|e| e.reward).sum::<f64>() / examples.len() as f64;
        let adapter_id = format!("mock-lora-{}", Ulid::new());
        let adapter_path = output_dir.join(format!("{adapter_id}.safetensors"));

        let data: HashMap<String, TensorView> = HashMap::new();
        let metadata: HashMap<String, String> = [("format".to_string(), "pt".to_string())]
            .into_iter()
            .collect();
        serialize_to_file(&data, Some(metadata), &adapter_path)
            .map_err(|e| LoraTrainingError::AdapterSerialization(e.to_string()))?;

        Ok(LoraTrainingResult {
            adapter_id,
            adapter_path,
            metrics: LoraMetrics {
                train_loss: 0.1,
                validation_loss: 0.2,
                average_reward,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crytex_core::services::LoraTrainingConfig;
    use std::path::PathBuf;

    fn example(kind: &str, reward: f64) -> TrainingExample {
        TrainingExample {
            id: format!("ex-{}", Ulid::new()),
            task_id: "t1".into(),
            project_id: Some("p1".into()),
            prompt_version_id: Some("pv1".into()),
            task_kind: kind.into(),
            agent_role: None,
            input_text: "Implement X".into(),
            output_text: "fn x() {}".into(),
            reward,
            created_at: 0,
        }
    }

    #[tokio::test]
    async fn mock_training_creates_safetensors() {
        let trainer = MockLoraTrainer::new();
        let output = PathBuf::from(format!(
            "{}/mock-train-test",
            std::env::temp_dir().to_string_lossy()
        ));
        let _ = tokio::fs::remove_dir_all(&output).await;

        let result = trainer
            .train(
                vec![example("codegen", 4.5), example("codegen", 5.0)],
                LoraTrainingConfig::default(),
                &output,
            )
            .await
            .unwrap();

        assert!(result.adapter_path.exists());
        assert!(
            result
                .adapter_path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .ends_with(".safetensors")
        );

        let _ = tokio::fs::remove_dir_all(&output).await;
    }

    #[tokio::test]
    async fn mock_training_returns_metrics() {
        let trainer = MockLoraTrainer::new();
        let output = PathBuf::from(format!(
            "{}/mock-train-metrics",
            std::env::temp_dir().to_string_lossy()
        ));
        let _ = tokio::fs::remove_dir_all(&output).await;

        let result = trainer
            .train(
                vec![example("codegen", 4.0), example("codegen", 5.0)],
                LoraTrainingConfig::default(),
                &output,
            )
            .await
            .unwrap();

        assert!((result.metrics.average_reward - 4.5).abs() < 0.001);

        let _ = tokio::fs::remove_dir_all(&output).await;
    }
}
