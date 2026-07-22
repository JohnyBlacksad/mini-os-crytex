//! LoRA training implementations for the mistral.rs backend.
//!
//! Currently this module provides a mock trainer that writes a minimal
//! PEFT-like adapter directory and returns deterministic metrics.  It exists
//! so the rest of the system (collection, triggering, routing, and selection)
//! can be built and tested before a real fine-tuning backend is integrated.

use async_trait::async_trait;
use crytex_core::models::TrainingExample;
use crytex_core::services::{
    AdapterMetadata, LoraMetrics, LoraTrainer, LoraTrainingConfig, LoraTrainingError,
    LoraTrainingObjective, LoraTrainingResult, validate_objective_examples,
};
use safetensors::tensor::{TensorView, serialize_to_file};
use std::collections::HashMap;
use std::path::Path;
use ulid::Ulid;

/// Mock trainer for integration tests.
///
/// Writes a minimal PEFT-like adapter directory to `output_dir` and returns
/// metrics derived from the input examples.
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
    fn backend_name(&self) -> &'static str {
        "mistral-mock"
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
        if examples.is_empty() {
            return Err(LoraTrainingError::NotEnoughExamples(0, 1));
        }
        validate_objective_examples(&config.objective, &examples)?;

        tokio::fs::create_dir_all(output_dir).await?;

        let average_reward = examples.iter().map(|e| e.reward).sum::<f64>() / examples.len() as f64;
        let adapter_id = format!("mock-lora-{}", Ulid::new());
        let adapter_path = output_dir.join(&adapter_id);
        tokio::fs::create_dir_all(&adapter_path).await?;
        let metadata = AdapterMetadata::from_examples(&config, &examples);
        let base_model = config
            .base_model_path
            .as_ref()
            .map(|path| {
                path.to_string_lossy()
                    .replace('\\', "\\\\")
                    .replace('"', "\\\"")
            })
            .or(config.base_model_id.clone())
            .unwrap_or_else(|| "unknown".to_string());
        let target_modules = config
            .target_modules
            .iter()
            .map(|module| format!("\"{}\"", module.replace('"', "\\\"")))
            .collect::<Vec<_>>()
            .join(",");
        let adapter_config = format!(
            "{{\"peft_type\":\"LORA\",\"base_model_name_or_path\":\"{base_model}\",\"r\":{},\"lora_alpha\":{},\"target_modules\":[{target_modules}]}}",
            config.rank, config.alpha
        );
        tokio::fs::write(adapter_path.join("adapter_config.json"), adapter_config).await?;
        tokio::fs::write(
            adapter_path.join("adapter_metadata.json"),
            serde_json::to_vec_pretty(&metadata)
                .map_err(|error| LoraTrainingError::Backend(error.to_string()))?,
        )
        .await?;

        let data: HashMap<String, TensorView> = HashMap::new();
        let safetensors_metadata: HashMap<String, String> =
            [("format".to_string(), "pt".to_string())]
                .into_iter()
                .collect();
        serialize_to_file(
            &data,
            Some(safetensors_metadata),
            &adapter_path.join("adapter_model.safetensors"),
        )
        .map_err(|e| LoraTrainingError::AdapterSerialization(e.to_string()))?;

        Ok(LoraTrainingResult {
            adapter_id,
            adapter_path,
            metrics: LoraMetrics {
                train_loss: 0.1,
                validation_loss: 0.2,
                average_reward,
            },
            metadata,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crytex_core::services::{LoraTrainingConfig, LoraTrainingObjective};
    use std::path::PathBuf;

    fn example(kind: &str, reward: f64) -> TrainingExample {
        TrainingExample {
            id: format!("ex-{}", Ulid::new()),
            task_id: "t1".into(),
            project_id: Some("p1".into()),
            prompt_version_id: Some("pv1".into()),
            task_kind: kind.into(),
            agent_role: None,
            model_id: None,
            rag_evidence_ids: Vec::new(),
            input_text: "Implement X".into(),
            output_text: "fn x() {}".into(),
            accepted_output: Some("fn x() {}".into()),
            rejected_output: None,
            critic_feedback: None,
            failure_type: None,
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

        assert!(result.adapter_path.is_dir());
        assert!(result.adapter_path.join("adapter_config.json").exists());
        assert!(
            result
                .adapter_path
                .join("adapter_model.safetensors")
                .exists()
        );
        assert!(result.adapter_path.join("adapter_metadata.json").exists());

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

    #[tokio::test]
    async fn mock_training_supports_preference_objective() {
        let trainer = MockLoraTrainer::new();
        let output = PathBuf::from(format!(
            "{}/mock-train-preference",
            std::env::temp_dir().to_string_lossy()
        ));
        let _ = tokio::fs::remove_dir_all(&output).await;
        let mut first = example("codegen", 4.0);
        first.agent_role = Some("coder-python".into());
        first.rejected_output = Some("bad code".into());

        let result = trainer
            .train(
                vec![first],
                LoraTrainingConfig {
                    objective: LoraTrainingObjective::Dpo,
                    role: Some("coder-python".into()),
                    base_model_id: Some("mistral-7b".into()),
                    ..Default::default()
                },
                &output,
            )
            .await
            .unwrap();

        assert_eq!(result.metadata.objective, LoraTrainingObjective::Dpo);
        assert_eq!(result.metadata.role.as_deref(), Some("coder-python"));
        assert_eq!(result.metadata.base_model, "mistral-7b");
        assert!(result.metadata.dataset_hash.starts_with("fnv1a64:"));

        let _ = tokio::fs::remove_dir_all(&output).await;
    }
}
