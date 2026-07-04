use std::sync::Arc;

use async_trait::async_trait;
use crytex_inference::{InferenceManager, InferenceRequest, Message as LlmMessage};

use crate::compress::CompressionError;
use crate::compressors::summarize::Summarizer;

/// Summarizer backed by an LLM via `crytex_inference::InferenceManager`.
#[derive(Clone)]
pub struct LlmSummarizer {
    inference: Arc<dyn InferenceManager>,
    model: String,
    system_prompt: String,
}

impl LlmSummarizer {
    pub fn new(inference: Arc<dyn InferenceManager>, model: impl Into<String>) -> Self {
        Self {
            inference,
            model: model.into(),
            system_prompt: "You are a context compressor. Summarize the following conversation history while preserving key facts, decisions, and user intent. Be concise.".into(),
        }
    }

    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }
}

impl std::fmt::Debug for LlmSummarizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmSummarizer")
            .field("model", &self.model)
            .field("system_prompt", &self.system_prompt)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl Summarizer for LlmSummarizer {
    async fn summarize(&self, text: &str, max_tokens: usize) -> Result<String, CompressionError> {
        let request = InferenceRequest {
            backend_id: None,
            model: self.model.clone(),
            messages: vec![
                LlmMessage {
                    role: "system".into(),
                    content: self.system_prompt.clone(),
                },
                LlmMessage {
                    role: "user".into(),
                    content: format!(
                        "Summarize this in at most {} tokens:\n\n{}",
                        max_tokens, text
                    ),
                },
            ],
            system_prompt: None,
            temperature: Some(0.3),
            max_tokens: Some(max_tokens),
            lora_adapter_id: None,
        };

        let response = self
            .inference
            .generate(request)
            .await
            .map_err(|e| CompressionError::Inference(e.to_string()))?;

        Ok(response.content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use crytex_inference::{
        BackendInfo, InferenceError, InferenceResponse, LoRAAdapter, ModelInfo, TokenUsage,
    };

    struct MockInference;

    #[async_trait]
    impl InferenceManager for MockInference {
        async fn generate(
            &self,
            _request: InferenceRequest,
        ) -> Result<InferenceResponse, InferenceError> {
            Ok(InferenceResponse {
                content: "mock summary".into(),
                usage: TokenUsage {
                    prompt_tokens: 10,
                    completion_tokens: 2,
                    total_tokens: 12,
                },
                finish_reason: "stop".into(),
            })
        }

        async fn embed(&self, _text: &str) -> Result<Vec<f32>, InferenceError> {
            Ok(vec![])
        }

        async fn register_lora(&self, _lora: LoRAAdapter) -> Result<(), InferenceError> {
            Ok(())
        }

        async fn swap_lora(&self, _lora_id: &str) -> Result<(), InferenceError> {
            Ok(())
        }

        fn available_backends(&self) -> Vec<BackendInfo> {
            vec![]
        }

        async fn list_models(&self) -> Result<Vec<ModelInfo>, InferenceError> {
            Ok(vec![])
        }
    }

    #[tokio::test]
    async fn llm_summarizer_delegates_to_inference() {
        let summarizer = LlmSummarizer::new(Arc::new(MockInference), "model");
        let summary = summarizer.summarize("long text", 64).await.unwrap();
        assert_eq!(summary, "mock summary");
    }
}
