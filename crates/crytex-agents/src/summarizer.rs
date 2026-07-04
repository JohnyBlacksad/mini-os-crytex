use crate::{extract_backend_id, extract_model};
use async_trait::async_trait;
use crytex_core::models::Task;
use crytex_core::services::InferenceService;
use crytex_core::services::{Agent, AgentError, ToolService};
use crytex_inference::{InferenceRequest, Message};
use serde_json::Value;
use std::sync::Arc;

pub struct SummarizerAgent;

impl SummarizerAgent {
    pub fn new() -> Self {
        Self
    }
}

impl Default for SummarizerAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Agent for SummarizerAgent {
    fn name(&self) -> &str {
        "summarizer"
    }

    fn capabilities(&self) -> Vec<String> {
        vec![
            "summarization".to_string(),
            "synthesis".to_string(),
            "reporting".to_string(),
        ]
    }

    async fn execute(
        &self,
        task: &Task,
        inference: Arc<dyn InferenceService>,
        _tools: Arc<dyn ToolService>,
    ) -> Result<Value, AgentError> {
        let system_prompt = task
            .payload
            .get("system_prompt_override")
            .and_then(|v| v.as_str())
            .unwrap_or("You are a summarizer. Condense information while preserving key points.")
            .to_string();
        let user_prompt = task
            .payload
            .get("prompt")
            .and_then(|v| v.as_str())
            .unwrap_or("Summarize the provided content.");

        let request = InferenceRequest {
            backend_id: extract_backend_id(&task.payload),
            model: extract_model(&task.payload),
            messages: vec![
                Message {
                    role: "system".to_string(),
                    content: system_prompt,
                },
                Message {
                    role: "user".to_string(),
                    content: user_prompt.to_string(),
                },
            ],
            system_prompt: None,
            temperature: Some(0.3),
            max_tokens: Some(1024),
            lora_adapter_id: task.lora_adapter_id.clone(),
        };

        let response = inference.generate(request).await?;

        Ok(serde_json::json!({
            "summary": response.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }
        }))
    }
}
