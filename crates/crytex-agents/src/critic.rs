use crate::{
    extract_backend_id, extract_model,
    prompts::{critic_system_prompt, critic_user_prompt, system_prompt_override},
    tooling::generate_with_tools,
};
use async_trait::async_trait;
use crytex_core::models::Task;
use crytex_core::services::InferenceService;
use crytex_core::services::{Agent, AgentError, ToolService};
use crytex_inference::{InferenceRequest, Message, TokenUsage};
use serde_json::Value;
use std::sync::Arc;

pub struct CriticAgent;

impl CriticAgent {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CriticAgent {
    fn default() -> Self {
        Self::new()
    }
}

fn parse_critic_output(content: &str, usage: &TokenUsage) -> Result<Value, serde_json::Error> {
    let trimmed = content.trim();
    let cleaned = if let Some(inner) = trimmed.strip_prefix("```json") {
        inner.trim().trim_end_matches("```").trim()
    } else if let Some(inner) = trimmed.strip_prefix("```") {
        inner.trim().trim_end_matches("```").trim()
    } else {
        trimmed
    };

    let mut value: Value = serde_json::from_str(cleaned)?;

    if value.get("comments").is_none() {
        value["comments"] = Value::Array(Vec::new());
    }

    if value.get("usage").is_none() {
        value["usage"] = serde_json::json!({
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        });
    }

    Ok(value)
}

#[async_trait]
impl Agent for CriticAgent {
    fn name(&self) -> &str {
        "critic"
    }

    fn capabilities(&self) -> Vec<String> {
        vec![
            "review".to_string(),
            "analysis".to_string(),
            "feedback".to_string(),
        ]
    }

    async fn execute(
        &self,
        task: &Task,
        inference: Arc<dyn InferenceService>,
        tools: Arc<dyn ToolService>,
    ) -> Result<Value, AgentError> {
        let override_prompt = system_prompt_override(&task.payload);
        let system_prompt = critic_system_prompt(&tools.list_tools(), override_prompt);
        let user_prompt = critic_user_prompt(&task.payload);

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
                    content: user_prompt,
                },
            ],
            system_prompt: None,
            temperature: Some(0.5),
            max_tokens: Some(2048),
            lora_adapter_id: task.lora_adapter_id.clone(),
        };

        let response = generate_with_tools(inference, tools, request)
            .await
            .map_err(|e| AgentError::Execution(e.to_string()))?;

        parse_critic_output(&response.content, &response.usage)
            .map_err(|e| AgentError::Execution(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;
    use crytex_core::models::{Task, TaskStatus};
    use crytex_core::services::{
        InferenceService, InferenceServiceError, ToolDescription, ToolService, ToolServiceError,
    };
    use crytex_inference::{
        BackendInfo, InferenceRequest, InferenceResponse, LoRAAdapter, ModelInfo, TokenUsage,
    };
    use serde_json::Value;

    use super::*;

    struct NoopToolService;

    #[async_trait]
    impl ToolService for NoopToolService {
        async fn invoke(&self, _name: &str, _args: Value) -> Result<Value, ToolServiceError> {
            Ok(Value::Null)
        }

        fn list_tools(&self) -> Vec<ToolDescription> {
            vec![]
        }
    }

    struct SingleResponseInference {
        content: String,
    }

    #[async_trait]
    impl InferenceService for SingleResponseInference {
        async fn generate(
            &self,
            _request: InferenceRequest,
        ) -> Result<InferenceResponse, InferenceServiceError> {
            Ok(InferenceResponse {
                content: self.content.clone(),
                usage: TokenUsage {
                    prompt_tokens: 10,
                    completion_tokens: 5,
                    total_tokens: 15,
                },
                finish_reason: "stop".to_string(),
            })
        }

        async fn embed(&self, _text: &str) -> Result<Vec<f32>, InferenceServiceError> {
            Ok(vec![])
        }
        fn available_backends(&self) -> Vec<BackendInfo> {
            vec![]
        }
        async fn register_lora(&self, _lora: LoRAAdapter) -> Result<(), InferenceServiceError> {
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

    fn sample_task() -> Task {
        Task {
            id: "t1".into(),
            project_id: "p1".into(),
            parent_id: Some("parent".into()),
            title: "review landing page".into(),
            description: None,
            kind: "review".into(),
            status: TaskStatus::Pending,
            assigned_agent: Some("critic".into()),
            priority: 0,
            payload: serde_json::json!({
                "prompt": "review the landing page implementation",
                "parent_result": {
                    "files_changed": [{"path": "src/lib.rs", "action": "created"}],
                    "summary": "Created src/lib.rs"
                }
            }),
            result: None,
            created_at: 0,
            started_at: None,
            finished_at: None,
            iteration_count: 0,
            priority_score: 0.0,
            critic_score: None,
            human_score: None,
            prompt_version_id: None,
            lora_adapter_id: None,
            trace_id: "trace-1".into(),
        }
    }

    #[tokio::test]
    async fn critic_agent_returns_structured_review() {
        let tools = Arc::new(NoopToolService);
        let inference = Arc::new(SingleResponseInference {
            content: r#"{"score":4.2,"approved":true,"comments":["looks good"]}"#.to_string(),
        });

        let agent = CriticAgent::new();
        let result = agent
            .execute(&sample_task(), inference, tools)
            .await
            .unwrap();

        assert_eq!(result["score"], 4.2);
        assert_eq!(result["approved"], true);
        assert_eq!(result["comments"], serde_json::json!(["looks good"]));
        assert_eq!(result["usage"]["prompt_tokens"], 10);
    }

    #[test]
    fn parse_critic_output_strips_markdown_fences() {
        let content = "```json\n{\"score\":3.0,\"approved\":false}\n```";
        let usage = TokenUsage {
            prompt_tokens: 1,
            completion_tokens: 2,
            total_tokens: 3,
        };
        let value = parse_critic_output(content, &usage).unwrap();
        assert_eq!(value["score"], 3.0);
        assert_eq!(value["approved"], false);
        assert_eq!(value["comments"], Value::Array(Vec::new()));
    }

    #[test]
    fn parse_critic_output_normalizes_missing_comments() {
        let content = "{\"score\":5.0,\"approved\":true}";
        let usage = TokenUsage {
            prompt_tokens: 1,
            completion_tokens: 1,
            total_tokens: 2,
        };
        let value = parse_critic_output(content, &usage).unwrap();
        assert_eq!(value["comments"], Value::Array(Vec::new()));
    }
}
