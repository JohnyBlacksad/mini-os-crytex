use async_trait::async_trait;
use crytex_core::models::Task;
use crytex_core::services::{Agent, AgentError, InferenceService, ToolService};
use serde_json::Value;
use std::sync::Arc;

use super::execute_specialized_critic;

pub struct CodeCriticAgent;

impl CodeCriticAgent {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CodeCriticAgent {
    fn default() -> Self {
        Self::new()
    }
}

const FOCUS: &str = r#"Evaluate correctness, edge cases, algorithm clarity, and maintainability. Look for bugs, logic errors, unhandled errors, unnecessary complexity, and violations of language idioms."#;

#[async_trait]
impl Agent for CodeCriticAgent {
    fn name(&self) -> &str {
        "critic-code"
    }

    fn capabilities(&self) -> Vec<String> {
        vec!["review".to_string(), "code".to_string()]
    }

    async fn execute(
        &self,
        task: &Task,
        inference: Arc<dyn InferenceService>,
        tools: Arc<dyn ToolService>,
    ) -> Result<Value, AgentError> {
        execute_specialized_critic("code", FOCUS, task, inference, tools).await
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
            parent_id: None,
            title: "review code".into(),
            description: None,
            kind: "review".into(),
            status: TaskStatus::Pending,
            assigned_agent: Some("critic-code".into()),
            priority: 0,
            payload: serde_json::json!({
                "prompt": "review the implementation",
                "parent_result": { "summary": "ok" }
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
    async fn code_critic_returns_score_and_comment() {
        let tools = Arc::new(NoopToolService);
        let inference = Arc::new(SingleResponseInference {
            content: r#"{"score":4.0,"comment":"add error handling"}"#.to_string(),
        });

        let agent = CodeCriticAgent::new();
        let result = agent
            .execute(&sample_task(), inference, tools)
            .await
            .unwrap();

        assert_eq!(result["dimension"], "code");
        assert_eq!(result["score"], 4.0);
        assert_eq!(result["comment"], "add error handling");
    }
}
