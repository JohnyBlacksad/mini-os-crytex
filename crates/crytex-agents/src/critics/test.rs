use async_trait::async_trait;
use crytex_core::models::Task;
use crytex_core::services::{Agent, AgentError, InferenceService, ToolService};
use serde_json::Value;
use std::sync::Arc;

use super::execute_specialized_critic;

pub struct TestCriticAgent;

impl TestCriticAgent {
    pub fn new() -> Self {
        Self
    }
}

impl Default for TestCriticAgent {
    fn default() -> Self {
        Self::new()
    }
}

const FOCUS: &str = r#"Evaluate test quality and coverage. Look for missing tests, flaky assertions, insufficient edge cases, and tests that do not actually verify the behavior they claim to cover."#;

#[async_trait]
impl Agent for TestCriticAgent {
    fn name(&self) -> &str {
        "critic-test"
    }

    fn capabilities(&self) -> Vec<String> {
        vec!["review".to_string(), "testing".to_string()]
    }

    async fn execute(
        &self,
        task: &Task,
        inference: Arc<dyn InferenceService>,
        tools: Arc<dyn ToolService>,
    ) -> Result<Value, AgentError> {
        execute_specialized_critic("test", FOCUS, task, inference, tools).await
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
            title: "review tests".into(),
            description: None,
            kind: "review".into(),
            status: TaskStatus::Pending,
            assigned_agent: Some("critic-test".into()),
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
    async fn test_critic_returns_score_and_comment() {
        let tools = Arc::new(NoopToolService);
        let inference = Arc::new(SingleResponseInference {
            content: r#"{"score":3.0,"comment":"missing edge case tests"}"#.to_string(),
        });

        let agent = TestCriticAgent::new();
        let result = agent
            .execute(&sample_task(), inference, tools)
            .await
            .unwrap();

        assert_eq!(result["dimension"], "test");
        assert_eq!(result["score"], 3.0);
        assert_eq!(result["comment"], "missing edge case tests");
    }
}
