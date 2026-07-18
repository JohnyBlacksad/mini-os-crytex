use crate::{
    extract_backend_id, extract_model,
    json::parse_llm_json_value,
    prompts::{architect_system_prompt, architect_user_prompt, system_prompt_override},
    tooling::ToolingEngine,
};
use async_trait::async_trait;
use crytex_core::models::Task;
use crytex_core::services::{Agent, AgentError, InferenceService, ToolService};
use crytex_inference::{InferenceRequest, Message};
use serde_json::Value;
use std::sync::Arc;

/// Architecture agent that explores the codebase and produces a structured plan.
pub struct ArchitectAgent;

impl ArchitectAgent {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ArchitectAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Agent for ArchitectAgent {
    fn name(&self) -> &str {
        "architect"
    }

    fn capabilities(&self) -> Vec<String> {
        vec![
            "design".to_string(),
            "planning".to_string(),
            "architecture".to_string(),
        ]
    }

    async fn execute(
        &self,
        task: &Task,
        inference: Arc<dyn InferenceService>,
        tools: Arc<dyn ToolService>,
    ) -> Result<Value, AgentError> {
        let override_prompt = system_prompt_override(&task.payload);
        let system_prompt = architect_system_prompt(&tools.list_tools(), override_prompt);
        let user_prompt = architect_user_prompt(&task.payload);

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
            max_tokens: Some(4096),
            lora_adapter_id: task.lora_adapter_id.clone(),
        };

        let engine = ToolingEngine::new(tools).with_max_iterations(5);
        let response = engine
            .run(inference, request)
            .await
            .map_err(|e| AgentError::Execution(e.to_string()))?;

        let mut result = parse_architect_output(&response.content)?;
        result["usage"] = serde_json::json!({
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        });

        Ok(result)
    }
}

/// Parse the final model output into a normalized architect result.
fn parse_architect_output(content: &str) -> Result<Value, AgentError> {
    let mut value = parse_llm_json_value(content)
        .map_err(|e| AgentError::Execution(format!("architect output is not valid JSON: {e}")))?;

    if value.get("plan").is_none() {
        value["plan"] = serde_json::json!({
            "goal": "",
            "assumptions": [],
            "subtasks": [],
        });
    }

    if value["plan"].get("subtasks").is_none() {
        value["plan"]["subtasks"] = serde_json::json!([]);
    }

    if value.get("summary").is_none() {
        value["summary"] = Value::String("No summary provided.".to_string());
    }

    Ok(value)
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

    struct OneShotInference {
        content: String,
    }

    #[async_trait]
    impl InferenceService for OneShotInference {
        async fn generate(
            &self,
            _request: InferenceRequest,
        ) -> Result<InferenceResponse, InferenceServiceError> {
            Ok(InferenceResponse {
                content: self.content.clone(),
                usage: TokenUsage {
                    prompt_tokens: 20,
                    completion_tokens: 10,
                    total_tokens: 30,
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
            title: "landing page".into(),
            description: None,
            kind: "codegen".into(),
            status: TaskStatus::Pending,
            assigned_agent: None,
            priority: 0,
            payload: serde_json::json!({ "prompt": "write a landing page" }),
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
    async fn architect_agent_returns_structured_plan() {
        let plan = serde_json::json!({
            "plan": {
                "goal": "write a landing page",
                "assumptions": ["static HTML is enough"],
                "subtasks": [
                    { "kind": "architecture", "agent": "architect", "title": "design", "description": "design", "prompt": "design" },
                    { "kind": "codegen", "agent": "coder", "title": "code", "description": "code", "prompt": "code" }
                ]
            },
            "summary": "Plan created",
        });
        let inference = Arc::new(OneShotInference {
            content: plan.to_string(),
        });
        let tools = Arc::new(NoopToolService);
        let agent = ArchitectAgent::new();

        let result = agent
            .execute(&sample_task(), inference, tools)
            .await
            .unwrap();

        let subtasks = result["plan"]["subtasks"]
            .as_array()
            .expect("subtasks array");
        assert_eq!(subtasks.len(), 2);
        assert_eq!(subtasks[0]["agent"], "architect");
        assert_eq!(subtasks[1]["agent"], "coder");
        assert_eq!(result["summary"], "Plan created");
    }

    #[test]
    fn parse_architect_output_normalizes_missing_fields() {
        let result = parse_architect_output("{\"plan\":{\"subtasks\":[]}}").unwrap();
        assert!(result["plan"]["subtasks"].is_array());
        assert_eq!(result["summary"], "No summary provided.");
    }
}
