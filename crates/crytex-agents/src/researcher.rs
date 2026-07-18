use crate::{
    extract_backend_id, extract_model,
    json::parse_llm_json_value,
    prompts::{researcher_system_prompt, researcher_user_prompt, system_prompt_override},
    tooling::generate_with_tools,
};
use async_trait::async_trait;
use crytex_core::models::Task;
use crytex_core::services::InferenceService;
use crytex_core::services::{Agent, AgentError, ToolService};
use crytex_inference::{InferenceRequest, Message, TokenUsage};
use serde_json::Value;
use std::sync::Arc;

pub struct ResearcherAgent;

impl ResearcherAgent {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ResearcherAgent {
    fn default() -> Self {
        Self::new()
    }
}

fn parse_researcher_output(content: &str, usage: &TokenUsage) -> Result<Value, serde_json::Error> {
    let mut value = parse_llm_json_value(content)?;

    if value.get("findings").is_none() {
        value["findings"] = Value::Array(Vec::new());
    }

    if value.get("sources").is_none() {
        value["sources"] = Value::Array(Vec::new());
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
impl Agent for ResearcherAgent {
    fn name(&self) -> &str {
        "researcher"
    }

    fn capabilities(&self) -> Vec<String> {
        vec![
            "deep_research".to_string(),
            "documentation".to_string(),
            "exploration".to_string(),
        ]
    }

    async fn execute(
        &self,
        task: &Task,
        inference: Arc<dyn InferenceService>,
        tools: Arc<dyn ToolService>,
    ) -> Result<Value, AgentError> {
        let override_prompt = system_prompt_override(&task.payload);
        let system_prompt = researcher_system_prompt(&tools.list_tools(), override_prompt);
        let user_prompt = researcher_user_prompt(&task.payload);

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
            temperature: Some(0.7),
            max_tokens: Some(4096),
            lora_adapter_id: task.lora_adapter_id.clone(),
        };

        let response = generate_with_tools(inference, tools, request)
            .await
            .map_err(|e| AgentError::Execution(e.to_string()))?;

        parse_researcher_output(&response.content, &response.usage)
            .map_err(|e| AgentError::Execution(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::Mutex;

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

    struct RecordingToolService {
        calls: Arc<Mutex<Vec<(String, Value)>>>,
    }

    impl RecordingToolService {
        fn new() -> Self {
            Self {
                calls: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn calls(&self) -> Vec<(String, Value)> {
            self.calls.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl ToolService for RecordingToolService {
        async fn invoke(&self, name: &str, args: Value) -> Result<Value, ToolServiceError> {
            self.calls
                .lock()
                .unwrap()
                .push((name.to_string(), args.clone()));
            if name == "search_code" {
                Ok(serde_json::json!({"matches": [{"path": "src/lib.rs"}]}))
            } else {
                Ok(Value::Null)
            }
        }

        fn list_tools(&self) -> Vec<ToolDescription> {
            vec![]
        }
    }

    struct TwoStepInference {
        first: String,
        second: String,
    }

    #[async_trait]
    impl InferenceService for TwoStepInference {
        async fn generate(
            &self,
            request: InferenceRequest,
        ) -> Result<InferenceResponse, InferenceServiceError> {
            let content = if request.messages.iter().any(|m| m.role == "assistant") {
                self.second.clone()
            } else {
                self.first.clone()
            };
            Ok(InferenceResponse {
                content,
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
            parent_id: Some("parent".into()),
            title: "research patterns".into(),
            description: None,
            kind: "research".into(),
            status: TaskStatus::Pending,
            assigned_agent: Some("researcher".into()),
            priority: 0,
            payload: serde_json::json!({"prompt": "find error handling patterns"}),
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
    async fn researcher_agent_runs_multiple_searches_and_returns_summary() {
        let service = Arc::new(RecordingToolService::new());
        let inference = Arc::new(TwoStepInference {
            first: r#"[
                {"tool":"search_code","args":{"query":"error handling"}},
                {"tool":"search_code","args":{"query":"Result type"}}
            ]"#
            .to_string(),
            second: r#"{"summary":"Common pattern is Result<T,E>","findings":["Use Result"],"sources":["error handling","Result type"]}"#.to_string(),
        });

        let agent = ResearcherAgent::new();
        let result = agent
            .execute(&sample_task(), inference, service.clone())
            .await
            .unwrap();

        let search_calls = service
            .calls()
            .into_iter()
            .filter(|(n, _)| n == "search_code")
            .count();
        assert!(
            search_calls >= 2,
            "researcher should perform at least two searches"
        );

        assert_eq!(result["summary"], "Common pattern is Result<T,E>");
        assert_eq!(result["findings"], serde_json::json!(["Use Result"]));
        assert_eq!(
            result["sources"],
            serde_json::json!(["error handling", "Result type"])
        );
    }

    #[test]
    fn parse_researcher_output_normalizes_missing_fields() {
        let content = "{\"summary\":\"ok\"}";
        let usage = TokenUsage {
            prompt_tokens: 1,
            completion_tokens: 1,
            total_tokens: 2,
        };
        let value = parse_researcher_output(content, &usage).unwrap();
        assert_eq!(value["findings"], Value::Array(Vec::new()));
        assert_eq!(value["sources"], Value::Array(Vec::new()));
        assert_eq!(value["usage"]["total_tokens"], 2);
    }
}
