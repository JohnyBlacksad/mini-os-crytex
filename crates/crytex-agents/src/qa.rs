use crate::{
    extract_backend_id, extract_model,
    json::parse_llm_json_value,
    prompts::{qa_system_prompt, qa_user_prompt, system_prompt_override},
    tooling::generate_with_tools,
};
use async_trait::async_trait;
use crytex_core::models::Task;
use crytex_core::services::InferenceService;
use crytex_core::services::{Agent, AgentError, ToolService};
use crytex_inference::{InferenceRequest, Message, TokenUsage};
use serde_json::Value;
use std::sync::Arc;

pub struct QaAgent;

impl QaAgent {
    pub fn new() -> Self {
        Self
    }
}

fn parse_qa_output(content: &str, usage: &TokenUsage) -> Result<Value, serde_json::Error> {
    let mut value = parse_llm_json_value(content)?;

    if value
        .get("summary")
        .and_then(Value::as_str)
        .is_none_or(str::is_empty)
    {
        value["summary"] = Value::String(default_qa_summary(&value));
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

fn default_qa_summary(value: &Value) -> String {
    let passed = value.get("passed").and_then(Value::as_bool).unwrap_or(false);
    let failure_count = value
        .get("failures")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or_default();
    if passed {
        "QA completed successfully.".to_string()
    } else if failure_count > 0 {
        format!("QA found {failure_count} failure(s).")
    } else {
        "QA completed without a model-provided summary.".to_string()
    }
}

impl Default for QaAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Agent for QaAgent {
    fn name(&self) -> &str {
        "qa"
    }

    fn capabilities(&self) -> Vec<String> {
        vec![
            "testing".to_string(),
            "review".to_string(),
            "validation".to_string(),
        ]
    }

    async fn execute(
        &self,
        task: &Task,
        inference: Arc<dyn InferenceService>,
        tools: Arc<dyn ToolService>,
    ) -> Result<Value, AgentError> {
        let override_prompt = system_prompt_override(&task.payload);
        let system_prompt = qa_system_prompt(&tools.list_tools(), override_prompt);
        let user_prompt = qa_user_prompt(&task.payload);

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
            temperature: Some(0.4),
            max_tokens: Some(2048),
            lora_adapter_id: task.lora_adapter_id.clone(),
        };

        let response = generate_with_tools(inference, tools, request)
            .await
            .map_err(|e| AgentError::Execution(e.to_string()))?;

        parse_qa_output(&response.content, &response.usage)
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
            if name == "run_command" {
                Ok(serde_json::json!({
                    "stdout": "test result: ok",
                    "stderr": "",
                    "exit_code": 0,
                }))
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
            title: "test landing page".into(),
            description: None,
            kind: "qa".into(),
            status: TaskStatus::Pending,
            assigned_agent: Some("qa".into()),
            priority: 0,
            payload: serde_json::json!({
                "prompt": "test the landing page",
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
    async fn qa_agent_runs_tests_and_returns_structured_result() {
        let service = Arc::new(RecordingToolService::new());
        let inference = Arc::new(TwoStepInference {
            first: r#"{"tool":"run_command","args":{"command":"cargo","args":["test"]}}"#.to_string(),
            second: r#"{"passed":true,"exit_code":0,"stdout":"test result: ok","stderr":"","summary":"All tests passed"}"#.to_string(),
        });

        let agent = QaAgent::new();
        let result = agent
            .execute(&sample_task(), inference, service.clone())
            .await
            .unwrap();

        let calls = service.calls();
        assert!(
            calls
                .iter()
                .any(|(n, args)| n == "run_command" && args["command"] == "cargo")
        );

        assert_eq!(result["passed"], true);
        assert_eq!(result["exit_code"], 0);
        assert_eq!(result["stdout"], "test result: ok");
        assert_eq!(result["summary"], "All tests passed");
    }

    #[test]
    fn parse_qa_output_strips_markdown_fences() {
        let content = "```json\n{\"passed\":true,\"summary\":\"ok\"}\n```";
        let usage = TokenUsage {
            prompt_tokens: 1,
            completion_tokens: 2,
            total_tokens: 3,
        };
        let value = parse_qa_output(content, &usage).unwrap();
        assert_eq!(value["passed"], true);
        assert_eq!(value["summary"], "ok");
    }

    #[test]
    fn parse_qa_output_injects_usage_when_missing() {
        let content = "{\"passed\":false,\"summary\":\"broken\",\"failures\":[\"x\"]}";
        let usage = TokenUsage {
            prompt_tokens: 7,
            completion_tokens: 8,
            total_tokens: 15,
        };
        let value = parse_qa_output(content, &usage).unwrap();
        assert_eq!(value["usage"]["prompt_tokens"], 7);
        assert_eq!(value["usage"]["completion_tokens"], 8);
        assert_eq!(value["usage"]["total_tokens"], 15);
    }

    #[test]
    fn parse_qa_output_injects_summary_when_missing() {
        let content = "{\"passed\":true,\"failures\":[]}";
        let usage = TokenUsage {
            prompt_tokens: 1,
            completion_tokens: 2,
            total_tokens: 3,
        };
        let value = parse_qa_output(content, &usage).unwrap();
        assert_eq!(value["summary"], "QA completed successfully.");
    }
}
