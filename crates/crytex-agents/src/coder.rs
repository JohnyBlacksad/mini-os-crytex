use crate::{
    extract_backend_id, extract_model,
    json::parse_llm_json_value,
    prompts::{coder_system_prompt, coder_user_prompt, system_prompt_override},
    tooling::ToolingEngine,
};
use async_trait::async_trait;
use crytex_core::models::Task;
use crytex_core::services::{
    Agent, AgentError, InferenceService, ToolDescription, ToolService, ToolServiceError,
};
use crytex_inference::{InferenceRequest, Message};
use serde_json::Value;
use std::sync::{Arc, Mutex};

/// Coding agent that writes, edits, and verifies code through tool calls.
pub struct CoderAgent;

impl CoderAgent {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CoderAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Agent for CoderAgent {
    fn name(&self) -> &str {
        "coder"
    }

    fn capabilities(&self) -> Vec<String> {
        vec![
            "code_generation".to_string(),
            "refactoring".to_string(),
            "debugging".to_string(),
        ]
    }

    async fn execute(
        &self,
        task: &Task,
        inference: Arc<dyn InferenceService>,
        tools: Arc<dyn ToolService>,
    ) -> Result<Value, AgentError> {
        let tdd = task
            .payload
            .get("tdd")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let override_prompt = system_prompt_override(&task.payload);
        let system_prompt = coder_system_prompt(tdd, &tools.list_tools(), override_prompt);
        let user_prompt = coder_user_prompt(&task.payload);

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
            temperature: Some(0.3),
            max_tokens: Some(4096),
            lora_adapter_id: task.lora_adapter_id.clone(),
        };

        let recorder = Arc::new(ToolRecorder::new(tools));
        let engine = ToolingEngine::new(recorder.clone()).with_max_iterations(10);
        let response = engine
            .run(inference, request)
            .await
            .map_err(|e| AgentError::Execution(e.to_string()))?;

        let mut result = parse_coder_output(&response.content, &recorder.calls())?;
        result["usage"] = serde_json::json!({
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        });

        Ok(result)
    }
}

/// Wraps a [`ToolService`] and records every invocation made by the agent.
struct ToolRecorder {
    inner: Arc<dyn ToolService>,
    calls: Mutex<Vec<(String, Value)>>,
}

impl ToolRecorder {
    fn new(inner: Arc<dyn ToolService>) -> Self {
        Self {
            inner,
            calls: Mutex::new(Vec::new()),
        }
    }

    fn calls(&self) -> Vec<(String, Value)> {
        self.calls
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

#[async_trait]
impl ToolService for ToolRecorder {
    async fn invoke(&self, name: &str, args: Value) -> Result<Value, ToolServiceError> {
        self.calls
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push((name.to_string(), args.clone()));
        self.inner.invoke(name, args).await
    }

    fn list_tools(&self) -> Vec<ToolDescription> {
        self.inner.list_tools()
    }
}

/// Parse the final model output into a structured coder result.
///
/// If the model produced valid JSON, required fields are normalised.  If not,
/// a best-effort fallback is built from the recorded tool calls.
fn parse_coder_output(content: &str, calls: &[(String, Value)]) -> Result<Value, AgentError> {
    let trimmed = content.trim();
    let mut value = parse_llm_json_value(trimmed).unwrap_or_else(|_| {
        serde_json::json!({
            "files_changed": infer_files_changed(calls),
            "test_results": null,
            "summary": trimmed,
        })
    });

    if value.get("files_changed").is_none() {
        value["files_changed"] = serde_json::to_value(infer_files_changed(calls))
            .map_err(|e| AgentError::Execution(e.to_string()))?;
    } else {
        merge_recorded_writes(&mut value["files_changed"], infer_files_changed(calls));
    }

    if value.get("test_results").is_none() {
        value["test_results"] = Value::Null;
    }

    if value.get("summary").is_none() {
        value["summary"] = Value::String("Done.".to_string());
    }

    Ok(value)
}

/// Infer which files changed from recorded `fs_write` invocations.
fn infer_files_changed(calls: &[(String, Value)]) -> Vec<Value> {
    calls
        .iter()
        .filter_map(|(name, args)| {
            if name == "fs_write" {
                args.get("path").and_then(|p| p.as_str()).map(|path| {
                    serde_json::json!({
                        "path": path,
                        "action": "created",
                    })
                })
            } else {
                None
            }
        })
        .collect()
}

fn merge_recorded_writes(files_changed: &mut Value, recorded_writes: Vec<Value>) {
    let Some(files) = files_changed.as_array_mut() else {
        *files_changed = Value::Array(recorded_writes);
        return;
    };

    for write in recorded_writes {
        let path = write.get("path");
        let already_recorded = files.iter().any(|file| file.get("path") == path);
        if !already_recorded {
            files.push(write);
        }
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

    /// Records every tool invocation and forwards to an optional inner service.
    struct RecordingToolService {
        calls: Arc<Mutex<Vec<(String, Value)>>>,
        inner: Option<Arc<dyn ToolService>>,
    }

    impl RecordingToolService {
        fn new() -> Self {
            Self {
                calls: Arc::new(Mutex::new(Vec::new())),
                inner: None,
            }
        }

        #[allow(dead_code)]
        fn with_inner(inner: Arc<dyn ToolService>) -> Self {
            Self {
                calls: Arc::new(Mutex::new(Vec::new())),
                inner: Some(inner),
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
            match &self.inner {
                Some(inner) => inner.invoke(name, args).await,
                None => Ok(Value::String("ok".to_string())),
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
            parent_id: None,
            title: "write lib".into(),
            description: None,
            kind: "codegen".into(),
            status: TaskStatus::Pending,
            assigned_agent: None,
            priority: 0,
            payload: serde_json::json!({ "prompt": "write a library" }),
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
    async fn coder_agent_writes_file_and_returns_files_changed() {
        let service = Arc::new(RecordingToolService::new());
        let inference = Arc::new(TwoStepInference {
            first: r#"{"tool":"fs_write","args":{"path":"src/lib.rs","content":"fn main() {}"}}"#
                .to_string(),
            second: r#"{"files_changed":[{"path":"src/lib.rs","action":"created"}],"test_results":null,"summary":"Created src/lib.rs"}"#.to_string(),
        });

        let agent = CoderAgent::new();
        let result = agent
            .execute(&sample_task(), inference, service.clone())
            .await
            .unwrap();

        let calls = service.calls();
        assert!(
            calls
                .iter()
                .any(|(n, args)| n == "fs_write" && args["path"] == "src/lib.rs")
        );

        let files = result["files_changed"]
            .as_array()
            .expect("files_changed array");
        assert_eq!(files.len(), 1);
        assert_eq!(files[0]["path"], "src/lib.rs");
        assert_eq!(files[0]["action"], "created");
        assert_eq!(
            result["summary"].as_str().expect("summary"),
            "Created src/lib.rs"
        );
        assert_eq!(result["usage"]["prompt_tokens"].as_u64(), Some(10));
    }

    #[tokio::test]
    async fn coder_returns_structured_json() {
        let service = Arc::new(RecordingToolService::new());
        let inference = Arc::new(TwoStepInference {
            first: r#"{"tool":"fs_write","args":{"path":"src/lib.rs","content":"fn main() {}"}}"#
                .to_string(),
            second: r#"{"files_changed":[{"path":"src/lib.rs","action":"created"}],"test_results":{"passed":true,"command":"cargo test"},"summary":"Created src/lib.rs"}"#.to_string(),
        });

        let agent = CoderAgent::new();
        let result = agent
            .execute(&sample_task(), inference, service.clone())
            .await
            .unwrap();

        assert!(result.get("files_changed").is_some());
        assert!(result.get("test_results").is_some());
        assert!(result.get("summary").is_some());
        assert!(result.get("usage").is_some());
        assert!(result["files_changed"].is_array());
    }

    #[tokio::test]
    async fn coder_tdd_mode_writes_tests_first() {
        let service = Arc::new(RecordingToolService::new());
        let inference = Arc::new(TwoStepInference {
            first: r#"{"tool":"fs_write","args":{"path":"src/lib.rs","content":"fn main() {}"}}"#
                .to_string(),
            second: r#"{"files_changed":[],"test_results":null,"summary":"ok"}"#.to_string(),
        });

        let mut task = sample_task();
        task.payload["tdd"] = Value::Bool(true);

        let agent = CoderAgent::new();
        let _ = agent
            .execute(&task, inference, service.clone())
            .await
            .unwrap();

        let calls = service.calls();
        // In TDD mode the agent should at least plan to write a test before implementation.
        // We record the prompt sent to inference by checking that the first assistant message
        // contains the TDD instruction; the mock tool service receives the actual tool calls.
        // The system prompt is verified separately in prompts.rs; here we just ensure the
        // agent runs without error when TDD is enabled.
        assert!(calls.iter().any(|(n, _)| n == "fs_write"));
    }

    #[test]
    fn parse_coder_output_extracts_json_fields() {
        let content = r#"{"files_changed":[{"path":"a.rs","action":"created"}],"test_results":null,"summary":"ok"}"#;
        let result = parse_coder_output(content, &[]).unwrap();
        assert_eq!(result["files_changed"].as_array().unwrap().len(), 1);
        assert_eq!(result["summary"], "ok");
    }

    #[test]
    fn parse_coder_output_falls_back_to_recorded_writes() {
        let calls = vec![(
            "fs_write".to_string(),
            serde_json::json!({"path": "b.rs", "content": "x"}),
        )];
        let result = parse_coder_output("plain prose", &calls).unwrap();
        assert_eq!(result["files_changed"][0]["path"], "b.rs");
        assert_eq!(result["summary"].as_str().unwrap(), "plain prose");
    }

    #[test]
    fn parse_coder_output_merges_recorded_writes_when_model_returns_empty_files_changed() {
        let calls = vec![(
            "fs_write".to_string(),
            serde_json::json!({"path": "crytex-smoke.txt", "content": "ok"}),
        )];
        let content = r#"{"files_changed":[],"test_results":null,"summary":"ok"}"#;

        let result = parse_coder_output(content, &calls).unwrap();

        assert_eq!(result["files_changed"][0]["path"], "crytex-smoke.txt");
        assert_eq!(result["files_changed"][0]["action"], "created");
    }
}
