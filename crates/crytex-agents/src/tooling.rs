//! Tool-calling loop for agents.
//!
//! Takes an LLM response, parses JSON tool calls, executes them through the
//! [`ToolService`], and feeds the results back into the conversation.

use std::sync::Arc;

use crytex_core::security::{SecurityScanner, SecurityThreat};
use crytex_core::services::{
    InferenceService, InferenceServiceError, ToolService, ToolServiceError,
};
use crytex_inference::{InferenceRequest, InferenceResponse, Message};
use crytex_tools::{ToolCall, parse_tool_calls};
use serde_json::json;
use thiserror::Error;

/// Errors returned by the tooling engine.
#[derive(Debug, Error)]
pub enum ToolingError {
    #[error("inference error: {0}")]
    Inference(#[from] InferenceServiceError),
    #[error("tool service error: {0}")]
    ToolService(#[from] ToolServiceError),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("security error: {threat}: {message}")]
    Security {
        threat: SecurityThreat,
        message: String,
    },
}

/// Executes an LLM request with a tool-calling loop.
pub struct ToolingEngine {
    tool_service: Arc<dyn ToolService>,
    max_iterations: usize,
    scanner: Option<Arc<dyn SecurityScanner>>,
}

/// Convenience helper: run a single inference request with tool calling.
pub async fn generate_with_tools(
    inference: Arc<dyn InferenceService>,
    tool_service: Arc<dyn ToolService>,
    request: InferenceRequest,
) -> Result<InferenceResponse, ToolingError> {
    ToolingEngine::new(tool_service)
        .run(inference, request)
        .await
}

impl ToolingEngine {
    /// Create a new engine bound to the given tool service.
    pub fn new(tool_service: Arc<dyn ToolService>) -> Self {
        Self {
            tool_service,
            max_iterations: 5,
            scanner: None,
        }
    }

    /// Set the maximum number of tool-calling iterations.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Attach a security scanner that inspects every tool call before invocation.
    pub fn with_scanner(mut self, scanner: Arc<dyn SecurityScanner>) -> Self {
        self.scanner = Some(scanner);
        self
    }

    /// Run `request` through the inference backend, repeatedly invoking tools
    /// until the model responds without a tool call or the iteration limit is
    /// reached.
    pub async fn run(
        &self,
        inference: Arc<dyn InferenceService>,
        mut request: InferenceRequest,
    ) -> Result<InferenceResponse, ToolingError> {
        let mut response = inference.generate(request.clone()).await?;

        for _ in 0..self.max_iterations {
            let calls = match parse_tool_calls(&response.content) {
                Ok(calls) if !calls.is_empty() => calls,
                _ => return Ok(response),
            };

            request.messages.push(Message {
                role: "assistant".to_string(),
                content: response.content.clone(),
            });

            let observations = self.execute_calls(calls).await?;
            request.messages.push(Message {
                role: "user".to_string(),
                content: observations,
            });

            response = inference.generate(request.clone()).await?;
        }

        Ok(response)
    }

    async fn execute_calls(&self, calls: Vec<ToolCall>) -> Result<String, ToolingError> {
        let mut results = Vec::new();
        for call in calls {
            if let Some(scanner) = &self.scanner {
                let findings = scanner.scan_tool_args(&call.name, &call.arguments);
                if let Some(finding) = findings.into_iter().next() {
                    return Err(ToolingError::Security {
                        threat: finding.threat,
                        message: finding.message,
                    });
                }
            }
            let output = self
                .tool_service
                .invoke(&call.name, call.arguments.clone())
                .await;
            let value = match output {
                Ok(v) => json!({ "ok": v }),
                Err(e) => json!({ "err": e.to_string() }),
            };
            results.push(json!({
                "tool": call.name,
                "result": value,
            }));
        }
        Ok(serde_json::to_string(&results)?)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;
    use crytex_core::services::{
        InferenceService, InferenceServiceError, ToolDescription, ToolService, ToolServiceError,
    };
    use crytex_inference::{
        BackendInfo, InferenceRequest, InferenceResponse, LoRAAdapter, Message, ModelInfo,
        TokenUsage,
    };
    use serde_json::Value;

    use super::*;

    struct MockToolService {
        result: Value,
    }

    #[async_trait]
    impl ToolService for MockToolService {
        async fn invoke(&self, _name: &str, _args: Value) -> Result<Value, ToolServiceError> {
            Ok(self.result.clone())
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
            _request: InferenceRequest,
        ) -> Result<InferenceResponse, InferenceServiceError> {
            // Return the first tool-call response once, then the final answer.
            let content = if _request.messages.iter().any(|m| m.role == "assistant") {
                self.second.clone()
            } else {
                self.first.clone()
            };
            Ok(InferenceResponse {
                content,
                usage: TokenUsage {
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    total_tokens: 0,
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

    fn request() -> InferenceRequest {
        InferenceRequest {
            backend_id: None,
            model: "mock".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: "read the file".to_string(),
            }],
            system_prompt: None,
            temperature: None,
            max_tokens: None,
            lora_adapter_id: None,
        }
    }

    #[tokio::test]
    async fn tooling_engine_invokes_tool_and_continues_dialog() {
        let engine = ToolingEngine::new(Arc::new(MockToolService {
            result: Value::String("file contents".to_string()),
        }));

        let inference = Arc::new(TwoStepInference {
            first: r#"{"tool":"fs_read","args":{"path":"src/main.rs"}}"#.to_string(),
            second: "I have read the file.".to_string(),
        });

        let result = engine.run(inference, request()).await.unwrap();

        assert_eq!(result.content, "I have read the file.");
    }

    #[tokio::test]
    async fn tooling_engine_returns_immediately_when_no_tool_call() {
        let engine = ToolingEngine::new(Arc::new(MockToolService {
            result: Value::Null,
        }));

        let inference = Arc::new(TwoStepInference {
            first: "No tool needed.".to_string(),
            second: "ignored".to_string(),
        });

        let result = engine.run(inference, request()).await.unwrap();

        assert_eq!(result.content, "No tool needed.");
    }
}
