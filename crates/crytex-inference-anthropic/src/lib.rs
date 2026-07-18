use async_trait::async_trait;
use crytex_inference::{
    BackendInfo, InferenceError, InferenceManager, InferenceRequest, InferenceResponse,
    LoRAAdapter, ModelInfo, TokenUsage, generation_http_error,
};
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};

/// Anthropic Messages API inference backend.
#[derive(Debug, Clone)]
pub struct AnthropicBackend {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    default_model: String,
}

impl AnthropicBackend {
    /// Creates a new Anthropic backend.
    ///
    /// `base_url` should not include a trailing slash, e.g. `https://api.anthropic.com/v1`.
    pub fn new(
        base_url: impl Into<String>,
        default_model: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.into(),
            api_key: api_key.into(),
            default_model: default_model.into(),
        }
    }

    fn build_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            "x-api-key",
            HeaderValue::from_str(&self.api_key).unwrap_or_else(|_| HeaderValue::from_static("")),
        );
        headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        headers
    }

    fn resolve_model(&self, request_model: &str) -> String {
        if request_model.is_empty() || request_model == "default" {
            self.default_model.clone()
        } else {
            request_model.to_string()
        }
    }

    fn build_messages(request: &InferenceRequest) -> Vec<AnthropicMessage> {
        request
            .messages
            .iter()
            .map(|m| AnthropicMessage {
                role: m.role.clone(),
                content: m.content.clone(),
            })
            .collect()
    }
}

#[async_trait]
impl InferenceManager for AnthropicBackend {
    async fn generate(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceResponse, InferenceError> {
        let model = self.resolve_model(&request.model);
        let messages = Self::build_messages(&request);
        let max_tokens = request.max_tokens.unwrap_or(4096);

        let body = MessagesRequest {
            model,
            max_tokens,
            messages,
            system: request.system_prompt,
            temperature: request.temperature,
        };

        let url = format!("{}/messages", self.base_url);
        let response = self
            .client
            .post(&url)
            .headers(self.build_headers())
            .json(&body)
            .send()
            .await
            .map_err(|e| InferenceError::GenerationFailed(format!("HTTP error: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let text = response
                .text()
                .await
                .unwrap_or_else(|_| "<failed to read body>".to_string());
            return Err(generation_http_error(status, &text));
        }

        let messages_response: MessagesResponse = response
            .json()
            .await
            .map_err(|e| InferenceError::GenerationFailed(format!("JSON decode error: {}", e)))?;

        let content = messages_response
            .content
            .into_iter()
            .filter_map(|block| {
                if block.content_type == "text" {
                    Some(block.text)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("");

        Ok(InferenceResponse {
            content,
            usage: TokenUsage {
                prompt_tokens: messages_response.usage.input_tokens,
                completion_tokens: messages_response.usage.output_tokens,
                total_tokens: messages_response.usage.input_tokens
                    + messages_response.usage.output_tokens,
            },
            finish_reason: messages_response
                .stop_reason
                .unwrap_or_else(|| "stop".to_string()),
        })
    }

    async fn embed(&self, _text: &str) -> Result<Vec<f32>, InferenceError> {
        Err(InferenceError::EmbeddingFailed(
            "Anthropic does not provide embeddings".to_string(),
        ))
    }

    async fn register_lora(&self, _lora: LoRAAdapter) -> Result<(), InferenceError> {
        Err(InferenceError::UnsupportedOperation(
            "Anthropic does not support LoRA".to_string(),
        ))
    }

    async fn swap_lora(&self, _lora_id: &str) -> Result<(), InferenceError> {
        Err(InferenceError::UnsupportedOperation(
            "Anthropic does not support LoRA".to_string(),
        ))
    }

    fn available_backends(&self) -> Vec<BackendInfo> {
        vec![BackendInfo {
            id: "anthropic".to_string(),
            name: format!("Anthropic ({})", self.default_model),
            capabilities: vec!["generate".to_string()],
        }]
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, InferenceError> {
        // Anthropic does not expose a public model list endpoint.
        Ok(vec![ModelInfo {
            id: self.default_model.clone(),
            name: self.default_model.clone(),
        }])
    }
}

#[derive(Debug, Serialize)]
struct MessagesRequest {
    model: String,
    max_tokens: usize,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct MessagesResponse {
    content: Vec<ContentBlock>,
    usage: AnthropicUsage,
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    content_type: String,
    text: String,
}

#[derive(Debug, Deserialize, Default)]
struct AnthropicUsage {
    input_tokens: usize,
    output_tokens: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crytex_inference::Message;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn request() -> InferenceRequest {
        InferenceRequest {
            backend_id: None,
            model: "default".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: "Hello".to_string(),
            }],
            system_prompt: Some("You are a tester".to_string()),
            temperature: Some(0.5),
            max_tokens: Some(16),
            lora_adapter_id: None,
        }
    }

    #[tokio::test]
    async fn generate_forwards_request_and_parses_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/messages"))
            .and(header("x-api-key", "test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "content": [{ "type": "text", "text": "Hi there" }],
                "usage": { "input_tokens": 10, "output_tokens": 2 },
                "stop_reason": "end_turn"
            })))
            .mount(&server)
            .await;

        let backend = AnthropicBackend::new(server.uri(), "claude-sonnet-4", "test-key");
        let response = backend.generate(request()).await.unwrap();

        assert_eq!(response.content, "Hi there");
        assert_eq!(response.usage.prompt_tokens, 10);
        assert_eq!(response.finish_reason, "end_turn");
    }

    #[tokio::test]
    async fn generate_returns_error_on_api_failure() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/messages"))
            .respond_with(ResponseTemplate::new(500).set_body_string("Internal error"))
            .mount(&server)
            .await;

        let backend = AnthropicBackend::new(server.uri(), "claude-sonnet-4", "key");
        let result = backend.generate(request()).await;
        assert!(result.is_err());
    }

    #[test]
    fn lora_is_unsupported() {
        let backend = AnthropicBackend::new("http://localhost", "model", "key");
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let err = backend
                .register_lora(LoRAAdapter {
                    id: "1".to_string(),
                    path: "/tmp".to_string(),
                    base_model: "base".to_string(),
                })
                .await
                .unwrap_err();
            assert!(
                matches!(err, InferenceError::UnsupportedOperation(message) if message.contains("Anthropic"))
            );
        });
    }
}
