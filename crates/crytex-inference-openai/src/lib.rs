use async_trait::async_trait;
use crytex_inference::{
    BackendInfo, InferenceError, InferenceManager, InferenceRequest, InferenceResponse,
    LoRAAdapter, ModelInfo, TokenUsage, embedding_http_error, generation_http_error,
    model_http_error,
};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// OpenAI-compatible inference backend.
///
/// Supports any provider implementing the `/v1/chat/completions` and
/// `/v1/embeddings` endpoints, including OpenAI, OpenRouter, vLLM, LM Studio,
/// and other compatible services.
#[derive(Debug, Clone)]
pub struct OpenAiBackend {
    client: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    default_model: String,
    headers: HashMap<String, String>,
}

impl OpenAiBackend {
    /// Creates a new backend.
    ///
    /// `base_url` should not include a trailing slash, e.g. `https://api.openai.com/v1`.
    pub fn new(
        base_url: impl Into<String>,
        default_model: impl Into<String>,
        api_key: Option<String>,
    ) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.into(),
            api_key,
            default_model: default_model.into(),
            headers: HashMap::new(),
        }
    }

    /// Sets custom HTTP headers sent with every request.
    pub fn with_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.headers = headers;
        self
    }

    fn build_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if let Some(key) = &self.api_key
            && let Ok(value) = HeaderValue::from_str(&format!("Bearer {}", key))
        {
            headers.insert(AUTHORIZATION, value);
        }
        for (name, value) in &self.headers {
            if let Ok(name) = HeaderName::from_bytes(name.as_bytes())
                && let Ok(value) = HeaderValue::from_str(value)
            {
                headers.insert(name, value);
            }
        }
        headers
    }

    fn resolve_model(&self, request_model: &str) -> String {
        if request_model.is_empty() || request_model == "default" {
            self.default_model.clone()
        } else {
            request_model.to_string()
        }
    }

    fn build_messages(request: &InferenceRequest) -> Vec<OpenAiMessage> {
        let mut messages = Vec::new();
        if let Some(system) = &request.system_prompt {
            messages.push(OpenAiMessage {
                role: "system".to_string(),
                content: system.clone(),
            });
        }
        messages.extend(request.messages.iter().map(|m| OpenAiMessage {
            role: m.role.clone(),
            content: m.content.clone(),
        }));
        messages
    }
}

#[async_trait]
impl InferenceManager for OpenAiBackend {
    async fn generate(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceResponse, InferenceError> {
        let model = self.resolve_model(&request.model);
        let messages = Self::build_messages(&request);

        let body = ChatCompletionRequest {
            model,
            messages,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            stream: Some(false),
        };

        let url = format!("{}/chat/completions", self.base_url);
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

        let completion: ChatCompletionResponse = response
            .json()
            .await
            .map_err(|e| InferenceError::GenerationFailed(format!("JSON decode error: {}", e)))?;

        let choice = completion.choices.into_iter().next().ok_or_else(|| {
            InferenceError::GenerationFailed("No completion choices returned".to_string())
        })?;

        let usage = completion.usage.unwrap_or_default();

        Ok(InferenceResponse {
            content: choice.message.content,
            usage: TokenUsage {
                prompt_tokens: usage.prompt_tokens,
                completion_tokens: usage.completion_tokens,
                total_tokens: usage.total_tokens,
            },
            finish_reason: choice.finish_reason.unwrap_or_else(|| "stop".to_string()),
        })
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>, InferenceError> {
        let body = EmbeddingRequest {
            model: self.default_model.clone(),
            input: text.to_string(),
        };

        let url = format!("{}/embeddings", self.base_url);
        let response = self
            .client
            .post(&url)
            .headers(self.build_headers())
            .json(&body)
            .send()
            .await
            .map_err(|e| InferenceError::EmbeddingFailed(format!("HTTP error: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let text = response
                .text()
                .await
                .unwrap_or_else(|_| "<failed to read body>".to_string());
            return Err(embedding_http_error(status, &text));
        }

        let embedding_response: EmbeddingResponse = response
            .json()
            .await
            .map_err(|e| InferenceError::EmbeddingFailed(format!("JSON decode error: {}", e)))?;

        embedding_response
            .data
            .into_iter()
            .next()
            .map(|d| d.embedding)
            .ok_or_else(|| InferenceError::EmbeddingFailed("No embedding returned".to_string()))
    }

    async fn register_lora(&self, _lora: LoRAAdapter) -> Result<(), InferenceError> {
        Err(InferenceError::UnsupportedOperation(
            "OpenAI-compatible backends do not support LoRA".to_string(),
        ))
    }

    async fn swap_lora(&self, _lora_id: &str) -> Result<(), InferenceError> {
        Err(InferenceError::UnsupportedOperation(
            "OpenAI-compatible backends do not support LoRA".to_string(),
        ))
    }

    fn available_backends(&self) -> Vec<BackendInfo> {
        vec![BackendInfo {
            id: "openai".to_string(),
            name: format!("OpenAI-compatible ({})", self.default_model),
            capabilities: vec!["generate".to_string(), "embed".to_string()],
        }]
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, InferenceError> {
        let response = self
            .client
            .get(format!("{}/models", self.base_url))
            .headers(self.build_headers())
            .send()
            .await
            .map_err(|e| InferenceError::ModelNotFound(format!("HTTP error: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let text = response
                .text()
                .await
                .unwrap_or_else(|_| "<failed to read body>".to_string());
            return Err(model_http_error(status, &text));
        }

        let models: ModelsResponse = response
            .json()
            .await
            .map_err(|e| InferenceError::ModelNotFound(format!("JSON decode error: {}", e)))?;

        Ok(models
            .data
            .into_iter()
            .map(|m| ModelInfo {
                id: m.id.clone(),
                name: m.id,
            })
            .collect())
    }
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAiMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: OpenAiMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    model: String,
    input: String,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct ModelsResponse {
    data: Vec<ModelEntry>,
}

#[derive(Debug, Deserialize)]
struct ModelEntry {
    id: String,
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
            .and(path("/chat/completions"))
            .and(header("Authorization", "Bearer test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{
                    "message": { "role": "assistant", "content": "Hi there" },
                    "finish_reason": "stop"
                }],
                "usage": { "prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12 }
            })))
            .mount(&server)
            .await;

        let backend = OpenAiBackend::new(server.uri(), "gpt-4o-mini", Some("test-key".to_string()));
        let response = backend.generate(request()).await.unwrap();

        assert_eq!(response.content, "Hi there");
        assert_eq!(response.usage.prompt_tokens, 10);
        assert_eq!(response.finish_reason, "stop");
    }

    #[tokio::test]
    async fn generate_forwards_custom_headers() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("X-Custom", "value"))
            .and(header("Authorization", "Bearer token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{
                    "message": { "role": "assistant", "content": "Hi there" },
                    "finish_reason": "stop"
                }],
                "usage": { "prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12 }
            })))
            .mount(&server)
            .await;

        let mut headers = std::collections::HashMap::new();
        headers.insert("X-Custom".to_string(), "value".to_string());
        headers.insert("Authorization".to_string(), "Bearer token".to_string());
        let backend = OpenAiBackend::new(server.uri(), "gpt-4o-mini", None).with_headers(headers);
        let response = backend.generate(request()).await.unwrap();

        assert_eq!(response.content, "Hi there");
    }

    #[tokio::test]
    async fn list_models_forwards_request_and_parses_response() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .and(header("Authorization", "Bearer test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [
                    { "id": "gpt-4o-mini" },
                    { "id": "gpt-4o" }
                ]
            })))
            .mount(&server)
            .await;

        let backend = OpenAiBackend::new(server.uri(), "default", Some("test-key".to_string()));
        let models = backend.list_models().await.unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "gpt-4o-mini");
        assert_eq!(models[1].id, "gpt-4o");
    }

    #[tokio::test]
    async fn generate_returns_error_on_api_failure() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(500).set_body_string("Internal error"))
            .mount(&server)
            .await;

        let backend = OpenAiBackend::new(server.uri(), "gpt-4o-mini", None);
        let result = backend.generate(request()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn embed_forwards_request_and_parses_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{ "embedding": [0.1, 0.2, 0.3] }]
            })))
            .mount(&server)
            .await;

        let backend = OpenAiBackend::new(server.uri(), "text-embedding-3-small", None);
        let embedding = backend.embed("hello").await.unwrap();
        assert_eq!(embedding, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn lora_is_unsupported() {
        let backend = OpenAiBackend::new("http://localhost", "model", None);
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
                matches!(err, InferenceError::UnsupportedOperation(message) if message.contains("OpenAI-compatible"))
            );
        });
    }
}
