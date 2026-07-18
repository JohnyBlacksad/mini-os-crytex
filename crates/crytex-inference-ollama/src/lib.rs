use async_trait::async_trait;
use crytex_inference::{
    BackendInfo, InferenceError, InferenceManager, InferenceRequest, InferenceResponse,
    LoRAAdapter, ModelInfo, TokenUsage, embedding_http_error, generation_http_error,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

const DEFAULT_TIMEOUT_SECONDS: u64 = 120;

pub struct OllamaBackend {
    client: reqwest::Client,
    url: String,
    model: String,
}

impl OllamaBackend {
    pub fn new(url: impl Into<String>, model: impl Into<String>) -> Self {
        let url = url.into();
        let model = model.into();
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(DEFAULT_TIMEOUT_SECONDS))
            .build()
            .unwrap_or_default();
        Self { client, url, model }
    }

    pub fn with_default_model(model: impl Into<String>) -> Self {
        Self::new("http://localhost:11434", model)
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    async fn chat(&self, request: InferenceRequest) -> Result<ChatResponse, InferenceError> {
        let mut messages = Vec::new();
        if let Some(system) = request.system_prompt {
            messages.push(OllamaMessage {
                role: "system".to_string(),
                content: system,
            });
        }
        for msg in request.messages {
            messages.push(OllamaMessage {
                role: msg.role,
                content: msg.content,
            });
        }

        let body = ChatRequest {
            model: request.model.clone(),
            messages,
            stream: false,
            think: Some(false),
            options: json!({
                "temperature": request.temperature.unwrap_or(0.7),
                "num_predict": request.max_tokens.unwrap_or(4096),
            }),
        };

        let response = self
            .client
            .post(format!("{}/api/chat", self.url))
            .json(&body)
            .send()
            .await
            .map_err(|e| InferenceError::GenerationFailed(format!("ollama request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let text = response
                .text()
                .await
                .unwrap_or_else(|_| "unknown error".to_string());
            return Err(generation_http_error(status, &text));
        }

        response.json::<ChatResponse>().await.map_err(|e| {
            InferenceError::GenerationFailed(format!("failed to parse ollama response: {e}"))
        })
    }

    async fn fetch_models(&self) -> Result<Vec<String>, InferenceError> {
        let response = self
            .client
            .get(format!("{}/api/tags", self.url))
            .send()
            .await
            .map_err(|e| InferenceError::ModelNotFound(format!("failed to list models: {e}")))?;

        if !response.status().is_success() {
            return Ok(vec![]);
        }

        let tags: ModelTagsResponse = response
            .json()
            .await
            .map_err(|e| InferenceError::ModelNotFound(format!("failed to parse models: {e}")))?;

        Ok(tags.models.into_iter().map(|m| m.name).collect())
    }
}

#[async_trait]
impl InferenceManager for OllamaBackend {
    async fn generate(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceResponse, InferenceError> {
        let model = if request.model.is_empty() || request.model == "default" {
            self.model.clone()
        } else {
            request.model.clone()
        };

        let mut ollama_request = request;
        ollama_request.model = model;

        let response = self.chat(ollama_request).await?;
        let content = response.message.content;
        let prompt_tokens = response.prompt_eval_count.unwrap_or(0);
        let completion_tokens = response.eval_count.unwrap_or(0);

        Ok(InferenceResponse {
            content,
            usage: TokenUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
            finish_reason: if response.done {
                "stop".to_string()
            } else {
                "length".to_string()
            },
        })
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>, InferenceError> {
        let body = json!({
            "model": self.model,
            "prompt": text,
        });

        let response = self
            .client
            .post(format!("{}/api/embeddings", self.url))
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                InferenceError::EmbeddingFailed(format!("ollama embed request failed: {e}"))
            })?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let text = response.text().await.unwrap_or_default();
            return Err(embedding_http_error(status, &text));
        }

        let embed: EmbeddingResponse = response.json().await.map_err(|e| {
            InferenceError::EmbeddingFailed(format!("failed to parse embed response: {e}"))
        })?;

        Ok(embed.embedding)
    }

    async fn register_lora(&self, _lora: LoRAAdapter) -> Result<(), InferenceError> {
        Err(InferenceError::UnsupportedOperation(
            "Ollama does not support runtime LoRA adapters".to_string(),
        ))
    }

    async fn swap_lora(&self, _lora_id: &str) -> Result<(), InferenceError> {
        Err(InferenceError::UnsupportedOperation(
            "Ollama does not support runtime LoRA adapters".to_string(),
        ))
    }

    fn available_backends(&self) -> Vec<BackendInfo> {
        let capabilities = vec!["generate".to_string(), "chat".to_string()];
        vec![BackendInfo {
            id: "ollama".to_string(),
            name: format!("Ollama ({})", self.model),
            capabilities,
        }]
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, InferenceError> {
        let names = self.fetch_models().await?;
        Ok(names
            .into_iter()
            .map(|name| ModelInfo {
                id: name.clone(),
                name,
            })
            .collect())
    }
}

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    think: Option<bool>,
    options: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct OllamaMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    #[allow(dead_code)]
    model: String,
    message: OllamaMessage,
    done: bool,
    #[serde(default)]
    prompt_eval_count: Option<usize>,
    #[serde(default)]
    eval_count: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ModelTagsResponse {
    models: Vec<OllamaModelInfo>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OllamaModelInfo {
    name: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_creation() {
        let backend = OllamaBackend::with_default_model("qwen2.5-coder:14b");
        assert_eq!(backend.model(), "qwen2.5-coder:14b");
        assert_eq!(backend.url(), "http://localhost:11434");
    }

    #[tokio::test]
    async fn lora_is_unsupported() {
        let backend = OllamaBackend::with_default_model("qwen2.5-coder:14b");
        let err = backend
            .register_lora(LoRAAdapter {
                id: "1".to_string(),
                path: "/tmp".to_string(),
                base_model: "base".to_string(),
            })
            .await
            .unwrap_err();

        assert!(
            matches!(err, InferenceError::UnsupportedOperation(message) if message.contains("Ollama"))
        );
    }
}
