use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod registry;

pub use registry::{BackendRegistry, RegistryError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct InferenceRequest {
    /// Optional backend id. If `None`, the caller should use the default backend.
    pub backend_id: Option<String>,
    pub model: String,
    pub messages: Vec<Message>,
    pub system_prompt: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
    /// Optional LoRA adapter id to apply for this request.
    /// When `None`, the backend uses its globally active adapter (if any).
    pub lora_adapter_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct InferenceResponse {
    pub content: String,
    pub usage: TokenUsage,
    pub finish_reason: String,
}

#[derive(Debug, Clone)]
pub struct BackendInfo {
    pub id: String,
    pub name: String,
    pub capabilities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct LoRAAdapter {
    pub id: String,
    pub path: String,
    pub base_model: String,
}

#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("model not found: {0}")]
    ModelNotFound(String),
    #[error("generation failed: {0}")]
    GenerationFailed(String),
    #[error("LoRA load failed: {0}")]
    LoRALoadFailed(String),
    #[error("embedding failed: {0}")]
    EmbeddingFailed(String),
    #[error("backend not available: {0}")]
    BackendNotAvailable(String),
    /// The backend reported a rate limit. The caller may retry after the
    /// suggested delay (in milliseconds) has elapsed.
    #[error("rate limited")]
    RateLimited {
        /// Suggested wait time before the next retry, if provided by the backend.
        retry_after_ms: Option<u64>,
    },
    /// A transient HTTP error (5xx) that may succeed on retry.
    #[error("transient HTTP error {status}: {body}")]
    Transient { status: u16, body: String },
}

/// Classify an HTTP status code from a generation request into an
/// [`InferenceError`]. 429 and 5xx are treated as retriable.
pub fn generation_http_error(status: u16, body: &str) -> InferenceError {
    match status {
        429 => InferenceError::RateLimited {
            retry_after_ms: None,
        },
        500..=599 => InferenceError::Transient {
            status,
            body: body.to_string(),
        },
        _ => InferenceError::GenerationFailed(format!("API error {}: {}", status, body)),
    }
}

/// Classify an HTTP status code from an embedding request.
pub fn embedding_http_error(status: u16, body: &str) -> InferenceError {
    match status {
        429 => InferenceError::RateLimited {
            retry_after_ms: None,
        },
        500..=599 => InferenceError::Transient {
            status,
            body: body.to_string(),
        },
        _ => InferenceError::EmbeddingFailed(format!("API error {}: {}", status, body)),
    }
}

/// Classify an HTTP status code from a model-listing request.
pub fn model_http_error(status: u16, body: &str) -> InferenceError {
    match status {
        429 => InferenceError::RateLimited {
            retry_after_ms: None,
        },
        500..=599 => InferenceError::Transient {
            status,
            body: body.to_string(),
        },
        _ => InferenceError::ModelNotFound(format!("API error {}: {}", status, body)),
    }
}

#[async_trait]
pub trait InferenceManager: Send + Sync {
    async fn generate(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceResponse, InferenceError>;
    async fn embed(&self, text: &str) -> Result<Vec<f32>, InferenceError>;
    async fn register_lora(&self, lora: LoRAAdapter) -> Result<(), InferenceError>;
    async fn swap_lora(&self, lora_id: &str) -> Result<(), InferenceError>;
    fn available_backends(&self) -> Vec<BackendInfo>;
    /// Lists models available through this backend.
    async fn list_models(&self) -> Result<Vec<ModelInfo>, InferenceError>;
}
