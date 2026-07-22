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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BackendCapabilityReport {
    pub id: String,
    pub name: String,
    pub generate: bool,
    pub chat: bool,
    pub embed: bool,
    pub rerank: bool,
    pub lora: bool,
    pub hot_swap: bool,
}

impl BackendInfo {
    pub fn capability_report(&self) -> BackendCapabilityReport {
        BackendCapabilityReport {
            id: self.id.clone(),
            name: self.name.clone(),
            generate: self.has_capability("generate"),
            chat: self.has_capability("chat"),
            embed: self.has_capability("embed"),
            rerank: self.has_capability("rerank"),
            lora: self.has_capability("lora"),
            hot_swap: self.has_capability("hot_swap") || self.has_capability("lora_hot_swap"),
        }
    }

    fn has_capability(&self, capability: &str) -> bool {
        self.capabilities.iter().any(|item| item == capability)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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
    #[error("unsupported operation: {0}")]
    UnsupportedOperation(String),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_info_builds_typed_capability_report() {
        let info = BackendInfo {
            id: "mistralrs".into(),
            name: "mistral.rs".into(),
            capabilities: vec![
                "generate".into(),
                "chat".into(),
                "embed".into(),
                "rerank".into(),
                "lora".into(),
                "hot_swap".into(),
            ],
        };

        let report = info.capability_report();

        assert_eq!(report.id, "mistralrs");
        assert!(report.generate);
        assert!(report.chat);
        assert!(report.embed);
        assert!(report.rerank);
        assert!(report.lora);
        assert!(report.hot_swap);
    }

    #[test]
    fn backend_info_does_not_infer_hot_swap_from_lora() {
        let info = BackendInfo {
            id: "mistralrs".into(),
            name: "mistral.rs".into(),
            capabilities: vec!["generate".into(), "lora".into()],
        };

        let report = info.capability_report();

        assert!(report.lora);
        assert!(!report.hot_swap);
    }
}
