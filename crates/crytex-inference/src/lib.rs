use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;
pub mod mock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct GenerationRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct GenerationResponse {
    pub content: String,
    pub usage: TokenUsage,
}

#[derive(Debug, Clone)]
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
}

#[async_trait]
pub trait InferenceBackend: Send + Sync {
    async fn generate(&self, request: GenerationRequest) -> Result<GenerationResponse, InferenceError>;
    async fn embed(&self, text: &str) -> Result<Vec<f32>, InferenceError>;
    async fn list_models(&self) -> Result<Vec<ModelInfo>, InferenceError>;
    async fn load_lora(&self, adapter: &LoRAAdapter) -> Result<(), InferenceError>;
    async fn unload_lora(&self, id: &str) -> Result<(), InferenceError>;
}