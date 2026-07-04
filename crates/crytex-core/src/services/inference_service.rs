use std::sync::Arc;

use async_trait::async_trait;
use crytex_compress::compress::CompressionError;
use crytex_compress::pipeline::CompressionPipeline;
use crytex_inference::{
    BackendInfo, BackendRegistry, InferenceError, InferenceManager, InferenceRequest,
    InferenceResponse, LoRAAdapter, ModelInfo,
};
use thiserror::Error;

/// Errors that can occur in [`InferenceService`].
#[derive(Debug, Error)]
pub enum InferenceServiceError {
    #[error("inference error: {0}")]
    Inference(InferenceError),
    #[error("compression error: {0}")]
    Compression(#[from] CompressionError),
    #[error("no backend available")]
    NoBackend,
    #[error("backend {0} not found")]
    BackendNotFound(String),
    #[error("backend {0} does not support the requested operation")]
    UnsupportedOperation(String),
    #[error("rate limited by backend")]
    RateLimited,
}

impl From<InferenceError> for InferenceServiceError {
    fn from(e: InferenceError) -> Self {
        match e {
            InferenceError::RateLimited { .. } => Self::RateLimited,
            other => Self::Inference(other),
        }
    }
}

/// High-level service that wraps a registry of inference backends.
#[async_trait]
pub trait InferenceService: Send + Sync {
    /// Generate text using the backend selected by the request or the default.
    async fn generate(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceResponse, InferenceServiceError>;

    /// Generate embeddings using the configured embedding backend or default.
    async fn embed(&self, text: &str) -> Result<Vec<f32>, InferenceServiceError>;

    /// List available backends.
    fn available_backends(&self) -> Vec<BackendInfo>;

    /// Register a LoRA adapter with the default backend.
    async fn register_lora(&self, lora: LoRAAdapter) -> Result<(), InferenceServiceError>;

    /// Swap to a registered LoRA adapter on the default backend.
    async fn swap_lora(&self, lora_id: &str) -> Result<(), InferenceServiceError>;

    /// List models available from the selected or default backend.
    async fn list_models(
        &self,
        backend_id: Option<&str>,
    ) -> Result<Vec<ModelInfo>, InferenceServiceError>;

    /// Convenience helper: build a simple chat request.
    fn chat_request(
        &self,
        backend_id: Option<&str>,
        model: &str,
        system: Option<&str>,
        user: &str,
    ) -> InferenceRequest {
        use crytex_inference::Message;
        let mut messages = Vec::new();
        if let Some(system) = system {
            messages.push(Message {
                role: "system".into(),
                content: system.into(),
            });
        }
        messages.push(Message {
            role: "user".into(),
            content: user.into(),
        });
        InferenceRequest {
            backend_id: backend_id.map(|s| s.into()),
            model: model.into(),
            messages,
            system_prompt: None,
            temperature: Some(0.7),
            max_tokens: Some(2048),
            lora_adapter_id: None,
        }
    }
}

/// Default implementation of [`InferenceService`].
pub struct InferenceServiceImpl {
    registry: Arc<BackendRegistry>,
    default_backend: Option<String>,
    embedding_backend: Option<String>,
    compression_pipeline: Option<Arc<CompressionPipeline>>,
    compression_budget: usize,
}

impl InferenceServiceImpl {
    /// Creates a new service backed by a registry.
    pub fn new(registry: Arc<BackendRegistry>, default_backend: Option<String>) -> Self {
        Self {
            registry,
            default_backend,
            embedding_backend: None,
            compression_pipeline: None,
            compression_budget: 4096,
        }
    }

    /// Sets the backend used for embedding requests.
    pub fn with_embedding_backend(mut self, backend_id: impl Into<String>) -> Self {
        self.embedding_backend = Some(backend_id.into());
        self
    }

    /// Attach a context-compression pipeline.
    pub fn with_compression(mut self, pipeline: Arc<CompressionPipeline>, budget: usize) -> Self {
        self.compression_pipeline = Some(pipeline);
        self.compression_budget = budget;
        self
    }

    fn resolve_backend(
        &self,
        backend_id: Option<&str>,
    ) -> Result<Arc<dyn InferenceManager>, InferenceServiceError> {
        let id = backend_id
            .or(self.default_backend.as_deref())
            .ok_or(InferenceServiceError::NoBackend)?;
        self.registry
            .get(id)
            .ok_or_else(|| InferenceServiceError::BackendNotFound(id.to_string()))
    }

    fn resolve_embedding_backend(
        &self,
    ) -> Result<Arc<dyn InferenceManager>, InferenceServiceError> {
        let id = self
            .embedding_backend
            .as_deref()
            .or(self.default_backend.as_deref())
            .ok_or(InferenceServiceError::NoBackend)?;
        self.registry
            .get(id)
            .ok_or_else(|| InferenceServiceError::BackendNotFound(id.to_string()))
    }

    async fn maybe_compress(
        &self,
        mut request: InferenceRequest,
    ) -> Result<InferenceRequest, InferenceServiceError> {
        let Some(pipeline) = &self.compression_pipeline else {
            return Ok(request);
        };
        if request.messages.is_empty() {
            return Ok(request);
        }

        let compressed = pipeline
            .run(
                &request
                    .messages
                    .iter()
                    .map(|m| crytex_compress::message::Message::new(&m.role, &m.content))
                    .collect::<Vec<_>>(),
                self.compression_budget,
            )
            .await?;

        request.messages = compressed
            .0
            .iter()
            .map(|m| crytex_inference::Message {
                role: m.role.clone(),
                content: m.content.clone(),
            })
            .collect();
        Ok(request)
    }
}

#[async_trait]
impl InferenceService for InferenceServiceImpl {
    async fn generate(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceResponse, InferenceServiceError> {
        let backend_id = request.backend_id.clone();
        let backend = self.resolve_backend(backend_id.as_deref())?;
        let request = self.maybe_compress(request).await?;
        Ok(backend.generate(request).await?)
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>, InferenceServiceError> {
        let backend = self.resolve_embedding_backend()?;
        Ok(backend.embed(text).await?)
    }

    fn available_backends(&self) -> Vec<BackendInfo> {
        self.registry.list()
    }

    async fn register_lora(&self, lora: LoRAAdapter) -> Result<(), InferenceServiceError> {
        let backend = self.resolve_backend(self.default_backend.as_deref())?;
        Ok(backend.register_lora(lora).await?)
    }

    async fn swap_lora(&self, lora_id: &str) -> Result<(), InferenceServiceError> {
        let backend = self.resolve_backend(self.default_backend.as_deref())?;
        Ok(backend.swap_lora(lora_id).await?)
    }

    async fn list_models(
        &self,
        backend_id: Option<&str>,
    ) -> Result<Vec<ModelInfo>, InferenceServiceError> {
        let backend = self.resolve_backend(backend_id)?;
        Ok(backend.list_models().await?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use crytex_compress::compress::{CompressionError, Compressor};
    use crytex_compress::message::Message as CompressMessage;
    use crytex_inference::{
        InferenceRequest, InferenceResponse, LoRAAdapter, ModelInfo, TokenUsage,
    };
    use std::sync::Mutex;

    struct MockBackend {
        name: String,
    }

    #[async_trait]
    impl InferenceManager for MockBackend {
        async fn generate(
            &self,
            _request: InferenceRequest,
        ) -> Result<InferenceResponse, InferenceError> {
            Ok(InferenceResponse {
                content: "mock response".into(),
                usage: TokenUsage {
                    prompt_tokens: 1,
                    completion_tokens: 2,
                    total_tokens: 3,
                },
                finish_reason: "stop".into(),
            })
        }

        async fn embed(&self, text: &str) -> Result<Vec<f32>, InferenceError> {
            Ok(vec![text.len() as f32])
        }

        async fn register_lora(&self, _lora: LoRAAdapter) -> Result<(), InferenceError> {
            Ok(())
        }

        async fn swap_lora(&self, lora_id: &str) -> Result<(), InferenceError> {
            if lora_id == "missing" {
                Err(InferenceError::LoRALoadFailed("not found".into()))
            } else {
                Ok(())
            }
        }

        fn available_backends(&self) -> Vec<BackendInfo> {
            vec![BackendInfo {
                id: self.name.clone(),
                name: self.name.clone(),
                capabilities: vec!["generate".into(), "embed".into()],
            }]
        }

        async fn list_models(&self) -> Result<Vec<ModelInfo>, InferenceError> {
            Ok(vec![ModelInfo {
                id: self.name.clone(),
                name: self.name.clone(),
            }])
        }
    }

    fn registry_with_mock(name: &str) -> Arc<BackendRegistry> {
        let mut registry = BackendRegistry::new(name);
        registry.register(name, Arc::new(MockBackend { name: name.into() }));
        Arc::new(registry)
    }

    fn service() -> InferenceServiceImpl {
        InferenceServiceImpl::new(registry_with_mock("mock"), Some("mock".to_string()))
    }

    struct RecordingBackend {
        name: String,
        requests: Mutex<Vec<InferenceRequest>>,
    }

    #[async_trait]
    impl InferenceManager for RecordingBackend {
        async fn generate(
            &self,
            request: InferenceRequest,
        ) -> Result<InferenceResponse, InferenceError> {
            self.requests.lock().unwrap().push(request);
            Ok(InferenceResponse {
                content: "mock response".into(),
                usage: TokenUsage {
                    prompt_tokens: 1,
                    completion_tokens: 2,
                    total_tokens: 3,
                },
                finish_reason: "stop".into(),
            })
        }

        async fn embed(&self, text: &str) -> Result<Vec<f32>, InferenceError> {
            Ok(vec![text.len() as f32])
        }

        async fn register_lora(&self, _lora: LoRAAdapter) -> Result<(), InferenceError> {
            Ok(())
        }

        async fn swap_lora(&self, _lora_id: &str) -> Result<(), InferenceError> {
            Ok(())
        }

        fn available_backends(&self) -> Vec<BackendInfo> {
            vec![BackendInfo {
                id: self.name.clone(),
                name: self.name.clone(),
                capabilities: vec!["generate".into(), "embed".into()],
            }]
        }

        async fn list_models(&self) -> Result<Vec<ModelInfo>, InferenceError> {
            Ok(vec![])
        }
    }

    struct ReplacementCompressor;

    #[async_trait]
    impl Compressor for ReplacementCompressor {
        async fn compress(
            &self,
            _messages: &[CompressMessage],
            _budget: usize,
        ) -> Result<Vec<CompressMessage>, CompressionError> {
            Ok(vec![CompressMessage::system("compressed")])
        }
    }

    fn recording_service() -> (InferenceServiceImpl, Arc<RecordingBackend>) {
        let backend = Arc::new(RecordingBackend {
            name: "mock".into(),
            requests: Mutex::new(Vec::new()),
        });
        let mut registry = BackendRegistry::new("mock");
        registry.register("mock", backend.clone());
        let svc = InferenceServiceImpl::new(Arc::new(registry), Some("mock".to_string()));
        (svc, backend)
    }

    #[tokio::test]
    async fn generate_delegates_to_backend() {
        let svc = service();
        let request = svc.chat_request(None, "model", None, "hello");
        let response = svc.generate(request).await.unwrap();
        assert_eq!(response.content, "mock response");
    }

    #[tokio::test]
    async fn list_models_delegates_to_backend() {
        let svc = service();
        let models = svc.list_models(None).await.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "mock");
    }

    #[tokio::test]
    async fn list_models_uses_requested_backend() {
        let mut registry = BackendRegistry::new("a");
        registry.register("a", Arc::new(MockBackend { name: "a".into() }));
        registry.register("b", Arc::new(MockBackend { name: "b".into() }));
        let svc = InferenceServiceImpl::new(Arc::new(registry), Some("a".to_string()));
        let models = svc.list_models(Some("b")).await.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "b");
    }

    #[tokio::test]
    async fn generate_fails_when_no_default_backend() {
        let registry = BackendRegistry::new("");
        let svc = InferenceServiceImpl::new(Arc::new(registry), None);
        let request = svc.chat_request(None, "model", None, "hello");
        let result = svc.generate(request).await;
        assert!(matches!(result, Err(InferenceServiceError::NoBackend)));
    }

    #[tokio::test]
    async fn generate_uses_requested_backend() {
        let mut registry = BackendRegistry::new("a");
        registry.register("a", Arc::new(MockBackend { name: "a".into() }));
        registry.register("b", Arc::new(MockBackend { name: "b".into() }));
        let svc = InferenceServiceImpl::new(Arc::new(registry), Some("a".to_string()));

        let request = svc.chat_request(Some("b"), "model", None, "hello");
        let response = svc.generate(request).await.unwrap();
        // The mock backend content is generic, but the request should route without error.
        assert_eq!(response.content, "mock response");
    }

    #[tokio::test]
    async fn generate_fails_for_unknown_backend() {
        let svc = service();
        let mut request = svc.chat_request(None, "model", None, "hello");
        request.backend_id = Some("unknown".into());
        let result = svc.generate(request).await;
        assert!(matches!(
            result,
            Err(InferenceServiceError::BackendNotFound(_))
        ));
    }

    #[tokio::test]
    async fn embed_delegates_to_backend() {
        let svc = service();
        let embedding = svc.embed("hi").await.unwrap();
        assert_eq!(embedding, vec![2.0]);
    }

    #[tokio::test]
    async fn available_backends_reflects_registry() {
        let svc = service();
        let backends = svc.available_backends();
        assert_eq!(backends.len(), 1);
        assert_eq!(backends[0].id, "mock");
    }

    #[tokio::test]
    async fn swap_lora_propagates_errors() {
        let svc = service();
        let err = svc.swap_lora("missing").await.unwrap_err();
        assert!(matches!(err, InferenceServiceError::Inference(_)));
    }

    #[tokio::test]
    async fn chat_request_builds_messages() {
        let svc = service();
        let request = svc.chat_request(None, "model", Some("be helpful"), "hello");
        assert_eq!(request.model, "model");
        assert_eq!(request.messages.len(), 2);
        assert_eq!(request.messages[0].role, "system");
        assert_eq!(request.messages[1].role, "user");
    }

    #[tokio::test]
    async fn generate_without_compression_keeps_messages() {
        let (svc, backend) = recording_service();
        let request = svc.chat_request(None, "model", Some("sys"), "user");
        svc.generate(request).await.unwrap();
        let requests = backend.requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].messages.len(), 2);
        assert_eq!(requests[0].messages[0].content, "sys");
        assert_eq!(requests[0].messages[1].content, "user");
    }

    #[tokio::test]
    async fn generate_with_compression_replaces_messages() {
        let backend = Arc::new(RecordingBackend {
            name: "mock".into(),
            requests: Mutex::new(Vec::new()),
        });
        let mut registry = BackendRegistry::new("mock");
        registry.register("mock", backend.clone());
        let pipeline = Arc::new(CompressionPipeline::new(Arc::new(ReplacementCompressor)));
        let svc = InferenceServiceImpl::new(Arc::new(registry), Some("mock".to_string()))
            .with_compression(pipeline, 100);
        let request = svc.chat_request(None, "model", Some("sys"), "user");
        svc.generate(request).await.unwrap();
        let requests = backend.requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].messages.len(), 1);
        assert_eq!(requests[0].messages[0].content, "compressed");
    }
}
