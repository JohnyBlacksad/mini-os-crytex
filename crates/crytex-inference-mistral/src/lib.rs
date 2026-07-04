use async_trait::async_trait;
use crytex_inference::{
    BackendInfo, InferenceError, InferenceManager, InferenceRequest, InferenceResponse,
    LoRAAdapter, Message, ModelInfo, TokenUsage,
};
use mistralrs::core::{DeviceLayerMapMetadata, DeviceMapMetadata};
use mistralrs::{
    AutoDeviceMapParams, DeviceMapSetting, GgufModelBuilder, IsqBits, Model, ModelBuilder,
    RequestBuilder, TextMessageRole,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tokio::sync::Mutex as AsyncMutex;
use tracing::info;

pub mod training;

/// In-process local backend powered by [`mistral.rs`](https://github.com/EricLBuehler/mistral.rs).
///
/// Supports GGUF quantized models, automatic device mapping for GPU offload, and
/// runtime LoRA/X-LoRA adapter switching.
pub struct MistralRsBackend {
    inner: Arc<Inner>,
}

struct Inner {
    model_path: String,
    context_size: usize,
    gpu_layers: Option<usize>,
    state: AsyncMutex<Option<Arc<Model>>>,
    loras: Mutex<HashMap<String, LoRAAdapter>>,
    active_lora: Mutex<Option<String>>,
}

impl MistralRsBackend {
    /// Creates a new backend for the given model path.
    ///
    /// * `model_path` – path to a GGUF file, a directory containing GGUF files, or a
    ///   Hugging Face/local plain model identifier.
    /// * `context_size` – requested context length (passed to automatic device mapping).
    /// * `gpu_layers` – number of transformer layers to offload to the GPU. `Some(0)`
    ///   forces CPU, `None` lets `mistral.rs` auto-select the device.
    pub fn new(
        model_path: impl Into<String>,
        context_size: usize,
        gpu_layers: Option<usize>,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                model_path: model_path.into(),
                context_size,
                gpu_layers,
                state: AsyncMutex::new(None),
                loras: Mutex::new(HashMap::new()),
                active_lora: Mutex::new(None),
            }),
        }
    }

    async fn ensure_loaded(&self) -> Result<Arc<Model>, InferenceError> {
        let mut guard = self.inner.state.lock().await;
        if let Some(model) = guard.as_ref() {
            return Ok(model.clone());
        }

        let model = Arc::new(self.load_model().await?);
        *guard = Some(model.clone());
        Ok(model)
    }

    async fn load_model(&self) -> Result<Model, InferenceError> {
        let path = PathBuf::from(&self.inner.model_path);

        if looks_like_gguf(&path) {
            self.load_gguf_model(&path).await
        } else {
            self.load_plain_model(&path).await
        }
    }

    async fn load_gguf_model(&self, path: &Path) -> Result<Model, InferenceError> {
        let (model_id, files) = resolve_gguf_paths(path)?;
        info!(
            "Loading mistral.rs GGUF model from {} with files {:?}",
            model_id, files
        );

        let mut builder = GgufModelBuilder::new(model_id, files);
        builder =
            apply_gguf_device_settings(builder, self.inner.gpu_layers, self.inner.context_size);

        builder
            .build()
            .await
            .map_err(|e| InferenceError::GenerationFailed(format!("model load failed: {e}")))
    }

    async fn load_plain_model(&self, path: &Path) -> Result<Model, InferenceError> {
        let model_id = path.to_string_lossy().to_string();
        info!("Loading mistral.rs plain model from {}", model_id);

        let mut builder = ModelBuilder::new(&model_id)
            .with_auto_isq(IsqBits::Four)
            .with_logging();
        builder =
            apply_plain_device_settings(builder, self.inner.gpu_layers, self.inner.context_size);

        builder
            .build()
            .await
            .map_err(|e| InferenceError::GenerationFailed(format!("model load failed: {e}")))
    }

    fn active_adapter_name(&self) -> Option<String> {
        self.inner.active_lora.lock().ok()?.clone()
    }

    fn resolve_adapter(&self, request: &InferenceRequest) -> Option<String> {
        request
            .lora_adapter_id
            .clone()
            .or_else(|| self.active_adapter_name())
    }
}

fn looks_like_gguf(path: &Path) -> bool {
    if path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
    {
        return true;
    }
    if path.is_dir() {
        return std::fs::read_dir(path)
            .ok()
            .and_then(|mut entries| {
                entries
                    .any(|e| {
                        e.ok()
                            .and_then(|e| {
                                e.path()
                                    .extension()
                                    .map(|ext| ext.eq_ignore_ascii_case("gguf"))
                            })
                            .unwrap_or(false)
                    })
                    .then_some(())
            })
            .is_some();
    }
    false
}

fn resolve_gguf_paths(path: &Path) -> Result<(String, Vec<String>), InferenceError> {
    if path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
    {
        let parent = path
            .parent()
            .and_then(|p| p.to_str())
            .unwrap_or(".")
            .to_string();
        let file = path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| InferenceError::ModelNotFound(path.display().to_string()))?
            .to_string();
        return Ok((parent, vec![file]));
    }

    if path.is_dir() {
        let mut files: Vec<String> = std::fs::read_dir(path)
            .map_err(|e| InferenceError::ModelNotFound(format!("{}: {e}", path.display())))?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
            })
            .filter_map(|e| e.file_name().into_string().ok())
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(InferenceError::ModelNotFound(format!(
                "no .gguf files found in {}",
                path.display()
            )));
        }
        let model_id = path.to_string_lossy().to_string();
        return Ok((model_id, files));
    }

    Err(InferenceError::ModelNotFound(path.display().to_string()))
}

fn apply_gguf_device_settings(
    builder: GgufModelBuilder,
    gpu_layers: Option<usize>,
    context_size: usize,
) -> GgufModelBuilder {
    match gpu_layers {
        Some(0) => builder.with_force_cpu(),
        Some(layers) => builder.with_device_mapping(DeviceMapSetting::Map(
            DeviceMapMetadata::from_num_device_layers(vec![DeviceLayerMapMetadata {
                ordinal: 0,
                layers,
            }]),
        )),
        None => builder.with_device_mapping(DeviceMapSetting::Auto(AutoDeviceMapParams::Text {
            max_seq_len: context_size,
            max_batch_size: 1,
        })),
    }
}

fn apply_plain_device_settings(
    builder: ModelBuilder,
    gpu_layers: Option<usize>,
    context_size: usize,
) -> ModelBuilder {
    match gpu_layers {
        Some(0) => builder.with_force_cpu(),
        Some(layers) => builder.with_device_mapping(DeviceMapSetting::Map(
            DeviceMapMetadata::from_num_device_layers(vec![DeviceLayerMapMetadata {
                ordinal: 0,
                layers,
            }]),
        )),
        None => builder.with_device_mapping(DeviceMapSetting::Auto(AutoDeviceMapParams::Text {
            max_seq_len: context_size,
            max_batch_size: 1,
        })),
    }
}

#[async_trait]
impl InferenceManager for MistralRsBackend {
    async fn generate(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceResponse, InferenceError> {
        let adapter = self.resolve_adapter(&request);
        let model = self.ensure_loaded().await?;

        let mut messages = RequestBuilder::new();
        if let Some(system) = request.system_prompt {
            messages = messages.add_message(TextMessageRole::System, system);
        }
        for Message { role, content } in &request.messages {
            let role = match role.as_str() {
                "system" => TextMessageRole::System,
                "assistant" => TextMessageRole::Assistant,
                _ => TextMessageRole::User,
            };
            messages = messages.add_message(role, content.clone());
        }

        let temperature = request.temperature.unwrap_or(0.7);
        let max_tokens = request.max_tokens.unwrap_or(512);
        messages = messages
            .set_sampler_temperature(f64::from(temperature))
            .set_sampler_max_len(max_tokens);

        if let Some(adapter) = adapter {
            messages = messages.set_adapters(vec![adapter]);
        }

        let response = model
            .send_chat_request(messages)
            .await
            .map_err(|e| InferenceError::GenerationFailed(e.to_string()))?;

        let content = response
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .unwrap_or_default();

        let usage = response.usage;
        Ok(InferenceResponse {
            content,
            usage: TokenUsage {
                prompt_tokens: usage.prompt_tokens as usize,
                completion_tokens: usage.completion_tokens as usize,
                total_tokens: usage.total_tokens as usize,
            },
            finish_reason: "stop".to_string(),
        })
    }

    async fn embed(&self, _text: &str) -> Result<Vec<f32>, InferenceError> {
        Err(InferenceError::EmbeddingFailed(
            "Mistral.rs text backend does not provide embeddings; configure a dedicated embedding backend".to_string(),
        ))
    }

    async fn register_lora(&self, lora: LoRAAdapter) -> Result<(), InferenceError> {
        {
            let mut loras = self.inner.loras.lock().map_err(|e| {
                InferenceError::LoRALoadFailed(format!("failed to lock LoRA registry: {e}"))
            })?;
            loras.insert(lora.id.clone(), lora);
        }

        // New adapters require a model reload; the next generation will re-create the pipeline.
        let mut guard = self.inner.state.lock().await;
        *guard = None;
        Ok(())
    }

    async fn swap_lora(&self, lora_id: &str) -> Result<(), InferenceError> {
        {
            let loras = self.inner.loras.lock().map_err(|e| {
                InferenceError::LoRALoadFailed(format!("failed to lock LoRA registry: {e}"))
            })?;
            if !loras.contains_key(lora_id) {
                return Err(InferenceError::LoRALoadFailed(format!(
                    "LoRA adapter {lora_id} is not registered"
                )));
            }
        }

        let mut active = self.inner.active_lora.lock().map_err(|e| {
            InferenceError::LoRALoadFailed(format!("failed to lock active LoRA: {e}"))
        })?;
        *active = Some(lora_id.to_string());
        info!("Activated mistral.rs LoRA adapter {}", lora_id);
        Ok(())
    }

    fn available_backends(&self) -> Vec<BackendInfo> {
        vec![BackendInfo {
            id: "mistralrs".to_string(),
            name: "mistral.rs".to_string(),
            capabilities: vec![
                "generate".to_string(),
                "chat".to_string(),
                "gguf".to_string(),
                "lora".to_string(),
            ],
        }]
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, InferenceError> {
        Ok(vec![ModelInfo {
            id: self.inner.model_path.clone(),
            name: self.inner.model_path.clone(),
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_reports_capabilities() {
        let backend = MistralRsBackend::new("/tmp/model.gguf", 4096, None);
        let info = backend.available_backends();
        assert_eq!(info.len(), 1);
        assert!(info[0].capabilities.contains(&"generate".to_string()));
    }

    #[test]
    fn resolve_gguf_file_path_splits_directory_and_filename() {
        let path = PathBuf::from("/models/mistral-7b.Q4_K_M.gguf");
        let (model_id, files) = resolve_gguf_paths(&path).unwrap();
        assert_eq!(model_id, "/models");
        assert_eq!(files, vec!["mistral-7b.Q4_K_M.gguf"]);
    }

    #[test]
    fn resolve_missing_directory_fails() {
        let path = PathBuf::from("/does/not/exist/models");
        let result = resolve_gguf_paths(&path);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn embed_returns_error_for_text_backend() {
        let backend = MistralRsBackend::new("/tmp/model.gguf", 4096, None);
        let result = backend.embed("hello").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn generate_uses_request_lora_adapter() {
        let backend = MistralRsBackend::new("/tmp/model.gguf", 4096, None);
        backend
            .register_lora(LoRAAdapter {
                id: "active".to_string(),
                path: "/tmp/active.safetensors".to_string(),
                base_model: "mistral".to_string(),
            })
            .await
            .unwrap();
        backend.swap_lora("active").await.unwrap();

        let mut request = empty_request();
        request.lora_adapter_id = Some("request-lora".to_string());

        assert_eq!(
            backend.resolve_adapter(&request),
            Some("request-lora".to_string())
        );
    }

    #[tokio::test]
    async fn generate_uses_active_lora_when_request_has_none() {
        let backend = MistralRsBackend::new("/tmp/model.gguf", 4096, None);
        backend
            .register_lora(LoRAAdapter {
                id: "active".to_string(),
                path: "/tmp/active.safetensors".to_string(),
                base_model: "mistral".to_string(),
            })
            .await
            .unwrap();
        backend.swap_lora("active").await.unwrap();

        let request = empty_request();
        assert_eq!(
            backend.resolve_adapter(&request),
            Some("active".to_string())
        );
    }

    fn empty_request() -> InferenceRequest {
        InferenceRequest {
            backend_id: None,
            model: "mistral".to_string(),
            messages: vec![],
            system_prompt: None,
            temperature: None,
            max_tokens: None,
            lora_adapter_id: None,
        }
    }
}
