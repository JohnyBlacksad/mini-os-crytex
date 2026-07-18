use async_trait::async_trait;
use crytex_inference::{
    BackendInfo, InferenceError, InferenceManager, InferenceRequest, InferenceResponse,
    LoRAAdapter, Message, ModelInfo, TokenUsage,
};
use mistralrs::core::{DeviceLayerMapMetadata, DeviceMapMetadata};
use mistralrs::{
    AutoDeviceMapParams, DeviceMapSetting, GgufModelBuilder, IsqBits, LoraModelBuilder, Model,
    RequestBuilder, TextMessageRole, TextModelBuilder,
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

#[derive(Debug, PartialEq)]
enum MistralLoadPlan {
    Plain {
        model_id: String,
        lora_adapter_paths: Vec<String>,
    },
    Gguf {
        model_id: String,
        files: Vec<String>,
    },
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

    pub fn cuda_gdn_kernel_available() -> bool {
        option_env!("MISTRALRS_SKIP_GDN_CUDA").is_none_or(|value| value != "1")
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
        match self.load_plan()? {
            MistralLoadPlan::Plain {
                model_id,
                lora_adapter_paths,
            } => self.load_plain_model(&model_id, lora_adapter_paths).await,
            MistralLoadPlan::Gguf { model_id, files } => {
                self.load_gguf_model(model_id, files).await
            }
        }
    }

    fn load_plan(&self) -> Result<MistralLoadPlan, InferenceError> {
        let path = PathBuf::from(&self.inner.model_path);
        let lora_adapter_paths = self.registered_lora_adapter_paths()?;

        if looks_like_gguf(&path) {
            if !lora_adapter_paths.is_empty() {
                return Err(InferenceError::UnsupportedOperation(
                    "GGUF LoRA adapter loading is not yet wired for local registered adapters"
                        .to_string(),
                ));
            }

            let (model_id, files) = resolve_gguf_paths(&path)?;
            return Ok(MistralLoadPlan::Gguf { model_id, files });
        }

        Ok(MistralLoadPlan::Plain {
            model_id: path.to_string_lossy().to_string(),
            lora_adapter_paths,
        })
    }

    fn registered_lora_adapter_paths(&self) -> Result<Vec<String>, InferenceError> {
        let loras = self.inner.loras.lock().map_err(|e| {
            InferenceError::LoRALoadFailed(format!("failed to lock LoRA registry: {e}"))
        })?;
        let mut adapters = loras.values().collect::<Vec<_>>();
        adapters.sort_by(|left, right| left.id.cmp(&right.id));
        Ok(adapters
            .into_iter()
            .map(|adapter| adapter.path.clone())
            .collect())
    }

    async fn load_gguf_model(
        &self,
        model_id: String,
        files: Vec<String>,
    ) -> Result<Model, InferenceError> {
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

    async fn load_plain_model(
        &self,
        model_id: &str,
        lora_adapter_paths: Vec<String>,
    ) -> Result<Model, InferenceError> {
        info!("Loading mistral.rs plain model from {}", model_id);

        let mut builder = TextModelBuilder::new(model_id)
            .with_auto_isq(IsqBits::Four)
            .with_logging();
        builder =
            apply_plain_device_settings(builder, self.inner.gpu_layers, self.inner.context_size);

        if lora_adapter_paths.is_empty() {
            builder
                .build()
                .await
                .map_err(|e| InferenceError::GenerationFailed(format!("model load failed: {e}")))
        } else {
            LoraModelBuilder::from_text_model_builder(builder, lora_adapter_paths)
                .build()
                .await
                .map_err(|e| InferenceError::GenerationFailed(format!("model load failed: {e}")))
        }
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

    fn ensure_registered_adapter(&self, adapter_id: &str) -> Result<(), InferenceError> {
        let loras = self.inner.loras.lock().map_err(|e| {
            InferenceError::LoRALoadFailed(format!("failed to lock LoRA registry: {e}"))
        })?;
        if loras.contains_key(adapter_id) {
            return Ok(());
        }

        Err(InferenceError::LoRALoadFailed(format!(
            "LoRA adapter {adapter_id} is not registered"
        )))
    }

    fn resolve_registered_adapter(
        &self,
        request: &InferenceRequest,
    ) -> Result<Option<String>, InferenceError> {
        let adapter = self.resolve_adapter(request);
        if let Some(adapter_id) = adapter.as_deref() {
            self.ensure_registered_adapter(adapter_id)?;
        }
        Ok(adapter)
    }

    async fn validate_lora_adapter_layout(lora: &LoRAAdapter) -> Result<(), InferenceError> {
        let adapter_path = Path::new(&lora.path);
        let metadata = tokio::fs::metadata(adapter_path).await.map_err(|e| {
            InferenceError::LoRALoadFailed(format!(
                "LoRA adapter {} metadata is unreadable at {}: {e}",
                lora.id,
                adapter_path.display()
            ))
        })?;
        if !metadata.is_dir() {
            return Err(InferenceError::LoRALoadFailed(format!(
                "LoRA adapter {} must be a directory containing adapter_config.json and adapter_model.safetensors",
                lora.id
            )));
        }

        let config_path = adapter_path.join("adapter_config.json");
        let weights_path = adapter_path.join("adapter_model.safetensors");
        let config = tokio::fs::read_to_string(&config_path).await.map_err(|e| {
            InferenceError::LoRALoadFailed(format!(
                "LoRA adapter {} adapter_config.json is unreadable at {}: {e}",
                lora.id,
                config_path.display()
            ))
        })?;
        let config: serde_json::Value = serde_json::from_str(&config).map_err(|e| {
            InferenceError::LoRALoadFailed(format!(
                "LoRA adapter {} adapter_config.json must be valid JSON at {}: {e}",
                lora.id,
                config_path.display()
            ))
        })?;
        if config
            .get("peft_type")
            .and_then(serde_json::Value::as_str)
            .is_none_or(|peft_type| !peft_type.eq_ignore_ascii_case("LORA"))
        {
            return Err(InferenceError::LoRALoadFailed(format!(
                "LoRA adapter {} adapter_config.json must declare peft_type=LORA",
                lora.id
            )));
        }

        let weights_metadata = tokio::fs::metadata(&weights_path).await.map_err(|e| {
            InferenceError::LoRALoadFailed(format!(
                "LoRA adapter {} adapter_model.safetensors is unreadable at {}: {e}",
                lora.id,
                weights_path.display()
            ))
        })?;
        if !weights_metadata.is_file() || weights_metadata.len() == 0 {
            return Err(InferenceError::LoRALoadFailed(format!(
                "LoRA adapter {} adapter_model.safetensors must be a non-empty file",
                lora.id
            )));
        }

        Ok(())
    }

    fn supports_lora_capability(&self) -> bool {
        !Self::is_gguf_model_path(Path::new(&self.inner.model_path))
    }

    fn is_gguf_model_path(path: &Path) -> bool {
        if path
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
        {
            return true;
        }

        path.is_dir()
            && std::fs::read_dir(path)
                .ok()
                .into_iter()
                .flatten()
                .filter_map(Result::ok)
                .any(|entry| {
                    entry
                        .path()
                        .extension()
                        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
                })
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
    builder: TextModelBuilder,
    gpu_layers: Option<usize>,
    context_size: usize,
) -> TextModelBuilder {
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
        let adapter = self.resolve_registered_adapter(&request)?;
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
        Self::validate_lora_adapter_layout(&lora).await?;

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
        let mut capabilities = vec![
            "generate".to_string(),
            "chat".to_string(),
            "gguf".to_string(),
        ];
        if self.supports_lora_capability() {
            capabilities.push("lora".to_string());
            capabilities.push("hot_swap".to_string());
        }

        vec![BackendInfo {
            id: "mistralrs".to_string(),
            name: "mistral.rs".to_string(),
            capabilities,
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
    use async_trait::async_trait;
    use crytex_core::bus::Event;
    use crytex_core::config::BackendKind;
    use crytex_core::services::hardware::{DeviceKind, HardwareDetector};
    use crytex_core::services::{
        EventHandler, EventService, ManagedModel, ManifestEntry, ModelManager, ModelManagerError,
        ModelManagerImpl, SystemHardwareDetector,
    };
    use std::sync::Arc;
    use std::time::Duration;
    use ulid::Ulid;

    struct FixedDetector(DeviceKind);

    impl HardwareDetector for FixedDetector {
        fn detect(&self) -> DeviceKind {
            self.0.clone()
        }
    }

    #[derive(Default)]
    struct SilentEventService;

    #[async_trait]
    impl EventService for SilentEventService {
        fn publish(&self, _event: Event) {}

        fn subscribe(&self) -> tokio::sync::broadcast::Receiver<Event> {
            let (tx, _) = tokio::sync::broadcast::channel(1);
            tx.subscribe()
        }

        async fn start_handler(&self, _handler: Arc<dyn EventHandler>) {}
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct MistralSmokeRuntime {
        gpu_layers: Option<usize>,
        requires_cuda_feature: bool,
        reason: String,
    }

    fn mistral_smoke_runtime(
        detector: &dyn HardwareDetector,
        mode: Option<&str>,
        gpu_layers_override: Option<usize>,
    ) -> MistralSmokeRuntime {
        if let Some(layers) = gpu_layers_override {
            return MistralSmokeRuntime {
                gpu_layers: Some(layers),
                requires_cuda_feature: layers > 0,
                reason: format!("user override: gpu_layers={layers}"),
            };
        }

        match mode.unwrap_or("auto").to_ascii_lowercase().as_str() {
            "cpu" => MistralSmokeRuntime {
                gpu_layers: Some(0),
                requires_cuda_feature: false,
                reason: "forced CPU by CRYTEX_MISTRAL_SMOKE_DEVICE=cpu".into(),
            },
            "gpu" => {
                let device = detector.detect();
                MistralSmokeRuntime {
                    gpu_layers: None,
                    requires_cuda_feature: matches!(device, DeviceKind::Cuda { .. }),
                    reason: format!("forced GPU by CRYTEX_MISTRAL_SMOKE_DEVICE=gpu: {device:?}"),
                }
            }
            _ => match detector.detect() {
                DeviceKind::Cpu => MistralSmokeRuntime {
                    gpu_layers: Some(0),
                    requires_cuda_feature: false,
                    reason: "auto selected CPU: no usable GPU detected".into(),
                },
                device @ DeviceKind::Cuda { .. } => MistralSmokeRuntime {
                    gpu_layers: None,
                    requires_cuda_feature: true,
                    reason: format!("auto selected CUDA GPU: {device:?}"),
                },
                device @ DeviceKind::Metal { .. } => MistralSmokeRuntime {
                    gpu_layers: None,
                    requires_cuda_feature: false,
                    reason: format!("auto selected Metal GPU: {device:?}"),
                },
            },
        }
    }

    fn env_usize(name: &str) -> Option<usize> {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
    }

    #[test]
    fn backend_reports_capabilities() {
        let backend = MistralRsBackend::new("/tmp/model.gguf", 4096, None);
        let info = backend.available_backends();
        assert_eq!(info.len(), 1);
        assert!(info[0].capabilities.contains(&"generate".to_string()));
    }

    #[test]
    fn gguf_backend_does_not_advertise_lora_until_supported() {
        let backend = MistralRsBackend::new("/tmp/model.gguf", 4096, None);
        let info = backend.available_backends();

        assert!(!info[0].capabilities.contains(&"lora".to_string()));
    }

    #[test]
    fn smoke_runtime_auto_uses_cuda_gpu_without_forcing_cpu() {
        let runtime = mistral_smoke_runtime(
            &FixedDetector(DeviceKind::Cuda {
                name: "RTX".into(),
                vram_mb: 16_303,
                driver_version: "596.36".into(),
            }),
            None,
            None,
        );

        assert_eq!(runtime.gpu_layers, None);
        assert!(runtime.requires_cuda_feature);
        assert!(runtime.reason.contains("CUDA"));
    }

    #[test]
    fn smoke_runtime_cpu_mode_forces_zero_gpu_layers() {
        let runtime = mistral_smoke_runtime(
            &FixedDetector(DeviceKind::Cuda {
                name: "RTX".into(),
                vram_mb: 16_303,
                driver_version: "596.36".into(),
            }),
            Some("cpu"),
            None,
        );

        assert_eq!(runtime.gpu_layers, Some(0));
        assert!(!runtime.requires_cuda_feature);
    }

    #[test]
    fn smoke_runtime_gpu_layers_override_is_respected() {
        let runtime = mistral_smoke_runtime(
            &FixedDetector(DeviceKind::Cuda {
                name: "RTX".into(),
                vram_mb: 16_303,
                driver_version: "596.36".into(),
            }),
            Some("auto"),
            Some(12),
        );

        assert_eq!(runtime.gpu_layers, Some(12));
        assert!(runtime.requires_cuda_feature);
    }

    #[test]
    fn plain_backend_advertises_lora_capability() {
        let backend = MistralRsBackend::new("hf-model", 4096, None);
        let info = backend.available_backends();

        assert!(info[0].capabilities.contains(&"lora".to_string()));
    }

    #[test]
    fn plain_backend_advertises_lora_hot_swap_capability() {
        let backend = MistralRsBackend::new("hf-model", 4096, None);
        let report = backend.available_backends()[0].capability_report();

        assert!(report.lora);
        assert!(report.hot_swap);
    }

    #[test]
    fn gguf_directory_backend_does_not_advertise_lora_until_supported() {
        let dir = std::env::temp_dir().join(format!("crytex-gguf-dir-{}", Ulid::new()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.gguf"), b"not a real model").unwrap();
        let backend = MistralRsBackend::new(dir.to_string_lossy(), 4096, None);
        let info = backend.available_backends();

        assert!(!info[0].capabilities.contains(&"lora".to_string()));

        let _ = std::fs::remove_dir_all(dir);
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
        let active_path = valid_lora_adapter_path("active").await;
        backend
            .register_lora(LoRAAdapter {
                id: "active".to_string(),
                path: active_path.to_string_lossy().to_string(),
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

        let _ = tokio::fs::remove_dir_all(active_path).await;
    }

    #[tokio::test]
    async fn generate_rejects_unregistered_request_lora_adapter_before_model_load() {
        let backend = MistralRsBackend::new("/tmp/model.gguf", 4096, None);
        let mut request = empty_request();
        request.lora_adapter_id = Some("missing".to_string());

        let err = backend.generate(request).await.unwrap_err();

        assert!(
            matches!(err, InferenceError::LoRALoadFailed(message) if message.contains("missing"))
        );
    }

    #[tokio::test]
    async fn register_lora_rejects_single_file_adapter_layout() {
        let backend = MistralRsBackend::new("hf-model", 4096, None);
        let adapter_file = std::env::temp_dir().join(format!("{}.safetensors", Ulid::new()));
        tokio::fs::write(&adapter_file, b"not a peft adapter")
            .await
            .unwrap();

        let err = backend
            .register_lora(LoRAAdapter {
                id: "coder".to_string(),
                path: adapter_file.to_string_lossy().to_string(),
                base_model: "hf-model".to_string(),
            })
            .await
            .unwrap_err();

        assert!(
            matches!(err, InferenceError::LoRALoadFailed(message) if message.contains("adapter_config.json"))
        );
        assert!(backend.registered_lora_adapter_paths().unwrap().is_empty());

        let _ = tokio::fs::remove_file(adapter_file).await;
    }

    #[tokio::test]
    async fn register_lora_rejects_malformed_adapter_config() {
        let backend = MistralRsBackend::new("hf-model", 4096, None);
        let adapter_path = valid_lora_adapter_path("malformed").await;
        tokio::fs::write(
            adapter_path.join("adapter_config.json"),
            "{\"peft_type\":\"LORA\"",
        )
        .await
        .unwrap();

        let err = backend
            .register_lora(LoRAAdapter {
                id: "coder".to_string(),
                path: adapter_path.to_string_lossy().to_string(),
                base_model: "hf-model".to_string(),
            })
            .await
            .unwrap_err();

        assert!(
            matches!(err, InferenceError::LoRALoadFailed(message) if message.contains("valid JSON"))
        );
        assert!(backend.registered_lora_adapter_paths().unwrap().is_empty());

        let _ = tokio::fs::remove_dir_all(adapter_path).await;
    }

    #[tokio::test]
    async fn plain_model_load_plan_uses_registered_lora_paths() {
        let backend = MistralRsBackend::new("hf-model", 4096, None);
        let adapter_path = valid_lora_adapter_path("coder").await;
        backend
            .register_lora(LoRAAdapter {
                id: "coder".to_string(),
                path: adapter_path.to_string_lossy().to_string(),
                base_model: "hf-model".to_string(),
            })
            .await
            .unwrap();

        let plan = backend.load_plan().unwrap();

        assert_eq!(
            plan,
            MistralLoadPlan::Plain {
                model_id: "hf-model".to_string(),
                lora_adapter_paths: vec![adapter_path.to_string_lossy().to_string()],
            }
        );

        let _ = tokio::fs::remove_dir_all(adapter_path).await;
    }

    #[tokio::test]
    async fn gguf_model_load_plan_rejects_registered_lora_until_supported() {
        let backend = MistralRsBackend::new("/tmp/model.gguf", 4096, None);
        let adapter_path = valid_lora_adapter_path("coder").await;
        backend
            .register_lora(LoRAAdapter {
                id: "coder".to_string(),
                path: adapter_path.to_string_lossy().to_string(),
                base_model: "hf-model".to_string(),
            })
            .await
            .unwrap();

        let err = backend.load_plan().unwrap_err();

        assert!(
            matches!(err, InferenceError::UnsupportedOperation(message) if message.contains("GGUF LoRA"))
        );

        let _ = tokio::fs::remove_dir_all(adapter_path).await;
    }

    #[tokio::test]
    async fn generate_uses_active_lora_when_request_has_none() {
        let backend = MistralRsBackend::new("/tmp/model.gguf", 4096, None);
        let active_path = valid_lora_adapter_path("active").await;
        backend
            .register_lora(LoRAAdapter {
                id: "active".to_string(),
                path: active_path.to_string_lossy().to_string(),
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

        let _ = tokio::fs::remove_dir_all(active_path).await;
    }

    #[tokio::test]
    #[ignore = "slow manual smoke: set CRYTEX_RUN_SLOW_MISTRAL_SMOKE=1 to download/load a 483 MB TinyLlama GGUF"]
    async fn real_hf_tiny_gguf_downloaded_model_generates_with_mistralrs() {
        if std::env::var("CRYTEX_RUN_SLOW_MISTRAL_SMOKE").as_deref() != Ok("1") {
            eprintln!(
                "skipping slow mistral.rs smoke; set CRYTEX_RUN_SLOW_MISTRAL_SMOKE=1 to run it"
            );
            return;
        }

        let runtime = mistral_smoke_runtime(
            &SystemHardwareDetector::new(),
            std::env::var("CRYTEX_MISTRAL_SMOKE_DEVICE").ok().as_deref(),
            env_usize("CRYTEX_MISTRAL_SMOKE_GPU_LAYERS"),
        );
        eprintln!("mistral.rs smoke runtime: {}", runtime.reason);
        if runtime.requires_cuda_feature && !cfg!(feature = "cuda") {
            eprintln!(
                "skipping CUDA smoke because crytex-inference-mistral was built without --features cuda"
            );
            return;
        }

        let config_dir = tempfile::tempdir().unwrap();
        let cache_dir = tempfile::tempdir().unwrap();
        let manager = ModelManagerImpl::new_standard(
            config_dir.path(),
            cache_dir.path(),
            Arc::new(SilentEventService),
            Arc::new(FixedDetector(DeviceKind::Cpu)),
        );

        manager
            .add_model(ManifestEntry {
                id: Some("hf-tinyllama-chat-q2-gguf".into()),
                name: Some("HF TinyLlama Chat Q2 GGUF".into()),
                repo: Some("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF".into()),
                filename: Some("tinyllama-1.1b-chat-v1.0.Q2_K.gguf".into()),
                quantization: Some("Q2_K".into()),
                backend: Some("mistral_rs".into()),
                params_b: Some(1.1),
            })
            .unwrap();

        let model = manager
            .download_model("hf-tinyllama-chat-q2-gguf")
            .await
            .unwrap();
        let local_path = downloaded_mistral_path(model).unwrap();
        let backend = MistralRsBackend::new(local_path.to_string_lossy(), 64, runtime.gpu_layers);

        let response = tokio::time::timeout(
            Duration::from_secs(180),
            backend.generate(InferenceRequest {
                backend_id: Some("mistralrs".into()),
                model: local_path.to_string_lossy().to_string(),
                messages: vec![Message {
                    role: "user".into(),
                    content: "Reply with the single word: ok".into(),
                }],
                system_prompt: Some("You are a concise test assistant.".into()),
                temperature: Some(0.0),
                max_tokens: Some(1),
                lora_adapter_id: None,
            }),
        )
        .await
        .expect("mistral.rs TinyLlama smoke timed out")
        .unwrap();

        assert!(
            !response.content.trim().is_empty(),
            "mistral.rs should return non-empty text for the downloaded GGUF"
        );
        assert!(response.usage.total_tokens > 0);
    }

    #[tokio::test]
    #[ignore = "slow manual smoke: set CRYTEX_RUN_SLOW_MISTRAL_GDN_SMOKE=1 to download/load a tiny Qwen3 Next GDN model"]
    async fn real_hf_tiny_qwen3_next_gdn_generates_with_mistralrs() {
        if std::env::var("CRYTEX_RUN_SLOW_MISTRAL_GDN_SMOKE").as_deref() != Ok("1") {
            eprintln!(
                "skipping slow mistral.rs GDN smoke; set CRYTEX_RUN_SLOW_MISTRAL_GDN_SMOKE=1 to run it"
            );
            return;
        }

        let runtime = mistral_smoke_runtime(
            &SystemHardwareDetector::new(),
            std::env::var("CRYTEX_MISTRAL_SMOKE_DEVICE").ok().as_deref(),
            env_usize("CRYTEX_MISTRAL_SMOKE_GPU_LAYERS"),
        );
        eprintln!("mistral.rs GDN smoke runtime: {}", runtime.reason);
        if runtime.requires_cuda_feature && !cfg!(feature = "cuda") {
            eprintln!(
                "skipping CUDA GDN smoke because crytex-inference-mistral was built without --features cuda"
            );
            return;
        }

        let model_id = "tiny-random/qwen3-next-moe";
        let backend = MistralRsBackend::new(model_id, 64, runtime.gpu_layers);

        let response = tokio::time::timeout(
            Duration::from_secs(240),
            backend.generate(InferenceRequest {
                backend_id: Some("mistralrs".into()),
                model: model_id.into(),
                messages: vec![Message {
                    role: "user".into(),
                    content: "Reply with the single word: ok".into(),
                }],
                system_prompt: Some("You are a concise test assistant.".into()),
                temperature: Some(0.0),
                max_tokens: Some(1),
                lora_adapter_id: None,
            }),
        )
        .await
        .expect("mistral.rs Qwen3 Next GDN smoke timed out")
        .unwrap();

        assert!(
            !response.content.trim().is_empty(),
            "mistral.rs should return non-empty text for the tiny Qwen3 Next GDN model"
        );
        assert!(response.usage.total_tokens > 0);
    }

    fn downloaded_mistral_path(model: ManagedModel) -> Result<PathBuf, ModelManagerError> {
        assert_eq!(model.preferred_backend, BackendKind::MistralRs);
        model
            .local_path
            .ok_or_else(|| ModelManagerError::Download("downloaded model has no path".into()))
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

    async fn valid_lora_adapter_path(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!("crytex-{name}-{}", Ulid::new()));
        tokio::fs::create_dir_all(&path).await.unwrap();
        tokio::fs::write(path.join("adapter_config.json"), "{\"peft_type\":\"LORA\"}")
            .await
            .unwrap();
        tokio::fs::write(path.join("adapter_model.safetensors"), b"adapter")
            .await
            .unwrap();
        path
    }
}
