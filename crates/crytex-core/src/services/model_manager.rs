//! Model lifecycle management.
//!
//! `ModelManager` is a trait seam. The default implementation (`ModelManagerImpl`)
//! composes small single-responsibility collaborators:
//!
//! - `ModelManifestSource` — reads the static user-editable manifest.
//! - `ModelRegistryStore` — reads/writes the runtime registry of downloaded models.
//! - `ModelDownloader` — performs the actual download and reports progress.
//! - `ModelRecommender` — suggests quantization, context size and backend from hardware.
//!
//! This keeps the manager open for extension (new download sources, new recommender
//! strategies) and closed for modification.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::bus::Event;
use crate::config::BackendKind;
use crate::services::{EventService, HardwareDetector};

/// Errors that can occur in [`ModelManager`].
#[derive(Debug, Error)]
pub enum ModelManagerError {
    #[error("model not found: {0}")]
    NotFound(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("TOML error: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("serialization error: {0}")]
    TomlSerialize(#[from] toml::ser::Error),
    #[error("download error: {0}")]
    Download(String),
    #[error("recommendation error: {0}")]
    Recommendation(String),
    #[error("resolve error: {0}")]
    Resolve(String),
}

/// Status of a managed model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelStatus {
    /// Model is known but not downloaded.
    Available,
    /// Model is currently being downloaded (progress 0.0-1.0).
    Downloading(f32),
    /// Model is downloaded and ready to use.
    Downloaded,
    /// Download or validation failed.
    Error(String),
}

/// A model that can be selected by the user.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagedModel {
    pub id: String,
    pub name: String,
    pub repo: Option<String>,
    pub filename: Option<String>,
    pub local_path: Option<PathBuf>,
    pub quantization: Option<Quantization>,
    pub preferred_backend: BackendKind,
    pub params_b: Option<f32>,
    pub status: ModelStatus,
}

/// Supported quantization levels.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Quantization {
    Q2K,
    Q3KS,
    Q4KM,
    Q5KM,
    Q8_0,
    FP16,
}

impl Quantization {
    pub fn as_str(&self) -> &'static str {
        match self {
            Quantization::Q2K => "Q2_K",
            Quantization::Q3KS => "Q3_K_S",
            Quantization::Q4KM => "Q4_K_M",
            Quantization::Q5KM => "Q5_K_M",
            Quantization::Q8_0 => "Q8_0",
            Quantization::FP16 => "FP16",
        }
    }

    /// Approximate GiB per 1B parameters for a quantized model.
    pub fn gib_per_b_params(&self) -> f32 {
        match self {
            Quantization::Q2K => 0.30,
            Quantization::Q3KS => 0.40,
            Quantization::Q4KM => 0.50,
            Quantization::Q5KM => 0.65,
            Quantization::Q8_0 => 1.00,
            Quantization::FP16 => 2.00,
        }
    }
}

impl std::str::FromStr for Quantization {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "Q2_K" => Ok(Quantization::Q2K),
            "Q3_K_S" => Ok(Quantization::Q3KS),
            "Q4_K_M" => Ok(Quantization::Q4KM),
            "Q5_K_M" => Ok(Quantization::Q5KM),
            "Q8_0" => Ok(Quantization::Q8_0),
            "FP16" => Ok(Quantization::FP16),
            other => Err(format!("unknown quantization: {other}")),
        }
    }
}

/// Recommended runtime configuration for a model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RecommendedConfig {
    pub backend: BackendKind,
    pub quantization: Quantization,
    pub gpu_layers: Option<usize>,
    pub context_size: usize,
}

/// Static user-editable manifest entry.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ManifestEntry {
    pub id: Option<String>,
    pub name: Option<String>,
    pub repo: Option<String>,
    pub filename: Option<String>,
    pub quantization: Option<String>,
    pub backend: Option<String>,
    pub params_b: Option<f32>,
}

#[derive(Debug, Default, Clone, Deserialize, Serialize)]
pub struct Manifest {
    #[serde(default)]
    pub models: Vec<ManifestEntry>,
}

/// Runtime registry entry for a downloaded model.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RegistryEntry {
    pub local_path: PathBuf,
    pub size: u64,
    pub sha256: Option<String>,
}

#[derive(Debug, Default, Clone, Deserialize, Serialize)]
pub struct Registry {
    #[serde(default)]
    pub models: HashMap<String, RegistryEntry>,
}

/// Source required to download a model.
#[derive(Debug, Clone)]
pub struct DownloadSource {
    pub model_id: String,
    pub repo: String,
    pub filename: String,
    pub target_dir: PathBuf,
}

/// Request for resolving a GGUF artifact from a HuggingFace model repo.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HfGgufResolveRequest {
    pub repo: String,
    pub preferred_quantization: Option<Quantization>,
    pub params_b: Option<f32>,
}

/// A selectable GGUF file variant in a HuggingFace model repo.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HfGgufVariant {
    pub repo: String,
    pub filename: String,
    pub quantization: Quantization,
}

/// Resolution result for a HuggingFace GGUF model repo.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HfGgufResolution {
    pub selected: HfGgufVariant,
    pub variants: Vec<HfGgufVariant>,
    pub recommendation: RecommendedConfig,
}

/// Trait for reading the static model manifest.
pub trait ModelManifestSource: Send + Sync {
    fn load(&self) -> Result<Manifest, ModelManagerError>;

    fn save(&self, _manifest: &Manifest) -> Result<(), ModelManagerError> {
        Err(ModelManagerError::Download(
            "manifest source is read-only".to_string(),
        ))
    }
}

/// Trait for reading/writing the runtime registry of downloaded models.
pub trait ModelRegistryStore: Send + Sync {
    fn load(&self) -> Result<Registry, ModelManagerError>;
    fn save(&self, registry: &Registry) -> Result<(), ModelManagerError>;
}

/// Trait for downloading model files with progress callbacks.
#[async_trait]
pub trait ModelDownloader: Send + Sync {
    async fn download(
        &self,
        source: &DownloadSource,
        on_progress: Arc<dyn Fn(f32) + Send + Sync>,
    ) -> Result<PathBuf, ModelManagerError>;
}

/// Trait for recommending runtime config from a model and hardware.
pub trait ModelRecommender: Send + Sync {
    fn recommend(&self, model: &ManagedModel) -> Result<RecommendedConfig, ModelManagerError>;
}

#[async_trait]
pub trait HfRepoFileLister: Send + Sync {
    async fn list_files(&self, repo: &str) -> Result<Vec<String>, ModelManagerError>;
}

/// High-level model lifecycle trait.
#[async_trait]
pub trait ModelManager: Send + Sync {
    /// List all known models (manifest + registry).
    fn list_models(&self) -> Result<Vec<ManagedModel>, ModelManagerError>;

    /// Get a single model by id.
    fn get_model(&self, id: &str) -> Result<ManagedModel, ModelManagerError>;

    /// Download a model and register it locally.
    async fn download_model(&self, id: &str) -> Result<ManagedModel, ModelManagerError>;

    /// Add or replace a user-managed model in the manifest.
    fn add_model(&self, _entry: ManifestEntry) -> Result<ManagedModel, ModelManagerError> {
        Err(ModelManagerError::Download(
            "model manifest editing is not supported".to_string(),
        ))
    }

    /// Recommend runtime configuration for a model.
    fn recommend_config(&self, id: &str) -> Result<RecommendedConfig, ModelManagerError>;

    /// Resolve a concrete GGUF file from a HuggingFace repo.
    async fn resolve_hf_gguf(
        &self,
        _request: HfGgufResolveRequest,
    ) -> Result<HfGgufResolution, ModelManagerError> {
        Err(ModelManagerError::Resolve(
            "HF GGUF resolution is not supported".to_string(),
        ))
    }
}

/// Default composable implementation of [`ModelManager`].
pub struct ModelManagerImpl {
    manifest_source: Arc<dyn ModelManifestSource>,
    registry_store: Arc<dyn ModelRegistryStore>,
    downloader: Arc<dyn ModelDownloader>,
    hf_repo_file_lister: Arc<dyn HfRepoFileLister>,
    recommender: Arc<dyn ModelRecommender>,
    event_service: Arc<dyn EventService>,
    models_dir: PathBuf,
}

impl ModelManagerImpl {
    pub fn new(
        manifest_source: Arc<dyn ModelManifestSource>,
        registry_store: Arc<dyn ModelRegistryStore>,
        downloader: Arc<dyn ModelDownloader>,
        hf_repo_file_lister: Arc<dyn HfRepoFileLister>,
        recommender: Arc<dyn ModelRecommender>,
        event_service: Arc<dyn EventService>,
        models_dir: PathBuf,
    ) -> Self {
        Self {
            manifest_source,
            registry_store,
            downloader,
            hf_repo_file_lister,
            recommender,
            event_service,
            models_dir,
        }
    }

    /// Convenience constructor for the standard file-backed + HF downloader setup.
    pub fn new_standard(
        config_dir: impl AsRef<Path>,
        cache_dir: impl AsRef<Path>,
        event_service: Arc<dyn EventService>,
        hardware_detector: Arc<dyn HardwareDetector>,
    ) -> Self {
        let config_dir = config_dir.as_ref();
        let cache_dir = cache_dir.as_ref();
        let manifest_path = config_dir.join("manifest.toml");
        let registry_path = cache_dir.join("registry.toml");
        let models_dir = cache_dir.join("models");

        Self::new(
            Arc::new(FileSystemManifestSource::new(manifest_path)),
            Arc::new(TomlRegistryStore::new(registry_path)),
            Arc::new(HfHubDownloader),
            Arc::new(HfHubRepoFileLister),
            Arc::new(HardwareModelRecommender::new(hardware_detector)),
            event_service,
            models_dir,
        )
    }
}

#[async_trait]
impl ModelManager for ModelManagerImpl {
    fn list_models(&self) -> Result<Vec<ManagedModel>, ModelManagerError> {
        let manifest = self.manifest_source.load()?;
        let registry = self.registry_store.load().unwrap_or_default();

        let mut models = Vec::new();
        for entry in manifest.models {
            let id = entry
                .id
                .clone()
                .or_else(|| entry.filename.clone())
                .unwrap_or_else(|| "unknown".to_string());
            let local_path = registry.models.get(&id).map(|e| e.local_path.clone());
            let status = if local_path.is_some() {
                ModelStatus::Downloaded
            } else {
                ModelStatus::Available
            };
            models.push(ManagedModel {
                id: id.clone(),
                name: entry.name.clone().unwrap_or_else(|| id.clone()),
                repo: entry.repo.clone(),
                filename: entry.filename.clone(),
                local_path,
                quantization: entry.quantization.as_deref().and_then(|s| s.parse().ok()),
                preferred_backend: parse_backend_kind(entry.backend.as_deref()),
                params_b: entry.params_b,
                status,
            });
        }

        for (id, entry) in &registry.models {
            if models.iter().any(|m| &m.id == id) {
                continue;
            }
            models.push(ManagedModel {
                id: id.clone(),
                name: id.clone(),
                repo: None,
                filename: None,
                local_path: Some(entry.local_path.clone()),
                quantization: None,
                preferred_backend: recommend_backend_from_path(&entry.local_path),
                params_b: None,
                status: ModelStatus::Downloaded,
            });
        }

        Ok(models)
    }

    fn get_model(&self, id: &str) -> Result<ManagedModel, ModelManagerError> {
        self.list_models()?
            .into_iter()
            .find(|m| m.id == id)
            .ok_or_else(|| ModelManagerError::NotFound(id.to_string()))
    }

    async fn download_model(&self, id: &str) -> Result<ManagedModel, ModelManagerError> {
        let model = self.get_model(id)?;

        if model.local_path.is_some() {
            return Ok(model);
        }

        let repo = model.repo.clone().ok_or_else(|| {
            ModelManagerError::Download(format!("model {} has no HuggingFace repo", id))
        })?;
        let filename = model
            .filename
            .clone()
            .ok_or_else(|| ModelManagerError::Download(format!("model {} has no filename", id)))?;

        let target_dir = self.models_dir.join(sanitize_id(id));
        tokio::fs::create_dir_all(&target_dir).await?;

        self.event_service.publish(Event::ModelDownloadProgress {
            model_id: id.to_string(),
            progress: 0.0,
        });

        let event_service = self.event_service.clone();
        let model_id = id.to_string();
        let on_progress: Arc<dyn Fn(f32) + Send + Sync> = Arc::new(move |progress: f32| {
            event_service.publish(Event::ModelDownloadProgress {
                model_id: model_id.clone(),
                progress,
            });
        });

        let source = DownloadSource {
            model_id: id.to_string(),
            repo,
            filename,
            target_dir,
        };

        let target_path = self.downloader.download(&source, on_progress).await?;

        self.event_service.publish(Event::ModelDownloadProgress {
            model_id: id.to_string(),
            progress: 1.0,
        });

        let mut registry = self.registry_store.load().unwrap_or_default();
        let size = tokio::fs::metadata(&target_path).await?.len();
        registry.models.insert(
            id.to_string(),
            RegistryEntry {
                local_path: target_path.clone(),
                size,
                sha256: None,
            },
        );
        self.registry_store.save(&registry)?;

        Ok(ManagedModel {
            local_path: Some(target_path),
            status: ModelStatus::Downloaded,
            ..model
        })
    }

    fn add_model(&self, entry: ManifestEntry) -> Result<ManagedModel, ModelManagerError> {
        let id = manifest_entry_id(&entry)?;
        let mut manifest = self.manifest_source.load()?;
        manifest
            .models
            .retain(|model| manifest_entry_id(model).ok().as_deref() != Some(id.as_str()));
        manifest.models.push(entry);
        self.manifest_source.save(&manifest)?;
        self.get_model(&id)
    }

    fn recommend_config(&self, id: &str) -> Result<RecommendedConfig, ModelManagerError> {
        let model = self.get_model(id)?;
        self.recommender.recommend(&model)
    }

    async fn resolve_hf_gguf(
        &self,
        request: HfGgufResolveRequest,
    ) -> Result<HfGgufResolution, ModelManagerError> {
        let files = self.hf_repo_file_lister.list_files(&request.repo).await?;
        let variants = list_gguf_variants(&request.repo, files);
        if variants.is_empty() {
            return Err(ModelManagerError::Resolve(format!(
                "repo {} has no quantized GGUF files",
                request.repo
            )));
        }

        let synthetic_model = ManagedModel {
            id: request.repo.clone(),
            name: request.repo.clone(),
            repo: Some(request.repo.clone()),
            filename: None,
            local_path: None,
            quantization: request.preferred_quantization,
            preferred_backend: BackendKind::MistralRs,
            params_b: request.params_b,
            status: ModelStatus::Available,
        };
        let recommendation = self.recommender.recommend(&synthetic_model)?;
        let selected =
            select_gguf_variant(&variants, recommendation.quantization).ok_or_else(|| {
                ModelManagerError::Resolve(format!(
                    "repo {} has no usable GGUF files",
                    request.repo
                ))
            })?;

        Ok(HfGgufResolution {
            selected,
            variants,
            recommendation,
        })
    }
}

/// File-system backed manifest source.
pub struct FileSystemManifestSource {
    path: PathBuf,
}

impl FileSystemManifestSource {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

impl ModelManifestSource for FileSystemManifestSource {
    fn load(&self) -> Result<Manifest, ModelManagerError> {
        if !self.path.exists() {
            return Ok(Manifest::default());
        }
        let contents = std::fs::read_to_string(&self.path)?;
        let manifest: Manifest = toml::from_str(&contents)?;
        Ok(manifest)
    }

    fn save(&self, manifest: &Manifest) -> Result<(), ModelManagerError> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let contents = toml::to_string_pretty(manifest)?;
        std::fs::write(&self.path, contents)?;
        Ok(())
    }
}

/// TOML-backed registry store.
pub struct TomlRegistryStore {
    path: PathBuf,
}

impl TomlRegistryStore {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

impl ModelRegistryStore for TomlRegistryStore {
    fn load(&self) -> Result<Registry, ModelManagerError> {
        if !self.path.exists() {
            return Ok(Registry::default());
        }
        let contents = std::fs::read_to_string(&self.path)?;
        let registry: Registry = toml::from_str(&contents)?;
        Ok(registry)
    }

    fn save(&self, registry: &Registry) -> Result<(), ModelManagerError> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let contents = toml::to_string_pretty(registry)?;
        std::fs::write(&self.path, contents)?;
        Ok(())
    }
}

/// HuggingFace Hub downloader.
pub struct HfHubDownloader;

#[async_trait]
impl ModelDownloader for HfHubDownloader {
    async fn download(
        &self,
        source: &DownloadSource,
        on_progress: Arc<dyn Fn(f32) + Send + Sync>,
    ) -> Result<PathBuf, ModelManagerError> {
        #[cfg(feature = "hf-hub")]
        {
            use hf_hub::api::tokio::Api;

            on_progress(0.0);
            let api = Api::new().map_err(|e| ModelManagerError::Download(e.to_string()))?;
            let repo = api.model(source.repo.clone());
            let downloaded = repo
                .download(&source.filename)
                .await
                .map_err(|e| ModelManagerError::Download(e.to_string()))?;

            let target_path = source.target_dir.join(&source.filename);
            tokio::fs::copy(&downloaded, &target_path).await?;
            on_progress(1.0);
            Ok(target_path)
        }

        #[cfg(not(feature = "hf-hub"))]
        {
            let _ = (source, on_progress);
            Err(ModelManagerError::Download(
                "hf-hub support is disabled".to_string(),
            ))
        }
    }
}

pub struct HfHubRepoFileLister;

#[async_trait]
impl HfRepoFileLister for HfHubRepoFileLister {
    async fn list_files(&self, repo: &str) -> Result<Vec<String>, ModelManagerError> {
        #[cfg(feature = "hf-hub")]
        {
            use hf_hub::api::tokio::Api;

            let api = Api::new().map_err(|e| ModelManagerError::Resolve(e.to_string()))?;
            let info = api
                .model(repo.to_string())
                .info()
                .await
                .map_err(|e| ModelManagerError::Resolve(e.to_string()))?;
            Ok(info
                .siblings
                .into_iter()
                .map(|sibling| sibling.rfilename)
                .collect())
        }

        #[cfg(not(feature = "hf-hub"))]
        {
            let _ = repo;
            Err(ModelManagerError::Resolve(
                "hf-hub support is disabled".to_string(),
            ))
        }
    }
}

/// Recommends config from detected hardware.
pub struct HardwareModelRecommender {
    hardware_detector: Arc<dyn HardwareDetector>,
}

fn list_gguf_variants(repo: &str, filenames: Vec<String>) -> Vec<HfGgufVariant> {
    let mut variants = filenames
        .into_iter()
        .filter(|filename| filename.to_ascii_lowercase().ends_with(".gguf"))
        .filter_map(|filename| {
            quantization_from_filename(&filename).map(|quantization| HfGgufVariant {
                repo: repo.to_string(),
                filename,
                quantization,
            })
        })
        .collect::<Vec<_>>();
    variants.sort_by(|left, right| {
        quantization_rank(right.quantization)
            .cmp(&quantization_rank(left.quantization))
            .then_with(|| left.filename.cmp(&right.filename))
    });
    variants
}

fn select_gguf_variant(
    variants: &[HfGgufVariant],
    preferred: Quantization,
) -> Option<HfGgufVariant> {
    variants
        .iter()
        .find(|variant| variant.quantization == preferred)
        .or_else(|| {
            variants
                .iter()
                .filter(|variant| {
                    quantization_rank(variant.quantization) <= quantization_rank(preferred)
                })
                .max_by_key(|variant| quantization_rank(variant.quantization))
        })
        .or_else(|| {
            variants
                .iter()
                .min_by_key(|variant| quantization_rank(variant.quantization))
        })
        .cloned()
}

fn quantization_from_filename(filename: &str) -> Option<Quantization> {
    let upper = filename.to_ascii_uppercase();
    [
        ("Q2_K", Quantization::Q2K),
        ("Q3_K_S", Quantization::Q3KS),
        ("Q4_K_M", Quantization::Q4KM),
        ("Q5_K_M", Quantization::Q5KM),
        ("Q8_0", Quantization::Q8_0),
        ("F16", Quantization::FP16),
        ("FP16", Quantization::FP16),
    ]
    .into_iter()
    .find_map(|(needle, quantization)| upper.contains(needle).then_some(quantization))
}

fn quantization_rank(quantization: Quantization) -> u8 {
    match quantization {
        Quantization::Q2K => 2,
        Quantization::Q3KS => 3,
        Quantization::Q4KM => 4,
        Quantization::Q5KM => 5,
        Quantization::Q8_0 => 8,
        Quantization::FP16 => 16,
    }
}

impl HardwareModelRecommender {
    pub fn new(hardware_detector: Arc<dyn HardwareDetector>) -> Self {
        Self { hardware_detector }
    }

    /// Approximate model size in GiB from params and quantization.
    fn approximate_size_gib(model: &ManagedModel, quantization: Quantization) -> f32 {
        let params = model.params_b.unwrap_or(7.0);
        params * quantization.gib_per_b_params()
    }

    fn cuda_gpu_layers(
        model: &ManagedModel,
        quantization: Quantization,
        usable_gib: f32,
    ) -> Option<usize> {
        let estimated_model_gib = Self::approximate_size_gib(model, quantization);
        (estimated_model_gib <= usable_gib * 0.7).then_some(999)
    }
}

impl ModelRecommender for HardwareModelRecommender {
    fn recommend(&self, model: &ManagedModel) -> Result<RecommendedConfig, ModelManagerError> {
        use crate::services::hardware::DeviceKind;

        let device = self.hardware_detector.detect();

        // Cloud-preferred models keep their backend; we only tune local ones.
        let backend = match model.preferred_backend {
            BackendKind::Anthropic
            | BackendKind::Ollama
            | BackendKind::OpenAiCompatible
            | BackendKind::Custom => model.preferred_backend,
            BackendKind::MistralRs => BackendKind::MistralRs,
            BackendKind::Onnx => BackendKind::Onnx,
        };

        let is_remote = !matches!(backend, BackendKind::MistralRs);
        if is_remote {
            return Ok(RecommendedConfig {
                backend,
                quantization: model.quantization.unwrap_or(Quantization::Q4KM),
                gpu_layers: None,
                context_size: 4096,
            });
        }

        let default_quantization = model.quantization.unwrap_or(Quantization::Q4KM);

        match device {
            DeviceKind::Cpu => Ok(RecommendedConfig {
                backend: BackendKind::MistralRs,
                quantization: default_quantization,
                gpu_layers: Some(0),
                context_size: 4096,
            }),
            DeviceKind::Metal { .. } => Ok(RecommendedConfig {
                backend: BackendKind::MistralRs,
                quantization: default_quantization,
                gpu_layers: None,
                context_size: 8192,
            }),
            DeviceKind::Cuda { vram_mb, .. } => {
                let vram_gib = vram_mb as f32 / 1024.0;
                // Leave 20% headroom for context / KV cache / OS.
                let usable_gib = vram_gib * 0.8;

                let candidates = [
                    Quantization::FP16,
                    Quantization::Q8_0,
                    Quantization::Q5KM,
                    Quantization::Q4KM,
                    Quantization::Q3KS,
                    Quantization::Q2K,
                ];

                let quantization = model.quantization.unwrap_or_else(|| {
                    candidates
                        .into_iter()
                        .find(|q| Self::approximate_size_gib(model, *q) <= usable_gib)
                        .unwrap_or(Quantization::Q4KM)
                });

                let context_size = if vram_mb >= 24_000 { 8192 } else { 4096 };

                Ok(RecommendedConfig {
                    backend: BackendKind::MistralRs,
                    quantization,
                    gpu_layers: Self::cuda_gpu_layers(model, quantization, usable_gib),
                    context_size,
                })
            }
        }
    }
}

fn parse_backend_kind(kind: Option<&str>) -> BackendKind {
    match kind.unwrap_or("mistral_rs").to_lowercase().as_str() {
        "ollama" => BackendKind::Ollama,
        "openai" | "open_ai_compatible" | "openai_compatible" => BackendKind::OpenAiCompatible,
        "anthropic" => BackendKind::Anthropic,
        "mistral" | "mistralrs" | "mistral.rs" | "llama" | "llamacpp" | "llama_cpp"
        | "llama-gguf" => BackendKind::MistralRs,
        "onnx" => BackendKind::Onnx,
        "custom" => BackendKind::Custom,
        _ => BackendKind::MistralRs,
    }
}

fn recommend_backend_from_path(path: &Path) -> BackendKind {
    if path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("gguf"))
        .unwrap_or(false)
    {
        BackendKind::MistralRs
    } else {
        BackendKind::OpenAiCompatible
    }
}

fn sanitize_id(id: &str) -> String {
    id.replace(|c: char| !c.is_alphanumeric() && c != '-' && c != '_', "_")
}

fn manifest_entry_id(entry: &ManifestEntry) -> Result<String, ModelManagerError> {
    entry
        .id
        .clone()
        .or_else(|| entry.filename.clone())
        .filter(|id| !id.trim().is_empty())
        .ok_or_else(|| ModelManagerError::Download("model id or filename is required".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::hardware::{DeviceKind, HardwareDetector};
    use std::sync::Mutex;

    struct FixedDetector(DeviceKind);

    impl HardwareDetector for FixedDetector {
        fn detect(&self) -> DeviceKind {
            self.0.clone()
        }
    }

    fn temp_dirs(suffix: &str) -> (PathBuf, PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "crytex-model-test-{}-{}",
            suffix,
            std::process::id()
        ));
        let config_dir = dir.join("config");
        let cache_dir = dir.join("cache");
        std::fs::create_dir_all(&config_dir).unwrap();
        std::fs::create_dir_all(&cache_dir).unwrap();
        (config_dir, cache_dir)
    }

    #[derive(Default)]
    struct InMemoryManifestSource {
        manifest: Mutex<Manifest>,
    }

    impl ModelManifestSource for InMemoryManifestSource {
        fn load(&self) -> Result<Manifest, ModelManagerError> {
            Ok(self.manifest.lock().unwrap().clone())
        }
    }

    impl Clone for InMemoryManifestSource {
        fn clone(&self) -> Self {
            Self {
                manifest: Mutex::new(self.manifest.lock().unwrap().clone()),
            }
        }
    }

    #[derive(Default)]
    struct InMemoryRegistryStore {
        registry: Mutex<Registry>,
    }

    impl ModelRegistryStore for InMemoryRegistryStore {
        fn load(&self) -> Result<Registry, ModelManagerError> {
            Ok(self.registry.lock().unwrap().clone())
        }

        fn save(&self, registry: &Registry) -> Result<(), ModelManagerError> {
            *self.registry.lock().unwrap() = registry.clone();
            Ok(())
        }
    }

    impl Clone for InMemoryRegistryStore {
        fn clone(&self) -> Self {
            Self {
                registry: Mutex::new(self.registry.lock().unwrap().clone()),
            }
        }
    }

    #[derive(Default)]
    struct MockDownloader {
        progress: Mutex<Vec<f32>>,
    }

    #[async_trait]
    impl ModelDownloader for MockDownloader {
        async fn download(
            &self,
            source: &DownloadSource,
            on_progress: Arc<dyn Fn(f32) + Send + Sync>,
        ) -> Result<PathBuf, ModelManagerError> {
            on_progress(0.0);
            on_progress(0.5);
            on_progress(1.0);
            self.progress.lock().unwrap().push(0.5);
            let target = source.target_dir.join(&source.filename);
            std::fs::create_dir_all(&source.target_dir)?;
            std::fs::write(&target, b"gguf")?;
            Ok(target)
        }
    }

    #[derive(Default)]
    struct MockEventService {
        events: Mutex<Vec<Event>>,
    }

    struct MockHfRepoFileLister {
        files: Vec<String>,
    }

    #[async_trait]
    impl HfRepoFileLister for MockHfRepoFileLister {
        async fn list_files(&self, _repo: &str) -> Result<Vec<String>, ModelManagerError> {
            Ok(self.files.clone())
        }
    }

    #[async_trait]
    impl EventService for MockEventService {
        fn publish(&self, event: Event) {
            self.events.lock().unwrap().push(event);
        }
        fn subscribe(&self) -> tokio::sync::broadcast::Receiver<Event> {
            let (tx, _) = tokio::sync::broadcast::channel(1);
            tx.subscribe()
        }
        async fn start_handler(&self, _handler: Arc<dyn crate::services::EventHandler>) {}
    }

    fn manager_with(
        manifest: Manifest,
        registry: Registry,
        hardware: DeviceKind,
    ) -> (ModelManagerImpl, Arc<MockEventService>) {
        let (_config_dir, cache_dir) = temp_dirs("unit");
        let events = Arc::new(MockEventService::default());
        let manifest_source: Arc<dyn ModelManifestSource> = Arc::new(InMemoryManifestSource {
            manifest: Mutex::new(manifest),
        });
        let registry_store: Arc<dyn ModelRegistryStore> = Arc::new(InMemoryRegistryStore {
            registry: Mutex::new(registry),
        });
        let downloader: Arc<dyn ModelDownloader> = Arc::new(MockDownloader::default());
        let hf_repo_file_lister: Arc<dyn HfRepoFileLister> =
            Arc::new(MockHfRepoFileLister { files: Vec::new() });
        let recommender: Arc<dyn ModelRecommender> = Arc::new(HardwareModelRecommender::new(
            Arc::new(FixedDetector(hardware)),
        ));

        (
            ModelManagerImpl::new(
                manifest_source,
                registry_store,
                downloader,
                hf_repo_file_lister,
                recommender,
                events.clone(),
                cache_dir.join("models"),
            ),
            events,
        )
    }

    #[test]
    fn empty_manager_returns_no_models() {
        let (mgr, _) = manager_with(Manifest::default(), Registry::default(), DeviceKind::Cpu);
        let models = mgr.list_models().unwrap();
        assert!(models.is_empty());
    }

    #[test]
    fn manifest_models_are_listed() {
        let manifest = Manifest {
            models: vec![ManifestEntry {
                id: Some("qwen-9b".into()),
                name: Some("Qwen 2.5 Coder 9B".into()),
                repo: Some("Qwen/Qwen2.5-Coder-9B-Instruct-GGUF".into()),
                filename: Some("qwen2.5-coder-9b-instruct-q4_k_m.gguf".into()),
                quantization: Some("Q4_K_M".into()),
                backend: Some("mistral_rs".into()),
                params_b: Some(9.0),
            }],
        };
        let (mgr, _) = manager_with(manifest, Registry::default(), DeviceKind::Cpu);
        let models = mgr.list_models().unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "qwen-9b");
        assert!(matches!(models[0].status, ModelStatus::Available));
        assert!(matches!(
            models[0].preferred_backend,
            BackendKind::MistralRs
        ));
        assert_eq!(models[0].quantization, Some(Quantization::Q4KM));
    }

    #[test]
    fn add_model_persists_manifest_entry() {
        let (config_dir, cache_dir) = temp_dirs("add-model");
        let events: Arc<dyn EventService> = Arc::new(MockEventService::default());
        let mgr = ModelManagerImpl::new_standard(
            &config_dir,
            &cache_dir,
            events,
            Arc::new(FixedDetector(DeviceKind::Cpu)),
        );

        let added = mgr
            .add_model(ManifestEntry {
                id: Some("qwen-coder-9b-q4".into()),
                name: Some("Qwen Coder 9B Q4".into()),
                repo: Some("Qwen/Qwen2.5-Coder-9B-Instruct-GGUF".into()),
                filename: Some("qwen2.5-coder-9b-instruct-q4_k_m.gguf".into()),
                quantization: Some("Q4_K_M".into()),
                backend: Some("mistral_rs".into()),
                params_b: Some(9.0),
            })
            .unwrap();

        assert_eq!(added.id, "qwen-coder-9b-q4");
        assert!(matches!(added.status, ModelStatus::Available));

        let reloaded = ModelManagerImpl::new_standard(
            &config_dir,
            &cache_dir,
            Arc::new(MockEventService::default()),
            Arc::new(FixedDetector(DeviceKind::Cpu)),
        );
        let models = reloaded.list_models().unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(
            models[0].repo.as_deref(),
            Some("Qwen/Qwen2.5-Coder-9B-Instruct-GGUF")
        );
        assert_eq!(models[0].quantization, Some(Quantization::Q4KM));
    }

    #[test]
    fn downloaded_models_are_marked() {
        let manifest = Manifest {
            models: vec![ManifestEntry {
                id: Some("qwen-9b".into()),
                name: Some("Qwen".into()),
                filename: Some("model.gguf".into()),
                ..Default::default() // needs Default on ManifestEntry
            }],
        };
        let (_config_dir, cache_dir) = temp_dirs("downloaded");
        let model_dir = cache_dir.join("models").join("qwen-9b");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("model.gguf"), b"gguf").unwrap();

        let registry = Registry {
            models: {
                let mut map = HashMap::new();
                map.insert(
                    "qwen-9b".to_string(),
                    RegistryEntry {
                        local_path: model_dir.join("model.gguf"),
                        size: 4,
                        sha256: None,
                    },
                );
                map
            },
        };

        let manifest_source = Arc::new(InMemoryManifestSource {
            manifest: Mutex::new(manifest),
        });
        let registry_store = Arc::new(InMemoryRegistryStore {
            registry: Mutex::new(registry),
        });
        let events: Arc<dyn EventService> = Arc::new(MockEventService::default());
        let mgr = ModelManagerImpl::new(
            manifest_source,
            registry_store,
            Arc::new(MockDownloader::default()),
            Arc::new(MockHfRepoFileLister { files: Vec::new() }),
            Arc::new(HardwareModelRecommender::new(Arc::new(FixedDetector(
                DeviceKind::Cpu,
            )))),
            events,
            cache_dir.join("models"),
        );

        let models = mgr.list_models().unwrap();
        assert_eq!(models.len(), 1);
        assert!(matches!(models[0].status, ModelStatus::Downloaded));
    }

    #[test]
    fn gguf_variant_listing_filters_and_extracts_quantization() {
        let variants = list_gguf_variants(
            "owner/repo",
            vec![
                "README.md".into(),
                "model.Q4_K_M.gguf".into(),
                "model.Q2_K.gguf".into(),
                "adapter.safetensors".into(),
            ],
        );

        assert_eq!(
            variants
                .iter()
                .map(|variant| (variant.filename.as_str(), variant.quantization))
                .collect::<Vec<_>>(),
            vec![
                ("model.Q4_K_M.gguf", Quantization::Q4KM),
                ("model.Q2_K.gguf", Quantization::Q2K),
            ]
        );
    }

    #[tokio::test]
    async fn resolve_hf_gguf_selects_requested_quantization_from_repo_files() {
        let (_config_dir, cache_dir) = temp_dirs("resolve-gguf");
        let events = Arc::new(MockEventService::default());
        let mgr = ModelManagerImpl::new(
            Arc::new(InMemoryManifestSource {
                manifest: Mutex::new(Manifest::default()),
            }),
            Arc::new(InMemoryRegistryStore {
                registry: Mutex::new(Registry::default()),
            }),
            Arc::new(MockDownloader::default()),
            Arc::new(MockHfRepoFileLister {
                files: vec![
                    "tiny.Q2_K.gguf".into(),
                    "tiny.Q4_K_M.gguf".into(),
                    "tiny.Q8_0.gguf".into(),
                ],
            }),
            Arc::new(HardwareModelRecommender::new(Arc::new(FixedDetector(
                DeviceKind::Cuda {
                    name: "RTX 5080".into(),
                    vram_mb: 16_000,
                    driver_version: "581".into(),
                },
            )))),
            events,
            cache_dir.join("models"),
        );

        let resolution = mgr
            .resolve_hf_gguf(HfGgufResolveRequest {
                repo: "owner/repo".into(),
                preferred_quantization: Some(Quantization::Q4KM),
                params_b: Some(1.1),
            })
            .await
            .unwrap();

        assert_eq!(resolution.selected.filename, "tiny.Q4_K_M.gguf");
        assert_eq!(resolution.selected.quantization, Quantization::Q4KM);
        assert_eq!(resolution.recommendation.quantization, Quantization::Q4KM);
        assert_eq!(resolution.variants.len(), 3);
    }

    #[tokio::test]
    async fn download_emits_progress_events() {
        let manifest = Manifest {
            models: vec![ManifestEntry {
                id: Some("qwen-9b".into()),
                name: Some("Qwen".into()),
                repo: Some("Qwen/Qwen2.5-Coder-9B-Instruct-GGUF".into()),
                filename: Some("model.gguf".into()),
                quantization: None,
                backend: None,
                params_b: None,
            }],
        };
        let (mgr, events) = manager_with(manifest, Registry::default(), DeviceKind::Cpu);

        let model = mgr.download_model("qwen-9b").await.unwrap();
        assert!(matches!(model.status, ModelStatus::Downloaded));

        let progress_events: Vec<_> = events
            .events
            .lock()
            .unwrap()
            .iter()
            .filter_map(|e| match e {
                Event::ModelDownloadProgress { model_id, progress } => {
                    Some((model_id.clone(), *progress))
                }
                _ => None,
            })
            .collect();
        assert!(!progress_events.is_empty());
        assert_eq!(progress_events.first().unwrap().1, 0.0);
        assert_eq!(progress_events.last().unwrap().1, 1.0);
    }

    #[cfg(feature = "hf-hub")]
    #[tokio::test]
    #[ignore = "network smoke: downloads a tiny Hugging Face config file"]
    async fn real_hf_download_persists_registry_and_reloads_as_downloaded() {
        let config_dir = tempfile::tempdir().unwrap();
        let cache_dir = tempfile::tempdir().unwrap();
        let events: Arc<dyn EventService> = Arc::new(MockEventService::default());
        let mgr = ModelManagerImpl::new_standard(
            config_dir.path(),
            cache_dir.path(),
            events,
            Arc::new(FixedDetector(DeviceKind::Cpu)),
        );

        mgr.add_model(ManifestEntry {
            id: Some("hf-tiny-gpt2-config".into()),
            name: Some("HF Tiny GPT-2 Config".into()),
            repo: Some("sshleifer/tiny-gpt2".into()),
            filename: Some("config.json".into()),
            quantization: None,
            backend: Some("custom".into()),
            params_b: None,
        })
        .unwrap();

        let downloaded = mgr.download_model("hf-tiny-gpt2-config").await.unwrap();
        let local_path = downloaded.local_path.clone().unwrap();

        assert!(matches!(downloaded.status, ModelStatus::Downloaded));
        assert_eq!(
            local_path.file_name().and_then(|name| name.to_str()),
            Some("config.json")
        );
        assert!(
            local_path.starts_with(cache_dir.path().join("models").join("hf-tiny-gpt2-config"))
        );
        assert!(tokio::fs::metadata(&local_path).await.unwrap().len() > 0);

        let registry_path = cache_dir.path().join("registry.toml");
        assert!(registry_path.exists());

        let reloaded = ModelManagerImpl::new_standard(
            config_dir.path(),
            cache_dir.path(),
            Arc::new(MockEventService::default()),
            Arc::new(FixedDetector(DeviceKind::Cpu)),
        );
        let model = reloaded.get_model("hf-tiny-gpt2-config").unwrap();

        assert!(matches!(model.status, ModelStatus::Downloaded));
        assert_eq!(model.local_path.as_deref(), Some(local_path.as_path()));
    }

    #[cfg(feature = "hf-hub")]
    #[tokio::test]
    #[ignore = "network smoke: downloads an 83 MB tiny GGUF model file"]
    async fn real_hf_tiny_gguf_download_reloads_as_mistral_runtime_candidate() {
        let config_dir = tempfile::tempdir().unwrap();
        let cache_dir = tempfile::tempdir().unwrap();
        let events: Arc<dyn EventService> = Arc::new(MockEventService::default());
        let mgr = ModelManagerImpl::new_standard(
            config_dir.path(),
            cache_dir.path(),
            events,
            Arc::new(FixedDetector(DeviceKind::Cpu)),
        );

        mgr.add_model(ManifestEntry {
            id: Some("hf-tiny-random-minicpm-q2-gguf".into()),
            name: Some("HF Tiny Random MiniCPM Q2 GGUF".into()),
            repo: Some("tensorblock/tiny-random-minicpm-GGUF".into()),
            filename: Some("tiny-random-minicpm-Q2_K.gguf".into()),
            quantization: Some("Q2_K".into()),
            backend: Some("mistral_rs".into()),
            params_b: Some(0.08),
        })
        .unwrap();

        let downloaded = mgr
            .download_model("hf-tiny-random-minicpm-q2-gguf")
            .await
            .unwrap();
        let local_path = downloaded.local_path.clone().unwrap();

        assert!(matches!(downloaded.status, ModelStatus::Downloaded));
        assert_eq!(downloaded.preferred_backend, BackendKind::MistralRs);
        assert_eq!(downloaded.quantization, Some(Quantization::Q2K));
        assert_eq!(
            local_path.file_name().and_then(|name| name.to_str()),
            Some("tiny-random-minicpm-Q2_K.gguf")
        );
        assert!(
            local_path.starts_with(
                cache_dir
                    .path()
                    .join("models")
                    .join("hf-tiny-random-minicpm-q2-gguf")
            )
        );
        assert!(tokio::fs::metadata(&local_path).await.unwrap().len() > 70 * 1024 * 1024);

        let reloaded = ModelManagerImpl::new_standard(
            config_dir.path(),
            cache_dir.path(),
            Arc::new(MockEventService::default()),
            Arc::new(FixedDetector(DeviceKind::Cpu)),
        );
        let model = reloaded
            .get_model("hf-tiny-random-minicpm-q2-gguf")
            .unwrap();

        assert!(matches!(model.status, ModelStatus::Downloaded));
        assert_eq!(model.preferred_backend, BackendKind::MistralRs);
        assert_eq!(model.local_path.as_deref(), Some(local_path.as_path()));
    }

    #[test]
    fn recommend_config_uses_highest_quantization_that_fits_vram() {
        let manifest = Manifest {
            models: vec![ManifestEntry {
                id: Some("qwen-9b".into()),
                name: Some("Qwen".into()),
                repo: Some("Qwen/Qwen2.5-Coder-9B-Instruct-GGUF".into()),
                filename: Some("model.gguf".into()),
                quantization: None,
                backend: None,
                params_b: Some(9.0),
            }],
        };
        // 24 GB VRAM, usable 0.8*24 = 19.2 GiB.
        // 9B FP16 = 18 GiB fits, so the highest quantization is selected.
        let (mgr, _) = manager_with(
            manifest,
            Registry::default(),
            DeviceKind::Cuda {
                name: "RTX 4090".into(),
                vram_mb: 24_000,
                driver_version: "531".into(),
            },
        );
        let cfg = mgr.recommend_config("qwen-9b").unwrap();
        assert_eq!(cfg.quantization, Quantization::FP16);
        assert_eq!(cfg.gpu_layers, None);
        assert_eq!(cfg.context_size, 8192);
    }

    #[test]
    fn recommend_config_preserves_explicit_gguf_quantization_and_pins_small_cuda_models_to_gpu() {
        let manifest = Manifest {
            models: vec![ManifestEntry {
                id: Some("tinyllama-q2".into()),
                name: Some("TinyLlama Q2".into()),
                repo: Some("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF".into()),
                filename: Some("tinyllama-1.1b-chat-v1.0.Q2_K.gguf".into()),
                quantization: Some("Q2_K".into()),
                backend: Some("mistral_rs".into()),
                params_b: Some(1.1),
            }],
        };
        let (mgr, _) = manager_with(
            manifest,
            Registry::default(),
            DeviceKind::Cuda {
                name: "RTX 5080".into(),
                vram_mb: 16_000,
                driver_version: "581".into(),
            },
        );

        let cfg = mgr.recommend_config("tinyllama-q2").unwrap();

        assert_eq!(cfg.quantization, Quantization::Q2K);
        assert_eq!(cfg.gpu_layers, Some(999));
        assert_eq!(cfg.context_size, 4096);
    }

    #[test]
    fn recommend_config_falls_back_to_q4_on_low_vram() {
        let manifest = Manifest {
            models: vec![ManifestEntry {
                id: Some("qwen-9b".into()),
                name: Some("Qwen".into()),
                filename: Some("model.gguf".into()),
                params_b: Some(9.0),
                ..Default::default()
            }],
        };
        // 6 GB VRAM: usable 4.8 GiB. Q4KM = 4.5 GiB fits, Q5KM = 5.85 GiB does not.
        let (mgr, _) = manager_with(
            manifest,
            Registry::default(),
            DeviceKind::Cuda {
                name: "RTX 3050".into(),
                vram_mb: 6_000,
                driver_version: "531".into(),
            },
        );
        let cfg = mgr.recommend_config("qwen-9b").unwrap();
        assert_eq!(cfg.quantization, Quantization::Q4KM);
        assert_eq!(cfg.context_size, 4096);
    }

    #[test]
    fn recommend_config_for_cpu_forces_gpu_layers_zero() {
        let manifest = Manifest {
            models: vec![ManifestEntry {
                id: Some("qwen-9b".into()),
                name: Some("Qwen".into()),
                filename: Some("model.gguf".into()),
                params_b: Some(9.0),
                ..Default::default()
            }],
        };
        let (mgr, _) = manager_with(manifest, Registry::default(), DeviceKind::Cpu);
        let cfg = mgr.recommend_config("qwen-9b").unwrap();
        assert_eq!(cfg.gpu_layers, Some(0));
    }
}
