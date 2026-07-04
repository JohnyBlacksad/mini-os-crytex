//! Local ONNX embedding and reranking backend powered by `fastembed`.
//!
//! Supports both fastembed's built-in model catalogue and user-supplied ONNX
//! models on disk. Models are downloaded on first use and cached for subsequent
//! runs.

pub mod reranker;
pub use reranker::{OnnxReranker, RerankerSource};

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use async_trait::async_trait;
use crytex_inference::{
    BackendInfo, InferenceError, InferenceManager, InferenceRequest, InferenceResponse,
    LoRAAdapter, ModelInfo,
};
use fastembed::{
    EmbeddingModel, InitOptions, InitOptionsUserDefined, TextEmbedding, TokenizerFiles,
    UserDefinedEmbeddingModel,
};

/// Source of an ONNX embedding model.
#[derive(Debug, Clone)]
pub enum EmbeddingSource {
    /// A model shipped with fastembed (e.g. `sentence-transformers/all-MiniLM-L6-v2`).
    Preset(EmbeddingModel),
    /// A user-supplied ONNX model directory on disk.
    Local(PathBuf),
}

/// ONNX-based embedding backend.
pub struct OnnxBackend {
    source: EmbeddingSource,
    model_id: String,
    model: OnceLock<Arc<Mutex<TextEmbedding>>>,
}

impl std::fmt::Debug for OnnxBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxBackend")
            .field("model_id", &self.model_id)
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

impl OnnxBackend {
    /// Create a backend with the default `all-MiniLM-L6-v2` model.
    pub fn new() -> Self {
        Self::with_model(EmbeddingModel::AllMiniLML6V2)
    }

    /// Create a backend with a specific FastEmbed-supported model.
    pub fn with_model(model: EmbeddingModel) -> Self {
        let model_id = model.to_string();
        Self {
            source: EmbeddingSource::Preset(model),
            model_id,
            model: OnceLock::new(),
        }
    }

    /// Create a backend from a user-supplied ONNX model directory.
    ///
    /// The directory must contain an `.onnx` model file and the tokenizer JSON
    /// files (`tokenizer.json`, `config.json`, `special_tokens_map.json`,
    /// `tokenizer_config.json`).
    pub fn from_local_path(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let model_id = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("local-onnx")
            .to_string();
        Self {
            source: EmbeddingSource::Local(path),
            model_id,
            model: OnceLock::new(),
        }
    }

    /// Create a backend from a model identifier.
    ///
    /// If `model` starts with `local:` or looks like a filesystem path, it is
    /// loaded as a user-defined ONNX directory. Otherwise it is matched against
    /// fastembed's built-in catalogue.
    pub fn from_name(model: &str) -> Result<Self, InferenceError> {
        if let Some(local) = model.strip_prefix("local:") {
            return Ok(Self::from_local_path(local));
        }
        if let Some(preset) = parse_embedding_model(model) {
            return Ok(Self::with_model(preset));
        }
        if is_existing_path(model) {
            return Ok(Self::from_local_path(model));
        }
        Err(InferenceError::EmbeddingFailed(format!(
            "unknown embedding model: {model}"
        )))
    }

    /// Return the embedding dimension advertised by the configured model.
    ///
    /// For preset models this reads fastembed's static metadata without
    /// downloading weights. For local models it attempts to read `hidden_size`
    /// from `config.json` and falls back to loading the model and embedding an
    /// empty string.
    pub fn dimension(&self) -> Result<usize, InferenceError> {
        match &self.source {
            EmbeddingSource::Preset(model) => {
                let info = TextEmbedding::get_model_info(model)
                    .map_err(|e| InferenceError::EmbeddingFailed(e.to_string()))?;
                Ok(info.dim)
            }
            EmbeddingSource::Local(path) => {
                if let Some(dim) = read_hidden_size_from_config(path) {
                    return Ok(dim);
                }
                let model = self.model()?;
                let embeddings = model
                    .lock()
                    .map_err(|e| InferenceError::EmbeddingFailed(e.to_string()))?
                    .embed(vec![""], None)
                    .map_err(|e| InferenceError::EmbeddingFailed(e.to_string()))?;
                embeddings
                    .into_iter()
                    .next()
                    .map(|v| v.len())
                    .ok_or_else(|| InferenceError::EmbeddingFailed("empty embedding result".into()))
            }
        }
    }

    fn load_model(&self) -> Result<Arc<Mutex<TextEmbedding>>, InferenceError> {
        let model = match &self.source {
            EmbeddingSource::Preset(model) => TextEmbedding::try_new(
                InitOptions::new(model.clone()).with_show_download_progress(true),
            )
            .map_err(|e| InferenceError::EmbeddingFailed(e.to_string()))?,
            EmbeddingSource::Local(path) => {
                let local_model = read_local_embedding_model(path)?;
                let tokenizer_files = read_tokenizer_files(path)?;
                let mut user_model =
                    UserDefinedEmbeddingModel::new(local_model.onnx_bytes, tokenizer_files);
                for (file_name, buffer) in local_model.external_data {
                    user_model = user_model.with_external_initializer(file_name, buffer);
                }
                TextEmbedding::try_new_from_user_defined(
                    user_model,
                    InitOptionsUserDefined::default(),
                )
                .map_err(|e| InferenceError::EmbeddingFailed(e.to_string()))?
            }
        };
        Ok(Arc::new(Mutex::new(model)))
    }

    fn model(&self) -> Result<Arc<Mutex<TextEmbedding>>, InferenceError> {
        if let Some(m) = self.model.get() {
            return Ok(m.clone());
        }
        let m = self.load_model()?;
        let _ = self.model.set(m.clone());
        Ok(m)
    }
}

impl Default for OnnxBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl InferenceManager for OnnxBackend {
    async fn generate(
        &self,
        _request: InferenceRequest,
    ) -> Result<InferenceResponse, InferenceError> {
        Err(InferenceError::GenerationFailed(
            "ONNX embedding backend does not support text generation".into(),
        ))
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>, InferenceError> {
        let model = self.model()?.clone();
        let text = text.to_string();
        let embeddings = tokio::task::spawn_blocking(move || {
            model
                .lock()
                .map_err(|e| InferenceError::EmbeddingFailed(e.to_string()))?
                .embed(vec![text], None)
                .map_err(|e| InferenceError::EmbeddingFailed(e.to_string()))
        })
        .await
        .map_err(|e| InferenceError::EmbeddingFailed(e.to_string()))??;

        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| InferenceError::EmbeddingFailed("empty embedding result".into()))
    }

    async fn register_lora(&self, _lora: LoRAAdapter) -> Result<(), InferenceError> {
        Err(InferenceError::LoRALoadFailed(
            "ONNX embedding backend does not support LoRA".into(),
        ))
    }

    async fn swap_lora(&self, _lora_id: &str) -> Result<(), InferenceError> {
        Err(InferenceError::LoRALoadFailed(
            "ONNX embedding backend does not support LoRA".into(),
        ))
    }

    fn available_backends(&self) -> Vec<BackendInfo> {
        vec![BackendInfo {
            id: "onnx".into(),
            name: format!("ONNX embedding ({})", self.model_id),
            capabilities: vec!["embed".into()],
        }]
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, InferenceError> {
        Ok(vec![ModelInfo {
            id: self.model_id.clone(),
            name: self.model_id.clone(),
        }])
    }
}

fn is_existing_path(s: &str) -> bool {
    Path::new(s).is_dir()
}

fn parse_embedding_model(name: &str) -> Option<EmbeddingModel> {
    if let Ok(model) = name.parse::<EmbeddingModel>() {
        return Some(model);
    }
    let supported = TextEmbedding::list_supported_models();
    supported
        .into_iter()
        .find(|info| info.model_code.eq_ignore_ascii_case(name))
        .map(|info| info.model)
}

fn find_onnx_file(dir: &Path) -> Result<PathBuf, InferenceError> {
    let candidate = dir.join("model.onnx");
    if candidate.is_file() {
        return Ok(candidate);
    }
    let mut onnx_files: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| InferenceError::EmbeddingFailed(format!("cannot read model dir: {e}")))?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry
                .path()
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("onnx"))
                .unwrap_or(false)
        })
        .map(|entry| entry.path())
        .collect();
    onnx_files.sort();
    onnx_files.into_iter().next().ok_or_else(|| {
        InferenceError::EmbeddingFailed(format!(
            "no .onnx model file found in {}",
            dir.display()
        ))
    })
}

struct LocalEmbeddingFiles {
    onnx_bytes: Vec<u8>,
    external_data: Vec<(String, Vec<u8>)>,
}

fn read_local_embedding_model(dir: &Path) -> Result<LocalEmbeddingFiles, InferenceError> {
    let onnx_path = find_onnx_file(dir)?;
    let onnx_bytes = std::fs::read(&onnx_path)
        .map_err(|e| InferenceError::EmbeddingFailed(format!("cannot read {}: {}", onnx_path.display(), e)))?;

    let mut external_data = Vec::new();
    let data_path = onnx_path.with_extension("onnx.data");
    if data_path.is_file() {
        let buffer = std::fs::read(&data_path).map_err(|e| {
            InferenceError::EmbeddingFailed(format!("cannot read {}: {}", data_path.display(), e))
        })?;
        let file_name = data_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("model.onnx.data")
            .to_string();
        external_data.push((file_name, buffer));
    }
    Ok(LocalEmbeddingFiles {
        onnx_bytes,
        external_data,
    })
}

fn read_tokenizer_files(dir: &Path) -> Result<TokenizerFiles, InferenceError> {
    fn read(dir: &Path, name: &str) -> Result<Vec<u8>, InferenceError> {
        let path = dir.join(name);
        std::fs::read(&path)
            .map_err(|e| InferenceError::EmbeddingFailed(format!("cannot read {}: {}", path.display(), e)))
    }
    Ok(TokenizerFiles {
        tokenizer_file: read(dir, "tokenizer.json")?,
        config_file: read(dir, "config.json")?,
        special_tokens_map_file: read(dir, "special_tokens_map.json")?,
        tokenizer_config_file: read(dir, "tokenizer_config.json")?,
    })
}

fn read_hidden_size_from_config(dir: &Path) -> Option<usize> {
    let config_path = dir.join("config.json");
    let contents = std::fs::read_to_string(&config_path).ok()?;
    let config: serde_json::Value = serde_json::from_str(&contents).ok()?;
    config
        .get("hidden_size")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .or_else(|| {
            config
                .get("dim")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
        })
        .or_else(|| {
            config
                .get("n_embd")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
        })
        .or_else(|| {
            config
                .get("d_model")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
        })
}

#[cfg(test)]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_reports_capabilities_without_download() {
        let backend = OnnxBackend::new();
        let info = backend.available_backends();
        assert_eq!(info.len(), 1);
        assert!(info[0].capabilities.contains(&"embed".into()));
        assert_eq!(backend.dimension().unwrap(), 384);
    }

    #[test]
    fn preset_nomic_dimension_is_768() {
        let backend = OnnxBackend::with_model(EmbeddingModel::NomicEmbedTextV15);
        assert_eq!(backend.dimension().unwrap(), 768);
    }

    #[test]
    fn from_name_matches_preset() {
        let backend = OnnxBackend::from_name("nomic-ai/nomic-embed-text-v1.5").unwrap();
        assert_eq!(backend.dimension().unwrap(), 768);
    }

    #[test]
    fn from_name_recognises_local_prefix() {
        let backend = OnnxBackend::from_name("local:C:\\models\\my-embed").unwrap();
        assert!(matches!(backend.source, EmbeddingSource::Local(_)));
        assert_eq!(backend.model_id, "my-embed");
    }

    #[tokio::test]
    #[ignore = "requires the ONNX model to be downloaded/cached (network)"]
    async fn backend_returns_384_dimensional_vector() {
        let backend = OnnxBackend::new();
        let vector = backend.embed("hello world").await.unwrap();
        assert_eq!(vector.len(), 384);
    }

    #[tokio::test]
    #[ignore = "requires the ONNX model to be downloaded/cached (network)"]
    async fn backend_handles_batch_embedding() {
        let backend = OnnxBackend::new();
        let texts = vec![
            "The cat sits on the mat",
            "A feline rests on a rug",
            "Dogs are great pets",
        ];
        let embeddings = tokio::task::spawn_blocking(move || {
            backend.model().unwrap().lock().unwrap().embed(texts, None)
        })
        .await
        .unwrap()
        .unwrap();
        assert_eq!(embeddings.len(), 3);
        for v in &embeddings {
            assert_eq!(v.len(), 384);
        }
    }

    #[tokio::test]
    #[ignore = "requires the ONNX model to be downloaded/cached (network)"]
    async fn cosine_cat_kitten_greater_than_cat_car() {
        let backend = OnnxBackend::new();
        let cat = backend.embed("A cat is resting").await.unwrap();
        let kitten = backend.embed("A kitten is sleeping").await.unwrap();
        let car = backend.embed("A car is fast").await.unwrap();

        let sim_cat_kitten = cosine_similarity(&cat, &kitten);
        let sim_cat_car = cosine_similarity(&cat, &car);

        assert!(
            sim_cat_kitten > sim_cat_car,
            "cat/kitten ({sim_cat_kitten}) should be more similar than cat/car ({sim_cat_car})"
        );
    }
}
