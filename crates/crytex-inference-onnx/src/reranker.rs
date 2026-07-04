//! Local ONNX reranker backend powered by `fastembed`.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use async_trait::async_trait;
use crytex_core::services::{
    Reranker, RerankerError, RerankPassage, RerankResult as CoreRerankResult,
};
use crytex_inference::InferenceError;
use fastembed::{
    OnnxSource, RerankInitOptions, RerankInitOptionsUserDefined, RerankerModel, TextRerank,
    TokenizerFiles, UserDefinedRerankingModel,
};

/// Source of an ONNX reranker model.
#[derive(Debug, Clone)]
pub enum RerankerSource {
    /// A model shipped with fastembed (e.g. `BAAI/bge-reranker-base`).
    Preset(RerankerModel),
    /// A user-supplied ONNX model directory on disk.
    Local(PathBuf),
}

/// ONNX-based reranker backend.
pub struct OnnxReranker {
    source: RerankerSource,
    model_id: String,
    model: OnceLock<Arc<Mutex<TextRerank>>>,
}

impl std::fmt::Debug for OnnxReranker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxReranker")
            .field("model_id", &self.model_id)
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

impl OnnxReranker {
    /// Create a reranker with the default `BAAI/bge-reranker-base` model.
    pub fn new() -> Self {
        Self::with_model(RerankerModel::BGERerankerBase)
    }

    /// Create a reranker with a specific FastEmbed-supported model.
    pub fn with_model(model: RerankerModel) -> Self {
        let model_id = model.to_string();
        Self {
            source: RerankerSource::Preset(model),
            model_id,
            model: OnceLock::new(),
        }
    }

    /// Create a reranker from a user-supplied ONNX model directory.
    pub fn from_local_path(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let model_id = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("local-reranker")
            .to_string();
        Self {
            source: RerankerSource::Local(path),
            model_id,
            model: OnceLock::new(),
        }
    }

    /// Create a reranker from a model identifier.
    ///
    /// If `model` starts with `local:` or looks like a filesystem path, it is
    /// loaded as a user-defined ONNX directory. Otherwise it is matched against
    /// fastembed's built-in catalogue.
    pub fn from_name(model: &str) -> Result<Self, InferenceError> {
        if let Some(local) = model.strip_prefix("local:") {
            return Ok(Self::from_local_path(local));
        }
        if let Some(preset) = parse_reranker_model(model) {
            return Ok(Self::with_model(preset));
        }
        if is_existing_path(model) {
            return Ok(Self::from_local_path(model));
        }
        Err(InferenceError::EmbeddingFailed(format!(
            "unknown reranker model: {model}"
        )))
    }

    fn load_model(&self) -> Result<Arc<Mutex<TextRerank>>, InferenceError> {
        let model = match &self.source {
            RerankerSource::Preset(model) => TextRerank::try_new(
                RerankInitOptions::new(model.clone()).with_show_download_progress(true),
            )
            .map_err(|e| InferenceError::EmbeddingFailed(e.to_string()))?,
            RerankerSource::Local(path) => {
                let onnx_source = find_onnx_source(path)?;
                let tokenizer_files = read_tokenizer_files(path)?;
                let user_model = UserDefinedRerankingModel::new(onnx_source, tokenizer_files);
                TextRerank::try_new_from_user_defined(
                    user_model,
                    RerankInitOptionsUserDefined::default(),
                )
                .map_err(|e| InferenceError::EmbeddingFailed(e.to_string()))?
            }
        };
        Ok(Arc::new(Mutex::new(model)))
    }

    fn model(&self) -> Result<Arc<Mutex<TextRerank>>, InferenceError> {
        if let Some(m) = self.model.get() {
            return Ok(m.clone());
        }
        let m = self.load_model()?;
        let _ = self.model.set(m.clone());
        Ok(m)
    }
}

impl Default for OnnxReranker {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Reranker for OnnxReranker {
    async fn rerank(
        &self,
        query: &str,
        passages: &[RerankPassage],
    ) -> Result<Vec<CoreRerankResult>, RerankerError> {
        if passages.is_empty() {
            return Ok(Vec::new());
        }

        let texts: Vec<String> = passages.iter().map(|p| p.text.clone()).collect();
        let model = self
            .model()
            .map_err(|e| RerankerError::RerankFailed(e.to_string()))?;
        let query = query.to_string();

        let results = tokio::task::spawn_blocking(move || {
            model
                .lock()
                .map_err(|e| RerankerError::RerankFailed(e.to_string()))?
                .rerank(query, texts, false, None)
                .map_err(|e| RerankerError::RerankFailed(e.to_string()))
        })
        .await
        .map_err(|e| RerankerError::RerankFailed(e.to_string()))??;

        Ok(results
            .into_iter()
            .map(|r| CoreRerankResult {
                id: passages[r.index].id.clone(),
                score: r.score,
                text: r.document.unwrap_or_default(),
                payload: passages[r.index].payload.clone(),
            })
            .collect())
    }
}

fn is_existing_path(s: &str) -> bool {
    Path::new(s).is_dir()
}

fn parse_reranker_model(name: &str) -> Option<RerankerModel> {
    if let Ok(model) = name.parse::<RerankerModel>() {
        return Some(model);
    }
    let supported = TextRerank::list_supported_models();
    supported
        .into_iter()
        .find(|info| info.model_code.eq_ignore_ascii_case(name))
        .map(|info| info.model)
}

fn find_onnx_source(dir: &Path) -> Result<OnnxSource, InferenceError> {
    let candidate = dir.join("model.onnx");
    if candidate.is_file() {
        return Ok(OnnxSource::File(candidate));
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
    let path = onnx_files.into_iter().next().ok_or_else(|| {
        InferenceError::EmbeddingFailed(format!(
            "no .onnx model file found in {}",
            dir.display()
        ))
    })?;
    Ok(OnnxSource::File(path))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reranker_reports_default_model() {
        let reranker = OnnxReranker::new();
        assert_eq!(reranker.model_id, "BAAI/bge-reranker-base");
    }

    #[test]
    fn from_name_matches_preset() {
        let reranker = OnnxReranker::from_name("BAAI/bge-reranker-base").unwrap();
        assert!(matches!(reranker.source, RerankerSource::Preset(_)));
    }

    #[test]
    fn from_name_recognises_local_prefix() {
        let reranker = OnnxReranker::from_name("local:C:\\models\\my-reranker").unwrap();
        assert!(matches!(reranker.source, RerankerSource::Local(_)));
        assert_eq!(reranker.model_id, "my-reranker");
    }
}
