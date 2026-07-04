use std::path::PathBuf;
use std::sync::Arc;

use crytex_core::config::{BackendConfig, BackendKind, CrytexConfig, SearchConfig};
use crytex_core::metrics::MetricsService;
use crytex_core::persistence::Persistence;
use crytex_core::services::RetryRateLimitBackend;
use crytex_core::services::caching::{CachedEmbedder, CachedVectorStore};
use crytex_core::indexer::ProjectIndexer;
use crytex_core::services::{
    Embedder, EventService, FusionStrategy, InferenceService, LoraEvolutionService,
    LoraEvolutionServiceImpl, LoraRouter, LoraRouterImpl, LoraTrainer, MemoryBankService,
    MemoryBankServiceImpl, Reranker, SparseEmbedder, VectorStore,
};
use crytex_core::services::hybrid::{HybridRetriever, build_fusion_strategy};
use crytex_inference::{InferenceError, InferenceManager};
use crytex_storage::vector::{
    edge::EdgeVectorStore, memory::MemoryVectorStore, qdrant::QdrantVectorStore,
};

/// Creates an inference backend from a configuration entry.
pub fn create_backend(config: &BackendConfig) -> Result<Arc<dyn InferenceManager>, InferenceError> {
    match config.kind {
        BackendKind::Ollama => {
            let url = config
                .url
                .clone()
                .unwrap_or_else(|| "http://localhost:11434".to_string());
            Ok(Arc::new(RetryRateLimitBackend::default_for(Arc::new(
                crytex_inference_ollama::OllamaBackend::new(url, &config.model),
            ))))
        }
        BackendKind::OpenAiCompatible => {
            let url = config
                .url
                .clone()
                .unwrap_or_else(|| "https://api.openai.com/v1".to_string());
            Ok(Arc::new(RetryRateLimitBackend::default_for(Arc::new(
                crytex_inference_openai::OpenAiBackend::new(
                    url,
                    &config.model,
                    config.api_key.clone(),
                ),
            ))))
        }
        BackendKind::Anthropic => {
            let url = config
                .url
                .clone()
                .unwrap_or_else(|| "https://api.anthropic.com/v1".to_string());
            let api_key = config.api_key.clone().ok_or_else(|| {
                InferenceError::GenerationFailed("Anthropic API key is required".to_string())
            })?;
            Ok(Arc::new(RetryRateLimitBackend::default_for(Arc::new(
                crytex_inference_anthropic::AnthropicBackend::new(url, &config.model, api_key),
            ))))
        }
        #[cfg(feature = "mistral")]
        BackendKind::MistralRs => {
            let context_size = config.context_size.unwrap_or(4096);
            Ok(Arc::new(RetryRateLimitBackend::default_for(Arc::new(
                crytex_inference_mistral::MistralRsBackend::new(
                    &config.model,
                    context_size,
                    config.gpu_layers,
                ),
            ))))
        }
        #[cfg(not(feature = "mistral"))]
        BackendKind::MistralRs => Err(InferenceError::BackendNotAvailable(
            "mistral.rs backend is not compiled; enable the `mistral` feature".to_string(),
        )),
        BackendKind::Custom => {
            let url = config
                .url
                .clone()
                .unwrap_or_else(|| "http://localhost:8000/v1".to_string());
            Ok(Arc::new(RetryRateLimitBackend::default_for(Arc::new(
                crytex_inference_openai::OpenAiBackend::new(
                    url,
                    &config.model,
                    config.api_key.clone(),
                )
                .with_headers(config.headers.clone()),
            ))))
        }
        BackendKind::Onnx => {
            let backend = crytex_inference_onnx::OnnxBackend::from_name(&config.model)?;
            Ok(Arc::new(RetryRateLimitBackend::default_for(Arc::new(backend))))
        }
    }
}

/// Creates a vector store from configuration.
///
/// Priority:
/// 1. External Qdrant server when `vector_store_url` is set.
/// 2. Embedded Qdrant Edge at `vector_store_path` or `<data_dir>/vectors`.
/// 3. In-memory fallback only if the embedded store cannot be initialized.
pub fn create_vector_store(
    config: &CrytexConfig,
    metrics: Option<Arc<dyn MetricsService>>,
) -> Arc<dyn VectorStore> {
    let raw: Arc<dyn VectorStore> = if let Some(url) = &config.inference.vector_store_url {
        match QdrantVectorStore::new(url) {
            Ok(store) => Arc::new(store),
            Err(e) => {
                eprintln!(
                    "Warning: failed to connect to Qdrant at {url}: {e}. \
                     Falling back to embedded vector store."
                );
                create_embedded_vector_store(config)
            }
        }
    } else {
        create_embedded_vector_store(config)
    };

    if config.cache.vector_search_cache_enabled {
        Arc::new(CachedVectorStore::new(raw, metrics, config.cache.clone()))
    } else {
        raw
    }
}

fn create_embedded_vector_store(config: &CrytexConfig) -> Arc<dyn VectorStore> {
    let path = config
        .inference
        .vector_store_path
        .clone()
        .unwrap_or_else(|| config.paths.data_dir.join("vectors"));
    match EdgeVectorStore::new(&path) {
        Ok(store) => Arc::new(store),
        Err(e) => {
            eprintln!(
                "Warning: failed to open embedded vector store at {}: {e}. \
                 Falling back to in-memory vector store.",
                path.display()
            );
            Arc::new(MemoryVectorStore::new())
        }
    }
}

/// Creates an embedder backed by the inference service.
pub async fn create_embedder(
    config: &CrytexConfig,
    inference: Arc<dyn InferenceService>,
    metrics: Option<Arc<dyn MetricsService>>,
) -> Arc<dyn Embedder> {
    let namespace = embedding_namespace(config, inference.clone()).await;
    let raw = Arc::new(InferenceEmbedder(inference));
    if config.cache.embedding_cache_enabled {
        Arc::new(CachedEmbedder::new(raw, namespace, metrics, &config.cache))
    } else {
        raw
    }
}

async fn embedding_namespace(
    config: &CrytexConfig,
    inference: Arc<dyn InferenceService>,
) -> String {
    let backend_meta = config
        .inference
        .embedding_backend
        .as_ref()
        .or(config.inference.default_backend.as_ref())
        .and_then(|id| config.inference.backend(id))
        .map(|b| format!("{}:{}", b.id, b.model));

    let dim = inference
        .embed("crytex::embedding_namespace_probe")
        .await
        .map(|v| v.len())
        .unwrap_or(0);

    backend_meta
        .map(|meta| format!("{}:dim{}", meta, dim))
        .unwrap_or_else(|| format!("default:dim{}", dim))
}

/// Thin wrapper exposing an inference service as an [`Embedder`].
struct InferenceEmbedder(Arc<dyn InferenceService>);

#[async_trait::async_trait]
impl Embedder for InferenceEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, crytex_core::services::EmbeddingError> {
        self.0
            .embed(text)
            .await
            .map_err(|e| crytex_core::services::EmbeddingError::EmbeddingFailed(e.to_string()))
    }

    async fn dimension(&self) -> Result<usize, crytex_core::services::EmbeddingError> {
        let vector = self
            .0
            .embed("crytex::embedder_dimension_probe")
            .await
            .map_err(|e| crytex_core::services::EmbeddingError::EmbeddingFailed(e.to_string()))?;
        Ok(vector.len())
    }
}

/// Creates the configured fusion strategy for hybrid search.
pub fn create_fusion_strategy(config: &SearchConfig) -> Arc<dyn FusionStrategy> {
    build_fusion_strategy(config.fusion_strategy, config.rrf_k)
}

/// Creates a hybrid retriever that fuses dense and sparse search results.
pub fn create_hybrid_retriever(
    config: &CrytexConfig,
    embedder: Arc<dyn Embedder>,
    vector_store: Arc<dyn VectorStore>,
    sparse_embedder: Option<Arc<dyn SparseEmbedder>>,
) -> Arc<HybridRetriever> {
    let fusion = create_fusion_strategy(&config.search);
    Arc::new(HybridRetriever::new(embedder, vector_store, sparse_embedder, fusion))
}

/// Creates a reranker from configuration.
///
/// Returns `None` if no `rerank_backend` is configured or if the selected
/// backend kind does not support reranking.
/// Creates a sparse (BM25) embedder from configuration.
///
/// Returns `None` if sparse embedding is disabled or if the underlying
/// qdrant-edge BM25 model cannot be initialized.
pub fn create_sparse_embedder(config: &CrytexConfig) -> Option<Arc<dyn SparseEmbedder>> {
    if !config.inference.sparse_embedding_enabled {
        return None;
    }
    let language = config.inference.sparse_embedding_language.clone();
    let embedder =
        crytex_storage::sparse_embedder::EdgeBm25SparseEmbedder::with_language(language).ok()?;
    Some(Arc::new(embedder))
}

/// Creates a project indexer wired for dense and (optionally) sparse indexing.
pub fn create_project_indexer(
    embedder: Arc<dyn Embedder>,
    vector_store: Arc<dyn VectorStore>,
    sparse_embedder: Option<Arc<dyn SparseEmbedder>>,
) -> ProjectIndexer {
    let mut indexer = ProjectIndexer::new(embedder, vector_store);
    if let Some(sparse) = sparse_embedder {
        indexer = indexer.with_sparse_embedder(sparse);
    }
    indexer
}

pub fn create_reranker(config: &CrytexConfig) -> Option<Arc<dyn Reranker>> {
    let backend_id = config.inference.rerank_backend.as_deref()?;
    let backend_config = config.inference.backend(backend_id)?;
    match backend_config.kind {
        BackendKind::Onnx => {
            let reranker =
                crytex_inference_onnx::OnnxReranker::from_name(&backend_config.model).ok()?;
            Some(Arc::new(reranker))
        }
        _ => None,
    }
}

/// Creates a LoRA trainer.
///
/// Uses the Candle-based trainer which performs real low-rank adapter training
/// in pure Rust and writes a `.safetensors` adapter file.
pub fn create_lora_trainer() -> Arc<dyn LoraTrainer> {
    Arc::new(crytex_inference_candle::CandleLoraTrainer::new())
}

/// Creates the domain service that collects golden examples and trains adapters.
#[allow(clippy::too_many_arguments)]
pub fn create_lora_evolution_service(
    persistence: Arc<dyn Persistence>,
    task_service: Arc<dyn crytex_core::services::TaskService>,
    prompt_version_repo: Arc<dyn crytex_core::persistence::PromptVersionRepository>,
    inference_service: Arc<dyn InferenceService>,
    event_service: Arc<dyn EventService>,
    embedder: Option<Arc<dyn Embedder>>,
    vector_store: Option<Arc<dyn VectorStore>>,
    adapters_dir: PathBuf,
    base_model: String,
) -> Arc<dyn LoraEvolutionService> {
    let mut service = LoraEvolutionServiceImpl::new(
        task_service,
        prompt_version_repo,
        persistence.clone(),
        persistence.clone(),
        inference_service,
        event_service,
        create_lora_trainer(),
        adapters_dir,
        base_model,
    )
    .with_threshold(50)
    .with_experience_repo(persistence.clone())
    .with_training_job_repo(persistence.clone());
    if let (Some(embedder), Some(vector_store)) = (embedder, vector_store) {
        service = service.with_vector_index(embedder, vector_store);
    }
    Arc::new(service)
}

/// Creates the router that selects an adapter for each task.
pub fn create_lora_router(
    evolution: Arc<dyn LoraEvolutionService>,
    embedder: Option<Arc<dyn Embedder>>,
    vector_store: Option<Arc<dyn VectorStore>>,
) -> Arc<dyn LoraRouter> {
    let mut router = LoraRouterImpl::new(evolution);
    if let (Some(embedder), Some(vector_store)) = (embedder, vector_store) {
        router = router.with_semantic_fallback(embedder, vector_store);
    }
    Arc::new(router)
}

/// Creates the session memory bank.
pub fn create_memory_bank_service(
    repository: Arc<dyn crytex_core::persistence::MemoryEntryRepository>,
    embedder: Option<Arc<dyn Embedder>>,
    vector_store: Option<Arc<dyn VectorStore>>,
) -> Arc<dyn MemoryBankService> {
    let mut service = MemoryBankServiceImpl::new(repository);
    if let (Some(embedder), Some(vector_store)) = (embedder, vector_store) {
        service = service.with_semantic_index(embedder, vector_store);
    }
    Arc::new(service)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_sparse_embedder_returns_some_when_enabled() {
        let mut config = CrytexConfig::default();
        config.inference.sparse_embedding_enabled = true;
        config.inference.sparse_embedding_language = Some("english".into());

        let embedder = create_sparse_embedder(&config);
        assert!(embedder.is_some(), "expected BM25 sparse embedder to be created");
    }

    #[test]
    fn create_sparse_embedder_returns_none_when_disabled() {
        let mut config = CrytexConfig::default();
        config.inference.sparse_embedding_enabled = false;

        let embedder = create_sparse_embedder(&config);
        assert!(embedder.is_none(), "expected no sparse embedder when disabled");
    }
}
