use std::path::PathBuf;
use std::sync::Arc;

use crytex_core::config::{BackendConfig, BackendKind, CrytexConfig, SearchConfig};
use crytex_core::indexer::ProjectIndexer;
use crytex_core::metrics::MetricsService;
use crytex_core::persistence::Persistence;
use crytex_core::services::RetryRateLimitBackend;
use crytex_core::services::caching::{CachedEmbedder, CachedVectorStore};
use crytex_core::services::hybrid::{HybridRetriever, build_fusion_strategy};
use crytex_core::services::{
    Embedder, EventService, FusionStrategy, InferenceService, LoraBenchmarkGate,
    LoraEvolutionService, LoraEvolutionServiceImpl, LoraRouter, LoraRouterImpl, LoraTrainer,
    MemoryBankService, MemoryBankServiceImpl, Reranker, SparseEmbedder, VectorStore,
};
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
            Ok(Arc::new(RetryRateLimitBackend::default_for(Arc::new(
                backend,
            ))))
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
    let raw: Arc<dyn VectorStore> = match select_vector_store_mode(config) {
        VectorStoreMode::ExternalQdrant(url) => match QdrantVectorStore::new(url) {
            Ok(store) => Arc::new(store),
            Err(e) => {
                eprintln!(
                    "Warning: failed to connect to Qdrant at {url}: {e}. \
                         Falling back to embedded vector store."
                );
                create_embedded_vector_store(config)
            }
        },
        VectorStoreMode::Embedded => create_embedded_vector_store(config),
        VectorStoreMode::ExternalDisabled => {
            eprintln!(
                "Warning: external vector DB is disabled by config.modules.external_vector_db. \
                 Using embedded vector store."
            );
            create_embedded_vector_store(config)
        }
    };

    if config.cache.vector_search_cache_enabled {
        Arc::new(CachedVectorStore::new(raw, metrics, config.cache.clone()))
    } else {
        raw
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum VectorStoreMode<'a> {
    ExternalQdrant(&'a str),
    Embedded,
    ExternalDisabled,
}

fn select_vector_store_mode(config: &CrytexConfig) -> VectorStoreMode<'_> {
    match (
        config.modules.external_vector_db,
        config.inference.vector_store_url.as_deref(),
    ) {
        (true, Some(url)) => VectorStoreMode::ExternalQdrant(url),
        (true, None) => VectorStoreMode::Embedded,
        (false, Some(_)) => VectorStoreMode::ExternalDisabled,
        (false, None) => VectorStoreMode::Embedded,
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
    Arc::new(HybridRetriever::new(
        embedder,
        vector_store,
        sparse_embedder,
        fusion,
    ))
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
    if !config.modules.reranker {
        return None;
    }
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
/// in pure Rust and writes a PEFT-like adapter directory.
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
    benchmark_gate: Option<Arc<dyn LoraBenchmarkGate>>,
) -> Arc<dyn LoraEvolutionService> {
    create_lora_evolution_service_with_trainer(
        persistence,
        task_service,
        prompt_version_repo,
        inference_service,
        event_service,
        embedder,
        vector_store,
        adapters_dir,
        base_model,
        create_lora_trainer(),
        benchmark_gate,
    )
}

/// Creates the LoRA evolution service with injectable trainer and benchmark gate.
///
/// Production uses [`create_lora_evolution_service`]; tests and alternate runtimes
/// can inject deterministic trainer/gate implementations without changing the
/// domain service.
#[allow(clippy::too_many_arguments)]
pub fn create_lora_evolution_service_with_trainer(
    persistence: Arc<dyn Persistence>,
    task_service: Arc<dyn crytex_core::services::TaskService>,
    prompt_version_repo: Arc<dyn crytex_core::persistence::PromptVersionRepository>,
    inference_service: Arc<dyn InferenceService>,
    event_service: Arc<dyn EventService>,
    embedder: Option<Arc<dyn Embedder>>,
    vector_store: Option<Arc<dyn VectorStore>>,
    adapters_dir: PathBuf,
    base_model: String,
    trainer: Arc<dyn LoraTrainer>,
    benchmark_gate: Option<Arc<dyn LoraBenchmarkGate>>,
) -> Arc<dyn LoraEvolutionService> {
    let mut service = LoraEvolutionServiceImpl::new(
        task_service,
        prompt_version_repo,
        persistence.clone(),
        persistence.clone(),
        inference_service,
        event_service,
        trainer,
        adapters_dir,
        base_model,
    )
    .with_threshold(50)
    .with_experience_repo(persistence.clone())
    .with_training_job_repo(persistence.clone());
    if let (Some(embedder), Some(vector_store)) = (embedder, vector_store) {
        service = service.with_vector_index(embedder, vector_store);
    }
    if let Some(gate) = benchmark_gate {
        service = service.with_benchmark_gate(gate);
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
    use async_trait::async_trait;
    use crytex_core::bus::EventBus;
    use crytex_core::models::{AgentLog, LoraAdapter, Project, Task, TaskStatus, TrainingExample};
    use crytex_core::persistence::{
        LoraAdapterRepository, ProjectRepository, TaskRepository, TrainingExampleRepository,
    };
    use crytex_core::services::{
        AuditLogEntry, AuditLogService, EventServiceImpl, InferenceServiceError,
        LoraBenchmarkDecision, LoraBenchmarkGate, LoraBenchmarkRequest, LoraTrainer,
        LoraTrainingConfig, LoraTrainingError, LoraTrainingResult, TaskServiceImpl,
    };
    use crytex_inference::{
        BackendInfo, InferenceRequest, InferenceResponse, LoRAAdapter as InferenceLoRAAdapter,
        ModelInfo, TokenUsage,
    };
    use std::path::Path;

    struct RejectingBenchmarkGate;

    #[async_trait]
    impl LoraBenchmarkGate for RejectingBenchmarkGate {
        async fn evaluate(
            &self,
            _request: LoraBenchmarkRequest,
        ) -> Result<LoraBenchmarkDecision, crytex_core::services::LoraEvolutionError> {
            Ok(LoraBenchmarkDecision {
                accepted: false,
                reason: "challenger failed held-out benchmark".into(),
                metadata: serde_json::json!({ "winner": "Baseline" }),
                quality_gates: Vec::new(),
            })
        }
    }

    struct MockTrainer;

    #[async_trait]
    impl LoraTrainer for MockTrainer {
        fn backend_name(&self) -> &'static str {
            "factory-mock"
        }

        async fn train(
            &self,
            examples: Vec<TrainingExample>,
            config: LoraTrainingConfig,
            output_dir: &Path,
        ) -> Result<LoraTrainingResult, LoraTrainingError> {
            tokio::fs::create_dir_all(output_dir).await?;
            let adapter_path = output_dir.join("candidate");
            tokio::fs::create_dir_all(&adapter_path).await?;
            let metadata =
                crytex_core::services::AdapterMetadata::from_examples(&config, &examples);
            tokio::fs::write(
                adapter_path.join("adapter_config.json"),
                br#"{"peft_type":"LORA","r":8,"lora_alpha":16}"#,
            )
            .await?;
            tokio::fs::write(
                adapter_path.join("adapter_model.safetensors"),
                b"small lora adapter",
            )
            .await?;
            tokio::fs::write(
                adapter_path.join("adapter_metadata.json"),
                serde_json::to_vec_pretty(&metadata)
                    .map_err(|error| LoraTrainingError::Backend(error.to_string()))?,
            )
            .await?;
            Ok(LoraTrainingResult {
                adapter_id: "candidate".into(),
                adapter_path,
                metrics: crytex_core::services::LoraMetrics {
                    train_loss: 0.10,
                    validation_loss: 0.12,
                    average_reward: 5.0,
                },
                metadata,
            })
        }
    }

    struct NoopInference;

    #[async_trait]
    impl InferenceService for NoopInference {
        async fn generate(
            &self,
            _request: InferenceRequest,
        ) -> Result<InferenceResponse, InferenceServiceError> {
            Ok(InferenceResponse {
                content: "ok".into(),
                usage: TokenUsage {
                    prompt_tokens: 1,
                    completion_tokens: 1,
                    total_tokens: 2,
                },
                finish_reason: "stop".into(),
            })
        }

        async fn embed(&self, _text: &str) -> Result<Vec<f32>, InferenceServiceError> {
            Ok(vec![0.0])
        }

        fn available_backends(&self) -> Vec<BackendInfo> {
            vec![]
        }

        async fn register_lora(
            &self,
            _lora: InferenceLoRAAdapter,
        ) -> Result<(), InferenceServiceError> {
            Ok(())
        }

        async fn swap_lora(&self, _lora_id: &str) -> Result<(), InferenceServiceError> {
            Ok(())
        }

        async fn list_models(
            &self,
            _backend_id: Option<&str>,
        ) -> Result<Vec<ModelInfo>, InferenceServiceError> {
            Ok(vec![])
        }
    }

    struct NoopAudit;

    #[async_trait]
    impl AuditLogService for NoopAudit {
        async fn log(
            &self,
            _entry: AuditLogEntry,
        ) -> Result<(), crytex_core::services::AuditError> {
            Ok(())
        }

        async fn list_by_task(
            &self,
            _task_id: &str,
        ) -> Result<Vec<AgentLog>, crytex_core::services::AuditError> {
            Ok(vec![])
        }

        async fn list_by_project(
            &self,
            _project_id: &str,
        ) -> Result<Vec<AgentLog>, crytex_core::services::AuditError> {
            Ok(vec![])
        }
    }

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("crytex-kernel-{name}-{}", ulid::Ulid::new()))
    }

    async fn seed_examples(repo: Arc<crytex_storage::Storage>, kind: &str) {
        repo.insert_project(&Project {
            id: "project-1".into(),
            name: "Kernel Factory Test".into(),
            root_path: ".".into(),
            created_at: 0,
            updated_at: 0,
            metadata: serde_json::Value::Null,
        })
        .await
        .unwrap();

        for idx in 0..50 {
            let task_id = format!("task-{idx}");
            repo.insert_task(&Task {
                id: task_id.clone(),
                project_id: "project-1".into(),
                parent_id: None,
                title: format!("Training task {idx}"),
                description: None,
                kind: kind.into(),
                status: TaskStatus::Completed,
                assigned_agent: Some("coder".into()),
                priority: 0,
                created_at: idx,
                started_at: Some(idx),
                finished_at: Some(idx + 1),
                payload: serde_json::json!({
                    "prompt": format!("Implement robust parser branch for realistic held-out example {idx}")
                }),
                result: Some(serde_json::json!({
                    "answer": format!("Parser branch implementation with validation and test coverage {idx}")
                })),
                iteration_count: 1,
                priority_score: 0.0,
                critic_score: Some(5.0),
                human_score: Some(5.0),
                prompt_version_id: None,
                lora_adapter_id: None,
                trace_id: format!("trace-{idx}"),
            })
            .await
            .unwrap();

            repo.insert_training_example(&TrainingExample {
                id: format!("example-{idx}"),
                task_id,
                project_id: Some("project-1".into()),
                prompt_version_id: None,
                task_kind: kind.into(),
                agent_role: None,
                model_id: None,
                rag_evidence_ids: Vec::new(),
                input_text: format!(
                    "Implement robust parser branch for realistic held-out example {idx}"
                ),
                output_text: format!(
                    "Parser branch implementation with validation and test coverage {idx}"
                ),
                accepted_output: Some(format!(
                    "Parser branch implementation with validation and test coverage {idx}"
                )),
                rejected_output: None,
                critic_feedback: None,
                failure_type: None,
                reward: 5.0,
                created_at: idx,
            })
            .await
            .unwrap();
        }
    }

    #[test]
    fn create_sparse_embedder_returns_some_when_enabled() {
        let mut config = CrytexConfig::default();
        config.inference.sparse_embedding_enabled = true;
        config.inference.sparse_embedding_language = Some("english".into());

        let embedder = create_sparse_embedder(&config);
        assert!(
            embedder.is_some(),
            "expected BM25 sparse embedder to be created"
        );
    }

    #[test]
    fn create_sparse_embedder_returns_none_when_disabled() {
        let mut config = CrytexConfig::default();
        config.inference.sparse_embedding_enabled = false;

        let embedder = create_sparse_embedder(&config);
        assert!(
            embedder.is_none(),
            "expected no sparse embedder when disabled"
        );
    }

    #[test]
    fn create_reranker_returns_none_when_module_disabled() {
        let mut config = CrytexConfig::default();
        config.modules.reranker = false;
        config.inference.rerank_backend = Some("rerank".into());
        config
            .inference
            .backends
            .push(BackendConfig::onnx("rerank", "bge-reranker-base"));

        let reranker = create_reranker(&config);

        assert!(reranker.is_none(), "disabled reranker must degrade to None");
    }

    #[test]
    fn select_vector_store_mode_uses_embedded_when_external_vector_db_disabled() {
        let mut config = CrytexConfig::default();
        config.modules.external_vector_db = false;
        config.inference.vector_store_url = Some("http://127.0.0.1:1".into());

        let mode = select_vector_store_mode(&config);

        assert_eq!(mode, VectorStoreMode::ExternalDisabled);
    }

    #[tokio::test]
    async fn create_lora_evolution_service_wires_benchmark_gate_before_promotion() {
        let db_path = temp_path("gate.sqlite");
        let adapters_dir = temp_path("adapters");
        let storage = Arc::new(
            crytex_storage::Storage::new(&db_path.to_string_lossy())
                .await
                .unwrap(),
        );
        let event_service: Arc<dyn EventService> =
            Arc::new(EventServiceImpl::new(Arc::new(EventBus::new())));
        let task_service: Arc<dyn crytex_core::services::TaskService> = Arc::new(
            TaskServiceImpl::new(storage.clone(), event_service.clone(), Arc::new(NoopAudit)),
        );
        seed_examples(storage.clone(), "codegen").await;

        let service = create_lora_evolution_service_with_trainer(
            storage.clone(),
            task_service,
            storage.clone(),
            Arc::new(NoopInference),
            event_service,
            None,
            None,
            adapters_dir.clone(),
            "mock-base".into(),
            Arc::new(MockTrainer),
            Some(Arc::new(RejectingBenchmarkGate)),
        );

        let result = service.train_and_register("codegen").await;

        assert!(
            matches!(
            result,
            Err(crytex_core::services::LoraEvolutionError::ValidationFailed(_, ref reason))
                if reason.contains("benchmark gate rejected")
            ),
            "expected benchmark rejection, got {result:?}"
        );
        let adapters: Vec<LoraAdapter> =
            storage.list_lora_adapters_by_kind("codegen").await.unwrap();
        assert!(adapters.is_empty(), "rejected LoRA must not be registered");
        assert!(
            !adapters_dir.join("codegen").join("candidate").exists(),
            "rejected LoRA artifact must be removed"
        );
        let _ = tokio::fs::remove_file(db_path).await;
        let _ = tokio::fs::remove_dir_all(adapters_dir).await;
    }
}
