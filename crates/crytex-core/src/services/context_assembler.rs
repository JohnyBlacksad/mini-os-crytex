//! Assemble prompt context under a token budget.
//!
//! The assembler combines a system prompt, semantically retrieved project
//! chunks, task history, and the current user query. When the resulting
//! messages exceed the token budget it first summarizes old history, then
//! drops the least relevant retrieved chunks, and finally truncates content.

use std::sync::Arc;

use async_trait::async_trait;
use crytex_compress::compress::Compressor;
use crytex_compress::compressors::summarize::Summarizer;
use crytex_compress::message::Message;
use crytex_compress::token::{CharTokenEstimator, TokenEstimator};
use thiserror::Error;

use crate::services::{
    build_fusion_strategy, Embedder, EmbeddingError, HybridRetriever, HybridSearchError,
    MemoryBankError, MemoryBankService, SearchResult, SparseEmbedder, VectorStore, VectorStoreError,
};

/// Errors produced while assembling context.
#[derive(Debug, Error)]
pub enum ContextAssemblerError {
    #[error("embedding failed: {0}")]
    Embedding(#[from] EmbeddingError),
    #[error("vector store failed: {0}")]
    VectorStore(#[from] VectorStoreError),
    #[error("hybrid search failed: {0}")]
    HybridSearch(#[from] HybridSearchError),
    #[error("memory bank failed: {0}")]
    MemoryBank(#[from] MemoryBankError),
    #[error("token estimation failed: {0}")]
    TokenEstimation(String),
    #[error("budget too small to fit mandatory messages")]
    BudgetTooSmall,
}

/// Input to the context assembler.
#[derive(Debug, Clone)]
pub struct ContextRequest {
    /// Mandatory system prompt.
    pub system_prompt: String,
    /// Current user query.
    pub user_query: String,
    /// Project to scope retrieval to.
    pub project_id: Option<String>,
    /// Previous conversation turns (oldest first).
    pub history: Vec<Message>,
    /// Maximum tokens for the assembled context.
    pub token_budget: usize,
    /// Number of chunks to retrieve from each collection.
    pub top_k: usize,
    /// When history consumes more than this ratio of the budget, summarize it.
    pub summarize_threshold_ratio: f32,
}

impl Default for ContextRequest {
    fn default() -> Self {
        Self {
            system_prompt: String::new(),
            user_query: String::new(),
            project_id: None,
            history: Vec::new(),
            token_budget: 4_096,
            top_k: 5,
            summarize_threshold_ratio: 0.6,
        }
    }
}

/// Builds a list of [`Message`]s that fit inside a token budget.
#[derive(Clone)]
pub struct ContextAssembler {
    embedder: Arc<dyn Embedder>,
    vector_store: Arc<dyn VectorStore>,
    sparse_embedder: Option<Arc<dyn SparseEmbedder>>,
    estimator: Arc<dyn TokenEstimator>,
    summarizer: Arc<dyn Summarizer>,
    memory_bank: Option<Arc<dyn MemoryBankService>>,
    hybrid_retriever: Arc<HybridRetriever>,
}

impl ContextAssembler {
    /// Create an assembler with a token estimator and a summarizing compressor.
    pub fn new(embedder: Arc<dyn Embedder>, vector_store: Arc<dyn VectorStore>) -> Self {
        Self::with_estimator(embedder, vector_store, Arc::new(CharTokenEstimator))
    }

    /// Create an assembler with a custom token estimator.
    pub fn with_estimator(
        embedder: Arc<dyn Embedder>,
        vector_store: Arc<dyn VectorStore>,
        estimator: Arc<dyn TokenEstimator>,
    ) -> Self {
        let summarizer: Arc<dyn Summarizer> = Arc::new(SimpleSummarizer);
        let sparse_embedder: Option<Arc<dyn SparseEmbedder>> = None;
        let fusion = build_fusion_strategy(crate::services::FusionStrategyKind::Rrf, 60.0);
        let hybrid_retriever = Arc::new(HybridRetriever::new(
            embedder.clone(),
            vector_store.clone(),
            sparse_embedder.clone(),
            fusion,
        ));
        Self {
            embedder,
            vector_store,
            sparse_embedder,
            estimator,
            summarizer,
            memory_bank: None,
            hybrid_retriever,
        }
    }

    /// Attach a sparse embedder (e.g. BM25) for hybrid retrieval.
    pub fn with_sparse_embedder(mut self, sparse_embedder: Arc<dyn SparseEmbedder>) -> Self {
        self.sparse_embedder = Some(sparse_embedder.clone());
        self.rebuild_hybrid_retriever();
        self
    }

    /// Replace the hybrid retriever used for chunk retrieval.
    pub fn with_hybrid_retriever(mut self, hybrid_retriever: Arc<HybridRetriever>) -> Self {
        self.hybrid_retriever = hybrid_retriever;
        self
    }

    fn rebuild_hybrid_retriever(&mut self) {
        let fusion = build_fusion_strategy(crate::services::FusionStrategyKind::Rrf, 60.0);
        self.hybrid_retriever = Arc::new(HybridRetriever::new(
            self.embedder.clone(),
            self.vector_store.clone(),
            self.sparse_embedder.clone(),
            fusion,
        ));
    }

    /// Attach a session memory bank as an additional retrieval source.
    pub fn with_memory_bank(mut self, memory_bank: Arc<dyn MemoryBankService>) -> Self {
        self.memory_bank = Some(memory_bank);
        self
    }

    /// Assemble context messages for `request`.
    pub async fn assemble(
        &self,
        request: ContextRequest,
    ) -> Result<Vec<Message>, ContextAssemblerError> {
        let mut messages = Vec::new();

        messages.push(Message::system(request.system_prompt.clone()));

        if let Some(memory_context) = self.retrieve_memory_context(&request).await? {
            messages.push(Message::system(memory_context));
        }

        let retrieved = self.retrieve_relevant_chunks(&request).await?;
        if !retrieved.is_empty() {
            messages.push(Message::system(format_retrieved_context(&retrieved)));
        }

        messages.extend(request.history.clone());
        messages.push(Message::user(request.user_query.clone()));

        let mandatory_tokens = self.estimate_messages(&messages)?;
        if mandatory_tokens > request.token_budget {
            self.fit_to_budget(&mut messages, &request).await?;
        }

        if self.estimate_messages(&messages)? > request.token_budget {
            return Err(ContextAssemblerError::BudgetTooSmall);
        }

        Ok(messages)
    }

    async fn retrieve_memory_context(
        &self,
        request: &ContextRequest,
    ) -> Result<Option<String>, ContextAssemblerError> {
        let (memory_bank, project_id) = match (&self.memory_bank, &request.project_id) {
            (Some(mb), Some(pid)) => (mb, pid),
            _ => return Ok(None),
        };

        let entries = memory_bank
            .recall_semantic(Some(project_id), &request.user_query, request.top_k)
            .await?;
        if entries.is_empty() {
            return Ok(None);
        }

        let parts: Vec<String> = entries
            .into_iter()
            .map(|e| format!("[{}] {}", e.kind, e.text))
            .collect();
        Ok(Some(format!("Session memory:\n{}", parts.join("\n\n"))))
    }

    async fn retrieve_relevant_chunks(
        &self,
        request: &ContextRequest,
    ) -> Result<Vec<SearchResult>, ContextAssemblerError> {
        let project_id = match &request.project_id {
            Some(id) => id,
            None => return Ok(Vec::new()),
        };
        if request.user_query.is_empty() {
            return Ok(Vec::new());
        }

        let results = self
            .hybrid_retriever
            .search(
                &request.user_query,
                project_id,
                &["code_chunks", "doc_chunks", "experience"],
                request.top_k,
                request.top_k,
            )
            .await?;
        Ok(results)
    }

    async fn fit_to_budget(
        &self,
        messages: &mut Vec<Message>,
        request: &ContextRequest,
    ) -> Result<(), ContextAssemblerError> {
        let history_indices: Vec<usize> = messages
            .iter()
            .enumerate()
            .filter(|(_, m)| m.role == "user" || m.role == "assistant")
            .map(|(i, _)| i)
            .collect();
        let history_tokens: usize = history_indices
            .iter()
            .map(|&i| {
                self.estimator
                    .estimate_message(&messages[i])
                    .unwrap_or(usize::MAX)
            })
            .sum();
        let threshold = (request.token_budget as f32 * request.summarize_threshold_ratio) as usize;

        if history_tokens > threshold {
            let total_tokens = self.estimate_messages(messages)?;
            let non_history_tokens = total_tokens.saturating_sub(history_tokens);
            let summary_budget = request
                .token_budget
                .saturating_sub(non_history_tokens)
                .max(20);

            let history_text = history_indices
                .iter()
                .map(|&i| format!("{}: {}", messages[i].role, messages[i].content))
                .collect::<Vec<_>>()
                .join("\n\n");

            let summary = self
                .summarizer
                .summarize(&history_text, summary_budget)
                .await
                .map_err(|e| ContextAssemblerError::TokenEstimation(e.to_string()))?;

            // Keep all non-history messages and insert a summary in place of history.
            let mut compressed: Vec<Message> = messages
                .iter()
                .enumerate()
                .filter(|(i, _)| !history_indices.contains(i))
                .map(|(_, m)| m.clone())
                .collect();
            let insert_at = compressed.len().saturating_sub(1);
            compressed.insert(
                insert_at,
                Message::system(format!("Summary of earlier conversation:\n{}", summary)),
            );
            *messages = compressed;
        }

        while self.estimate_messages(messages)? > request.token_budget {
            if !drop_least_relevant_chunk(messages) {
                break;
            }
        }

        let truncator =
            crytex_compress::compressors::TruncateCompressor::new(self.estimator.clone());
        *messages = truncator
            .compress(messages, request.token_budget)
            .await
            .map_err(|e| ContextAssemblerError::TokenEstimation(e.to_string()))?;

        Ok(())
    }

    fn estimate_messages(&self, messages: &[Message]) -> Result<usize, ContextAssemblerError> {
        self.estimator
            .estimate_messages(messages)
            .map_err(|e| ContextAssemblerError::TokenEstimation(e.to_string()))
    }
}

impl std::fmt::Debug for ContextAssembler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContextAssembler").finish_non_exhaustive()
    }
}

fn format_retrieved_context(results: &[SearchResult]) -> String {
    let parts: Vec<String> = results
        .iter()
        .map(|r| {
            let text = r.payload.get("text").and_then(|v| v.as_str()).unwrap_or("");
            let source = r
                .payload
                .get("source")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            format!("[{}] {}", source, text)
        })
        .collect();
    format!("Relevant context:\n{}", parts.join("\n\n"))
}

/// A simple summarizer that keeps the leading portion of the text.
///
/// It is deterministic and requires no external model, making tests fast and
/// offline-safe. Production deployments can inject a real LLM-based summarizer.
#[derive(Clone, Debug, Default)]
struct SimpleSummarizer;

#[async_trait]
impl Summarizer for SimpleSummarizer {
    async fn summarize(
        &self,
        text: &str,
        max_tokens: usize,
    ) -> Result<String, crytex_compress::compress::CompressionError> {
        let char_budget = max_tokens.saturating_mul(3);
        if text.len() <= char_budget {
            return Ok(text.to_string());
        }
        let prefix = &text[..char_budget];
        Ok(format!("{}...", prefix))
    }
}

fn drop_least_relevant_chunk(messages: &mut Vec<Message>) -> bool {
    if let Some(pos) = messages
        .iter()
        .position(|m| m.role == "system" && m.content.starts_with("Relevant context:"))
    {
        messages.remove(pos);
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::MemoryEntry;
    use crate::services::{
        hybrid::{HybridRetriever, ReciprocalRankFusion},
        MemoryBankError, MemoryBankService, MockEmbedder, MockSparseEmbedder, SearchOptions,
        SparseVector, VectorPoint, VectorStoreError,
    };
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    #[derive(Debug, Default)]
    struct TestVectorStore {
        collections: Mutex<HashMap<String, Vec<VectorPoint>>>,
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 {
            0.0
        } else {
            dot / (na * nb)
        }
    }

    #[async_trait::async_trait]
    impl VectorStore for TestVectorStore {
        async fn create_collection(
            &self,
            collection: &str,
            _dim: usize,
        ) -> Result<(), VectorStoreError> {
            self.collections
                .lock()
                .unwrap()
                .entry(collection.into())
                .or_default();
            Ok(())
        }

        async fn delete_collection(&self, collection: &str) -> Result<(), VectorStoreError> {
            self.collections.lock().unwrap().remove(collection);
            Ok(())
        }

        async fn upsert(
            &self,
            collection: &str,
            points: Vec<VectorPoint>,
        ) -> Result<(), VectorStoreError> {
            self.collections
                .lock()
                .unwrap()
                .entry(collection.into())
                .or_default()
                .extend(points);
            Ok(())
        }

        async fn search(
            &self,
            collection: &str,
            vector: &[f32],
            options: SearchOptions,
        ) -> Result<Vec<SearchResult>, VectorStoreError> {
            let cols = self.collections.lock().unwrap();
            let points = cols.get(collection).cloned().unwrap_or_default();
            let mut results: Vec<SearchResult> = points
                .iter()
                .map(|p| SearchResult {
                    id: p.id.clone(),
                    score: cosine(vector, &p.vector),
                    payload: p.payload.clone(),
                })
                .collect();
            results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            results.truncate(options.limit);
            Ok(results)
        }
    }

    fn make_assembler() -> ContextAssembler {
        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new(8));
        let store: Arc<dyn VectorStore> = Arc::new(TestVectorStore::default());
        ContextAssembler::new(embedder, store)
    }

    fn make_assembler_with_store(store: Arc<dyn VectorStore>) -> ContextAssembler {
        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new(8));
        ContextAssembler::new(embedder, store)
    }

    #[tokio::test]
    async fn assembler_fits_within_token_budget() {
        let assembler = make_assembler();
        let request = ContextRequest {
            system_prompt: "You are a helpful assistant.".into(),
            user_query: "hello".into(),
            token_budget: 20,
            ..Default::default()
        };

        let messages = assembler.assemble(request).await.unwrap();
        let tokens = assembler.estimate_messages(&messages).unwrap();
        assert!(tokens <= 20, "assembled context exceeds budget: {}", tokens);
    }

    #[tokio::test]
    async fn assembler_includes_top_n_relevant_chunks() {
        let store = Arc::new(TestVectorStore::default());
        let embedder = MockEmbedder::new(8);

        // Seed a code chunk whose vector is identical to the query vector.
        let query = "handle_error";
        let vector = embedder.embed(query).await.unwrap();
        store
            .upsert(
                "code_chunks",
                vec![VectorPoint {
                    id: "chunk-1".into(),
                    vector,
                    payload: serde_json::json!({
                        "project_id": "proj-1",
                        "source": "main.rs",
                        "text": "fn handle_error() {}",
                    }),
                }],
            )
            .await
            .unwrap();

        let assembler = make_assembler_with_store(store);
        let request = ContextRequest {
            system_prompt: "You are a coder.".into(),
            user_query: query.into(),
            project_id: Some("proj-1".into()),
            token_budget: 200,
            top_k: 3,
            ..Default::default()
        };

        let messages = assembler.assemble(request).await.unwrap();
        let context = messages
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(context.contains("fn handle_error"));
    }

    #[tokio::test]
    async fn assembler_summarizes_old_history_when_budget_tight() {
        let assembler = make_assembler();
        let history = (0..20)
            .map(|i| {
                if i % 2 == 0 {
                    Message::user(format!("user message number {}", i))
                } else {
                    Message::assistant(format!("assistant response number {}", i))
                }
            })
            .collect();

        let request = ContextRequest {
            system_prompt: "You are a helpful assistant.".into(),
            user_query: "what is the answer?".into(),
            history,
            token_budget: 150,
            summarize_threshold_ratio: 0.5,
            ..Default::default()
        };

        let messages = assembler.assemble(request).await.unwrap();
        let context = messages
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            context.contains("Summary of earlier conversation"),
            "expected history to be summarized, got: {}",
            context
        );
    }

    #[derive(Default)]
    struct TestMemoryBank {
        entries: std::sync::Mutex<Vec<MemoryEntry>>,
    }

    #[async_trait::async_trait]
    impl MemoryBankService for TestMemoryBank {
        async fn remember(&self, _entry: &MemoryEntry) -> Result<(), MemoryBankError> {
            Ok(())
        }
        async fn recall(
            &self,
            _project_id: Option<&str>,
            _kind: Option<&str>,
            _limit: usize,
        ) -> Result<Vec<MemoryEntry>, MemoryBankError> {
            Ok(self.entries.lock().unwrap().clone())
        }
        async fn recall_semantic(
            &self,
            _project_id: Option<&str>,
            _query: &str,
            limit: usize,
        ) -> Result<Vec<MemoryEntry>, MemoryBankError> {
            let entries = self.entries.lock().unwrap();
            Ok(entries.iter().take(limit).cloned().collect())
        }
        async fn summarize_session(
            &self,
            _session_id: &str,
        ) -> Result<Option<String>, MemoryBankError> {
            Ok(None)
        }
        async fn mental_model_for_project(
            &self,
            _project_id: &str,
        ) -> Result<serde_json::Value, MemoryBankError> {
            Ok(serde_json::Value::Null)
        }
    }

    #[derive(Debug, Default)]
    struct HybridTestVectorStore;

    #[async_trait::async_trait]
    impl VectorStore for HybridTestVectorStore {
        async fn create_collection(
            &self,
            _collection: &str,
            _dim: usize,
        ) -> Result<(), VectorStoreError> {
            Ok(())
        }
        async fn delete_collection(&self, _collection: &str) -> Result<(), VectorStoreError> {
            Ok(())
        }
        async fn upsert(
            &self,
            _collection: &str,
            _points: Vec<VectorPoint>,
        ) -> Result<(), VectorStoreError> {
            Ok(())
        }
        async fn search(
            &self,
            _collection: &str,
            _vector: &[f32],
            options: SearchOptions,
        ) -> Result<Vec<SearchResult>, VectorStoreError> {
            let mut results = vec![SearchResult {
                id: "dense-chunk".into(),
                score: 0.9,
                payload: serde_json::json!({
                    "project_id": "proj-1",
                    "source": "main.rs",
                    "text": "dense match",
                }),
            }];
            results.truncate(options.limit);
            Ok(results)
        }
        async fn supports_sparse(&self) -> bool {
            true
        }
        async fn search_sparse(
            &self,
            _collection: &str,
            _vector: &SparseVector,
            options: SearchOptions,
        ) -> Result<Vec<SearchResult>, VectorStoreError> {
            let mut results = vec![SearchResult {
                id: "sparse-chunk".into(),
                score: 5.0,
                payload: serde_json::json!({
                    "project_id": "proj-1",
                    "source": "lib.rs",
                    "text": "sparse match",
                }),
            }];
            results.truncate(options.limit);
            Ok(results)
        }
    }

    #[tokio::test]
    async fn assembler_uses_hybrid_retriever_to_fuse_dense_and_sparse() {
        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new(8));
        let sparse_embedder: Arc<dyn SparseEmbedder> = Arc::new(MockSparseEmbedder);
        let store: Arc<dyn VectorStore> = Arc::new(HybridTestVectorStore);

        let hybrid = Arc::new(HybridRetriever::new(
            embedder.clone(),
            store.clone(),
            Some(sparse_embedder),
            Arc::new(ReciprocalRankFusion::default()),
        ));

        let assembler = ContextAssembler::new(embedder, store).with_hybrid_retriever(hybrid);
        let request = ContextRequest {
            system_prompt: "You are a coder.".into(),
            user_query: "find matches".into(),
            project_id: Some("proj-1".into()),
            token_budget: 200,
            top_k: 5,
            ..Default::default()
        };

        let messages = assembler.assemble(request).await.unwrap();
        let context = messages
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        assert!(context.contains("dense match"));
        assert!(context.contains("sparse match"));
    }

    #[tokio::test]
    async fn assembler_includes_memory_bank_context() {
        let store = Arc::new(TestVectorStore::default());
        let memory_bank = Arc::new(TestMemoryBank::default());
        memory_bank.entries.lock().unwrap().push(MemoryEntry {
            id: "m1".into(),
            project_id: Some("proj-1".into()),
            session_id: None,
            kind: "goal".into(),
            text: "Always propagate errors with ?".into(),
            metadata: serde_json::Value::Null,
            created_at: 1,
        });

        let assembler = make_assembler_with_store(store).with_memory_bank(memory_bank);
        let request = ContextRequest {
            system_prompt: "You are a coder.".into(),
            user_query: "error handling".into(),
            project_id: Some("proj-1".into()),
            token_budget: 200,
            top_k: 3,
            ..Default::default()
        };

        let messages = assembler.assemble(request).await.unwrap();
        let context = messages
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(context.contains("Session memory:"));
        assert!(context.contains("Always propagate errors with ?"));
    }
}
