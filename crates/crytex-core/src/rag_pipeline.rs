//! Explainable RAG pipeline for project context retrieval.
//!
//! The pipeline keeps every stage observable: dense candidates, sparse
//! candidates, fused ranking, optional rerank, prompt-injection risk, token
//! budget selection, and the final context handed to an agent.

use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::services::{
    Embedder, EmbeddingError, FusionStrategyKind, RerankPassage, Reranker, RerankerError,
    SearchOptions, SearchResult, SparseEmbedder, VectorStore, VectorStoreError,
    build_fusion_strategy,
};

#[derive(Debug, Error)]
pub enum RagPipelineError {
    #[error("embedding failed: {0}")]
    Embedding(#[from] EmbeddingError),
    #[error("vector store failed: {0}")]
    VectorStore(#[from] VectorStoreError),
    #[error("reranker failed: {0}")]
    Reranker(#[from] RerankerError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Clone)]
pub struct RagPipelineRequest {
    pub project_id: String,
    pub query: String,
    pub top_k: usize,
    pub token_budget: usize,
    pub rerank: bool,
    pub explain: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RagPipelineResponse {
    pub project_id: String,
    pub query: String,
    pub selected_context: String,
    pub diagnostics: RagDiagnostics,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct RagDiagnostics {
    pub dense_candidates: Vec<RagCandidateEvidence>,
    pub sparse_candidates: Vec<RagCandidateEvidence>,
    pub fused_candidates: Vec<RagCandidateEvidence>,
    pub reranked_candidates: Vec<RagCandidateEvidence>,
    pub selected: Vec<RagCandidateEvidence>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RagCandidateEvidence {
    pub id: String,
    pub score: f32,
    pub stage: String,
    pub relative_path: Option<String>,
    pub language: Option<String>,
    pub symbol_id: Option<String>,
    pub prompt_injection: Option<PromptInjectionSeverity>,
    pub reason: String,
    pub text_preview: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PromptInjectionSeverity {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IncrementalReindexReport {
    pub status: String,
    pub changed_files: usize,
    pub removed_files: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CrashSafeRebuildReport {
    pub status: String,
    pub active_path: String,
    pub staging_path: String,
}

pub struct RagPipeline {
    embedder: Arc<dyn Embedder>,
    vector_store: Arc<dyn VectorStore>,
    sparse_embedder: Option<Arc<dyn SparseEmbedder>>,
    reranker: Option<Arc<dyn Reranker>>,
}

impl RagPipeline {
    pub fn new(embedder: Arc<dyn Embedder>, vector_store: Arc<dyn VectorStore>) -> Self {
        Self {
            embedder,
            vector_store,
            sparse_embedder: None,
            reranker: None,
        }
    }

    pub fn with_sparse_embedder(mut self, sparse_embedder: Arc<dyn SparseEmbedder>) -> Self {
        self.sparse_embedder = Some(sparse_embedder);
        self
    }

    pub fn with_reranker(mut self, reranker: Arc<dyn Reranker>) -> Self {
        self.reranker = Some(reranker);
        self
    }

    pub async fn search(
        &self,
        request: RagPipelineRequest,
    ) -> Result<RagPipelineResponse, RagPipelineError> {
        let dense = self.dense_candidates(&request).await?;
        let sparse = self.sparse_candidates(&request).await?;
        let fused = fuse_candidates(dense.clone(), sparse.clone(), request.top_k);
        let reranked = self.rerank_candidates(&request, fused.clone()).await?;
        let ranked = if request.rerank { &reranked } else { &fused };
        let selected = select_for_budget(ranked, request.token_budget);
        let selected_context = format_selected_context(&selected);

        Ok(RagPipelineResponse {
            project_id: request.project_id,
            query: request.query,
            selected_context,
            diagnostics: RagDiagnostics {
                dense_candidates: dense
                    .iter()
                    .map(|result| candidate_evidence(result, "dense"))
                    .collect(),
                sparse_candidates: sparse
                    .iter()
                    .map(|result| candidate_evidence(result, "sparse"))
                    .collect(),
                fused_candidates: fused
                    .iter()
                    .map(|result| candidate_evidence(result, "fused"))
                    .collect(),
                reranked_candidates: if request.rerank {
                    reranked
                        .iter()
                        .map(|result| candidate_evidence(result, "reranked"))
                        .collect()
                } else {
                    Vec::new()
                },
                selected: selected
                    .iter()
                    .map(|result| candidate_evidence(result, "selected"))
                    .collect(),
            },
        })
    }

    async fn dense_candidates(
        &self,
        request: &RagPipelineRequest,
    ) -> Result<Vec<SearchResult>, RagPipelineError> {
        let vector = self.embedder.embed(&request.query).await?;
        let mut results = Vec::new();
        for collection in ["code_chunks", "doc_chunks", "experience"] {
            match self
                .vector_store
                .search(collection, &vector, search_options(request))
                .await
            {
                Ok(mut hits) => results.append(&mut hits),
                Err(VectorStoreError::Collection(_)) => {}
                Err(error) => return Err(error.into()),
            }
        }
        results.sort_by(|left, right| right.score.total_cmp(&left.score));
        results.truncate(request.top_k);
        Ok(results)
    }

    async fn sparse_candidates(
        &self,
        request: &RagPipelineRequest,
    ) -> Result<Vec<SearchResult>, RagPipelineError> {
        let Some(sparse_embedder) = &self.sparse_embedder else {
            return Ok(Vec::new());
        };
        if !self.vector_store.supports_sparse().await {
            return Ok(Vec::new());
        }
        let sparse = sparse_embedder.embed_query(&request.query).await?;
        let mut results = Vec::new();
        for collection in ["code_chunks", "doc_chunks", "experience"] {
            match self
                .vector_store
                .search_sparse(collection, &sparse, search_options(request))
                .await
            {
                Ok(mut hits) => results.append(&mut hits),
                Err(VectorStoreError::Collection(_)) => {}
                Err(error) => return Err(error.into()),
            }
        }
        results.sort_by(|left, right| right.score.total_cmp(&left.score));
        results.truncate(request.top_k);
        Ok(results)
    }

    async fn rerank_candidates(
        &self,
        request: &RagPipelineRequest,
        candidates: Vec<SearchResult>,
    ) -> Result<Vec<SearchResult>, RagPipelineError> {
        if !request.rerank || candidates.len() < 2 {
            return Ok(candidates);
        }
        let Some(reranker) = &self.reranker else {
            return Ok(candidates);
        };
        let passages = candidates
            .iter()
            .map(|candidate| RerankPassage {
                id: candidate.id.clone(),
                text: text_payload(&candidate.payload).to_string(),
                payload: Some(candidate.payload.clone()),
            })
            .collect::<Vec<_>>();
        let reranked = reranker.rerank(&request.query, &passages).await?;
        Ok(reranked
            .into_iter()
            .map(|result| SearchResult {
                id: result.id,
                score: result.score,
                payload: result.payload.unwrap_or_else(|| {
                    serde_json::json!({
                        "text": result.text,
                    })
                }),
            })
            .collect())
    }

    pub async fn recover_rebuild(
        active_path: &Path,
        staging_path: &Path,
    ) -> Result<CrashSafeRebuildReport, RagPipelineError> {
        let tmp_manifest = staging_path.join("manifest.json.tmp");
        let status = if tmp_manifest.exists() {
            let _ = std::fs::remove_file(tmp_manifest);
            "recovered_active_index"
        } else if staging_path.join("manifest.json").exists() {
            let active_manifest = active_path.join("manifest.json");
            std::fs::create_dir_all(active_path)?;
            std::fs::rename(staging_path.join("manifest.json"), active_manifest)?;
            "promoted_staged_index"
        } else {
            "active_index_unchanged"
        };

        Ok(CrashSafeRebuildReport {
            status: status.into(),
            active_path: active_path.display().to_string(),
            staging_path: staging_path.display().to_string(),
        })
    }
}

fn search_options(request: &RagPipelineRequest) -> SearchOptions {
    SearchOptions {
        limit: request.top_k.max(1),
        filter: Some(serde_json::json!({
            "project_id": {"match": {"value": request.project_id}}
        })),
        score_threshold: None,
    }
}

fn fuse_candidates(
    dense: Vec<SearchResult>,
    sparse: Vec<SearchResult>,
    limit: usize,
) -> Vec<SearchResult> {
    use crate::services::{RankedResult, RetrieverSource};
    let fusion = build_fusion_strategy(FusionStrategyKind::Rrf, 60.0);
    let dense_ranked = dense
        .into_iter()
        .enumerate()
        .map(|(index, result)| RankedResult {
            result,
            source: RetrieverSource::Dense,
            rank: index + 1,
        })
        .collect();
    let sparse_ranked = sparse
        .into_iter()
        .enumerate()
        .map(|(index, result)| RankedResult {
            result,
            source: RetrieverSource::Sparse,
            rank: index + 1,
        })
        .collect();
    let mut fused = fusion.fuse(vec![dense_ranked, sparse_ranked]);
    fused.truncate(limit.max(1));
    fused
}

fn select_for_budget(candidates: &[SearchResult], token_budget: usize) -> Vec<SearchResult> {
    let mut selected = Vec::new();
    let mut used = 0usize;
    for candidate in candidates {
        let estimated_tokens = text_payload(&candidate.payload).chars().count().div_ceil(4);
        if used + estimated_tokens > token_budget.max(1) && !selected.is_empty() {
            break;
        }
        selected.push(candidate.clone());
        used += estimated_tokens;
    }
    selected
}

fn format_selected_context(selected: &[SearchResult]) -> String {
    selected
        .iter()
        .map(|candidate| {
            format!(
                "[{}] {}",
                string_payload(&candidate.payload, "relative_path").unwrap_or("unknown"),
                text_payload(&candidate.payload)
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn candidate_evidence(result: &SearchResult, stage: &str) -> RagCandidateEvidence {
    let text = text_payload(&result.payload);
    RagCandidateEvidence {
        id: result.id.clone(),
        score: result.score,
        stage: stage.into(),
        relative_path: string_payload(&result.payload, "relative_path").map(ToString::to_string),
        language: string_payload(&result.payload, "language").map(ToString::to_string),
        symbol_id: string_payload(&result.payload, "symbol_id").map(ToString::to_string),
        prompt_injection: prompt_injection_severity(&result.payload),
        reason: selection_reason(result, stage),
        text_preview: text.chars().take(240).collect(),
    }
}

fn selection_reason(result: &SearchResult, stage: &str) -> String {
    let source = string_payload(&result.payload, "relative_path").unwrap_or("unknown source");
    let security = prompt_injection_severity(&result.payload)
        .map(|severity| format!("; untrusted document risk: {severity:?}"))
        .unwrap_or_default();
    format!(
        "{stage} candidate from {source} scored {:.4}{security}",
        result.score
    )
}

fn prompt_injection_severity(payload: &serde_json::Value) -> Option<PromptInjectionSeverity> {
    payload
        .get("security_findings")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .find(|finding| {
            finding.get("threat").and_then(serde_json::Value::as_str) == Some("prompt_injection")
        })
        .and_then(|finding| finding.get("severity").and_then(serde_json::Value::as_str))
        .map(|severity| match severity {
            "high" => PromptInjectionSeverity::High,
            "medium" => PromptInjectionSeverity::Medium,
            _ => PromptInjectionSeverity::Low,
        })
}

fn text_payload(payload: &serde_json::Value) -> &str {
    string_payload(payload, "text").unwrap_or_default()
}

fn string_payload<'a>(payload: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    payload.get(key).and_then(serde_json::Value::as_str)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::indexer::ProjectIndexer;
    use crate::services::{
        MockEmbedder, MockSparseEmbedder, RerankPassage, RerankResult, Reranker, RerankerError,
        SearchOptions, SearchResult, SparseVector, SparseVectorPoint, VectorPoint, VectorStore,
        VectorStoreError,
    };
    use std::collections::HashMap;
    use std::sync::Mutex;

    #[derive(Debug, Default)]
    struct PreferPdfReranker;

    #[async_trait::async_trait]
    impl Reranker for PreferPdfReranker {
        async fn rerank(
            &self,
            _query: &str,
            passages: &[RerankPassage],
        ) -> Result<Vec<RerankResult>, RerankerError> {
            let mut results = passages
                .iter()
                .map(|passage| RerankResult {
                    id: passage.id.clone(),
                    score: if passage.text.contains("pdf rerank target") {
                        10.0
                    } else {
                        1.0
                    },
                    text: passage.text.clone(),
                    payload: passage.payload.clone(),
                })
                .collect::<Vec<_>>();
            results.sort_by(|left, right| right.score.total_cmp(&left.score));
            Ok(results)
        }
    }

    #[derive(Debug, Default)]
    struct TestVectorStore {
        dense: Mutex<HashMap<String, Vec<VectorPoint>>>,
        sparse: Mutex<HashMap<String, Vec<SparseVectorPoint>>>,
    }

    #[async_trait::async_trait]
    impl VectorStore for TestVectorStore {
        async fn create_collection(
            &self,
            collection: &str,
            _dim: usize,
        ) -> Result<(), VectorStoreError> {
            self.dense
                .lock()
                .unwrap()
                .entry(collection.into())
                .or_default();
            Ok(())
        }

        async fn delete_collection(&self, collection: &str) -> Result<(), VectorStoreError> {
            self.dense.lock().unwrap().remove(collection);
            self.sparse.lock().unwrap().remove(collection);
            Ok(())
        }

        async fn upsert(
            &self,
            collection: &str,
            points: Vec<VectorPoint>,
        ) -> Result<(), VectorStoreError> {
            self.dense
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
            _vector: &[f32],
            options: SearchOptions,
        ) -> Result<Vec<SearchResult>, VectorStoreError> {
            let mut results = self
                .dense
                .lock()
                .unwrap()
                .get(collection)
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .filter(|point| payload_matches(&point.payload, options.filter.as_ref()))
                .map(|point| SearchResult {
                    id: point.id,
                    score: 1.0,
                    payload: point.payload,
                })
                .collect::<Vec<_>>();
            results.truncate(options.limit);
            Ok(results)
        }

        async fn supports_sparse(&self) -> bool {
            true
        }

        async fn create_sparse_collection(
            &self,
            collection: &str,
            _dim: usize,
        ) -> Result<(), VectorStoreError> {
            self.dense
                .lock()
                .unwrap()
                .entry(collection.into())
                .or_default();
            self.sparse
                .lock()
                .unwrap()
                .entry(collection.into())
                .or_default();
            Ok(())
        }

        async fn upsert_with_sparse(
            &self,
            collection: &str,
            points: Vec<SparseVectorPoint>,
        ) -> Result<(), VectorStoreError> {
            self.sparse
                .lock()
                .unwrap()
                .entry(collection.into())
                .or_default()
                .extend(points.clone());
            self.dense
                .lock()
                .unwrap()
                .entry(collection.into())
                .or_default()
                .extend(points.into_iter().map(|point| VectorPoint {
                    id: point.id,
                    vector: point.vector,
                    payload: point.payload,
                }));
            Ok(())
        }

        async fn search_sparse(
            &self,
            collection: &str,
            _vector: &SparseVector,
            options: SearchOptions,
        ) -> Result<Vec<SearchResult>, VectorStoreError> {
            let mut results = self
                .sparse
                .lock()
                .unwrap()
                .get(collection)
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .filter(|point| payload_matches(&point.payload, options.filter.as_ref()))
                .map(|point| SearchResult {
                    id: point.id,
                    score: 2.0,
                    payload: point.payload,
                })
                .collect::<Vec<_>>();
            results.truncate(options.limit);
            Ok(results)
        }

        async fn delete_by_filter(
            &self,
            collection: &str,
            filter: serde_json::Value,
        ) -> Result<(), VectorStoreError> {
            if let Some(points) = self.dense.lock().unwrap().get_mut(collection) {
                points.retain(|point| !payload_matches(&point.payload, Some(&filter)));
            }
            if let Some(points) = self.sparse.lock().unwrap().get_mut(collection) {
                points.retain(|point| !payload_matches(&point.payload, Some(&filter)));
            }
            Ok(())
        }
    }

    fn payload_matches(payload: &serde_json::Value, filter: Option<&serde_json::Value>) -> bool {
        let Some(expected_project) = filter
            .and_then(|filter| filter.get("project_id"))
            .and_then(|project| project.get("match"))
            .and_then(|match_value| match_value.get("value"))
        else {
            return true;
        };
        payload.get("project_id") == Some(expected_project)
    }

    #[tokio::test]
    async fn rag_pipeline_explains_dense_sparse_fused_reranked_and_selected_context() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir(root.join(".git")).unwrap();
        std::fs::create_dir(root.join("docs")).unwrap();
        std::fs::write(
            root.join("main.rs"),
            "fn retry_policy() { println!(\"RAG_SENTINEL dense code\"); }\n",
        )
        .unwrap();
        std::fs::write(
            root.join("docs/guide.md"),
            "RAG_SENTINEL sparse markdown pdf rerank target selected context",
        )
        .unwrap();

        let embedder: Arc<dyn crate::services::Embedder> = Arc::new(MockEmbedder::new(16));
        let sparse_embedder: Arc<dyn crate::services::SparseEmbedder> =
            Arc::new(MockSparseEmbedder);
        let vector_store: Arc<dyn crate::services::VectorStore> =
            Arc::new(TestVectorStore::default());
        ProjectIndexer::new(embedder.clone(), vector_store.clone())
            .with_sparse_embedder(sparse_embedder.clone())
            .index("proj-1", root)
            .await
            .unwrap();

        let response = super::RagPipeline::new(embedder, vector_store)
            .with_sparse_embedder(sparse_embedder)
            .with_reranker(Arc::new(PreferPdfReranker))
            .search(super::RagPipelineRequest {
                project_id: "proj-1".into(),
                query: "RAG_SENTINEL retry policy".into(),
                top_k: 8,
                token_budget: 128,
                rerank: true,
                explain: true,
            })
            .await
            .unwrap();

        assert!(!response.diagnostics.dense_candidates.is_empty());
        assert!(!response.diagnostics.sparse_candidates.is_empty());
        assert!(!response.diagnostics.fused_candidates.is_empty());
        assert!(!response.diagnostics.reranked_candidates.is_empty());
        assert!(!response.selected_context.is_empty());
        assert!(
            response
                .diagnostics
                .selected
                .iter()
                .all(|candidate| !candidate.reason.is_empty())
        );
    }

    #[tokio::test]
    async fn crash_safe_rebuild_uses_staging_manifest_before_swapping_active_index() {
        let dir = tempfile::tempdir().unwrap();
        let active = dir.path().join("active");
        let staging = dir.path().join("staging");
        std::fs::create_dir(&active).unwrap();
        std::fs::write(active.join("manifest.json"), r#"{"generation":1}"#).unwrap();
        std::fs::create_dir(&staging).unwrap();
        std::fs::write(staging.join("manifest.json.tmp"), r#"{"generation":2}"#).unwrap();

        let report = super::RagPipeline::recover_rebuild(&active, &staging)
            .await
            .unwrap();

        assert_eq!(report.status, "recovered_active_index");
        assert!(active.join("manifest.json").exists());
        assert!(!active.join("manifest.json.tmp").exists());
    }
}
