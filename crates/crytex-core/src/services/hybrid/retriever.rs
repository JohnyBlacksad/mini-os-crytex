//! Hybrid retriever that runs dense and sparse searches and fuses the results.

use std::sync::Arc;

use thiserror::Error;

use super::{FusionStrategy, RankedList, RankedResult, RetrieverSource};
use crate::services::{
    Embedder, EmbeddingError, SearchOptions, SearchResult, SparseEmbedder, VectorStore,
    VectorStoreError,
};

/// Errors that can occur during hybrid retrieval.
#[derive(Debug, Error)]
pub enum HybridSearchError {
    #[error("embedding failed: {0}")]
    Embedding(#[from] EmbeddingError),
    #[error("vector store failed: {0}")]
    VectorStore(#[from] VectorStoreError),
}

/// Runs dense and (optionally) sparse searches across collections and fuses
/// the ranked lists through a [`FusionStrategy`].
pub struct HybridRetriever {
    vector_store: Arc<dyn VectorStore>,
    embedder: Arc<dyn Embedder>,
    sparse_embedder: Option<Arc<dyn SparseEmbedder>>,
    fusion: Arc<dyn FusionStrategy>,
}

impl HybridRetriever {
    /// Create a new hybrid retriever.
    pub fn new(
        embedder: Arc<dyn Embedder>,
        vector_store: Arc<dyn VectorStore>,
        sparse_embedder: Option<Arc<dyn SparseEmbedder>>,
        fusion: Arc<dyn FusionStrategy>,
    ) -> Self {
        Self {
            vector_store,
            embedder,
            sparse_embedder,
            fusion,
        }
    }

    /// Search across `collections` and return the top `final_limit` fused results.
    ///
    /// For each collection a dense search is always performed. If a sparse
    /// embedder is attached and the vector store reports sparse support, a
    /// sparse search is also run and the two lists are fused.
    pub async fn search(
        &self,
        query: &str,
        project_id: &str,
        collections: &[&str],
        per_collection_limit: usize,
        final_limit: usize,
    ) -> Result<Vec<SearchResult>, HybridSearchError> {
        let query_vector = self.embedder.embed(query).await?;
        let supports_sparse = self.vector_store.supports_sparse().await;
        let sparse_vector = if supports_sparse {
            if let Some(embedder) = &self.sparse_embedder {
                Some(embedder.embed_query(query).await?)
            } else {
                None
            }
        } else {
            None
        };
        let filter = Some(project_filter(project_id));

        let mut all_lists: Vec<RankedList> = Vec::new();
        for collection in collections {
            let options = SearchOptions {
                limit: per_collection_limit,
                filter: filter.clone(),
                score_threshold: None,
            };

            let dense_results = match self
                .vector_store
                .search(collection, &query_vector, options.clone())
                .await
            {
                Ok(results) => results,
                Err(VectorStoreError::Collection(_)) => Vec::new(),
                Err(error) => return Err(error.into()),
            };
            all_lists.push(to_ranked_list(dense_results, RetrieverSource::Dense));

            if let Some(ref sv) = sparse_vector {
                let sparse_results = match self
                    .vector_store
                    .search_sparse(collection, sv, options)
                    .await
                {
                    Ok(results) => results,
                    Err(VectorStoreError::Collection(_)) => Vec::new(),
                    Err(error) => return Err(error.into()),
                };
                all_lists.push(to_ranked_list(sparse_results, RetrieverSource::Sparse));
            }
        }

        let mut fused = self.fusion.fuse(all_lists);
        fused.truncate(final_limit);
        Ok(fused)
    }
}

fn project_filter(project_id: &str) -> serde_json::Value {
    serde_json::json!({"project_id": {"match": {"value": project_id}}})
}

fn to_ranked_list(results: Vec<SearchResult>, source: RetrieverSource) -> RankedList {
    results
        .into_iter()
        .enumerate()
        .map(|(idx, mut result)| {
            append_retrieval_evidence(&mut result.payload, source, idx + 1, result.score);
            RankedResult {
                result,
                source,
                rank: idx + 1,
            }
        })
        .collect()
}

fn append_retrieval_evidence(
    payload: &mut serde_json::Value,
    source: RetrieverSource,
    rank: usize,
    score: f32,
) {
    let item = serde_json::json!({
        "source": retriever_source_name(source),
        "rank": rank,
        "score": score,
    });
    if let Some(items) = payload
        .get_mut("retrieval_evidence")
        .and_then(serde_json::Value::as_array_mut)
    {
        items.push(item);
    } else {
        payload["retrieval_evidence"] = serde_json::json!([item]);
    }
}

fn retriever_source_name(source: RetrieverSource) -> &'static str {
    match source {
        RetrieverSource::Dense => "dense",
        RetrieverSource::Sparse => "sparse",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::{
        MockEmbedder, MockSparseEmbedder, ReciprocalRankFusion, SparseVector, VectorPoint,
    };
    use std::collections::HashMap;
    use std::sync::Mutex;

    #[derive(Debug, Default)]
    struct StubVectorStore {
        data: Mutex<HashMap<String, Vec<VectorPoint>>>,
        sparse_supported: bool,
    }

    #[async_trait::async_trait]
    impl VectorStore for StubVectorStore {
        async fn create_collection(
            &self,
            collection: &str,
            _dim: usize,
        ) -> Result<(), VectorStoreError> {
            self.data
                .lock()
                .unwrap()
                .entry(collection.into())
                .or_default();
            Ok(())
        }

        async fn delete_collection(&self, collection: &str) -> Result<(), VectorStoreError> {
            self.data.lock().unwrap().remove(collection);
            Ok(())
        }

        async fn upsert(
            &self,
            collection: &str,
            points: Vec<VectorPoint>,
        ) -> Result<(), VectorStoreError> {
            self.data
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
            let cols = self.data.lock().unwrap();
            let points = cols.get(collection).cloned().ok_or_else(|| {
                VectorStoreError::Collection(format!("collection {collection} does not exist"))
            })?;
            let mut results: Vec<SearchResult> = points
                .iter()
                .map(|p| SearchResult {
                    id: p.id.clone(),
                    score: 1.0,
                    payload: p.payload.clone(),
                })
                .collect();
            results.truncate(options.limit);
            Ok(results)
        }

        async fn supports_sparse(&self) -> bool {
            self.sparse_supported
        }

        async fn search_sparse(
            &self,
            collection: &str,
            _vector: &SparseVector,
            options: SearchOptions,
        ) -> Result<Vec<SearchResult>, VectorStoreError> {
            let cols = self.data.lock().unwrap();
            let points = cols.get(collection).cloned().ok_or_else(|| {
                VectorStoreError::Collection(format!("collection {collection} does not exist"))
            })?;
            let mut results: Vec<SearchResult> = points
                .iter()
                .map(|p| SearchResult {
                    id: p.id.clone(),
                    score: 2.0,
                    payload: p.payload.clone(),
                })
                .collect();
            results.truncate(options.limit);
            Ok(results)
        }
    }

    #[derive(Debug)]
    struct RecordingFusion {
        lists: Mutex<Vec<RankedList>>,
        result: Vec<SearchResult>,
    }

    impl FusionStrategy for RecordingFusion {
        fn fuse(&self, lists: Vec<RankedList>) -> Vec<SearchResult> {
            self.lists.lock().unwrap().extend(lists);
            self.result.clone()
        }
    }

    #[tokio::test]
    async fn hybrid_retriever_passes_dense_and_sparse_lists_to_fusion() {
        let store = Arc::new(StubVectorStore {
            sparse_supported: true,
            ..Default::default()
        });
        store
            .upsert(
                "code_chunks",
                vec![VectorPoint {
                    id: "chunk-1".into(),
                    vector: vec![1.0, 0.0],
                    payload: serde_json::json!({"project_id": "proj-1"}),
                }],
            )
            .await
            .unwrap();

        let fusion = Arc::new(RecordingFusion {
            lists: Mutex::new(Vec::new()),
            result: vec![SearchResult {
                id: "chunk-1".into(),
                score: 0.0,
                payload: serde_json::Value::Null,
            }],
        });

        let retriever = HybridRetriever::new(
            Arc::new(MockEmbedder::new(2)),
            store,
            Some(Arc::new(MockSparseEmbedder)),
            fusion.clone(),
        );

        let result = retriever
            .search("query", "proj-1", &["code_chunks"], 5, 5)
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, "chunk-1");

        let lists = fusion.lists.lock().unwrap();
        assert_eq!(lists.len(), 2);
        assert_eq!(lists[0][0].source, RetrieverSource::Dense);
        assert_eq!(lists[1][0].source, RetrieverSource::Sparse);
    }

    #[tokio::test]
    async fn hybrid_retriever_adds_dense_and_sparse_rank_evidence() {
        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new(8));
        let sparse_embedder: Arc<dyn SparseEmbedder> = Arc::new(MockSparseEmbedder);
        let store = Arc::new(StubVectorStore {
            sparse_supported: true,
            ..Default::default()
        });
        let query_vector = embedder.embed("query").await.unwrap();
        store
            .upsert(
                "code_chunks",
                vec![VectorPoint {
                    id: "same".into(),
                    vector: query_vector,
                    payload: serde_json::json!({
                        "project_id": "proj-1",
                        "text": "dense and sparse match",
                    }),
                }],
            )
            .await
            .unwrap();
        let retriever = HybridRetriever::new(
            embedder,
            store,
            Some(sparse_embedder),
            Arc::new(ReciprocalRankFusion::default()),
        );
        let results = retriever
            .search("query", "proj-1", &["code_chunks"], 5, 5)
            .await
            .unwrap();

        let evidence = results[0].payload["retrieval_evidence"]
            .as_array()
            .expect("retrieval evidence should be present");
        assert!(evidence.iter().any(|item| item["source"] == "dense"));
        assert!(evidence.iter().any(|item| item["source"] == "sparse"));
    }

    #[tokio::test]
    async fn hybrid_retriever_skips_sparse_when_store_does_not_support_it() {
        let store = Arc::new(StubVectorStore {
            sparse_supported: false,
            ..Default::default()
        });
        store
            .upsert(
                "code_chunks",
                vec![VectorPoint {
                    id: "chunk-1".into(),
                    vector: vec![1.0, 0.0],
                    payload: serde_json::json!({"project_id": "proj-1"}),
                }],
            )
            .await
            .unwrap();

        let fusion = Arc::new(RecordingFusion {
            lists: Mutex::new(Vec::new()),
            result: Vec::new(),
        });

        let retriever = HybridRetriever::new(
            Arc::new(MockEmbedder::new(2)),
            store,
            Some(Arc::new(MockSparseEmbedder)),
            fusion.clone(),
        );

        retriever
            .search("query", "proj-1", &["code_chunks"], 5, 5)
            .await
            .unwrap();

        let lists = fusion.lists.lock().unwrap();
        assert_eq!(lists.len(), 1);
        assert_eq!(lists[0][0].source, RetrieverSource::Dense);
    }

    #[tokio::test]
    async fn hybrid_retriever_ignores_missing_collections() {
        let store = Arc::new(StubVectorStore {
            sparse_supported: true,
            ..Default::default()
        });
        store
            .upsert(
                "doc_chunks",
                vec![VectorPoint {
                    id: "doc-1".into(),
                    vector: vec![1.0, 0.0],
                    payload: serde_json::json!({
                        "project_id": "proj-1",
                        "text": "RAG_ONLY_SECRET_CONTEXT",
                    }),
                }],
            )
            .await
            .unwrap();

        let retriever = HybridRetriever::new(
            Arc::new(MockEmbedder::new(2)),
            store,
            Some(Arc::new(MockSparseEmbedder)),
            Arc::new(super::super::ReciprocalRankFusion::default()),
        );

        let result = retriever
            .search(
                "payment retry adapter",
                "proj-1",
                &["code_chunks", "doc_chunks", "experience"],
                5,
                5,
            )
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, "doc-1");
    }
}
