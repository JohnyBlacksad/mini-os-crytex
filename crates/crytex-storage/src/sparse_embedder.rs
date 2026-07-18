//! Sparse (BM25) embedder backed by qdrant-edge's built-in `EdgeBm25`.

use std::sync::Arc;

use async_trait::async_trait;
use crytex_core::services::{EmbeddingError, SparseEmbedder, SparseVector};
use qdrant_edge::bm25_embed::{EdgeBm25, EdgeBm25Config};

/// A `SparseEmbedder` implementation that uses qdrant-edge's native BM25 model.
///
/// `embed_document` produces TF-weighted sparse vectors suitable for indexing,
/// while `embed_query` produces unit-weighted query vectors. When stored in a
/// qdrant-edge collection configured with `Modifier::Idf`, the resulting dot
/// product yields a standard BM25 score.
#[derive(Debug, Clone)]
pub struct EdgeBm25SparseEmbedder {
    bm25: Arc<EdgeBm25>,
}

impl EdgeBm25SparseEmbedder {
    /// Create an embedder from a raw qdrant-edge BM25 configuration.
    pub fn new(config: EdgeBm25Config) -> Result<Self, EmbeddingError> {
        let bm25 =
            EdgeBm25::new(config).map_err(|e| EmbeddingError::EmbeddingFailed(e.to_string()))?;
        Ok(Self {
            bm25: Arc::new(bm25),
        })
    }

    /// Convenience constructor with an optional language (e.g. `"english"`).
    pub fn with_language(language: Option<String>) -> Result<Self, EmbeddingError> {
        let config = EdgeBm25Config {
            language,
            ..Default::default()
        };
        Self::new(config)
    }
}

#[async_trait]
impl SparseEmbedder for EdgeBm25SparseEmbedder {
    async fn embed_document(&self, text: &str) -> Result<SparseVector, EmbeddingError> {
        let sparse = self.bm25.embed_document(text);
        Ok(SparseVector {
            indices: sparse.indices,
            values: sparse.values,
        })
    }

    async fn embed_query(&self, text: &str) -> Result<SparseVector, EmbeddingError> {
        let sparse = self.bm25.embed_query(text);
        Ok(SparseVector {
            indices: sparse.indices,
            values: sparse.values,
        })
    }
}
