//! Abstract vector-store contract used by retrieval, memory, and indexing services.
//!
//! Concrete implementations live in `crytex-storage` (in-memory fallback and Qdrant).

use serde_json::Value;
use thiserror::Error;

/// A single point stored in a vector collection.
#[derive(Debug, Clone)]
pub struct VectorPoint {
    pub id: String,
    pub vector: Vec<f32>,
    pub payload: Value,
}

/// A sparse vector for lexical/BM25 retrieval.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseVector {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

/// A single point carrying both a dense and a sparse vector.
#[derive(Debug, Clone)]
pub struct SparseVectorPoint {
    pub id: String,
    pub vector: Vec<f32>,
    pub sparse_vector: SparseVector,
    pub payload: Value,
}

/// One search result returned by a vector collection.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
    pub payload: Value,
}

/// Options controlling a vector search.
#[derive(Debug, Clone, Default)]
pub struct SearchOptions {
    /// Maximum number of results to return.
    pub limit: usize,
    /// Optional payload filter (format is implementation-defined; Qdrant uses its JSON DSL).
    pub filter: Option<Value>,
    /// Minimum similarity score (inclusive). Values below the threshold are dropped.
    pub score_threshold: Option<f32>,
}

/// Errors returned by a vector store.
#[derive(Debug, Error)]
pub enum VectorStoreError {
    #[error("collection error: {0}")]
    Collection(String),
    #[error("upsert error: {0}")]
    Upsert(String),
    #[error("search error: {0}")]
    Search(String),
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("unsupported operation: {0}")]
    Unsupported(String),
}

/// Async vector-store contract.
#[async_trait::async_trait]
pub trait VectorStore: Send + Sync {
    /// Create a collection with the given vector dimension.
    async fn create_collection(&self, collection: &str, dim: usize)
    -> Result<(), VectorStoreError>;

    /// Returns `true` if this store supports sparse vectors.
    async fn supports_sparse(&self) -> bool {
        false
    }

    /// Create a collection that holds both dense and sparse vectors.
    ///
    /// The default implementation returns `Unsupported`. Stores that can host
    /// sparse vectors (e.g. qdrant-edge) must override this method.
    async fn create_sparse_collection(
        &self,
        collection: &str,
        dim: usize,
    ) -> Result<(), VectorStoreError> {
        let _ = (collection, dim);
        Err(VectorStoreError::Unsupported(
            "sparse collections are not supported by this store".into(),
        ))
    }

    /// Delete a collection and all its points.
    async fn delete_collection(&self, collection: &str) -> Result<(), VectorStoreError>;

    /// Upsert points into a collection. Existing points with the same id are overwritten.
    async fn upsert(
        &self,
        collection: &str,
        points: Vec<VectorPoint>,
    ) -> Result<(), VectorStoreError>;

    /// Upsert points carrying both dense and sparse vectors.
    ///
    /// The default implementation returns `Unsupported`.
    async fn upsert_with_sparse(
        &self,
        collection: &str,
        points: Vec<SparseVectorPoint>,
    ) -> Result<(), VectorStoreError> {
        let _ = (collection, points);
        Err(VectorStoreError::Unsupported(
            "sparse upsert is not supported by this store".into(),
        ))
    }

    /// Search a collection by cosine similarity against `vector`.
    async fn search(
        &self,
        collection: &str,
        vector: &[f32],
        options: SearchOptions,
    ) -> Result<Vec<SearchResult>, VectorStoreError>;

    /// Search a collection by sparse vector similarity.
    ///
    /// The default implementation returns `Unsupported`.
    async fn search_sparse(
        &self,
        collection: &str,
        vector: &SparseVector,
        options: SearchOptions,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let _ = (collection, vector, options);
        Err(VectorStoreError::Unsupported(
            "sparse search is not supported by this store".into(),
        ))
    }

    /// Delete points matching `filter` from a collection.
    ///
    /// The filter uses the same JSON DSL as [`SearchOptions::filter`]. The
    /// default implementation returns `Unsupported`; stores that support
    /// filtered deletion must override it.
    async fn delete_by_filter(
        &self,
        collection: &str,
        filter: serde_json::Value,
    ) -> Result<(), VectorStoreError> {
        let _ = (collection, filter);
        Err(VectorStoreError::Unsupported(
            "delete by filter is not supported by this store".into(),
        ))
    }
}
