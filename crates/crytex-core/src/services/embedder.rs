//! Embedding abstraction used by indexing and memory services.

use std::sync::Arc;

use async_trait::async_trait;
use thiserror::Error;

use super::vector_store::SparseVector;

/// Errors that can occur when producing an embedding vector.
#[derive(Debug, Error)]
pub enum EmbeddingError {
    #[error("embedding failed: {0}")]
    EmbeddingFailed(String),
}

/// Something that turns text into a dense vector.
#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError>;

    /// Return the dimensionality of the vectors produced by this embedder.
    async fn dimension(&self) -> Result<usize, EmbeddingError>;
}

#[async_trait]
impl Embedder for Arc<dyn crate::services::InferenceService> {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        self.embed(text)
            .await
            .map_err(|e| EmbeddingError::EmbeddingFailed(e.to_string()))
    }

    async fn dimension(&self) -> Result<usize, EmbeddingError> {
        let vector = self.embed("").await.map_err(|e| EmbeddingError::EmbeddingFailed(e.to_string()))?;
        Ok(vector.len())
    }
}

/// Deterministic mock embedder for tests. The vector encodes the input length
/// so cosine comparisons are stable and independent of any model download.
#[derive(Debug, Clone, Default)]
pub struct MockEmbedder {
    pub dim: usize,
}

impl MockEmbedder {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

/// Something that turns text into a sparse vector (e.g. BM25 or SPLADE).
///
/// The two methods are separate because BM25 applies different weighting to
/// documents (term-frequency) and queries (unit weight).
#[async_trait]
pub trait SparseEmbedder: Send + Sync {
    async fn embed_document(&self, text: &str) -> Result<SparseVector, EmbeddingError>;
    async fn embed_query(&self, text: &str) -> Result<SparseVector, EmbeddingError>;
}

/// Deterministic mock sparse embedder for tests.
#[derive(Debug, Clone, Default)]
pub struct MockSparseEmbedder;

#[async_trait]
impl SparseEmbedder for MockSparseEmbedder {
    async fn embed_document(&self, text: &str) -> Result<SparseVector, EmbeddingError> {
        Ok(Self::embed(text))
    }

    async fn embed_query(&self, text: &str) -> Result<SparseVector, EmbeddingError> {
        Ok(Self::embed(text))
    }
}

impl MockSparseEmbedder {
    fn embed(text: &str) -> SparseVector {
        let tokens: Vec<&str> = text.split_whitespace().collect();
        let mut indices = Vec::with_capacity(tokens.len());
        let mut values = Vec::with_capacity(tokens.len());
        for (i, token) in tokens.iter().enumerate() {
            indices.push(hash_token(token));
            values.push(i as f32 + 1.0);
        }
        SparseVector { indices, values }
    }
}

fn hash_token(token: &str) -> u32 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    token.hash(&mut hasher);
    hasher.finish() as u32
}

#[async_trait]
impl Embedder for MockEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let mut vec = vec![0.0f32; self.dim];
        let len = text.len().min(self.dim);
        for (i, byte) in text.bytes().take(len).enumerate() {
            vec[i] = byte as f32 / 255.0;
        }
        Ok(vec)
    }

    async fn dimension(&self) -> Result<usize, EmbeddingError> {
        Ok(self.dim)
    }
}
