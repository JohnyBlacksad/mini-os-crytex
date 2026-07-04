//! Reranker abstraction for second-stage retrieval.
//!
//! A reranker takes a query and a set of candidate passages and returns them
//! reordered by relevance. This is intentionally separate from the generation
//! backend interface: most inference backends do not support reranking.

use async_trait::async_trait;
use serde_json::Value;
use thiserror::Error;

/// Errors that can occur while reranking.
#[derive(Debug, Error)]
pub enum RerankerError {
    #[error("reranking failed: {0}")]
    RerankFailed(String),
}

/// A candidate passage to be reranked.
#[derive(Debug, Clone)]
pub struct RerankPassage {
    /// Original identifier of the passage (e.g. vector point id).
    pub id: String,
    /// Text content used by the cross-encoder.
    pub text: String,
    /// Optional payload preserved in the result.
    pub payload: Option<Value>,
}

/// A reranked passage with its relevance score.
#[derive(Debug, Clone)]
pub struct RerankResult {
    pub id: String,
    pub score: f32,
    pub text: String,
    pub payload: Option<Value>,
}

/// Something that scores query-passage relevance.
#[async_trait]
pub trait Reranker: Send + Sync {
    /// Rerank `passages` for `query` and return them sorted by descending score.
    async fn rerank(
        &self,
        query: &str,
        passages: &[RerankPassage],
    ) -> Result<Vec<RerankResult>, RerankerError>;
}
