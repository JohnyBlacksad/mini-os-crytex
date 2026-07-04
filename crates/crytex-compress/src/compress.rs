use async_trait::async_trait;

use crate::message::Message;

/// Errors that can occur during compression.
#[derive(Debug, thiserror::Error)]
pub enum CompressionError {
    #[error("token estimation error: {0}")]
    TokenEstimation(#[from] crate::token::TokenError),
    #[error("CCR store error: {0}")]
    CcrStore(#[from] crate::ccr::CcrStoreError),
    #[error("inference error: {0}")]
    Inference(String),
    #[error("budget too small: required at least {required}, got {budget}")]
    BudgetTooSmall { required: usize, budget: usize },
    #[error("compression failed: {0}")]
    Other(String),
}

/// A strategy that compresses a list of messages to fit a token budget.
#[async_trait]
pub trait Compressor: Send + Sync {
    /// Compress `messages` so the estimated token count is at most `budget`.
    ///
    /// The returned messages should preserve the original order and keep
    /// the most semantically important content.
    async fn compress(
        &self,
        messages: &[Message],
        budget: usize,
    ) -> Result<Vec<Message>, CompressionError>;
}

/// A strategy that compresses a single text blob based on its content type.
///
/// Used by the pipeline for content-aware compression (diffs, logs, search
/// results) before generic message-level compression is applied.
pub trait ContentCompressor: Send + Sync + std::fmt::Debug {
    /// Compress `content`. `query` is an optional user query used for
    /// relevance scoring. `budget` is a target token count.
    fn compress(&self, content: &str, query: Option<&str>, budget: usize) -> String;

    /// Compress with optional CCR storage. The default implementation ignores
    /// the store; compressors that offload content can override this.
    fn compress_with_store(
        &self,
        content: &str,
        query: Option<&str>,
        budget: usize,
        _store: Option<&dyn crate::ccr::CcrStore>,
    ) -> Result<String, CompressionError> {
        Ok(self.compress(content, query, budget))
    }
}
