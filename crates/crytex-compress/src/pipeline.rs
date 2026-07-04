use std::sync::Arc;

use async_trait::async_trait;

use crate::compress::{CompressionError, Compressor};
use crate::content::ContentType;
use crate::message::Message;
use crate::token::TokenEstimator;

/// A compressor selected by content type.
pub struct ContentRouter {
    default: Arc<dyn Compressor>,
    by_type: std::collections::HashMap<ContentType, Arc<dyn Compressor>>,
}

impl ContentRouter {
    pub fn new(default: Arc<dyn Compressor>) -> Self {
        Self {
            default,
            by_type: std::collections::HashMap::new(),
        }
    }

    pub fn with_compressor(
        mut self,
        content_type: ContentType,
        compressor: Arc<dyn Compressor>,
    ) -> Self {
        self.by_type.insert(content_type, compressor);
        self
    }
}

impl std::fmt::Debug for ContentRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContentRouter")
            .field("by_type", &self.by_type.keys().collect::<Vec<_>>())
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl Compressor for ContentRouter {
    async fn compress(
        &self,
        messages: &[Message],
        budget: usize,
    ) -> Result<Vec<Message>, CompressionError> {
        let content_type = crate::content::detect_messages_content_type(messages);
        let compressor = self.by_type.get(&content_type).unwrap_or(&self.default);
        compressor.compress(messages, budget).await
    }
}

/// Statistics returned by the compression pipeline.
#[derive(Debug, Clone, Default)]
pub struct CompressionStats {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub dropped_messages: usize,
}

/// High-level pipeline with stats.
pub struct CompressionPipeline {
    compressor: Arc<dyn Compressor>,
    estimator: Arc<dyn TokenEstimator>,
}

impl CompressionPipeline {
    /// Create a pipeline using the default [`CharTokenEstimator`].
    pub fn new(compressor: Arc<dyn Compressor>) -> Self {
        Self::with_estimator(compressor, Arc::new(crate::token::CharTokenEstimator))
    }

    /// Create a pipeline with an explicit token estimator.
    pub fn with_estimator(
        compressor: Arc<dyn Compressor>,
        estimator: Arc<dyn TokenEstimator>,
    ) -> Self {
        Self {
            compressor,
            estimator,
        }
    }

    pub async fn run(
        &self,
        messages: &[Message],
        budget: usize,
    ) -> Result<(Vec<Message>, CompressionStats), CompressionError> {
        let input_tokens = self.estimator.estimate_messages(messages)?;
        let output = self.compressor.compress(messages, budget).await?;
        let output_len = output.len();
        let output_tokens = self.estimator.estimate_messages(&output)?;

        Ok((
            output,
            CompressionStats {
                input_tokens,
                output_tokens,
                dropped_messages: messages.len().saturating_sub(output_len),
            },
        ))
    }
}

impl std::fmt::Debug for CompressionPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompressionPipeline")
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compressors::TruncateCompressor;
    use crate::token::{CharTokenEstimator, TokenError};

    #[derive(Debug, Clone)]
    struct FixedEstimator(usize);

    impl TokenEstimator for FixedEstimator {
        fn estimate_text(&self, _text: &str) -> Result<usize, TokenError> {
            Ok(self.0)
        }

        fn estimate_message(
            &self,
            _message: &crate::message::Message,
        ) -> Result<usize, TokenError> {
            Ok(self.0)
        }
    }

    #[tokio::test]
    async fn pipeline_returns_stats() {
        let compressor = Arc::new(TruncateCompressor::new(Arc::new(CharTokenEstimator)));
        let pipeline = CompressionPipeline::new(compressor);

        let messages = vec![
            crate::message::Message::user("a"),
            crate::message::Message::user("b"),
            crate::message::Message::user("c"),
        ];

        let (compressed, stats) = pipeline.run(&messages, 5).await.unwrap();
        assert!(stats.input_tokens >= stats.output_tokens);
        assert!(stats.dropped_messages > 0 || compressed.len() == messages.len());
    }

    #[tokio::test]
    async fn pipeline_uses_token_estimator_for_stats() {
        let compressor = Arc::new(TruncateCompressor::new(Arc::new(CharTokenEstimator)));
        let pipeline =
            CompressionPipeline::with_estimator(compressor, Arc::new(FixedEstimator(100)));

        let messages = vec![
            crate::message::Message::user("a"),
            crate::message::Message::user("b"),
            crate::message::Message::user("c"),
        ];

        let (compressed, stats) = pipeline.run(&messages, 5).await.unwrap();
        assert_eq!(stats.input_tokens, 100 * 3);
        assert_eq!(stats.output_tokens, 100 * compressed.len());
    }
}
