use std::sync::Arc;

use async_trait::async_trait;

use crate::compress::{CompressionError, Compressor};
use crate::message::Message;
use crate::token::TokenEstimator;

/// Configuration for the summarization compressor.
#[derive(Debug, Clone)]
pub struct SummarizeConfig {
    /// Approximate tokens reserved for the final summary message.
    pub summary_budget: usize,
    /// Approximate tokens per chunk when summarizing in pieces.
    pub chunk_budget: usize,
}

impl Default for SummarizeConfig {
    fn default() -> Self {
        Self {
            summary_budget: 256,
            chunk_budget: 1024,
        }
    }
}

/// Summarization backend used to compress text.
#[async_trait]
pub trait Summarizer: Send + Sync {
    /// Summarize the provided text into a shorter form.
    async fn summarize(&self, text: &str, max_tokens: usize) -> Result<String, CompressionError>;
}

/// Compressor that replaces the middle of the conversation with a summary.
#[derive(Clone)]
pub struct SummarizeCompressor {
    estimator: Arc<dyn TokenEstimator>,
    summarizer: Arc<dyn Summarizer>,
    config: SummarizeConfig,
}

impl SummarizeCompressor {
    pub fn new(
        estimator: Arc<dyn TokenEstimator>,
        summarizer: Arc<dyn Summarizer>,
        config: SummarizeConfig,
    ) -> Self {
        Self {
            estimator,
            summarizer,
            config,
        }
    }
}

impl std::fmt::Debug for SummarizeCompressor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SummarizeCompressor")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl Compressor for SummarizeCompressor {
    async fn compress(
        &self,
        messages: &[Message],
        budget: usize,
    ) -> Result<Vec<Message>, CompressionError> {
        if messages.is_empty() {
            return Ok(vec![]);
        }

        let required = self
            .estimator
            .estimate_messages(messages)
            .unwrap_or(usize::MAX);
        if required <= budget {
            return Ok(messages.to_vec());
        }

        // Keep first (system) and last (current turn) messages intact.
        let first = messages.first().cloned();
        let last = messages.last().cloned();

        let middle: Vec<Message> = messages
            .iter()
            .skip(first.as_ref().map_or(0, |_| 1))
            .take(
                messages
                    .len()
                    .saturating_sub(first.as_ref().map_or(0, |_| 1) + 1),
            )
            .cloned()
            .collect();

        if middle.is_empty() {
            // Fall back to truncation if there is no middle to summarize.
            let truncator = crate::compressors::TruncateCompressor::new(self.estimator.clone());
            return truncator.compress(messages, budget).await;
        }

        let middle_text = middle
            .iter()
            .map(|m| format!("{}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n\n");

        let summary = self
            .summarizer
            .summarize(&middle_text, self.config.summary_budget)
            .await?;

        let mut result = Vec::new();
        if let Some(first) = first {
            result.push(first);
        }
        result.push(Message::new(
            "system",
            format!("Summary of earlier conversation:\n{}", summary),
        ));
        if let Some(last) = last {
            result.push(last);
        }

        // If the summary still doesn't fit, truncate.
        let final_tokens = self.estimator.estimate_messages(&result)?;
        if final_tokens > budget {
            let truncator = crate::compressors::TruncateCompressor::new(self.estimator.clone());
            return truncator.compress(&result, budget).await;
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::CharTokenEstimator;

    struct MockSummarizer;

    #[async_trait]
    impl Summarizer for MockSummarizer {
        async fn summarize(
            &self,
            _text: &str,
            _max_tokens: usize,
        ) -> Result<String, CompressionError> {
            Ok("short summary".into())
        }
    }

    #[tokio::test]
    async fn summarizes_middle() {
        let estimator = Arc::new(CharTokenEstimator);
        let compressor = SummarizeCompressor::new(
            estimator,
            Arc::new(MockSummarizer),
            SummarizeConfig::default(),
        );

        let messages = vec![
            Message::system("you are helpful"),
            Message::user("question 1"),
            Message::assistant("answer 1 with lots of detail"),
            Message::user("question 2"),
            Message::assistant("answer 2 with lots of detail"),
            Message::user("current question"),
        ];

        let compressed = compressor.compress(&messages, 50).await.unwrap();
        assert_eq!(compressed.first().unwrap().role, "system");
        assert_eq!(compressed.last().unwrap().content, "current question");
        assert!(compressed.iter().any(|m| m.content.contains("Summary")));
    }
}
