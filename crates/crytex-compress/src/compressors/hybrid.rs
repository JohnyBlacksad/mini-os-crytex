use std::sync::Arc;

use async_trait::async_trait;

use crate::compress::{CompressionError, Compressor};
use crate::content::ContentType;
use crate::message::Message;
use crate::token::TokenEstimator;

/// Configuration for the hybrid compressor.
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// Number of recent messages to always keep verbatim.
    pub keep_recent: usize,
    /// Budget (in tokens) allocated to the verbatim recent zone.
    pub recent_budget: usize,
    /// Budget allocated to summarized older messages.
    pub summary_budget: usize,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            keep_recent: 4,
            recent_budget: 2048,
            summary_budget: 512,
        }
    }
}

/// Hybrid compressor: keeps recent messages verbatim, summarizes older ones.
#[derive(Clone)]
pub struct HybridCompressor {
    estimator: Arc<dyn TokenEstimator>,
    summarizer: Arc<dyn crate::compressors::summarize::Summarizer>,
    config: HybridConfig,
}

impl HybridCompressor {
    pub fn new(
        estimator: Arc<dyn TokenEstimator>,
        summarizer: Arc<dyn crate::compressors::summarize::Summarizer>,
        config: HybridConfig,
    ) -> Self {
        Self {
            estimator,
            summarizer,
            config,
        }
    }
}

impl std::fmt::Debug for HybridCompressor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HybridCompressor")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl Compressor for HybridCompressor {
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

        // Find system message.
        let system = messages.iter().find(|m| m.role == "system").cloned();

        // Keep the most recent messages verbatim.
        let keep_count = self.config.keep_recent.min(messages.len());
        let recent_start = messages.len().saturating_sub(keep_count);
        let recent = messages[recent_start..].to_vec();

        // Older messages to summarize.
        let older_start = if system.is_some() { 1 } else { 0 };
        let older_end = recent_start.max(older_start);
        let older = &messages[older_start..older_end];

        let mut result = Vec::new();
        if let Some(system) = system {
            result.push(system);
        }

        if !older.is_empty() {
            let older_text = older
                .iter()
                .map(|m| format!("{}: {}", m.role, m.content))
                .collect::<Vec<_>>()
                .join("\n\n");
            let summary = self
                .summarizer
                .summarize(&older_text, self.config.summary_budget)
                .await?;
            result.push(Message::new(
                "system",
                format!("Summary of earlier conversation:\n{}", summary),
            ));
        }

        result.extend(recent);

        // Final safety truncation.
        let final_tokens = self.estimator.estimate_messages(&result)?;
        if final_tokens > budget {
            let truncator = crate::compressors::TruncateCompressor::new(self.estimator.clone());
            return truncator.compress(&result, budget).await;
        }

        Ok(result)
    }
}

/// Content-aware compressor that selects a strategy based on detected type.
#[derive(Clone)]
pub struct ContentAwareCompressor {
    estimator: Arc<dyn TokenEstimator>,
    summarizer: Arc<dyn crate::compressors::summarize::Summarizer>,
}

impl ContentAwareCompressor {
    pub fn new(
        estimator: Arc<dyn TokenEstimator>,
        summarizer: Arc<dyn crate::compressors::summarize::Summarizer>,
    ) -> Self {
        Self {
            estimator,
            summarizer,
        }
    }
}

impl std::fmt::Debug for ContentAwareCompressor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContentAwareCompressor")
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl Compressor for ContentAwareCompressor {
    async fn compress(
        &self,
        messages: &[Message],
        budget: usize,
    ) -> Result<Vec<Message>, CompressionError> {
        let content_type = crate::content::detect_messages_content_type(messages);

        match content_type {
            // Logs and search results compress well via summarization.
            ContentType::Log | ContentType::SearchResults | ContentType::PlainText => {
                HybridCompressor::new(
                    self.estimator.clone(),
                    self.summarizer.clone(),
                    HybridConfig::default(),
                )
                .compress(messages, budget)
                .await
            }
            // Structured data: summarize middle, keep recent.
            ContentType::Json | ContentType::SourceCode | ContentType::Diff => {
                HybridCompressor::new(
                    self.estimator.clone(),
                    self.summarizer.clone(),
                    HybridConfig {
                        keep_recent: 6,
                        recent_budget: 3072,
                        summary_budget: 256,
                    },
                )
                .compress(messages, budget)
                .await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::CharTokenEstimator;

    struct MockSummarizer;

    #[async_trait]
    impl crate::compressors::summarize::Summarizer for MockSummarizer {
        async fn summarize(
            &self,
            text: &str,
            _max_tokens: usize,
        ) -> Result<String, CompressionError> {
            Ok(format!("summary of {} chars", text.len()))
        }
    }

    #[tokio::test]
    async fn hybrid_keeps_recent() {
        let estimator = Arc::new(CharTokenEstimator);
        let compressor =
            HybridCompressor::new(estimator, Arc::new(MockSummarizer), HybridConfig::default());

        let messages = vec![
            Message::system("you are helpful"),
            Message::user("old question number one with lots of text that consumes tokens"),
            Message::assistant("answer number one with lots of text that consumes tokens"),
            Message::user("old question number two with lots of text that consumes tokens"),
            Message::assistant("answer number two with lots of text that consumes tokens"),
            Message::user("recent question"),
            Message::assistant("recent answer"),
        ];

        let compressed = compressor.compress(&messages, 95).await.unwrap();
        assert_eq!(compressed.first().unwrap().role, "system");
        assert_eq!(compressed.last().unwrap().content, "recent answer");
        assert!(compressed.iter().any(|m| m.content.contains("Summary")));
    }

    #[tokio::test]
    async fn content_aware_selects_compressor() {
        let estimator = Arc::new(CharTokenEstimator);
        let compressor = ContentAwareCompressor::new(estimator, Arc::new(MockSummarizer));

        let messages = vec![
            Message::system("you are helpful"),
            Message::user("some plain text that is long enough to trigger detection logic"),
            Message::assistant("another long response that goes on and on and on"),
        ];

        let compressed = compressor.compress(&messages, 100).await.unwrap();
        assert!(!compressed.is_empty());
    }
}
