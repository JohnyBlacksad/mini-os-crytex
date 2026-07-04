use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use crate::ccr::CcrStore;
use crate::compress::{CompressionError, Compressor, ContentCompressor};
use crate::compressors::budget::BudgetEnforcer;
use crate::content::{ContentType, detect_content_type};
use crate::message::Message;
use crate::token::TokenEstimator;

/// A compressor that applies content-aware compression to individual messages
/// (diffs, logs, search results) before falling back to a generic compressor
/// for the final token budget.
pub struct SmartCompressor {
    by_type: HashMap<ContentType, Arc<dyn ContentCompressor>>,
    fallback: Arc<dyn Compressor>,
    store: Option<Arc<dyn CcrStore>>,
    estimator: Option<Arc<dyn TokenEstimator>>,
}

impl std::fmt::Debug for SmartCompressor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SmartCompressor")
            .field("by_type", &self.by_type.keys().collect::<Vec<_>>())
            .finish_non_exhaustive()
    }
}

impl SmartCompressor {
    pub fn new(fallback: Arc<dyn Compressor>) -> Self {
        Self {
            by_type: HashMap::new(),
            fallback,
            store: None,
            estimator: None,
        }
    }

    /// Provide a token estimator. When set, every content compressor added via
    /// [`with_compressor`] is wrapped in a [`BudgetEnforcer`] so that per-message
    /// budgets are actually respected.
    pub fn with_token_estimator(mut self, estimator: Arc<dyn TokenEstimator>) -> Self {
        self.estimator = Some(estimator);
        self
    }

    pub fn with_compressor(
        mut self,
        content_type: ContentType,
        compressor: Arc<dyn ContentCompressor>,
    ) -> Self {
        let compressor: Arc<dyn ContentCompressor> = match &self.estimator {
            Some(est) => Arc::new(BudgetEnforcer::new(compressor, est.clone())),
            None => compressor,
        };
        self.by_type.insert(content_type, compressor);
        self
    }

    pub fn with_ccr_store(mut self, store: Arc<dyn CcrStore>) -> Self {
        self.store = Some(store);
        self
    }
}

#[async_trait]
impl Compressor for SmartCompressor {
    async fn compress(
        &self,
        messages: &[Message],
        budget: usize,
    ) -> Result<Vec<Message>, CompressionError> {
        if messages.is_empty() {
            return Ok(Vec::new());
        }

        // Extract a query from the most recent user message.
        let query = messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .map(|m| m.content.as_str());

        let mut compressed: Vec<Message> = messages.to_vec();
        let per_message_budget = budget.saturating_div(messages.len().max(1));

        let store_ref = self.store.as_deref();
        for msg in &mut compressed {
            let ty = detect_content_type(&msg.content);
            if let Some(compressor) = self.by_type.get(&ty) {
                let new_content = compressor.compress_with_store(
                    &msg.content,
                    query,
                    per_message_budget,
                    store_ref,
                )?;
                if new_content.len() < msg.content.len() {
                    msg.content = new_content;
                }
            }
        }

        // Apply the generic fallback compressor to enforce the overall budget.
        self.fallback.compress(&compressed, budget).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compressors::{DiffCompressor, TruncateCompressor};

    #[tokio::test]
    async fn smart_compressor_routes_diff() {
        let fallback = Arc::new(TruncateCompressor::new(Arc::new(
            crate::token::CharTokenEstimator,
        )));
        let smart = SmartCompressor::new(fallback)
            .with_compressor(ContentType::Diff, Arc::new(DiffCompressor::default()));

        let mut diff = String::from("diff --git a/f.txt b/f.txt\n--- a/f.txt\n+++ b/f.txt\n");
        for i in 0..100 {
            diff.push_str(&format!(
                "@@ -{0},5 +{0},5 @@\n ctx1\n ctx2\n-old{i}\n+new{i}\n ctx3\n ctx4\n",
                i + 1
            ));
        }

        let messages = vec![
            Message::system("You are helpful"),
            Message::user("summarize the diff"),
            Message::assistant(diff),
        ];

        let out = smart.compress(&messages, 500).await.unwrap();
        assert!(out.last().unwrap().content.len() < messages.last().unwrap().content.len());
    }
}
