use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;

use crate::compress::{CompressionError, Compressor};
use crate::message::Message;
use crate::scoring::RelevanceScorer;
use crate::token::TokenEstimator;

/// Compressor that drops messages until the conversation fits the budget.
///
/// By default it drops oldest messages first while preserving the system
/// message and the most recent message. When a [`RelevanceScorer`] is supplied,
/// it instead drops the least relevant messages (still protecting system and
/// the most recent user message).
#[derive(Clone)]
pub struct TruncateCompressor {
    estimator: Arc<dyn TokenEstimator>,
    scorer: Option<Arc<dyn RelevanceScorer>>,
}

impl TruncateCompressor {
    pub fn new(estimator: Arc<dyn TokenEstimator>) -> Self {
        Self {
            estimator,
            scorer: None,
        }
    }

    /// Attach a relevance scorer. When present, the compressor will prefer to
    /// keep messages that are most relevant to the latest user query.
    pub fn with_relevance_scorer(mut self, scorer: Arc<dyn RelevanceScorer>) -> Self {
        self.scorer = Some(scorer);
        self
    }
}

impl std::fmt::Debug for TruncateCompressor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TruncateCompressor")
            .field("has_scorer", &self.scorer.is_some())
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl Compressor for TruncateCompressor {
    async fn compress(
        &self,
        messages: &[Message],
        budget: usize,
    ) -> Result<Vec<Message>, CompressionError> {
        if messages.is_empty() {
            return Ok(vec![]);
        }

        let system_index = messages.iter().position(|m| m.role == "system");
        let mut result = if let Some(scorer) = &self.scorer {
            if let Some(query) = messages
                .iter()
                .rev()
                .find(|m| m.role == "user")
                .map(|m| m.content.as_str())
            {
                drop_by_relevance(messages, query, scorer.as_ref(), budget, &*self.estimator)
                    .await?
            } else {
                messages.to_vec()
            }
        } else {
            messages.to_vec()
        };

        // If still over budget (or no scorer was used), fall back to dropping
        // oldest messages.
        while self.estimator.estimate_messages(&result)? > budget && result.len() > 1 {
            let drop_index = if system_index.is_some() && result.len() > 2 {
                let system_present = result[0].role == "system";
                if system_present { 1 } else { 0 }
            } else {
                0
            };

            if drop_index >= result.len() - 1 {
                break;
            }
            result.remove(drop_index);
        }

        // If even a single message doesn't fit, truncate its content.
        while self.estimator.estimate_messages(&result)? > budget && !result.is_empty() {
            let last = result.len() - 1;
            let current = &result[last];
            let char_budget = budget.saturating_mul(4);
            if current.content.len() > char_budget {
                let mut truncated = current.clone();
                truncated.content = current.content[..char_budget].to_string();
                result[last] = truncated;
            } else if result.len() > 1 {
                result.remove(last.saturating_sub(1));
            } else {
                break;
            }
        }

        Ok(result)
    }
}

async fn drop_by_relevance(
    messages: &[Message],
    query: &str,
    scorer: &dyn RelevanceScorer,
    budget: usize,
    estimator: &dyn TokenEstimator,
) -> Result<Vec<Message>, CompressionError> {
    let last_index = messages.len().saturating_sub(1);
    let system_index = messages.iter().position(|m| m.role == "system");

    // Score every message once. This is sequential so that a single embedder
    // backend does not get flooded; caching inside the scorer avoids duplicates.
    let mut scored: Vec<(usize, f64)> = Vec::with_capacity(messages.len());
    for (idx, msg) in messages.iter().enumerate() {
        let score = if Some(idx) == system_index || idx == last_index {
            f64::INFINITY
        } else {
            scorer.score(query, &msg.content).await?
        };
        scored.push((idx, score));
    }

    let mut kept: HashSet<usize> = (0..messages.len()).collect();

    while estimator.estimate_messages(
        &messages
            .iter()
            .enumerate()
            .filter(|(i, _)| kept.contains(i))
            .map(|(_, m)| m.clone())
            .collect::<Vec<_>>(),
    )? > budget
    {
        // Find the droppable message with the lowest relevance score.
        let to_drop = scored
            .iter()
            .filter(|(idx, _)| kept.contains(idx))
            .filter(|(idx, _)| Some(*idx) != system_index && *idx != last_index)
            .min_by(|a, b| crate::scoring::cmp_f64_asc(a.1, b.1));

        match to_drop {
            Some((idx, _)) => {
                kept.remove(idx);
            }
            None => break,
        }
    }

    Ok(messages
        .iter()
        .enumerate()
        .filter(|(i, _)| kept.contains(i))
        .map(|(_, m)| m.clone())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Message;
    use crate::scoring::KeywordRelevanceScorer;
    use crate::token::CharTokenEstimator;

    #[tokio::test]
    async fn keeps_recent_and_system() {
        let estimator = Arc::new(CharTokenEstimator);
        let compressor = TruncateCompressor::new(estimator);

        let messages = vec![
            Message::system("you are helpful"),
            Message::user("old question 1"),
            Message::assistant("old answer 1"),
            Message::user("old question 2"),
            Message::user("recent question"),
        ];

        let compressed = compressor.compress(&messages, 20).await.unwrap();
        assert_eq!(compressed.first().unwrap().role, "system");
        assert_eq!(compressed.last().unwrap().content, "recent question");
    }

    #[tokio::test]
    async fn empty_input() {
        let compressor = TruncateCompressor::new(Arc::new(CharTokenEstimator));
        let compressed = compressor.compress(&[], 10).await.unwrap();
        assert!(compressed.is_empty());
    }

    #[tokio::test]
    async fn relevance_scorer_keeps_matching_messages() {
        let estimator = Arc::new(CharTokenEstimator);
        let scorer: Arc<dyn RelevanceScorer> = Arc::new(KeywordRelevanceScorer::new());
        let compressor = TruncateCompressor::new(estimator).with_relevance_scorer(scorer);

        let messages = vec![
            Message::system("you are helpful"),
            Message::user("tell me about rust"),
            Message::assistant("rust is fast"),
            Message::assistant("unrelated paragraph about gardening"),
            Message::user("rust details"),
        ];

        // Tight budget forces dropping the least-relevant middle messages.
        let compressed = compressor.compress(&messages, 28).await.unwrap();
        assert_eq!(compressed.first().unwrap().role, "system");
        assert_eq!(compressed.last().unwrap().content, "rust details");
        // The gardening message is less relevant and should have been dropped.
        assert!(
            compressed
                .iter()
                .all(|m| m.content != "unrelated paragraph about gardening"),
            "least-relevant message should be dropped"
        );
    }
}
