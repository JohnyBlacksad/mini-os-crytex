/// Errors that can occur during token estimation.
#[derive(Debug, thiserror::Error)]
pub enum TokenError {
    #[error("token estimation failed: {0}")]
    EstimationFailed(String),
}

/// Estimates token counts for text or messages.
pub trait TokenEstimator: Send + Sync {
    /// Estimate tokens for a single piece of text.
    fn estimate_text(&self, text: &str) -> Result<usize, TokenError>;

    /// Estimate tokens for a full message (role + content + overhead).
    fn estimate_message(&self, message: &crate::message::Message) -> Result<usize, TokenError> {
        // Conservative overhead for message framing.
        let overhead = 4;
        Ok(self.estimate_text(&message.role)? + self.estimate_text(&message.content)? + overhead)
    }

    /// Estimate tokens for a slice of messages.
    fn estimate_messages(&self, messages: &[crate::message::Message]) -> Result<usize, TokenError> {
        messages.iter().try_fold(0, |acc, m| {
            self.estimate_message(m).map(|count| acc + count)
        })
    }
}

/// Simple character-based estimator: 1 token ≈ 4 characters.
#[derive(Debug, Clone, Default)]
pub struct CharTokenEstimator;

impl TokenEstimator for CharTokenEstimator {
    fn estimate_text(&self, text: &str) -> Result<usize, TokenError> {
        Ok(text.len().div_ceil(4))
    }
}

/// Word-based estimator: 1 token ≈ 0.75 words.
#[derive(Debug, Clone, Default)]
pub struct WordTokenEstimator;

impl TokenEstimator for WordTokenEstimator {
    fn estimate_text(&self, text: &str) -> Result<usize, TokenError> {
        let words = text.split_whitespace().count();
        Ok((words * 4).div_ceil(3))
    }
}

/// Tiktoken-backed estimator (requires `tiktoken` feature).
#[cfg(feature = "tiktoken")]
#[derive(Clone)]
pub struct TiktokenEstimator {
    bpe: std::sync::Arc<tiktoken_rs::CoreBPE>,
}

#[cfg(feature = "tiktoken")]
impl std::fmt::Debug for TiktokenEstimator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TiktokenEstimator").finish_non_exhaustive()
    }
}

#[cfg(feature = "tiktoken")]
impl TiktokenEstimator {
    pub fn cl100k() -> Self {
        Self {
            bpe: std::sync::Arc::new(tiktoken_rs::cl100k_base().unwrap()),
        }
    }
}

#[cfg(feature = "tiktoken")]
impl TokenEstimator for TiktokenEstimator {
    fn estimate_text(&self, text: &str) -> Result<usize, TokenError> {
        Ok(self.bpe.encode_with_special_tokens(text).len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Message;

    #[test]
    fn char_estimator_basic() {
        let est = CharTokenEstimator;
        assert_eq!(est.estimate_text("hello world").unwrap(), 3);
    }

    #[test]
    fn word_estimator_basic() {
        let est = WordTokenEstimator;
        // 2 words -> ~2.67 tokens -> 3
        assert_eq!(est.estimate_text("hello world").unwrap(), 3);
    }

    #[test]
    fn message_estimator_includes_overhead() {
        let est = CharTokenEstimator;
        let msg = Message::user("hello world");
        // role (4 chars -> 1) + content (11 chars -> 3) + overhead 4 = 8
        assert_eq!(est.estimate_message(&msg).unwrap(), 8);
    }
}
