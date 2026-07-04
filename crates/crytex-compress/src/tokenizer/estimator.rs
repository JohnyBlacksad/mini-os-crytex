use super::{Backend, Tokenizer};

#[derive(Debug, Clone, Copy)]
pub struct EstimatingCounter {
    chars_per_token: f64,
}

impl Default for EstimatingCounter {
    fn default() -> Self {
        Self {
            chars_per_token: 4.0,
        }
    }
}

impl EstimatingCounter {
    pub fn new(chars_per_token: f64) -> Self {
        assert!(
            chars_per_token > 0.0,
            "chars_per_token must be positive, got {chars_per_token}"
        );
        Self { chars_per_token }
    }

    pub fn chars_per_token(&self) -> f64 {
        self.chars_per_token
    }
}

impl Tokenizer for EstimatingCounter {
    fn count_text(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }
        let chars = text.chars().count() as f64;
        let raw = (chars / self.chars_per_token + 0.5) as usize;
        raw.max(1)
    }

    fn backend(&self) -> Backend {
        Backend::Estimation
    }
}
