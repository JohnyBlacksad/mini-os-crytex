use std::sync::Arc;

use crate::compress::ContentCompressor;
use crate::token::TokenEstimator;

/// Wraps a [`ContentCompressor`] and enforces a token budget on its output.
///
/// The inner compressor is invoked with the full budget hint. If its output
/// still exceeds the budget, the result is progressively trimmed until it
/// fits: first by dropping middle lines, then as a last resort by character
/// truncation.
#[derive(Clone)]
pub struct BudgetEnforcer {
    inner: Arc<dyn ContentCompressor>,
    estimator: Arc<dyn TokenEstimator>,
}

impl std::fmt::Debug for BudgetEnforcer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BudgetEnforcer")
            .field("inner", &self.inner)
            .finish_non_exhaustive()
    }
}

impl BudgetEnforcer {
    pub fn new(inner: Arc<dyn ContentCompressor>, estimator: Arc<dyn TokenEstimator>) -> Self {
        Self { inner, estimator }
    }
}

impl ContentCompressor for BudgetEnforcer {
    fn compress(&self, content: &str, query: Option<&str>, budget: usize) -> String {
        let output = self.inner.compress(content, query, budget);
        fit_to_budget(&output, &*self.estimator, budget)
    }

    fn compress_with_store(
        &self,
        content: &str,
        query: Option<&str>,
        budget: usize,
        store: Option<&dyn crate::ccr::CcrStore>,
    ) -> Result<String, crate::compress::CompressionError> {
        let output = self
            .inner
            .compress_with_store(content, query, budget, store)?;
        Ok(fit_to_budget(&output, &*self.estimator, budget))
    }
}

fn fits(text: &str, estimator: &dyn TokenEstimator, budget: usize) -> bool {
    estimator
        .estimate_text(text)
        .map(|tokens| tokens <= budget)
        .unwrap_or(false)
}

fn fit_to_budget(text: &str, estimator: &dyn TokenEstimator, budget: usize) -> String {
    if budget == 0 || fits(text, estimator, budget) {
        return text.to_string();
    }

    let lines: Vec<&str> = text.lines().collect();
    if lines.len() >= 4 {
        let mut keep = lines.len() / 2;
        while keep >= 1 {
            let head = lines.iter().take(keep).copied().collect::<Vec<_>>();
            let tail = lines
                .iter()
                .rev()
                .take(keep)
                .rev()
                .copied()
                .collect::<Vec<_>>();
            let omitted = lines.len().saturating_sub(keep * 2);
            let candidate = if omitted > 0 {
                format!(
                    "{}\n... {} lines omitted ...\n{}",
                    head.join("\n"),
                    omitted,
                    tail.join("\n")
                )
            } else {
                head.join("\n")
            };
            if fits(&candidate, estimator, budget) {
                return candidate;
            }
            keep /= 2;
        }
    }

    // Last resort: character-level truncation using a conservative 4 chars/token.
    let target_chars = budget.saturating_mul(4).max(1);
    let mut truncated = String::with_capacity(target_chars + 32);
    for (idx, ch) in text.chars().enumerate() {
        if idx >= target_chars {
            break;
        }
        truncated.push(ch);
    }
    truncated.push_str("\n… [truncated for budget]");
    truncated
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compress::ContentCompressor;
    use crate::token::{CharTokenEstimator, TokenError};

    #[derive(Debug, Clone)]
    struct LongCompressor;

    impl ContentCompressor for LongCompressor {
        fn compress(&self, _content: &str, _query: Option<&str>, _budget: usize) -> String {
            (0..100)
                .map(|i| format!("line {}: a b c d e f g h", i))
                .collect::<Vec<_>>()
                .join("\n")
        }
    }

    #[derive(Debug, Clone)]
    struct FixedEstimator(usize);

    impl TokenEstimator for FixedEstimator {
        fn estimate_text(&self, _text: &str) -> Result<usize, TokenError> {
            Ok(self.0)
        }
    }

    #[test]
    fn passes_through_when_under_budget() {
        let inner = Arc::new(LongCompressor);
        let enforcer = BudgetEnforcer::new(inner, Arc::new(FixedEstimator(1)));
        let out = enforcer.compress("x", None, 1000);
        assert!(out.contains("line 99"));
    }

    #[test]
    fn trims_output_over_budget() {
        let inner = Arc::new(LongCompressor);
        // FixedEstimator returns 1 token per text, so the long output will be
        // considered over any budget < 1? Actually 1 > budget 10? Wait budget=10,
        // estimate=1 <= 10, so it would pass. Use estimator that returns the
        // character-based count: CharTokenEstimator.
        let enforcer = BudgetEnforcer::new(inner, Arc::new(CharTokenEstimator));
        let out = enforcer.compress("x", None, 10);
        assert!(out.len() < LongCompressor.compress("x", None, 10).len());
        assert!(out.contains("lines omitted") || out.contains("truncated for budget"));
    }
}
