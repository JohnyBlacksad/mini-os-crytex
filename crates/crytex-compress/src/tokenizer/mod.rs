//! Reference-quality token counting ported from Headroom.
//!
//! Provides a [`Tokenizer`] trait with three backends:
//! - [`TiktokenCounter`] — byte-identical to Python `tiktoken` for OpenAI /
//!   o-series models.
//! - [`HfTokenizer`] — any model with a public `tokenizer.json` on HuggingFace.
//! - [`EstimatingCounter`] — calibrated chars-per-token fallback for Anthropic,
//!   Gemini, Cohere, etc.
//!
//! The [`get_tokenizer`] registry selects the best backend for a model name.

mod estimator;
#[cfg(feature = "tokenizers")]
mod hf_impl;
mod registry;
#[cfg(feature = "tiktoken")]
mod tiktoken_impl;

pub use estimator::EstimatingCounter;
#[cfg(feature = "tokenizers")]
pub use hf_impl::{HfTokenizer, HfTokenizerError};
pub use registry::{
    Backend, clear_hf_registrations, detect_backend, get_tokenizer, register_hf, try_register_hf,
};
#[cfg(feature = "tiktoken")]
pub use tiktoken_impl::{TiktokenCounter, TiktokenError};

use crate::token::{TokenError, TokenEstimator};
use std::sync::Arc;

/// Thread-safe token counter.
pub trait Tokenizer: Send + Sync + std::fmt::Debug {
    /// Number of tokens assigned to `text`.
    fn count_text(&self, text: &str) -> usize;

    /// Which backend produced the count.
    fn backend(&self) -> Backend;
}

/// Adapter that makes any [`Tokenizer`] implement our [`TokenEstimator`] trait.
#[derive(Debug, Clone)]
pub struct TokenizerEstimator {
    inner: Arc<dyn Tokenizer>,
}

impl TokenizerEstimator {
    pub fn new(inner: Arc<dyn Tokenizer>) -> Self {
        Self { inner }
    }

    pub fn for_model(model: &str) -> Self {
        Self::new(Arc::from(registry::get_tokenizer(model)))
    }
}

impl TokenEstimator for TokenizerEstimator {
    fn estimate_text(&self, text: &str) -> Result<usize, TokenError> {
        Ok(self.inner.count_text(text))
    }
}
