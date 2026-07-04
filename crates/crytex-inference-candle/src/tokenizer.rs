//! Tokenizer support for the Candle LoRA trainer.
//!
//! The default byte-level tokenizer needs no external files and works for any
//! UTF-8 text.  When a Hugging Face `tokenizer.json` is available it can be
//! loaded instead.

use std::path::Path;

/// Errors that can occur while encoding text.
#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error("tokenizer file not found: {0}")]
    NotFound(String),
    #[error("failed to load tokenizer: {0}")]
    Load(String),
    #[error("encoding failed: {0}")]
    Encode(String),
    #[error("decoding failed: {0}")]
    Decode(String),
}

/// A tokenizer that can be used during training and generation.
pub trait Tokenizer: Send + Sync {
    /// Encode a string into token ids.  The returned ids must be usable as
    /// indices into the model vocabulary.
    fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError>;

    /// Decode a sequence of token ids back into a string.
    fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError>;
}

/// Byte-level tokenizer: token id == byte value.  EOS is reserved as id 0.
pub struct ByteTokenizer {
    vocab_size: usize,
}

impl ByteTokenizer {
    pub fn new(vocab_size: usize) -> Self {
        Self { vocab_size }
    }
}

impl Tokenizer for ByteTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        Ok(text
            .as_bytes()
            .iter()
            .map(|b| (*b as u32).min(self.vocab_size as u32 - 1))
            .collect())
    }

    fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        let bytes: Vec<u8> = ids.iter().map(|id| *id as u8).collect();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}

/// Hugging Face tokenizer wrapper.
pub struct HfTokenizer {
    inner: tokenizers::Tokenizer,
}

impl HfTokenizer {
    pub fn from_file(path: &Path) -> Result<Self, TokenizerError> {
        if !path.exists() {
            return Err(TokenizerError::NotFound(path.display().to_string()));
        }
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| TokenizerError::Load(e.to_string()))?;
        Ok(Self { inner })
    }
}

impl Tokenizer for HfTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(|e| TokenizerError::Encode(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        self.inner
            .decode(ids, true)
            .map_err(|e| TokenizerError::Decode(e.to_string()))
    }
}

/// Create the appropriate tokenizer for a training run.
pub fn build_tokenizer(
    vocab_size: usize,
    tokenizer_path: Option<&Path>,
) -> Result<Box<dyn Tokenizer>, TokenizerError> {
    match tokenizer_path {
        Some(path) => Ok(Box::new(HfTokenizer::from_file(path)?)),
        None => Ok(Box::new(ByteTokenizer::new(vocab_size))),
    }
}
