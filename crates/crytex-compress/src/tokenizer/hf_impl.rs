#[cfg(feature = "tokenizers")]
use std::path::Path;
use std::sync::Arc;

use thiserror::Error;
#[cfg(feature = "tokenizers")]
use tokenizers::Tokenizer as HfInner;

use super::{Backend, Tokenizer};

#[derive(Debug, Error)]
pub enum HfTokenizerError {
    #[error("failed to load tokenizer for `{name}`: {source}")]
    Load {
        name: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("failed to download `{repo}` from HuggingFace Hub: {source}")]
    Hub {
        repo: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

#[derive(Clone)]
pub struct HfTokenizer {
    name: String,
    #[cfg(feature = "tokenizers")]
    inner: Arc<HfInner>,
}

impl std::fmt::Debug for HfTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HfTokenizer")
            .field("name", &self.name)
            .finish()
    }
}

impl HfTokenizer {
    #[cfg(feature = "tokenizers")]
    pub fn from_bytes(name: impl Into<String>, bytes: &[u8]) -> Result<Self, HfTokenizerError> {
        let name = name.into();
        let inner = HfInner::from_bytes(bytes).map_err(|e| HfTokenizerError::Load {
            name: name.clone(),
            source: e,
        })?;
        Ok(Self {
            name,
            inner: Arc::new(inner),
        })
    }

    #[cfg(not(feature = "tokenizers"))]
    pub fn from_bytes(name: impl Into<String>, _bytes: &[u8]) -> Result<Self, HfTokenizerError> {
        Ok(Self { name: name.into() })
    }

    #[cfg(feature = "tokenizers")]
    pub fn from_file(
        name: impl Into<String>,
        path: impl AsRef<Path>,
    ) -> Result<Self, HfTokenizerError> {
        let name = name.into();
        let inner = HfInner::from_file(path.as_ref()).map_err(|e| HfTokenizerError::Load {
            name: name.clone(),
            source: e,
        })?;
        Ok(Self {
            name,
            inner: Arc::new(inner),
        })
    }

    #[cfg(not(feature = "tokenizers"))]
    pub fn from_file(
        name: impl Into<String>,
        _path: impl AsRef<Path>,
    ) -> Result<Self, HfTokenizerError> {
        Ok(Self { name: name.into() })
    }

    #[cfg(feature = "tokenizers")]
    pub fn from_pretrained(repo: &str) -> Result<Self, HfTokenizerError> {
        let api = hf_hub::api::sync::Api::new().map_err(|e| HfTokenizerError::Hub {
            repo: repo.to_string(),
            source: e.into(),
        })?;
        let path = api
            .model(repo.to_string())
            .get("tokenizer.json")
            .map_err(|e| HfTokenizerError::Hub {
                repo: repo.to_string(),
                source: e.into(),
            })?;
        Self::from_file(repo, path)
    }

    #[cfg(not(feature = "tokenizers"))]
    pub fn from_pretrained(repo: &str) -> Result<Self, HfTokenizerError> {
        Err(HfTokenizerError::Hub {
            repo: repo.to_string(),
            source: Box::new(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "tokenizers feature disabled",
            )),
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

impl Tokenizer for HfTokenizer {
    #[cfg(feature = "tokenizers")]
    fn count_text(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }
        match self.inner.encode(text, false) {
            Ok(enc) => enc.len(),
            Err(_) => 0,
        }
    }

    #[cfg(not(feature = "tokenizers"))]
    fn count_text(&self, _text: &str) -> usize {
        0
    }

    fn backend(&self) -> Backend {
        Backend::HuggingFace
    }
}
