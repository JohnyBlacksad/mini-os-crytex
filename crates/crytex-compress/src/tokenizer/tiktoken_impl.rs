#[cfg(feature = "tiktoken")]
use std::sync::{Arc, LazyLock};

use thiserror::Error;
#[cfg(feature = "tiktoken")]
use tiktoken_rs::CoreBPE;

use super::{Backend, Tokenizer};

#[derive(Debug, Error)]
pub enum TiktokenError {
    #[error("unknown encoding for model `{0}`")]
    UnknownEncoding(String),
}

#[cfg(feature = "tiktoken")]
static O200K: LazyLock<Arc<CoreBPE>> =
    LazyLock::new(|| Arc::new(tiktoken_rs::o200k_base().expect("o200k_base init")));
#[cfg(feature = "tiktoken")]
static CL100K: LazyLock<Arc<CoreBPE>> =
    LazyLock::new(|| Arc::new(tiktoken_rs::cl100k_base().expect("cl100k_base init")));
#[cfg(feature = "tiktoken")]
static P50K: LazyLock<Arc<CoreBPE>> =
    LazyLock::new(|| Arc::new(tiktoken_rs::p50k_base().expect("p50k_base init")));
#[cfg(feature = "tiktoken")]
static R50K: LazyLock<Arc<CoreBPE>> =
    LazyLock::new(|| Arc::new(tiktoken_rs::r50k_base().expect("r50k_base init")));

pub struct TiktokenCounter {
    model: String,
    encoding_name: &'static str,
    #[cfg(feature = "tiktoken")]
    bpe: Arc<CoreBPE>,
}

impl std::fmt::Debug for TiktokenCounter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TiktokenCounter")
            .field("model", &self.model)
            .field("encoding", &self.encoding_name)
            .finish()
    }
}

impl TiktokenCounter {
    pub fn for_model(model: &str) -> Result<Self, TiktokenError> {
        let encoding_name = encoding_for(model)?;
        #[cfg(feature = "tiktoken")]
        let bpe = match encoding_name {
            "o200k_base" => O200K.clone(),
            "cl100k_base" => CL100K.clone(),
            "p50k_base" => P50K.clone(),
            "r50k_base" => R50K.clone(),
            _ => return Err(TiktokenError::UnknownEncoding(model.to_string())),
        };
        Ok(Self {
            model: model.to_string(),
            encoding_name,
            #[cfg(feature = "tiktoken")]
            bpe,
        })
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn encoding_name(&self) -> &'static str {
        self.encoding_name
    }
}

impl Tokenizer for TiktokenCounter {
    #[cfg(feature = "tiktoken")]
    fn count_text(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }
        self.bpe.encode_ordinary(text).len()
    }

    #[cfg(not(feature = "tiktoken"))]
    fn count_text(&self, _text: &str) -> usize {
        0
    }

    fn backend(&self) -> Backend {
        Backend::Tiktoken
    }
}

fn encoding_for(model: &str) -> Result<&'static str, TiktokenError> {
    let m = model.to_ascii_lowercase();

    if m.starts_with("gpt-4o") || m.starts_with("o1") || m.starts_with("o3") {
        return Ok("o200k_base");
    }

    if m.starts_with("gpt-4") || m.starts_with("gpt-3.5") || m.starts_with("text-embedding") {
        return Ok("cl100k_base");
    }

    if m.starts_with("code-")
        || m.starts_with("text-davinci-002")
        || m.starts_with("text-davinci-003")
    {
        return Ok("p50k_base");
    }

    if m.starts_with("text-davinci")
        || m.starts_with("davinci")
        || m.starts_with("curie")
        || m.starts_with("babbage")
        || m.starts_with("ada")
    {
        return Ok("r50k_base");
    }

    Err(TiktokenError::UnknownEncoding(model.to_string()))
}
