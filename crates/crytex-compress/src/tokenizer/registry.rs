use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

#[cfg(feature = "tiktoken")]
use super::TiktokenCounter;
use super::{EstimatingCounter, Tokenizer};
#[cfg(feature = "tokenizers")]
use super::{HfTokenizer, HfTokenizerError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Tiktoken,
    HuggingFace,
    Estimation,
}

pub fn detect_backend(model: &str) -> Backend {
    let m = model.to_ascii_lowercase();

    if m.starts_with("gpt-4o")
        || m.starts_with("gpt-4")
        || m.starts_with("gpt-3.5")
        || m.starts_with("o1")
        || m.starts_with("o3")
        || m.starts_with("text-embedding")
        || m.starts_with("text-davinci")
        || m.starts_with("davinci")
        || m.starts_with("curie")
        || m.starts_with("babbage")
        || m.starts_with("ada")
        || m.starts_with("code-")
    {
        return Backend::Tiktoken;
    }

    Backend::Estimation
}

pub fn get_tokenizer(model: &str) -> Box<dyn Tokenizer> {
    if let Some(hf) = lookup_hf(model) {
        return Box::new(hf);
    }
    match detect_backend(model) {
        #[cfg(feature = "tiktoken")]
        Backend::Tiktoken => match TiktokenCounter::for_model(model) {
            Ok(t) => Box::new(t),
            Err(_) => Box::new(default_estimator_for(model)),
        },
        #[cfg(not(feature = "tiktoken"))]
        Backend::Tiktoken => Box::new(default_estimator_for(model)),
        Backend::HuggingFace | Backend::Estimation => Box::new(default_estimator_for(model)),
    }
}

fn default_estimator_for(model: &str) -> EstimatingCounter {
    let m = model.to_ascii_lowercase();
    if m.starts_with("claude-") {
        EstimatingCounter::new(3.5)
    } else if m.starts_with("gemini") || m.starts_with("palm") || m.starts_with("command") {
        EstimatingCounter::new(4.0)
    } else {
        EstimatingCounter::default()
    }
}

fn hf_table() -> &'static RwLock<HashMap<String, HfTokenizer>> {
    static TABLE: OnceLock<RwLock<HashMap<String, HfTokenizer>>> = OnceLock::new();
    TABLE.get_or_init(|| RwLock::new(HashMap::new()))
}

#[cfg(feature = "tokenizers")]
pub fn register_hf(prefix: impl Into<String>, tokenizer: HfTokenizer) {
    let key = prefix.into().to_ascii_lowercase();
    hf_table()
        .write()
        .expect("hf registry poisoned")
        .insert(key, tokenizer);
}

#[cfg(not(feature = "tokenizers"))]
pub fn register_hf(_prefix: impl Into<String>, _tokenizer: HfTokenizer) {}

pub fn clear_hf_registrations() {
    hf_table().write().expect("hf registry poisoned").clear();
}

#[cfg(feature = "tokenizers")]
pub fn try_register_hf(prefix: &str, repo: &str) -> Result<(), HfTokenizerError> {
    let t = HfTokenizer::from_pretrained(repo)?;
    register_hf(prefix, t);
    Ok(())
}

#[cfg(not(feature = "tokenizers"))]
pub fn try_register_hf(_prefix: &str, repo: &str) -> Result<(), HfTokenizerError> {
    Err(HfTokenizerError::Hub {
        repo: repo.to_string(),
        source: Box::new(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "tokenizers feature disabled",
        )),
    })
}

fn lookup_hf(model: &str) -> Option<HfTokenizer> {
    #[cfg(feature = "tokenizers")]
    {
        let m = model.to_ascii_lowercase();
        let table = hf_table().read().expect("hf registry poisoned");
        table
            .iter()
            .filter(|(prefix, _)| m.starts_with(prefix.as_str()))
            .max_by_key(|(prefix, _)| prefix.len())
            .map(|(_, t)| t.clone())
    }
    #[cfg(not(feature = "tokenizers"))]
    {
        let _ = model;
        None
    }
}
