use std::sync::Arc;

use async_trait::async_trait;

use crate::compress::CompressionError;

/// Abstraction over any source of text embeddings.
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Embed `text` into a dense vector.
    async fn embed(&self, text: &str) -> Result<Vec<f32>, CompressionError>;
}

/// Adapter from `crytex-inference`'s `InferenceManager` to the local `Embedder`
/// trait so the compression layer can request embeddings without knowing backend
/// details.
pub struct InferenceEmbedder {
    inner: Arc<dyn crytex_inference::InferenceManager>,
}

impl InferenceEmbedder {
    pub fn new(inner: Arc<dyn crytex_inference::InferenceManager>) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl Embedder for InferenceEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, CompressionError> {
        self.inner
            .embed(text)
            .await
            .map_err(|e| CompressionError::Inference(e.to_string()))
    }
}

/// Cosine similarity between two equal-length vectors.
///
/// Returns `0.0` for empty or mismatched vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let xf = x as f64;
        let yf = y as f64;
        dot += xf * yf;
        norm_a += xf * xf;
        norm_b += yf * yf;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}
