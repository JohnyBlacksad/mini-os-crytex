use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use dashmap::DashMap;

use crate::compress::CompressionError;
use crate::embed::{Embedder, cosine_similarity};

/// Scores how relevant `candidate` text is to `query`.
///
/// Implementations may be lexical (fast, local) or embedding-based (richer but
/// requiring an inference backend).
#[async_trait]
pub trait RelevanceScorer: Send + Sync {
    async fn score(&self, query: &str, candidate: &str) -> Result<f64, CompressionError>;
}

/// Purely lexical scorer using Jaccard similarity over word sets.
///
/// Requires no model or network; always available.
#[derive(Debug, Clone, Default)]
pub struct KeywordRelevanceScorer;

impl KeywordRelevanceScorer {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl RelevanceScorer for KeywordRelevanceScorer {
    async fn score(&self, query: &str, candidate: &str) -> Result<f64, CompressionError> {
        let q = word_set(query);
        let c = word_set(candidate);
        Ok(jaccard(&q, &c))
    }
}

/// Embedding-based scorer.
///
/// Embeds both query and candidate, then returns cosine similarity.
/// Embeddings are cached per text so repeated candidates are cheap.
pub struct EmbeddingRelevanceScorer {
    embedder: Arc<dyn Embedder>,
    cache: DashMap<String, Vec<f32>>,
}

impl EmbeddingRelevanceScorer {
    pub fn new(embedder: Arc<dyn Embedder>) -> Self {
        Self {
            embedder,
            cache: DashMap::new(),
        }
    }

    async fn cached_embed(&self, text: &str) -> Result<Vec<f32>, CompressionError> {
        if let Some(cached) = self.cache.get(text) {
            return Ok(cached.clone());
        }
        let emb = self.embedder.embed(text).await?;
        self.cache.insert(text.to_string(), emb.clone());
        Ok(emb)
    }
}

#[async_trait]
impl RelevanceScorer for EmbeddingRelevanceScorer {
    async fn score(&self, query: &str, candidate: &str) -> Result<f64, CompressionError> {
        let q = self.cached_embed(query).await?;
        let c = self.cached_embed(candidate).await?;
        Ok(cosine_similarity(&q, &c))
    }
}

/// Tries embedding-based scoring first, then falls back to keyword scoring if
/// embedding fails.
pub struct HybridRelevanceScorer {
    embedding: EmbeddingRelevanceScorer,
    fallback: KeywordRelevanceScorer,
}

impl HybridRelevanceScorer {
    pub fn new(embedder: Arc<dyn Embedder>) -> Self {
        Self {
            embedding: EmbeddingRelevanceScorer::new(embedder),
            fallback: KeywordRelevanceScorer::new(),
        }
    }
}

#[async_trait]
impl RelevanceScorer for HybridRelevanceScorer {
    async fn score(&self, query: &str, candidate: &str) -> Result<f64, CompressionError> {
        match self.embedding.score(query, candidate).await {
            Ok(score) => Ok(score),
            Err(_) => self.fallback.score(query, candidate).await,
        }
    }
}

fn word_set(text: &str) -> HashSet<String> {
    text.to_ascii_lowercase()
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|w| w.len() > 2)
        .collect()
}

fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let intersection = a.intersection(b).count();
    let union = a.len() + b.len() - intersection;
    if union == 0 {
        return 0.0;
    }
    intersection as f64 / union as f64
}

/// Compares two `f64` values in descending order (higher values come first).
///
/// Unlike `partial_cmp`, this uses `total_cmp` and therefore never panics on `NaN`.
pub fn cmp_f64_desc(a: f64, b: f64) -> std::cmp::Ordering {
    b.total_cmp(&a)
}

/// Compares two `f64` values in ascending order (lower values come first).
///
/// Unlike `partial_cmp`, this uses `total_cmp` and therefore never panics on `NaN`.
pub fn cmp_f64_asc(a: f64, b: f64) -> std::cmp::Ordering {
    a.total_cmp(&b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn keyword_scores_identical_high() {
        let s = KeywordRelevanceScorer::new();
        let score = s.score("hello world", "hello world").await.unwrap();
        assert!(score > 0.99);
    }

    #[tokio::test]
    async fn keyword_scores_unrelated_low() {
        let s = KeywordRelevanceScorer::new();
        let score = s.score("rust compiler", "weather forecast").await.unwrap();
        assert!(score < 0.1);
    }

    #[test]
    fn cosine_same_vector() {
        let v = vec![1.0_f32, 2.0, 3.0];
        let score = cosine_similarity(&v, &v);
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        let score = cosine_similarity(&a, &b);
        assert!(score.abs() < 1e-6);
    }

    #[test]
    fn cmp_f64_desc_with_nan_does_not_panic() {
        let mut values = [f64::NAN, 1.0, 2.0, f64::NAN, 0.5];
        values.sort_by(|&a, &b| cmp_f64_desc(a, b));
        assert_eq!(values.len(), 5);
        // NaN is treated as the largest value by total_cmp, so in descending order it ends up first.
        assert!(values.first().unwrap().is_nan());
    }

    #[test]
    fn cmp_f64_asc_with_nan_does_not_panic() {
        let mut values = [f64::NAN, 1.0, 2.0, f64::NAN, 0.5];
        values.sort_by(|&a, &b| cmp_f64_asc(a, b));
        assert_eq!(values.len(), 5);
        assert!(values.last().unwrap().is_nan());
    }
}
