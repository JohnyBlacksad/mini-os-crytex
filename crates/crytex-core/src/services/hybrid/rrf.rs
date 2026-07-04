//! Reciprocal Rank Fusion (RRF).
//!
//! RRF merges ranked lists by summing reciprocal rank contributions:
//!
//! ```text
//! score(d) = Σ weight(source) / (k + rank_source(d))
//! ```
//!
//! It ignores raw scores, which makes it robust when combining retrievers
//! whose scores live on incompatible scales (e.g. cosine similarity and BM25).

use std::collections::HashMap;

use super::{FusionStrategy, RankedList, RetrieverSource};
use crate::services::SearchResult;

/// Reciprocal Rank Fusion strategy.
#[derive(Debug, Clone)]
pub struct ReciprocalRankFusion {
    /// Smoothing constant. The original RRF paper uses `k = 60`.
    k: f64,
    /// Optional per-source weights. Missing entries default to `1.0`.
    weights: HashMap<RetrieverSource, f64>,
}

impl ReciprocalRankFusion {
    /// Create a new RRF strategy with the given smoothing constant.
    pub fn new(k: f64) -> Self {
        Self {
            k,
            weights: HashMap::new(),
        }
    }

    /// Set a weight for a retriever source.
    pub fn with_weight(mut self, source: RetrieverSource, weight: f64) -> Self {
        self.weights.insert(source, weight);
        self
    }

    fn weight(&self, source: RetrieverSource) -> f64 {
        self.weights.get(&source).copied().unwrap_or(1.0)
    }
}

impl Default for ReciprocalRankFusion {
    fn default() -> Self {
        Self::new(60.0)
    }
}

impl FusionStrategy for ReciprocalRankFusion {
    fn fuse(&self, lists: Vec<RankedList>) -> Vec<SearchResult> {
        let mut scores: HashMap<String, (SearchResult, f64)> = HashMap::new();

        for list in lists {
            for ranked in list {
                let contrib = self.weight(ranked.source) / (self.k + ranked.rank as f64);
                let entry = scores
                    .entry(ranked.result.id.clone())
                    .or_insert_with(|| (ranked.result.clone(), 0.0));
                entry.1 += contrib;
            }
        }

        let mut results: Vec<SearchResult> = scores
            .into_values()
            .map(|(mut result, score)| {
                result.score = score as f32;
                result
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }
}
