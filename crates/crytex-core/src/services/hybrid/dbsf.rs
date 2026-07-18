//! Distribution-Based Score Fusion (DBSF).
//!
//! DBSF normalizes each retriever's raw scores using its mean and standard
//! deviation, then sums the normalized scores. This is useful when score
//! magnitudes are meaningful and their distributions are stable.
//!
//! ```text
//! z(source, d) = clamp((score(source, d) - mean(source)) / std(source), -3, 3)
//! fused_score(d) = Σ weight(source) * z(source, d)
//! ```

use std::collections::HashMap;

use super::{FusionStrategy, RankedList, RetrieverSource};
use crate::services::SearchResult;

/// Statistics of a score distribution used by DBSF.
#[derive(Debug, Clone, Copy)]
pub struct ScoreDistribution {
    /// Arithmetic mean of the scores.
    pub mean: f64,
    /// Population standard deviation of the scores.
    pub std: f64,
    /// Minimum observed score.
    pub min: f64,
    /// Maximum observed score.
    pub max: f64,
}

impl ScoreDistribution {
    /// Compute distribution statistics from a slice of scores.
    pub fn from_scores(scores: &[f64]) -> Self {
        if scores.is_empty() {
            return Self {
                mean: 0.0,
                std: 0.0,
                min: 0.0,
                max: 0.0,
            };
        }

        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64;
        let std = variance.sqrt();
        let min = scores.iter().copied().fold(f64::INFINITY, f64::min);
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        Self {
            mean,
            std,
            min,
            max,
        }
    }
}

/// Distribution-Based Score Fusion strategy.
#[derive(Debug, Clone, Default)]
pub struct DistributionBasedScoreFusion {
    /// Optional per-source weights. Missing entries default to `1.0`.
    weights: HashMap<RetrieverSource, f64>,
}

impl DistributionBasedScoreFusion {
    /// Create a new DBSF strategy with unit weights.
    pub fn new() -> Self {
        Self::default()
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

impl FusionStrategy for DistributionBasedScoreFusion {
    fn fuse(&self, lists: Vec<RankedList>) -> Vec<SearchResult> {
        let mut scores: HashMap<String, (SearchResult, f64)> = HashMap::new();

        for list in lists {
            if list.is_empty() {
                continue;
            }

            let source = list[0].source;
            let raw_scores: Vec<f64> = list.iter().map(|r| r.result.score as f64).collect();
            let dist = ScoreDistribution::from_scores(&raw_scores);

            for ranked in list {
                let normalized = if dist.std == 0.0 {
                    0.0
                } else {
                    ((ranked.result.score as f64 - dist.mean) / dist.std).clamp(-3.0, 3.0)
                };
                let contrib = self.weight(source) * normalized;
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
