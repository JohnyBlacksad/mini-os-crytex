//! Hybrid retrieval: fusing dense and sparse search results.
//!
//! The module provides rank- and score-based fusion strategies that combine
//! results from multiple retrievers without requiring comparable raw scores.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

pub mod dbsf;
pub mod retriever;
pub mod rrf;

pub use dbsf::{DistributionBasedScoreFusion, ScoreDistribution};
pub use retriever::{HybridRetriever, HybridSearchError};
pub use rrf::ReciprocalRankFusion;

use crate::services::SearchResult;

/// Identifies which retriever produced a ranked list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetrieverSource {
    /// Dense vector (cosine similarity) retriever.
    Dense,
    /// Sparse vector (BM25/SPLADE) retriever.
    Sparse,
}

/// A [`SearchResult`] together with its rank inside one retriever's list.
#[derive(Debug, Clone)]
pub struct RankedResult {
    /// The underlying search result.
    pub result: SearchResult,
    /// Which retriever produced the result.
    pub source: RetrieverSource,
    /// 1-based rank inside the retriever's list.
    pub rank: usize,
}

/// A ranked list produced by a single retriever.
pub type RankedList = Vec<RankedResult>;

/// Strategy for merging ranked result lists into a single ranking.
pub trait FusionStrategy: Send + Sync {
    /// Merge several ranked lists into one ranked list of search results.
    fn fuse(&self, lists: Vec<RankedList>) -> Vec<SearchResult>;
}

/// Selectable fusion algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FusionStrategyKind {
    /// Reciprocal Rank Fusion (default).
    #[default]
    Rrf,
    /// Distribution-Based Score Fusion.
    Dbsf,
}

impl FusionStrategyKind {
    /// String identifier used in logs and configuration.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Rrf => "rrf",
            Self::Dbsf => "dbsf",
        }
    }
}

/// Build a concrete [`FusionStrategy`] from its kind.
pub fn build_fusion_strategy(kind: FusionStrategyKind, rrf_k: f64) -> Arc<dyn FusionStrategy> {
    match kind {
        FusionStrategyKind::Rrf => Arc::new(ReciprocalRankFusion::new(rrf_k)),
        FusionStrategyKind::Dbsf => Arc::new(DistributionBasedScoreFusion::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(id: &str, score: f32) -> SearchResult {
        SearchResult {
            id: id.to_string(),
            score,
            payload: serde_json::Value::Null,
        }
    }

    fn ranked(id: &str, score: f32, source: RetrieverSource, rank: usize) -> RankedResult {
        RankedResult {
            result: result(id, score),
            source,
            rank,
        }
    }

    #[test]
    fn rrf_combines_dense_and_sparse_ranks() {
        // Dense ranking: a > b > c
        let dense = vec![
            ranked("a", 0.9, RetrieverSource::Dense, 1),
            ranked("b", 0.8, RetrieverSource::Dense, 2),
            ranked("c", 0.7, RetrieverSource::Dense, 3),
        ];
        // Sparse ranking: b > d (scores are huge, but RRF ignores magnitude)
        let sparse = vec![
            ranked("b", 100.0, RetrieverSource::Sparse, 1),
            ranked("d", 90.0, RetrieverSource::Sparse, 2),
        ];

        let strategy = ReciprocalRankFusion::default();
        let fused = strategy.fuse(vec![dense, sparse]);
        let ids: Vec<_> = fused.iter().map(|r| r.id.as_str()).collect();

        // b is top in sparse and second in dense -> wins.
        // a is top only in dense -> second.
        // d is second only in sparse -> third.
        // c is third only in dense -> fourth.
        assert_eq!(ids, vec!["b", "a", "d", "c"]);
    }

    #[test]
    fn dbsf_normalizes_incompatible_score_scales() {
        let dense = vec![
            ranked("a", 0.9, RetrieverSource::Dense, 1),
            ranked("b", 0.8, RetrieverSource::Dense, 2),
            ranked("c", 0.7, RetrieverSource::Dense, 3),
        ];
        // Sparse scores are on a completely different scale but have the same ordering.
        let sparse = vec![
            ranked("b", 30.0, RetrieverSource::Sparse, 1),
            ranked("d", 20.0, RetrieverSource::Sparse, 2),
            ranked("e", 10.0, RetrieverSource::Sparse, 3),
        ];

        let strategy = DistributionBasedScoreFusion::new();
        let fused = strategy.fuse(vec![dense, sparse]);
        let ids: Vec<_> = fused.iter().map(|r| r.id.as_str()).collect();

        // b is top in both lists.
        assert_eq!(fused[0].id, "b");
        assert!(ids.contains(&"a"));
        assert!(ids.contains(&"d"));
    }

    #[test]
    fn build_fusion_strategy_selects_rrf() {
        let strategy = build_fusion_strategy(FusionStrategyKind::Rrf, 60.0);
        let list = vec![vec![ranked("x", 1.0, RetrieverSource::Dense, 1)]];
        let fused = strategy.fuse(list);
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].id, "x");
    }

    #[test]
    fn build_fusion_strategy_selects_dbsf() {
        let strategy = build_fusion_strategy(FusionStrategyKind::Dbsf, 60.0);
        let list = vec![vec![ranked("y", 1.0, RetrieverSource::Dense, 1)]];
        let fused = strategy.fuse(list);
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].id, "y");
    }
}
