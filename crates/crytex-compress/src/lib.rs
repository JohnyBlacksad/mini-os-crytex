//! Context Compression Layer (CCL) for Crytex.
//!
//! Inspired by Headroom, this crate provides token-budget-aware
//! compression of conversation context. It can detect content types,
//! estimate token counts, and apply strategies such as truncation,
//! summarization, and hybrid compression.

pub mod cache_aligner;
pub mod ccr;
pub mod compress;
pub mod compressors;
pub mod content;
pub mod dedup;
pub mod embed;
pub mod llm;
pub mod message;
pub mod pipeline;
pub mod scoring;
pub mod sizing;
pub mod token;
pub mod token_economy;
pub mod tokenizer;
pub mod tree_sitter_detector;
pub mod unidiff_detector;

pub use ccr::{CcrStore, DiskCcrStore, InMemoryCcrStore, compute_key};
pub use compress::{CompressionError, Compressor};
pub use compressors::{
    hybrid::{ContentAwareCompressor, HybridCompressor, HybridConfig},
    smart_crusher::{SmartCrusher, SmartCrusherConfig},
    summarize::{SummarizeCompressor, SummarizeConfig, Summarizer},
    truncate::TruncateCompressor,
};
pub use content::{ContentType, detect_content_type, detect_messages_content_type};
pub use embed::{Embedder, InferenceEmbedder, cosine_similarity};
pub use llm::LlmSummarizer;
pub use message::Message;
pub use pipeline::{CompressionPipeline, CompressionStats, ContentRouter};
pub use scoring::{
    EmbeddingRelevanceScorer, HybridRelevanceScorer, KeywordRelevanceScorer, RelevanceScorer,
};
pub use sizing::{SizingBias, optimal_k};
pub use token::{CharTokenEstimator, TokenError, TokenEstimator, WordTokenEstimator};
pub use token_economy::{
    ArtifactKind, ArtifactOffload, CompressionQualityBenchmark, CompressionQualityReport,
    ModelTokenProfile, SharedContext, SharedContextEntry, SharedContextStats,
    TokenBudgetAllocation, TokenBudgetPlanner, TokenEconomyEngine, TokenEconomyMetrics,
    TokenEconomyReport, TokenEconomyRequest,
};
pub use tokenizer::{Backend, Tokenizer, TokenizerEstimator, get_tokenizer};
pub use unidiff_detector::{detect_diff, is_diff};

#[cfg(feature = "tiktoken")]
pub use token::TiktokenEstimator;
