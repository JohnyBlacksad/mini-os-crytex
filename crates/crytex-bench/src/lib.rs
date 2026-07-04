#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

pub mod ab_test;
pub mod error;
pub mod golden_set;
pub mod harness;
pub mod models;
pub mod repository;
pub mod runner;
pub mod scorer;

pub use ab_test::{ABTest, ABTestReport, ABWinner};
pub use error::BenchError;
pub use golden_set::{GoldenSet, GoldenSetLoader};
pub use harness::{BenchmarkHarness, BenchmarkRunRequest, DefaultBenchmarkHarness};
pub use models::{
    BenchmarkCase, BenchmarkResult, BenchmarkRun, BenchmarkRunSummary, BenchmarkVariant,
};
pub use repository::{BenchmarkResultRepository, MemoryBenchmarkResultRepository};
pub use runner::{
    AgentBenchmarkRunner, BenchmarkRunOutput, BenchmarkRunner, WorkflowBenchmarkRunner,
};
pub use scorer::{
    CompositeScorer, ExactMatchScorer, JsonSchemaScorer, LlmJudgeScorer, SandboxTestScorer, Score,
    Scorer,
};
