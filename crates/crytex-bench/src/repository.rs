use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;
use crytex_core::models::{BenchmarkResult, BenchmarkRun, BenchmarkRunSummary};
use crytex_core::persistence::PersistenceError;

pub use crytex_core::persistence::BenchmarkResultRepository;

/// In-memory implementation of [`BenchmarkResultRepository`] for unit tests.
#[derive(Default)]
pub struct MemoryBenchmarkResultRepository {
    runs: Mutex<HashMap<String, BenchmarkRun>>,
    results: Mutex<HashMap<String, Vec<BenchmarkResult>>>,
}

impl MemoryBenchmarkResultRepository {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl BenchmarkResultRepository for MemoryBenchmarkResultRepository {
    async fn insert_run(&self, run: &BenchmarkRun) -> Result<(), PersistenceError> {
        let mut runs = self
            .runs
            .lock()
            .map_err(|_| PersistenceError::Database("mutex poisoned".into()))?;
        runs.insert(run.summary.id.clone(), run.clone());
        Ok(())
    }

    async fn get_run(&self, id: &str) -> Result<Option<BenchmarkRun>, PersistenceError> {
        let runs = self
            .runs
            .lock()
            .map_err(|_| PersistenceError::Database("mutex poisoned".into()))?;
        Ok(runs.get(id).cloned())
    }

    async fn list_runs(&self, limit: usize) -> Result<Vec<BenchmarkRunSummary>, PersistenceError> {
        let runs = self
            .runs
            .lock()
            .map_err(|_| PersistenceError::Database("mutex poisoned".into()))?;
        let mut summaries: Vec<_> = runs.values().map(|r| r.summary.clone()).collect();
        summaries.sort_by(|a, b| b.id.cmp(&a.id));
        summaries.truncate(limit);
        Ok(summaries)
    }

    async fn insert_result(
        &self,
        _run_id: &str,
        result: &BenchmarkResult,
    ) -> Result<(), PersistenceError> {
        let mut results = self
            .results
            .lock()
            .map_err(|_| PersistenceError::Database("mutex poisoned".into()))?;
        results
            .entry(result.run_id.clone())
            .or_default()
            .push(result.clone());
        Ok(())
    }

    async fn list_results(&self, run_id: &str) -> Result<Vec<BenchmarkResult>, PersistenceError> {
        let results = self
            .results
            .lock()
            .map_err(|_| PersistenceError::Database("mutex poisoned".into()))?;
        Ok(results.get(run_id).cloned().unwrap_or_default())
    }
}
