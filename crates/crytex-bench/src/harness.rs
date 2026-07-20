use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use crytex_core::bus::Event;
use crytex_core::services::EventService;
use futures::future::join_all;
use serde_json::Value;
use tokio::sync::Semaphore;
use ulid::Ulid;

use crate::error::BenchError;
use crate::golden_set::GoldenSet;
use crate::models::{BenchmarkResult, BenchmarkRun, BenchmarkRunSummary, BenchmarkVariant};
use crate::repository::BenchmarkResultRepository;
use crate::runner::BenchmarkRunner;
use crate::scorer::Scorer;

/// Request to execute a full benchmark run.
#[derive(Clone)]
pub struct BenchmarkRunRequest {
    pub name: String,
    pub golden_set_path: std::path::PathBuf,
    pub variant: BenchmarkVariant,
    pub scorer: Arc<dyn Scorer>,
    pub runner: Arc<dyn BenchmarkRunner>,
    pub max_concurrency: usize,
    pub project_id: Option<String>,
}

/// Orchestrates benchmark runs over golden sets.
#[async_trait]
pub trait BenchmarkHarness: Send + Sync {
    async fn run(&self, request: BenchmarkRunRequest) -> Result<BenchmarkRun, BenchError>;
}

/// Default harness implementation.
#[derive(Clone)]
pub struct DefaultBenchmarkHarness {
    repo: Arc<dyn BenchmarkResultRepository>,
    event_service: Arc<dyn EventService>,
}

impl DefaultBenchmarkHarness {
    pub fn new(
        repo: Arc<dyn BenchmarkResultRepository>,
        event_service: Arc<dyn EventService>,
    ) -> Self {
        Self {
            repo,
            event_service,
        }
    }
}

#[async_trait]
impl BenchmarkHarness for DefaultBenchmarkHarness {
    async fn run(&self, request: BenchmarkRunRequest) -> Result<BenchmarkRun, BenchError> {
        let started_at = Utc::now();
        let cases = GoldenSet::load_validated(&request.golden_set_path).await?;
        if cases.is_empty() {
            return Err(BenchError::Harness("golden set contains no cases".into()));
        }

        let run_id = Ulid::new().to_string();
        let max_concurrency = request.max_concurrency.max(1);
        let semaphore = Arc::new(Semaphore::new(max_concurrency));
        let cases = Arc::new(cases);

        let mut handles = Vec::with_capacity(cases.len());
        for case in cases.iter() {
            let permit =
                semaphore.clone().acquire_owned().await.map_err(|e| {
                    BenchError::Harness(format!("concurrency semaphore closed: {e}"))
                })?;
            let runner = request.runner.clone();
            let variant = request.variant.clone();
            let case = case.clone();
            let handle = tokio::spawn(async move {
                let _permit = permit;
                runner.run(&case, &variant).await
            });
            handles.push(handle);
        }

        let outputs = join_all(handles)
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| BenchError::Harness(format!("benchmark task join failed: {e}")))?;

        let mut results = Vec::with_capacity(outputs.len());
        let mut pass_count = 0usize;
        let mut total_latency_ms = 0u64;
        let mut total_tokens = 0usize;

        for (output, case) in outputs.into_iter().zip(cases.iter()) {
            let output = output?;
            let score = request.scorer.score(case, &output.result).await?;
            if score.passed {
                pass_count += 1;
            }
            total_latency_ms += output.latency_ms;
            if let Some(usage) = &output.token_usage {
                total_tokens += usage.total_tokens;
            }

            let result = BenchmarkResult {
                id: Ulid::new().to_string(),
                run_id: run_id.clone(),
                case_id: case.id.clone(),
                case_input: case.input.clone(),
                expected: case.expected.clone(),
                actual: output.result,
                passed: score.passed,
                score_value: score.value,
                latency_ms: output.latency_ms,
                token_usage: output.token_usage,
                explanation: score.explanation,
                metadata: score.metadata,
            };
            results.push(result);
        }

        let total_cases = cases.len();
        let fail_count = total_cases - pass_count;
        let pass_rate = pass_count as f64 / total_cases as f64;
        let mean_latency_ms = total_latency_ms as f64 / total_cases as f64;

        let summary = BenchmarkRunSummary {
            id: run_id.clone(),
            name: request.name,
            golden_set_path: request.golden_set_path,
            variant_name: request.variant.name.clone(),
            pass_count,
            fail_count,
            total_cases,
            pass_rate,
            mean_latency_ms,
            total_tokens,
        };

        let run = BenchmarkRun {
            summary,
            project_id: request.project_id,
            variant: request.variant,
            scorer_kind: std::any::type_name_of_val(request.scorer.as_ref()).to_string(),
            started_at,
            finished_at: Some(Utc::now()),
            results,
            metadata: Value::Object(serde_json::Map::new()),
        };

        self.repo.insert_run(&run).await?;
        for result in &run.results {
            self.repo.insert_result(&run.summary.id, result).await?;
        }

        self.event_service.publish(Event::BenchmarkRunCompleted {
            run_id,
            name: run.summary.name.clone(),
            pass_rate,
        });

        Ok(run)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::BenchmarkCase;
    use crate::repository::MemoryBenchmarkResultRepository;
    use crate::runner::BenchmarkRunOutput;
    use crate::scorer::ExactMatchScorer;
    use async_trait::async_trait;
    use crytex_core::bus::EventBus;
    use crytex_core::models::{BenchmarkResult, BenchmarkRun, BenchmarkRunSummary};
    use crytex_core::persistence::PersistenceError;
    use crytex_core::services::EventServiceImpl;
    use std::collections::HashMap;
    use std::sync::Mutex;

    struct DummyRunner;

    struct RunFirstRepository {
        runs: Mutex<HashMap<String, BenchmarkRun>>,
        results: Mutex<HashMap<String, Vec<BenchmarkResult>>>,
    }

    impl RunFirstRepository {
        fn new() -> Self {
            Self {
                runs: Mutex::new(HashMap::new()),
                results: Mutex::new(HashMap::new()),
            }
        }
    }

    #[async_trait]
    impl BenchmarkResultRepository for RunFirstRepository {
        async fn insert_run(&self, run: &BenchmarkRun) -> Result<(), PersistenceError> {
            self.runs
                .lock()
                .map_err(|error| PersistenceError::Database(error.to_string()))?
                .insert(run.summary.id.clone(), run.clone());
            Ok(())
        }

        async fn get_run(&self, id: &str) -> Result<Option<BenchmarkRun>, PersistenceError> {
            Ok(self
                .runs
                .lock()
                .map_err(|error| PersistenceError::Database(error.to_string()))?
                .get(id)
                .cloned())
        }

        async fn list_runs(
            &self,
            _limit: usize,
        ) -> Result<Vec<BenchmarkRunSummary>, PersistenceError> {
            Ok(vec![])
        }

        async fn insert_result(
            &self,
            run_id: &str,
            result: &BenchmarkResult,
        ) -> Result<(), PersistenceError> {
            let has_run = self
                .runs
                .lock()
                .map_err(|error| PersistenceError::Database(error.to_string()))?
                .contains_key(run_id);
            if !has_run {
                return Err(PersistenceError::Database(format!("missing run {run_id}")));
            }
            self.results
                .lock()
                .map_err(|error| PersistenceError::Database(error.to_string()))?
                .entry(run_id.to_string())
                .or_default()
                .push(result.clone());
            Ok(())
        }

        async fn list_results(
            &self,
            run_id: &str,
        ) -> Result<Vec<BenchmarkResult>, PersistenceError> {
            Ok(self
                .results
                .lock()
                .map_err(|error| PersistenceError::Database(error.to_string()))?
                .get(run_id)
                .cloned()
                .unwrap_or_default())
        }
    }

    #[async_trait]
    impl BenchmarkRunner for DummyRunner {
        async fn run(
            &self,
            case: &BenchmarkCase,
            _variant: &BenchmarkVariant,
        ) -> Result<BenchmarkRunOutput, BenchError> {
            Ok(BenchmarkRunOutput {
                task_id: None,
                result: case.input.clone(),
                latency_ms: 10,
                token_usage: None,
            })
        }
    }

    #[tokio::test]
    async fn harness_runs_and_persists_results() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gs.jsonl");
        let data = r#"{"id":"a","input":{"answer":"compute add forty two"},"expected":{"answer":"compute add forty two"}}
{"id":"b","input":{"answer":"compute subtract one"},"expected":{"answer":"compute subtract two"}}"#;
        tokio::fs::write(&path, data).await.unwrap();

        let repo: Arc<dyn BenchmarkResultRepository> =
            Arc::new(MemoryBenchmarkResultRepository::new());
        let bus = Arc::new(EventBus::new());
        let event_service: Arc<dyn EventService> = Arc::new(EventServiceImpl::new(bus));
        let harness = DefaultBenchmarkHarness::new(repo.clone(), event_service);

        let request = BenchmarkRunRequest {
            name: "test".into(),
            golden_set_path: path,
            variant: BenchmarkVariant::default(),
            scorer: Arc::new(ExactMatchScorer),
            runner: Arc::new(DummyRunner),
            max_concurrency: 1,
            project_id: None,
        };

        let run = harness.run(request).await.unwrap();
        assert_eq!(run.summary.total_cases, 2);
        assert_eq!(run.summary.pass_count, 1);
        assert_eq!(run.summary.fail_count, 1);

        let persisted = repo.get_run(&run.summary.id).await.unwrap().unwrap();
        assert_eq!(persisted.summary.pass_rate, 0.5);
    }

    #[tokio::test]
    async fn harness_persists_run_before_results_for_fk_backends() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gs.jsonl");
        tokio::fs::write(
            &path,
            r#"{"id":"a","input":{"answer":"compute validated parser behavior for foreign key benchmark ordering"},"expected":{"answer":"compute validated parser behavior for foreign key benchmark ordering"}}"#,
        )
        .await
        .unwrap();

        let repo: Arc<dyn BenchmarkResultRepository> = Arc::new(RunFirstRepository::new());
        let bus = Arc::new(EventBus::new());
        let event_service: Arc<dyn EventService> = Arc::new(EventServiceImpl::new(bus));
        let harness = DefaultBenchmarkHarness::new(repo.clone(), event_service);

        let request = BenchmarkRunRequest {
            name: "fk-order".into(),
            golden_set_path: path,
            variant: BenchmarkVariant::default(),
            scorer: Arc::new(ExactMatchScorer),
            runner: Arc::new(DummyRunner),
            max_concurrency: 1,
            project_id: None,
        };

        let run = harness.run(request).await.unwrap();
        assert_eq!(repo.list_results(&run.summary.id).await.unwrap().len(), 1);
    }
}
