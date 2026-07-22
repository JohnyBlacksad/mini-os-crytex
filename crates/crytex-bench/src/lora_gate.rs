use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use crytex_core::services::{
    LoraBenchmarkDecision, LoraBenchmarkGate, LoraBenchmarkRequest, LoraEvolutionError,
};

use crate::ab_test::{ABTest, ABWinner};
use crate::golden_set::GoldenSet;
use crate::harness::{BenchmarkHarness, BenchmarkRunRequest};
use crate::models::BenchmarkVariant;
use crate::repository::BenchmarkResultRepository;
use crate::runner::BenchmarkRunner;
use crate::scorer::Scorer;

type BenchmarkRunnerFactory = dyn Fn(&str) -> Arc<dyn BenchmarkRunner> + Send + Sync;

/// Runs baseline/challenger benchmarks before a LoRA adapter is promoted.
pub struct BenchLoraBenchmarkGate {
    harness: Arc<dyn BenchmarkHarness>,
    repo: Arc<dyn BenchmarkResultRepository>,
    golden_set_path: PathBuf,
    runner_factory: Arc<BenchmarkRunnerFactory>,
    scorer: Arc<dyn Scorer>,
    max_concurrency: usize,
    significance_level: f64,
    min_delta_pass_rate: f64,
    project_id: Option<String>,
    leakage_similarity_threshold: f64,
}

impl BenchLoraBenchmarkGate {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        harness: Arc<dyn BenchmarkHarness>,
        repo: Arc<dyn BenchmarkResultRepository>,
        golden_set_path: PathBuf,
        runner: Arc<dyn BenchmarkRunner>,
        scorer: Arc<dyn Scorer>,
    ) -> Self {
        let runner_factory = Arc::new(move |_task_kind: &str| runner.clone());
        Self::new_with_runner_factory(harness, repo, golden_set_path, runner_factory, scorer)
    }

    pub fn new_with_runner_factory(
        harness: Arc<dyn BenchmarkHarness>,
        repo: Arc<dyn BenchmarkResultRepository>,
        golden_set_path: PathBuf,
        runner_factory: Arc<BenchmarkRunnerFactory>,
        scorer: Arc<dyn Scorer>,
    ) -> Self {
        Self {
            harness,
            repo,
            golden_set_path,
            runner_factory,
            scorer,
            max_concurrency: 1,
            significance_level: 0.05,
            min_delta_pass_rate: 0.0,
            project_id: None,
            leakage_similarity_threshold: 0.8,
        }
    }

    pub fn with_max_concurrency(mut self, max_concurrency: usize) -> Self {
        self.max_concurrency = max_concurrency.max(1);
        self
    }

    pub fn with_significance_level(mut self, significance_level: f64) -> Self {
        self.significance_level = significance_level;
        self
    }

    pub fn with_min_delta_pass_rate(mut self, min_delta_pass_rate: f64) -> Self {
        self.min_delta_pass_rate = min_delta_pass_rate;
        self
    }

    pub fn with_project_id(mut self, project_id: impl Into<String>) -> Self {
        self.project_id = Some(project_id.into());
        self
    }

    pub fn with_leakage_similarity_threshold(mut self, threshold: f64) -> Self {
        self.leakage_similarity_threshold = threshold;
        self
    }

    async fn run_variant(
        &self,
        task_kind: &str,
        name: String,
        variant: BenchmarkVariant,
    ) -> Result<String, LoraEvolutionError> {
        let runner = (self.runner_factory)(task_kind);
        let run = self
            .harness
            .run(BenchmarkRunRequest {
                name,
                golden_set_path: self.golden_set_path.clone(),
                variant,
                scorer: self.scorer.clone(),
                runner,
                max_concurrency: self.max_concurrency,
                project_id: self.project_id.clone(),
            })
            .await
            .map_err(|e| {
                LoraEvolutionError::ValidationFailed(
                    "benchmark".into(),
                    format!("LoRA benchmark run failed: {e}"),
                )
            })?;
        Ok(run.summary.id)
    }
}

#[async_trait]
impl LoraBenchmarkGate for BenchLoraBenchmarkGate {
    async fn evaluate(
        &self,
        request: LoraBenchmarkRequest,
    ) -> Result<LoraBenchmarkDecision, LoraEvolutionError> {
        let cases = GoldenSet::load_validated(&self.golden_set_path)
            .await
            .map_err(|e| {
                LoraEvolutionError::ValidationFailed(
                    request.task_kind.clone(),
                    format!("LoRA held-out benchmark is invalid: {e}"),
                )
            })?;
        let training_texts = request
            .training_fingerprints
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        if let Err(error) = GoldenSet::validate_no_training_leakage(
            &cases,
            &training_texts,
            self.leakage_similarity_threshold,
        ) {
            return Ok(LoraBenchmarkDecision {
                accepted: false,
                reason: format!("held-out benchmark leakage check failed: {error}"),
                metadata: serde_json::json!({
                    "leakage_check": {
                        "passed": false,
                        "training_fingerprint_count": request.training_fingerprints.len(),
                        "similarity_threshold": self.leakage_similarity_threshold,
                        "error": error.to_string()
                    }
                }),
            });
        }

        let baseline_run_id = self
            .run_variant(
                &request.task_kind,
                format!("{} baseline", request.task_kind),
                BenchmarkVariant {
                    name: "baseline".into(),
                    agent_role: request.agent_role.clone(),
                    lora_adapter_id: request.baseline_adapter_id.clone(),
                    prompt_version_id: None,
                    backend_id: None,
                },
            )
            .await?;

        let challenger_run_id = self
            .run_variant(
                &request.task_kind,
                format!(
                    "{} challenger {}",
                    request.task_kind, request.challenger_adapter_id
                ),
                BenchmarkVariant {
                    name: "challenger".into(),
                    agent_role: request.agent_role.clone(),
                    lora_adapter_id: Some(request.challenger_adapter_id.clone()),
                    prompt_version_id: None,
                    backend_id: None,
                },
            )
            .await?;

        let report = ABTest::new(baseline_run_id.clone(), challenger_run_id.clone())
            .with_significance(self.significance_level)
            .compare(self.repo.as_ref())
            .await
            .map_err(|e| {
                LoraEvolutionError::ValidationFailed(
                    request.task_kind.clone(),
                    format!("LoRA benchmark comparison failed: {e}"),
                )
            })?;

        let accepted = report.winner == ABWinner::Challenger
            && report.delta_pass_rate >= self.min_delta_pass_rate;
        let reason = format!(
            "winner={:?}, delta_pass_rate={:.4}, p_value={:.4}",
            report.winner, report.delta_pass_rate, report.mc_nemar_p_value
        );

        let metadata = serde_json::json!({
            "leakage_check": {
                "passed": true,
                "training_fingerprint_count": request.training_fingerprints.len(),
                "similarity_threshold": self.leakage_similarity_threshold
            },
            "baseline_run_id": baseline_run_id,
            "challenger_run_id": challenger_run_id,
            "winner": format!("{:?}", report.winner),
            "delta_pass_rate": report.delta_pass_rate,
            "mc_nemar_p_value": report.mc_nemar_p_value,
            "significance_level": report.significance_level,
            "baseline_pass_rate": report.baseline.pass_rate,
            "challenger_pass_rate": report.challenger.pass_rate
        });

        Ok(LoraBenchmarkDecision {
            accepted,
            reason,
            metadata,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::DefaultBenchmarkHarness;
    use crate::models::{BenchmarkCase, BenchmarkRunSummary};
    use crate::repository::MemoryBenchmarkResultRepository;
    use crate::runner::BenchmarkRunOutput;
    use crate::scorer::ExactMatchScorer;
    use async_trait::async_trait;
    use crytex_core::bus::EventBus;
    use crytex_core::models::{
        LoraAdapter, PromptVersion, Task, TaskStatus, TrainingExample, TrainingJob,
        TrainingJobStatus,
    };
    use crytex_core::persistence::{
        LoraAdapterRepository, PersistenceError, PromptVersionRepository,
        TrainingExampleRepository, TrainingJobRepository,
    };
    use crytex_core::services::{
        CreateTaskRequest, EventHandler, EventService, EventServiceImpl, InferenceService,
        InferenceServiceError, LoraEvolutionService, LoraEvolutionServiceImpl, LoraMetrics,
        LoraTrainer, LoraTrainingConfig, LoraTrainingError, LoraTrainingResult, TaskError,
        TaskService,
    };
    use crytex_inference::{
        BackendInfo, InferenceRequest, InferenceResponse, LoRAAdapter as InferenceLoRAAdapter,
        ModelInfo, TokenUsage,
    };
    use std::collections::HashMap;
    use std::path::Path;
    use std::sync::Mutex;

    struct LoraSensitiveRunner;

    #[async_trait]
    impl BenchmarkRunner for LoraSensitiveRunner {
        async fn run(
            &self,
            case: &BenchmarkCase,
            variant: &BenchmarkVariant,
        ) -> Result<BenchmarkRunOutput, crate::BenchError> {
            let result = if variant.lora_adapter_id.as_deref() == Some("candidate-lora") {
                case.expected.clone().unwrap()
            } else {
                serde_json::json!({"answer": "wrong baseline output"})
            };

            Ok(BenchmarkRunOutput {
                task_id: None,
                result,
                latency_ms: 1,
                token_usage: None,
            })
        }
    }

    struct RegressingRunner;

    #[async_trait]
    impl BenchmarkRunner for RegressingRunner {
        async fn run(
            &self,
            case: &BenchmarkCase,
            variant: &BenchmarkVariant,
        ) -> Result<BenchmarkRunOutput, crate::BenchError> {
            let result = if variant.name == "baseline" {
                case.expected.clone().unwrap()
            } else {
                serde_json::json!({"answer": "wrong challenger output"})
            };

            Ok(BenchmarkRunOutput {
                task_id: None,
                result,
                latency_ms: 1,
                token_usage: None,
            })
        }
    }

    struct CandidateSensitiveRunner;

    #[async_trait]
    impl BenchmarkRunner for CandidateSensitiveRunner {
        async fn run(
            &self,
            case: &BenchmarkCase,
            variant: &BenchmarkVariant,
        ) -> Result<BenchmarkRunOutput, crate::BenchError> {
            let result = if variant.lora_adapter_id.as_deref() == Some("codegen-v2") {
                case.expected.clone().unwrap()
            } else {
                serde_json::json!({"answer": "baseline missed the held-out behavior"})
            };

            Ok(BenchmarkRunOutput {
                task_id: None,
                result,
                latency_ms: 1,
                token_usage: None,
            })
        }
    }

    struct DummyTaskService {
        tasks: Mutex<HashMap<String, Task>>,
    }

    impl DummyTaskService {
        fn new() -> Self {
            Self {
                tasks: Mutex::new(HashMap::new()),
            }
        }
    }

    #[async_trait]
    impl TaskService for DummyTaskService {
        async fn submit(&self, _request: CreateTaskRequest) -> Result<Task, TaskError> {
            unimplemented!()
        }

        async fn add_dependency(
            &self,
            _dep: crytex_core::models::TaskDependency,
        ) -> Result<(), TaskError> {
            unimplemented!()
        }

        async fn get(&self, id: &str) -> Result<Option<Task>, TaskError> {
            Ok(self.tasks.lock().unwrap().get(id).cloned())
        }

        async fn list_by_project(&self, _project_id: &str) -> Result<Vec<Task>, TaskError> {
            unimplemented!()
        }

        async fn list_ready(&self) -> Result<Vec<Task>, TaskError> {
            unimplemented!()
        }

        async fn set_status(&self, _id: &str, _status: TaskStatus) -> Result<Task, TaskError> {
            unimplemented!()
        }

        async fn cancel(&self, _id: &str) -> Result<Task, TaskError> {
            unimplemented!()
        }

        async fn set_result(
            &self,
            _id: &str,
            _result: serde_json::Value,
        ) -> Result<Task, TaskError> {
            unimplemented!()
        }

        async fn set_critic_score(&self, _id: &str, _score: f64) -> Result<Task, TaskError> {
            unimplemented!()
        }

        async fn set_human_score(&self, _id: &str, _score: f64) -> Result<Task, TaskError> {
            unimplemented!()
        }

        async fn retry(&self, _id: &str, _feedback: Option<&str>) -> Result<Task, TaskError> {
            unimplemented!()
        }

        async fn load_all_tasks(&self) -> Result<Vec<Task>, TaskError> {
            unimplemented!()
        }

        async fn update_task(&self, _task: &Task) -> Result<(), TaskError> {
            unimplemented!()
        }
    }

    #[derive(Default)]
    struct DummyPromptRepo;

    #[async_trait]
    impl PromptVersionRepository for DummyPromptRepo {
        async fn insert_prompt_version(
            &self,
            _version: &PromptVersion,
        ) -> Result<(), PersistenceError> {
            Ok(())
        }

        async fn update_prompt_version(
            &self,
            _version: &PromptVersion,
        ) -> Result<(), PersistenceError> {
            Ok(())
        }

        async fn get_prompt_version(
            &self,
            _id: &str,
        ) -> Result<Option<PromptVersion>, PersistenceError> {
            Ok(None)
        }

        async fn list_prompt_versions_by_agent(
            &self,
            _agent: &str,
        ) -> Result<Vec<PromptVersion>, PersistenceError> {
            Ok(vec![])
        }

        async fn get_active_prompt_version(
            &self,
            _agent: &str,
        ) -> Result<Option<PromptVersion>, PersistenceError> {
            Ok(None)
        }

        async fn set_active_prompt_version(
            &self,
            _id: &str,
            _agent: &str,
        ) -> Result<(), PersistenceError> {
            Ok(())
        }
    }

    #[derive(Default)]
    struct InMemoryExamples {
        examples: Mutex<Vec<TrainingExample>>,
    }

    #[async_trait]
    impl TrainingExampleRepository for InMemoryExamples {
        async fn insert_training_example(
            &self,
            example: &TrainingExample,
        ) -> Result<(), PersistenceError> {
            self.examples.lock().unwrap().push(example.clone());
            Ok(())
        }

        async fn list_training_examples_by_kind(
            &self,
            task_kind: &str,
        ) -> Result<Vec<TrainingExample>, PersistenceError> {
            Ok(self
                .examples
                .lock()
                .unwrap()
                .iter()
                .filter(|example| example.task_kind == task_kind)
                .cloned()
                .collect())
        }

        async fn count_training_examples_by_kind(
            &self,
            task_kind: &str,
        ) -> Result<usize, PersistenceError> {
            Ok(self
                .examples
                .lock()
                .unwrap()
                .iter()
                .filter(|example| example.task_kind == task_kind)
                .count())
        }

        async fn list_training_examples_by_project(
            &self,
            _project_id: &str,
        ) -> Result<Vec<TrainingExample>, PersistenceError> {
            Ok(vec![])
        }

        async fn list_training_examples_by_role(
            &self,
            agent_role: &str,
        ) -> Result<Vec<TrainingExample>, PersistenceError> {
            Ok(self
                .examples
                .lock()
                .unwrap()
                .iter()
                .filter(|example| example.agent_role.as_deref() == Some(agent_role))
                .cloned()
                .collect())
        }

        async fn count_training_examples_by_role(
            &self,
            agent_role: &str,
        ) -> Result<usize, PersistenceError> {
            Ok(self
                .examples
                .lock()
                .unwrap()
                .iter()
                .filter(|example| example.agent_role.as_deref() == Some(agent_role))
                .count())
        }
    }

    #[derive(Default)]
    struct InMemoryAdapters {
        adapters: Mutex<Vec<LoraAdapter>>,
    }

    #[async_trait]
    impl LoraAdapterRepository for InMemoryAdapters {
        async fn insert_lora_adapter(&self, adapter: &LoraAdapter) -> Result<(), PersistenceError> {
            self.adapters.lock().unwrap().push(adapter.clone());
            Ok(())
        }

        async fn get_lora_adapter(
            &self,
            id: &str,
        ) -> Result<Option<LoraAdapter>, PersistenceError> {
            Ok(self
                .adapters
                .lock()
                .unwrap()
                .iter()
                .find(|adapter| adapter.id == id)
                .cloned())
        }

        async fn list_lora_adapters_by_kind(
            &self,
            task_kind: &str,
        ) -> Result<Vec<LoraAdapter>, PersistenceError> {
            Ok(self
                .adapters
                .lock()
                .unwrap()
                .iter()
                .filter(|adapter| adapter.task_kind.as_deref() == Some(task_kind))
                .cloned()
                .collect())
        }

        async fn list_lora_adapters_by_project(
            &self,
            _project_id: &str,
        ) -> Result<Vec<LoraAdapter>, PersistenceError> {
            Ok(vec![])
        }

        async fn list_lora_adapters_by_role(
            &self,
            agent_role: &str,
        ) -> Result<Vec<LoraAdapter>, PersistenceError> {
            Ok(self
                .adapters
                .lock()
                .unwrap()
                .iter()
                .filter(|adapter| adapter.agent_role.as_deref() == Some(agent_role))
                .cloned()
                .collect())
        }

        async fn set_lora_adapter_active(
            &self,
            id: &str,
            active: bool,
        ) -> Result<(), PersistenceError> {
            if let Some(adapter) = self
                .adapters
                .lock()
                .unwrap()
                .iter_mut()
                .find(|adapter| adapter.id == id)
            {
                adapter.active = active;
            }
            Ok(())
        }
    }

    #[derive(Default)]
    struct InMemoryTrainingJobs {
        jobs: Mutex<Vec<TrainingJob>>,
    }

    #[async_trait]
    impl TrainingJobRepository for InMemoryTrainingJobs {
        async fn insert_training_job(&self, job: &TrainingJob) -> Result<(), PersistenceError> {
            self.jobs.lock().unwrap().push(job.clone());
            Ok(())
        }

        async fn update_training_job(&self, job: &TrainingJob) -> Result<(), PersistenceError> {
            let mut jobs = self.jobs.lock().unwrap();
            if let Some(existing) = jobs.iter_mut().find(|existing| existing.id == job.id) {
                *existing = job.clone();
            } else {
                jobs.push(job.clone());
            }
            Ok(())
        }

        async fn get_training_job(
            &self,
            _id: &str,
        ) -> Result<Option<TrainingJob>, PersistenceError> {
            Ok(None)
        }

        async fn list_training_jobs_by_kind(
            &self,
            task_kind: &str,
        ) -> Result<Vec<TrainingJob>, PersistenceError> {
            Ok(self
                .jobs
                .lock()
                .unwrap()
                .iter()
                .filter(|job| job.task_kind == task_kind)
                .cloned()
                .collect())
        }
    }

    #[derive(Default)]
    struct DummyInference {
        registered: Mutex<Vec<String>>,
    }

    #[async_trait]
    impl InferenceService for DummyInference {
        async fn generate(
            &self,
            _request: InferenceRequest,
        ) -> Result<InferenceResponse, InferenceServiceError> {
            Ok(InferenceResponse {
                content: "ok".into(),
                usage: TokenUsage {
                    prompt_tokens: 1,
                    completion_tokens: 1,
                    total_tokens: 2,
                },
                finish_reason: "stop".into(),
            })
        }

        async fn embed(&self, _text: &str) -> Result<Vec<f32>, InferenceServiceError> {
            Ok(vec![0.0])
        }

        fn available_backends(&self) -> Vec<BackendInfo> {
            vec![]
        }

        async fn register_lora(
            &self,
            lora: InferenceLoRAAdapter,
        ) -> Result<(), InferenceServiceError> {
            self.registered.lock().unwrap().push(lora.id);
            Ok(())
        }

        async fn swap_lora(&self, _lora_id: &str) -> Result<(), InferenceServiceError> {
            Ok(())
        }

        async fn list_models(
            &self,
            _backend_id: Option<&str>,
        ) -> Result<Vec<ModelInfo>, InferenceServiceError> {
            Ok(vec![])
        }
    }

    #[derive(Default)]
    struct DummyEvents;

    #[async_trait]
    impl EventService for DummyEvents {
        fn publish(&self, _event: crytex_core::bus::Event) {}

        fn subscribe(&self) -> tokio::sync::broadcast::Receiver<crytex_core::bus::Event> {
            let (tx, _rx) = tokio::sync::broadcast::channel(1);
            tx.subscribe()
        }

        async fn start_handler(&self, _handler: Arc<dyn EventHandler>) {}
    }

    struct DeterministicTrainer;

    #[async_trait]
    impl LoraTrainer for DeterministicTrainer {
        async fn train(
            &self,
            examples: Vec<TrainingExample>,
            _config: LoraTrainingConfig,
            output_dir: &Path,
        ) -> Result<LoraTrainingResult, LoraTrainingError> {
            tokio::fs::create_dir_all(output_dir).await?;
            let adapter_path = output_dir.join("candidate");
            tokio::fs::create_dir_all(&adapter_path).await?;
            tokio::fs::write(
                adapter_path.join("adapter_config.json"),
                serde_json::json!({
                    "peft_type": "LORA",
                    "base_model_name_or_path": "mistral-7b",
                    "r": 8,
                    "lora_alpha": 16,
                    "target_modules": ["q_proj", "v_proj"]
                })
                .to_string(),
            )
            .await?;
            tokio::fs::write(
                adapter_path.join("adapter_model.safetensors"),
                b"deterministic adapter",
            )
            .await?;
            let average_reward =
                examples.iter().map(|example| example.reward).sum::<f64>() / examples.len() as f64;
            Ok(LoraTrainingResult {
                adapter_id: "candidate".into(),
                adapter_path,
                metrics: LoraMetrics {
                    train_loss: 0.10,
                    validation_loss: 0.11,
                    average_reward,
                },
            })
        }
    }

    fn training_example(idx: i64) -> TrainingExample {
        TrainingExample {
            id: format!("example-{idx}"),
            task_id: format!("task-{idx}"),
            project_id: Some("project-1".into()),
            prompt_version_id: None,
            task_kind: "codegen".into(),
            agent_role: None,
            model_id: None,
            rag_evidence_ids: Vec::new(),
            input_text: format!("Implement deterministic parsing behavior for approved case {idx}"),
            output_text: format!("Correct parser implementation and tests for approved case {idx}"),
            accepted_output: Some(format!(
                "Correct parser implementation and tests for approved case {idx}"
            )),
            rejected_output: None,
            critic_feedback: None,
            failure_type: None,
            reward: 5.0,
            created_at: idx,
        }
    }

    fn pending_codegen_task() -> Task {
        Task {
            id: "select-task".into(),
            project_id: "project-1".into(),
            parent_id: None,
            title: "select adapter".into(),
            description: None,
            kind: "codegen".into(),
            status: TaskStatus::Pending,
            assigned_agent: Some("coder".into()),
            priority: 0,
            created_at: 0,
            started_at: None,
            finished_at: None,
            payload: serde_json::Value::Null,
            result: None,
            iteration_count: 0,
            priority_score: 0.0,
            critic_score: None,
            human_score: None,
            prompt_version_id: None,
            lora_adapter_id: None,
            trace_id: "trace-select".into(),
        }
    }

    fn active_baseline_adapter() -> LoraAdapter {
        LoraAdapter {
            id: "codegen-v1".into(),
            project_id: None,
            name: "codegen-v1".into(),
            file_path: "baseline.safetensors".into(),
            base_model: "mistral-7b".into(),
            task_kind: Some("codegen".into()),
            agent_role: None,
            metrics: serde_json::json!({}),
            created_at: 1,
            active: true,
        }
    }

    async fn golden_set() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let lines = (0..6)
            .map(|idx| {
                serde_json::json!({
                    "id": format!("case-{idx}"),
                    "input": { "prompt": format!("solve heldout task {idx}") },
                    "expected": { "answer": format!("correct heldout answer {idx}") }
                })
                .to_string()
            })
            .collect::<Vec<_>>()
            .join("\n");
        tokio::fs::write(dir.path().join("golden.jsonl"), lines)
            .await
            .unwrap();
        dir
    }

    async fn leaked_golden_set() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let line = serde_json::json!({
            "id": "leaked-heldout",
            "input": { "prompt": "Implement deterministic parser branch from training corpus" },
            "expected": { "answer": "Parser branch implementation copied from training corpus" },
            "tags": ["heldout", "lora"]
        })
        .to_string();
        tokio::fs::write(dir.path().join("golden.jsonl"), line)
            .await
            .unwrap();
        dir
    }

    #[tokio::test]
    async fn gate_accepts_challenger_that_wins_ab_benchmark() {
        let dir = golden_set().await;
        let repo: Arc<dyn BenchmarkResultRepository> =
            Arc::new(MemoryBenchmarkResultRepository::new());
        let event_service: Arc<dyn EventService> =
            Arc::new(EventServiceImpl::new(Arc::new(EventBus::new())));
        let harness: Arc<dyn BenchmarkHarness> =
            Arc::new(DefaultBenchmarkHarness::new(repo.clone(), event_service));
        let gate = BenchLoraBenchmarkGate::new(
            harness,
            repo,
            dir.path().join("golden.jsonl"),
            Arc::new(LoraSensitiveRunner),
            Arc::new(ExactMatchScorer),
        )
        .with_significance_level(0.05)
        .with_min_delta_pass_rate(0.5);

        let decision = gate
            .evaluate(LoraBenchmarkRequest {
                task_kind: "codegen".into(),
                agent_role: Some("coder".into()),
                baseline_adapter_id: None,
                challenger_adapter_id: "candidate-lora".into(),
                challenger_adapter_path: dir.path().join("candidate.safetensors"),
                base_model: "mistral-7b".into(),
                challenger_metrics: serde_json::json!({}),
                validation_reward: 5.0,
                training_fingerprints: vec![],
            })
            .await
            .unwrap();

        assert!(decision.accepted, "{}", decision.reason);
        assert_eq!(
            decision.metadata["winner"],
            serde_json::Value::String("Challenger".into())
        );
        assert!(decision.metadata["delta_pass_rate"].as_f64().unwrap() >= 0.5);
    }

    #[tokio::test]
    async fn gate_rejects_challenger_that_loses_ab_benchmark() {
        let dir = golden_set().await;
        let repo: Arc<dyn BenchmarkResultRepository> =
            Arc::new(MemoryBenchmarkResultRepository::new());
        let event_service: Arc<dyn EventService> =
            Arc::new(EventServiceImpl::new(Arc::new(EventBus::new())));
        let harness: Arc<dyn BenchmarkHarness> =
            Arc::new(DefaultBenchmarkHarness::new(repo.clone(), event_service));
        let gate = BenchLoraBenchmarkGate::new(
            harness,
            repo,
            dir.path().join("golden.jsonl"),
            Arc::new(RegressingRunner),
            Arc::new(ExactMatchScorer),
        )
        .with_significance_level(0.05);

        let decision = gate
            .evaluate(LoraBenchmarkRequest {
                task_kind: "codegen".into(),
                agent_role: Some("coder".into()),
                baseline_adapter_id: Some("baseline-lora".into()),
                challenger_adapter_id: "candidate-lora".into(),
                challenger_adapter_path: dir.path().join("candidate.safetensors"),
                base_model: "mistral-7b".into(),
                challenger_metrics: serde_json::json!({}),
                validation_reward: 5.0,
                training_fingerprints: vec![],
            })
            .await
            .unwrap();

        assert!(!decision.accepted, "{}", decision.reason);
        assert_eq!(
            decision.metadata["winner"],
            serde_json::Value::String("Baseline".into())
        );
    }

    #[tokio::test]
    async fn gate_builds_runner_for_requested_task_kind() {
        let dir = golden_set().await;
        let repo: Arc<dyn BenchmarkResultRepository> =
            Arc::new(MemoryBenchmarkResultRepository::new());
        let event_service: Arc<dyn EventService> =
            Arc::new(EventServiceImpl::new(Arc::new(EventBus::new())));
        let harness: Arc<dyn BenchmarkHarness> =
            Arc::new(DefaultBenchmarkHarness::new(repo.clone(), event_service));
        let requested_kinds = Arc::new(std::sync::Mutex::new(Vec::new()));
        let captured = requested_kinds.clone();
        let gate = BenchLoraBenchmarkGate::new_with_runner_factory(
            harness,
            repo,
            dir.path().join("golden.jsonl"),
            Arc::new(move |task_kind| {
                captured.lock().unwrap().push(task_kind.to_string());
                Arc::new(LoraSensitiveRunner) as Arc<dyn BenchmarkRunner>
            }),
            Arc::new(ExactMatchScorer),
        );

        let decision = gate
            .evaluate(LoraBenchmarkRequest {
                task_kind: "architecture".into(),
                agent_role: Some("architect".into()),
                baseline_adapter_id: None,
                challenger_adapter_id: "candidate-lora".into(),
                challenger_adapter_path: dir.path().join("candidate.safetensors"),
                base_model: "mistral-7b".into(),
                challenger_metrics: serde_json::json!({}),
                validation_reward: 5.0,
                training_fingerprints: vec![],
            })
            .await
            .unwrap();

        assert!(decision.accepted, "{}", decision.reason);
        assert_eq!(
            requested_kinds.lock().unwrap().as_slice(),
            ["architecture", "architecture"]
        );
    }

    #[tokio::test]
    async fn gate_rejects_held_out_benchmark_that_leaks_training_corpus() {
        let dir = leaked_golden_set().await;
        let repo: Arc<dyn BenchmarkResultRepository> =
            Arc::new(MemoryBenchmarkResultRepository::new());
        let event_service: Arc<dyn EventService> =
            Arc::new(EventServiceImpl::new(Arc::new(EventBus::new())));
        let harness: Arc<dyn BenchmarkHarness> =
            Arc::new(DefaultBenchmarkHarness::new(repo.clone(), event_service));
        let gate = BenchLoraBenchmarkGate::new(
            harness,
            repo,
            dir.path().join("golden.jsonl"),
            Arc::new(LoraSensitiveRunner),
            Arc::new(ExactMatchScorer),
        );

        let decision = gate
            .evaluate(LoraBenchmarkRequest {
                task_kind: "codegen".into(),
                agent_role: Some("coder".into()),
                baseline_adapter_id: None,
                challenger_adapter_id: "candidate-lora".into(),
                challenger_adapter_path: dir.path().join("candidate.safetensors"),
                base_model: "mistral-7b".into(),
                challenger_metrics: serde_json::json!({}),
                validation_reward: 5.0,
                training_fingerprints: vec![
                    "Implement deterministic parser branch from training corpus Parser branch implementation copied from training corpus".into(),
                ],
            })
            .await
            .unwrap();

        assert!(!decision.accepted);
        assert!(decision.reason.contains("leakage"));
        assert_eq!(decision.metadata["leakage_check"]["passed"], false);
    }

    #[tokio::test]
    async fn concrete_gate_drives_lora_evolution_promotion_and_selection() {
        let golden_dir = golden_set().await;
        let examples = Arc::new(InMemoryExamples::default());
        for idx in 0..12 {
            examples
                .insert_training_example(&training_example(idx))
                .await
                .unwrap();
        }
        let adapters = Arc::new(InMemoryAdapters::default());
        adapters
            .insert_lora_adapter(&active_baseline_adapter())
            .await
            .unwrap();
        let jobs = Arc::new(InMemoryTrainingJobs::default());
        let inference = Arc::new(DummyInference::default());
        let benchmark_repo: Arc<dyn BenchmarkResultRepository> =
            Arc::new(MemoryBenchmarkResultRepository::new());
        let event_service: Arc<dyn EventService> = Arc::new(DummyEvents);
        let harness: Arc<dyn BenchmarkHarness> = Arc::new(DefaultBenchmarkHarness::new(
            benchmark_repo.clone(),
            event_service.clone(),
        ));
        let gate = Arc::new(BenchLoraBenchmarkGate::new(
            harness,
            benchmark_repo.clone(),
            golden_dir.path().join("golden.jsonl"),
            Arc::new(CandidateSensitiveRunner),
            Arc::new(ExactMatchScorer),
        ));

        let service = LoraEvolutionServiceImpl::new(
            Arc::new(DummyTaskService::new()),
            Arc::new(DummyPromptRepo),
            examples,
            adapters.clone(),
            inference.clone(),
            event_service,
            Arc::new(DeterministicTrainer),
            golden_dir.path().join("adapters"),
            "mistral-7b".into(),
        )
        .with_threshold(12)
        .with_training_job_repo(jobs.clone())
        .with_benchmark_gate(gate);

        let promoted = service.train_and_register("codegen").await.unwrap();
        let selected = service
            .select_lora(&pending_codegen_task(), "project-1")
            .await
            .unwrap();
        let adapter_records = adapters
            .list_lora_adapters_by_kind("codegen")
            .await
            .unwrap();
        let benchmark_runs: Vec<BenchmarkRunSummary> = benchmark_repo.list_runs(10).await.unwrap();
        let training_jobs = jobs.list_training_jobs_by_kind("codegen").await.unwrap();

        assert_eq!(promoted.id, "codegen-v2");
        assert_eq!(selected, Some("codegen-v2".into()));
        assert_eq!(
            inference.registered.lock().unwrap().as_slice(),
            ["codegen-v2"]
        );
        assert!(
            adapter_records
                .iter()
                .any(|adapter| adapter.id == "codegen-v1" && !adapter.active)
        );
        assert!(
            adapter_records
                .iter()
                .any(|adapter| adapter.id == "codegen-v2" && adapter.active)
        );
        assert_eq!(benchmark_runs.len(), 2);
        assert!(
            benchmark_runs
                .iter()
                .any(|run| run.name == "codegen baseline" && run.pass_rate == 0.0)
        );
        assert!(
            benchmark_runs
                .iter()
                .any(|run| run.name.starts_with("codegen challenger") && run.pass_rate == 1.0)
        );
        assert_eq!(training_jobs.len(), 1);
        assert_eq!(training_jobs[0].status, TrainingJobStatus::Succeeded);
        assert_eq!(training_jobs[0].adapter_id.as_deref(), Some("codegen-v2"));
        assert_eq!(
            promoted.metrics["benchmark_gate"]["accepted"],
            serde_json::Value::Bool(true)
        );
        assert_eq!(
            promoted.metrics["benchmark_gate"]["winner"],
            serde_json::Value::String("Challenger".into())
        );
        assert!(
            promoted.metrics["benchmark_gate"]["baseline_run_id"]
                .as_str()
                .is_some_and(|id| !id.is_empty())
        );
        assert!(
            promoted.metrics["benchmark_gate"]["challenger_run_id"]
                .as_str()
                .is_some_and(|id| !id.is_empty())
        );
        assert_eq!(
            training_jobs[0].metrics["benchmark_gate"],
            promoted.metrics["benchmark_gate"]
        );
    }
}
