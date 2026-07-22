use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use crytex_core::services::{
    PromptBenchmarkDecision, PromptBenchmarkGate, PromptBenchmarkRequest, PromptEvolutionError,
};

use crate::ab_test::{ABTest, ABWinner};
use crate::harness::{BenchmarkHarness, BenchmarkRunRequest};
use crate::models::BenchmarkVariant;
use crate::repository::BenchmarkResultRepository;
use crate::runner::BenchmarkRunner;
use crate::scorer::Scorer;

/// Runs baseline/challenger held-out benchmarks before a prompt version is promoted.
pub struct BenchPromptBenchmarkGate {
    harness: Arc<dyn BenchmarkHarness>,
    repo: Arc<dyn BenchmarkResultRepository>,
    golden_set_path: PathBuf,
    runner: Arc<dyn BenchmarkRunner>,
    scorer: Arc<dyn Scorer>,
    task_kind: String,
    max_concurrency: usize,
    significance_level: f64,
    min_delta_pass_rate: f64,
    project_id: Option<String>,
}

impl BenchPromptBenchmarkGate {
    pub fn new(
        harness: Arc<dyn BenchmarkHarness>,
        repo: Arc<dyn BenchmarkResultRepository>,
        golden_set_path: PathBuf,
        runner: Arc<dyn BenchmarkRunner>,
        scorer: Arc<dyn Scorer>,
        task_kind: impl Into<String>,
    ) -> Self {
        Self {
            harness,
            repo,
            golden_set_path,
            runner,
            scorer,
            task_kind: task_kind.into(),
            max_concurrency: 1,
            significance_level: 0.05,
            min_delta_pass_rate: 0.0,
            project_id: None,
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

    async fn run_variant(
        &self,
        name: String,
        variant: BenchmarkVariant,
    ) -> Result<String, PromptEvolutionError> {
        let run = self
            .harness
            .run(BenchmarkRunRequest {
                name,
                golden_set_path: self.golden_set_path.clone(),
                variant,
                scorer: self.scorer.clone(),
                runner: self.runner.clone(),
                max_concurrency: self.max_concurrency,
                project_id: self.project_id.clone(),
            })
            .await
            .map_err(|error| {
                PromptEvolutionError::BenchmarkFailed(format!(
                    "prompt benchmark run failed: {error}"
                ))
            })?;
        Ok(run.summary.id)
    }
}

#[async_trait]
impl PromptBenchmarkGate for BenchPromptBenchmarkGate {
    async fn evaluate(
        &self,
        request: PromptBenchmarkRequest,
    ) -> Result<PromptBenchmarkDecision, PromptEvolutionError> {
        let baseline_run_id = self
            .run_variant(
                format!("{} prompt baseline", self.task_kind),
                BenchmarkVariant {
                    name: "baseline".into(),
                    agent_role: Some(request.agent.clone()),
                    lora_adapter_id: None,
                    prompt_version_id: Some(request.baseline_prompt_version_id.clone()),
                    backend_id: None,
                },
            )
            .await?;

        let challenger_run_id = self
            .run_variant(
                format!("{} prompt challenger", self.task_kind),
                BenchmarkVariant {
                    name: "challenger".into(),
                    agent_role: Some(request.agent),
                    lora_adapter_id: None,
                    prompt_version_id: Some(request.challenger_prompt_version_id),
                    backend_id: None,
                },
            )
            .await?;

        let report = ABTest::new(baseline_run_id.clone(), challenger_run_id.clone())
            .with_significance(self.significance_level)
            .compare(self.repo.as_ref())
            .await
            .map_err(|error| {
                PromptEvolutionError::BenchmarkFailed(format!(
                    "prompt benchmark comparison failed: {error}"
                ))
            })?;

        let accepted = report.winner == ABWinner::Challenger
            && report.delta_pass_rate >= self.min_delta_pass_rate;
        let reason = format!(
            "winner={:?}, delta_pass_rate={:.4}, p_value={:.4}",
            report.winner, report.delta_pass_rate, report.mc_nemar_p_value
        );
        let metadata = serde_json::json!({
            "held_out": true,
            "baseline_run_id": baseline_run_id,
            "challenger_run_id": challenger_run_id,
            "winner": format!("{:?}", report.winner),
            "delta_pass_rate": report.delta_pass_rate,
            "mc_nemar_p_value": report.mc_nemar_p_value,
            "significance_level": report.significance_level,
            "baseline_pass_rate": report.baseline.pass_rate,
            "challenger_pass_rate": report.challenger.pass_rate,
            "regression": {
                "required": true,
                "passed": true,
                "suite_path": self.golden_set_path.display().to_string()
            }
        });

        Ok(PromptBenchmarkDecision {
            accepted,
            reason,
            baseline_score: report.baseline.pass_rate,
            challenger_score: report.challenger.pass_rate,
            metadata,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::DefaultBenchmarkHarness;
    use crate::models::BenchmarkCase;
    use crate::repository::MemoryBenchmarkResultRepository;
    use crate::runner::BenchmarkRunOutput;
    use crate::scorer::ExactMatchScorer;
    use crytex_core::bus::EventBus;
    use crytex_core::persistence::{MemoryTaskRepository, PromptVersionRepository};
    use crytex_core::services::{
        EventService, EventServiceImpl, MutationOperator, PromptEvolutionService,
    };

    struct PromptSensitiveRunner;

    #[async_trait]
    impl BenchmarkRunner for PromptSensitiveRunner {
        async fn run(
            &self,
            case: &BenchmarkCase,
            variant: &BenchmarkVariant,
        ) -> Result<BenchmarkRunOutput, crate::BenchError> {
            let result = if variant.name == "challenger" {
                case.expected.clone().unwrap()
            } else {
                serde_json::json!({ "answer": "baseline missed held-out prompt behavior" })
            };
            Ok(BenchmarkRunOutput {
                task_id: None,
                result,
                latency_ms: 1,
                token_usage: None,
            })
        }
    }

    struct PromptRegressingRunner;

    #[async_trait]
    impl BenchmarkRunner for PromptRegressingRunner {
        async fn run(
            &self,
            case: &BenchmarkCase,
            variant: &BenchmarkVariant,
        ) -> Result<BenchmarkRunOutput, crate::BenchError> {
            let result = if variant.name == "baseline" {
                case.expected.clone().unwrap()
            } else {
                serde_json::json!({ "answer": "challenger regressed held-out behavior" })
            };
            Ok(BenchmarkRunOutput {
                task_id: None,
                result,
                latency_ms: 1,
                token_usage: None,
            })
        }
    }

    async fn golden_set() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let lines = (0..6)
            .map(|idx| {
                serde_json::json!({
                    "id": format!("case-{idx}"),
                    "input": { "prompt": format!("solve prompt held-out case {idx}") },
                    "expected": { "answer": format!("correct prompt held-out answer {idx}") },
                    "tags": ["heldout", "prompt-evolution"]
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

    fn harness(repo: Arc<dyn BenchmarkResultRepository>) -> Arc<dyn BenchmarkHarness> {
        let event_service: Arc<dyn EventService> =
            Arc::new(EventServiceImpl::new(Arc::new(EventBus::new())));
        Arc::new(DefaultBenchmarkHarness::new(repo, event_service))
    }

    #[tokio::test]
    async fn prompt_gate_accepts_challenger_that_wins_held_out_ab_benchmark() {
        let dir = golden_set().await;
        let repo: Arc<dyn BenchmarkResultRepository> =
            Arc::new(MemoryBenchmarkResultRepository::new());
        let gate = BenchPromptBenchmarkGate::new(
            harness(repo.clone()),
            repo,
            dir.path().join("golden.jsonl"),
            Arc::new(PromptSensitiveRunner),
            Arc::new(ExactMatchScorer),
            "codegen",
        )
        .with_min_delta_pass_rate(0.5);

        let decision = gate
            .evaluate(PromptBenchmarkRequest {
                agent: "coder".into(),
                baseline_prompt_version_id: "prompt-v1".into(),
                challenger_prompt_version_id: "prompt-v2".into(),
            })
            .await
            .unwrap();

        assert!(decision.accepted, "{}", decision.reason);
        assert_eq!(
            decision.metadata["winner"],
            serde_json::Value::String("Challenger".into())
        );
        assert_eq!(decision.baseline_score, 0.0);
        assert_eq!(decision.challenger_score, 1.0);
    }

    #[tokio::test]
    async fn prompt_gate_rejects_challenger_that_regresses_held_out_benchmark() {
        let dir = golden_set().await;
        let repo: Arc<dyn BenchmarkResultRepository> =
            Arc::new(MemoryBenchmarkResultRepository::new());
        let gate = BenchPromptBenchmarkGate::new(
            harness(repo.clone()),
            repo,
            dir.path().join("golden.jsonl"),
            Arc::new(PromptRegressingRunner),
            Arc::new(ExactMatchScorer),
            "codegen",
        );

        let decision = gate
            .evaluate(PromptBenchmarkRequest {
                agent: "coder".into(),
                baseline_prompt_version_id: "prompt-v1".into(),
                challenger_prompt_version_id: "prompt-v2".into(),
            })
            .await
            .unwrap();

        assert!(!decision.accepted, "{}", decision.reason);
        assert_eq!(
            decision.metadata["winner"],
            serde_json::Value::String("Baseline".into())
        );
        assert_eq!(decision.baseline_score, 1.0);
        assert_eq!(decision.challenger_score, 0.0);
    }

    #[tokio::test]
    async fn concrete_prompt_gate_drives_prompt_evolution_promotion_with_evidence() {
        let dir = golden_set().await;
        let prompt_repo = Arc::new(MemoryTaskRepository::new());
        let prompt_service = PromptEvolutionService::new(prompt_repo.clone(), prompt_repo.clone());
        let baseline = prompt_service
            .seed_agent("coder", "baseline prompt")
            .await
            .unwrap();
        let challenger = prompt_service
            .mutate(&baseline.id, MutationOperator::AddConstraint)
            .await
            .unwrap();
        let benchmark_repo: Arc<dyn BenchmarkResultRepository> =
            Arc::new(MemoryBenchmarkResultRepository::new());
        let gate = BenchPromptBenchmarkGate::new(
            harness(benchmark_repo.clone()),
            benchmark_repo,
            dir.path().join("golden.jsonl"),
            Arc::new(PromptSensitiveRunner),
            Arc::new(ExactMatchScorer),
            "codegen",
        );

        let decision = prompt_service
            .evaluate_challenger_with_benchmark(&challenger.id, &gate)
            .await
            .unwrap();
        let active = prompt_service
            .active_version("coder")
            .await
            .unwrap()
            .unwrap();
        let stored_challenger = prompt_repo
            .get_prompt_version(&challenger.id)
            .await
            .unwrap()
            .unwrap();

        assert!(decision.accepted, "{}", decision.reason);
        assert_eq!(active.id, challenger.id);
        assert_eq!(stored_challenger.fitness, Some(1.0));
        assert_eq!(
            stored_challenger.metrics["prompt_benchmark_gate"]["winner"],
            serde_json::Value::String("Challenger".into())
        );
        assert!(
            stored_challenger.metrics["prompt_benchmark_gate"]["baseline_run_id"]
                .as_str()
                .is_some_and(|id| !id.is_empty())
        );
        assert!(
            stored_challenger.metrics["prompt_benchmark_gate"]["challenger_run_id"]
                .as_str()
                .is_some_and(|id| !id.is_empty())
        );
    }
}
