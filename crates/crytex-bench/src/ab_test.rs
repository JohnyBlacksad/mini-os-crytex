use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::error::BenchError;
use crate::models::BenchmarkRunSummary;
use crate::repository::BenchmarkResultRepository;

/// Result of comparing two benchmark runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ABWinner {
    Baseline,
    Challenger,
    Tie,
    Inconclusive,
}

/// Per-case comparison between baseline and challenger.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CaseComparison {
    pub case_id: String,
    pub baseline_passed: bool,
    pub challenger_passed: bool,
    pub baseline_score: f64,
    pub challenger_score: f64,
}

/// A/B test report with McNemar significance for binary outcomes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ABTestReport {
    pub baseline: BenchmarkRunSummary,
    pub challenger: BenchmarkRunSummary,
    pub delta_pass_rate: f64,
    pub mc_nemar_p_value: f64,
    pub significance_level: f64,
    pub winner: ABWinner,
    pub per_case_comparison: Vec<CaseComparison>,
    pub bootstrap_ci: Option<(f64, f64)>,
}

/// Compares two benchmark runs on the same golden set.
#[derive(Debug, Clone)]
pub struct ABTest {
    pub baseline_run_id: String,
    pub challenger_run_id: String,
    pub significance_level: f64,
    pub bootstrap_samples: usize,
}

impl ABTest {
    pub fn new(baseline_run_id: String, challenger_run_id: String) -> Self {
        Self {
            baseline_run_id,
            challenger_run_id,
            significance_level: 0.05,
            bootstrap_samples: 1000,
        }
    }

    pub fn with_significance(mut self, alpha: f64) -> Self {
        self.significance_level = alpha;
        self
    }

    pub async fn compare(
        &self,
        repo: &dyn BenchmarkResultRepository,
    ) -> Result<ABTestReport, BenchError> {
        let baseline_run = repo
            .get_run(&self.baseline_run_id)
            .await?
            .ok_or_else(|| BenchError::ABTest("baseline run not found".into()))?;
        let challenger_run = repo
            .get_run(&self.challenger_run_id)
            .await?
            .ok_or_else(|| BenchError::ABTest("challenger run not found".into()))?;

        let baseline_results = repo.list_results(&self.baseline_run_id).await?;
        let challenger_results = repo.list_results(&self.challenger_run_id).await?;

        let baseline_by_case: HashMap<_, _> = baseline_results
            .iter()
            .map(|r| (r.case_id.clone(), r))
            .collect();
        let challenger_by_case: HashMap<_, _> = challenger_results
            .iter()
            .map(|r| (r.case_id.clone(), r))
            .collect();

        let shared_cases: HashSet<_> = baseline_by_case
            .keys()
            .filter(|k| challenger_by_case.contains_key(*k))
            .cloned()
            .collect();

        if shared_cases.is_empty() {
            return Err(BenchError::ABTest(
                "baseline and challenger have no shared case ids".into(),
            ));
        }

        let mut comparisons = Vec::new();
        let mut both_pass = 0usize;
        let mut _both_fail = 0usize;
        let mut baseline_only = 0usize;
        let mut challenger_only = 0usize;

        for case_id in &shared_cases {
            let b = baseline_by_case[case_id];
            let c = challenger_by_case[case_id];
            comparisons.push(CaseComparison {
                case_id: case_id.clone(),
                baseline_passed: b.passed,
                challenger_passed: c.passed,
                baseline_score: b.score_value,
                challenger_score: c.score_value,
            });

            match (b.passed, c.passed) {
                (true, true) => both_pass += 1,
                (false, false) => _both_fail += 1,
                (true, false) => baseline_only += 1,
                (false, true) => challenger_only += 1,
            }
        }

        comparisons.sort_by(|a, b| a.case_id.cmp(&b.case_id));

        let n = shared_cases.len();
        let baseline_passes = both_pass + baseline_only;
        let challenger_passes = both_pass + challenger_only;
        let baseline_rate = baseline_passes as f64 / n as f64;
        let challenger_rate = challenger_passes as f64 / n as f64;
        let delta = challenger_rate - baseline_rate;

        let p_value = mc_nemar_exact_p_value(baseline_only, challenger_only);

        let score_deltas: Vec<f64> = comparisons
            .iter()
            .map(|c| c.challenger_score - c.baseline_score)
            .collect();
        let bootstrap_ci = bootstrap_mean_ci(&score_deltas, self.bootstrap_samples);

        let winner = if p_value < self.significance_level {
            if delta > 0.0 {
                ABWinner::Challenger
            } else if delta < 0.0 {
                ABWinner::Baseline
            } else {
                ABWinner::Tie
            }
        } else {
            ABWinner::Inconclusive
        };

        Ok(ABTestReport {
            baseline: baseline_run.summary,
            challenger: challenger_run.summary,
            delta_pass_rate: delta,
            mc_nemar_p_value: p_value,
            significance_level: self.significance_level,
            winner,
            per_case_comparison: comparisons,
            bootstrap_ci,
        })
    }
}

/// Two-tailed exact binomial McNemar p-value.
///
/// `b` = baseline passed but challenger failed.
/// `c` = challenger passed but baseline failed.
fn mc_nemar_exact_p_value(b: usize, c: usize) -> f64 {
    let n = b + c;
    if n == 0 {
        return 1.0;
    }
    let k = b.min(c);
    let observed_log_p = binomial_log_p(n, k);

    let mut p_value = 0.0;
    for i in 0..=n {
        let log_p = binomial_log_p(n, i);
        if log_p <= observed_log_p + 1e-12 {
            p_value += log_p.exp();
        }
    }
    p_value.min(1.0)
}

fn log_factorial(n: usize) -> f64 {
    (1..=n).map(|i| (i as f64).ln()).sum()
}

fn binomial_log_p(n: usize, k: usize) -> f64 {
    let comb = log_factorial(n) - log_factorial(k) - log_factorial(n - k);
    comb - (n as f64) * 2.0f64.ln()
}

/// Simple percentile bootstrap 95% confidence interval for the mean difference.
fn bootstrap_mean_ci(deltas: &[f64], samples: usize) -> Option<(f64, f64)> {
    if deltas.is_empty() {
        return None;
    }

    let mut rng = rand::thread_rng();
    let n = deltas.len();
    let mut means = Vec::with_capacity(samples);
    for _ in 0..samples {
        let sum: f64 = (0..n)
            .map(|_| deltas[rand::Rng::gen_range(&mut rng, 0..n)])
            .sum();
        means.push(sum / n as f64);
    }
    means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let lower_idx = ((samples as f64) * 0.025).floor() as usize;
    let upper_idx = ((samples as f64) * 0.975).ceil() as usize;
    let lower = means[lower_idx.min(samples - 1)];
    let upper = means[upper_idx.min(samples - 1)];
    Some((lower, upper))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::BenchmarkResult;
    use crate::repository::MemoryBenchmarkResultRepository;
    use std::path::PathBuf;

    fn summary(id: &str) -> BenchmarkRunSummary {
        BenchmarkRunSummary {
            id: id.into(),
            name: format!("run-{id}"),
            golden_set_path: PathBuf::from("x.jsonl"),
            variant_name: "v".into(),
            pass_count: 0,
            fail_count: 0,
            total_cases: 0,
            pass_rate: 0.0,
            mean_latency_ms: 0.0,
            total_tokens: 0,
        }
    }

    fn result(run_id: &str, case_id: &str, passed: bool) -> BenchmarkResult {
        BenchmarkResult {
            id: format!("{run_id}-{case_id}"),
            run_id: run_id.into(),
            case_id: case_id.into(),
            case_input: serde_json::Value::Null,
            expected: None,
            actual: serde_json::Value::Null,
            passed,
            score_value: if passed { 1.0 } else { 0.0 },
            latency_ms: 0,
            token_usage: None,
            explanation: None,
            metadata: serde_json::Value::Null,
        }
    }

    #[tokio::test]
    async fn ab_test_detects_challenger_improvement() {
        let repo = MemoryBenchmarkResultRepository::new();
        let baseline = crate::models::BenchmarkRun {
            summary: summary("base"),
            project_id: None,
            variant: crate::models::BenchmarkVariant::default(),
            scorer_kind: "exact".into(),
            started_at: chrono::Utc::now(),
            finished_at: None,
            results: vec![],
            metadata: serde_json::Value::Null,
        };
        let challenger = crate::models::BenchmarkRun {
            summary: summary("chall"),
            project_id: None,
            variant: crate::models::BenchmarkVariant::default(),
            scorer_kind: "exact".into(),
            started_at: chrono::Utc::now(),
            finished_at: None,
            results: vec![],
            metadata: serde_json::Value::Null,
        };
        repo.insert_run(&baseline).await.unwrap();
        repo.insert_run(&challenger).await.unwrap();

        for i in 0..10 {
            repo.insert_result("base", &result("base", &format!("c{i}"), true))
                .await
                .unwrap();
        }
        // Challenger regresses 6 cases, with no compensating fixes -> significant baseline win.
        for i in 0..4 {
            repo.insert_result("chall", &result("chall", &format!("c{i}"), true))
                .await
                .unwrap();
        }
        for i in 4..10 {
            repo.insert_result("chall", &result("chall", &format!("c{i}"), false))
                .await
                .unwrap();
        }

        let ab = ABTest::new("base".into(), "chall".into()).with_significance(0.05);
        let report = ab.compare(&repo).await.unwrap();
        assert_eq!(report.winner, ABWinner::Baseline);
        assert!(report.mc_nemar_p_value < 0.05);
    }

    #[test]
    fn mc_nemar_p_value_edge_cases() {
        assert!((mc_nemar_exact_p_value(0, 0) - 1.0).abs() < 1e-9);
        assert!((mc_nemar_exact_p_value(6, 0) - 0.031_25).abs() < 1e-9);
    }
}
