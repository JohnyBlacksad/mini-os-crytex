//! Prompt evolution service.
//!
//! Manages a population of system prompt versions per agent, computes fitness
//! from recorded experiences, and applies deterministic mutation operators.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use ulid::Ulid;

use crate::models::PromptVersion;
use crate::persistence::{ExperienceRepository, PersistenceError, PromptVersionRepository};

/// Errors returned by [`PromptEvolutionService`].
#[derive(Debug, Error)]
pub enum PromptEvolutionError {
    #[error("persistence error: {0}")]
    Persistence(#[from] PersistenceError),
    #[error("prompt version not found: {0}")]
    VersionNotFound(String),
    #[error("no population for agent: {0}")]
    EmptyPopulation(String),
    #[error("no active baseline prompt for agent: {0}")]
    NoActiveBaseline(String),
    #[error("prompt benchmark failed: {0}")]
    BenchmarkFailed(String),
    #[error("prompt benchmark rejected: {0}")]
    BenchmarkRejected(String),
}

/// Input passed to a held-out prompt benchmark gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptBenchmarkRequest {
    pub agent: String,
    pub baseline_prompt_version_id: String,
    pub challenger_prompt_version_id: String,
}

/// Decision returned by a held-out prompt benchmark gate.
#[derive(Debug, Clone, PartialEq)]
pub struct PromptBenchmarkDecision {
    pub accepted: bool,
    pub reason: String,
    pub baseline_score: f64,
    pub challenger_score: f64,
    pub metadata: Value,
}

/// Stable status of a prompt version in the evolution lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptDecisionKind {
    Proposed,
    Promoted,
    Rejected,
    RolledBack,
}

impl PromptDecisionKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Proposed => "proposed",
            Self::Promoted => "promoted",
            Self::Rejected => "rejected",
            Self::RolledBack => "rolled_back",
        }
    }
}

/// Structured proposal returned by `crytex prompts propose`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromptProposal {
    pub agent: String,
    pub baseline_version_id: String,
    pub challenger: PromptVersion,
    pub operator: String,
    pub diagnostics: Value,
}

/// Auditable benchmark/promotion decision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromptEvolutionDecisionReport {
    pub agent: String,
    pub baseline_version_id: Option<String>,
    pub challenger_version_id: String,
    pub decision_kind: PromptDecisionKind,
    pub accepted: bool,
    pub reason: String,
    pub baseline_score: Option<f64>,
    pub challenger_score: Option<f64>,
    pub regression_passed: bool,
    pub diagnostics: Value,
}

/// Typed failure classes used before deciding whether prompts or LoRA should improve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptFailureKind {
    Schema,
    Format,
    Quality,
    Safety,
    ToolUse,
    Other,
}

/// First remediation owner for a failed task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureRoute {
    PromptEvolution,
    Lora,
    Critic,
}

/// Routes failures to the module that can improve them first.
pub struct PromptFailureRouter;

impl PromptFailureRouter {
    pub fn route(kind: PromptFailureKind) -> FailureRoute {
        match kind {
            PromptFailureKind::Schema | PromptFailureKind::Format => FailureRoute::PromptEvolution,
            PromptFailureKind::Quality => FailureRoute::Lora,
            PromptFailureKind::Safety | PromptFailureKind::ToolUse | PromptFailureKind::Other => {
                FailureRoute::Critic
            }
        }
    }
}

/// Evaluates a baseline prompt against a challenger on a held-out benchmark.
#[async_trait]
pub trait PromptBenchmarkGate: Send + Sync {
    async fn evaluate(
        &self,
        request: PromptBenchmarkRequest,
    ) -> Result<PromptBenchmarkDecision, PromptEvolutionError>;
}

/// Deterministic mutation operators for a system prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutationOperator {
    /// Rephrase the final instruction block.
    Rephrase,
    /// Append an extra constraint.
    AddConstraint,
    /// Inject a short JSON example.
    InjectExample,
    /// Change the tone directive.
    ChangeTone,
}

impl MutationOperator {
    /// All supported operators, in a stable order.
    pub fn all() -> &'static [MutationOperator] {
        &[
            MutationOperator::Rephrase,
            MutationOperator::AddConstraint,
            MutationOperator::InjectExample,
            MutationOperator::ChangeTone,
        ]
    }

    /// Apply the operator to a prompt string.
    pub fn apply(&self, prompt: &str) -> String {
        match self {
            MutationOperator::Rephrase => format!(
                "{}\n\n[Rephrased instruction] Express the same requirements using different wording while preserving every rule above.",
                prompt
            ),
            MutationOperator::AddConstraint => format!(
                "{}\n\n[Added constraint] Every response must begin with a one-sentence summary of the intended outcome.",
                prompt
            ),
            MutationOperator::InjectExample => format!(
                "{}\n\n[Injected example] Example of a valid response snippet: {{ \"summary\": \"...\", \"result\": \"...\" }}.",
                prompt
            ),
            MutationOperator::ChangeTone => format!("[Tone: precise and concise]\n\n{}", prompt),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Rephrase => "rephrase",
            Self::AddConstraint => "add-constraint",
            Self::InjectExample => "inject-example",
            Self::ChangeTone => "change-tone",
        }
    }
}

/// Evolves system prompts for agents based on reward feedback.
pub struct PromptEvolutionService<R, E> {
    prompt_repo: Arc<R>,
    experience_repo: Arc<E>,
}

impl<R, E> PromptEvolutionService<R, E>
where
    R: PromptVersionRepository,
    E: ExperienceRepository,
{
    pub fn new(prompt_repo: Arc<R>, experience_repo: Arc<E>) -> Self {
        Self {
            prompt_repo,
            experience_repo,
        }
    }

    /// Create the initial active version for an agent from a base prompt.
    pub async fn seed_agent(
        &self,
        agent: &str,
        base_prompt: &str,
    ) -> Result<PromptVersion, PromptEvolutionError> {
        let version = PromptVersion {
            id: Ulid::new().to_string(),
            agent: agent.to_string(),
            project_id: None,
            system_prompt: base_prompt.to_string(),
            fitness: None,
            parent_id: None,
            metrics: Value::Null,
            created_at: Utc::now().timestamp_millis(),
            active: true,
        };
        self.prompt_repo.insert_prompt_version(&version).await?;
        self.prompt_repo
            .set_active_prompt_version(&version.id, agent)
            .await?;
        Ok(version)
    }

    /// Create a mutated child of an existing prompt version.
    pub async fn mutate(
        &self,
        parent_id: &str,
        operator: MutationOperator,
    ) -> Result<PromptVersion, PromptEvolutionError> {
        let parent = self
            .prompt_repo
            .get_prompt_version(parent_id)
            .await?
            .ok_or_else(|| PromptEvolutionError::VersionNotFound(parent_id.to_string()))?;

        let child = PromptVersion {
            id: Ulid::new().to_string(),
            agent: parent.agent.clone(),
            project_id: parent.project_id.clone(),
            system_prompt: operator.apply(&parent.system_prompt),
            fitness: None,
            parent_id: Some(parent_id.to_string()),
            metrics: Value::Null,
            created_at: Utc::now().timestamp_millis(),
            active: false,
        };

        self.prompt_repo.insert_prompt_version(&child).await?;
        Ok(child)
    }

    /// Create a challenger from the active prompt. The challenger is never activated here.
    pub async fn propose(
        &self,
        agent: &str,
        operator: MutationOperator,
    ) -> Result<PromptProposal, PromptEvolutionError> {
        let baseline = self
            .prompt_repo
            .get_active_prompt_version(agent)
            .await?
            .ok_or_else(|| PromptEvolutionError::NoActiveBaseline(agent.to_string()))?;
        let mut challenger = self.mutate(&baseline.id, operator).await?;
        let diagnostics = prompt_decision_diagnostics(
            PromptDecisionKind::Proposed,
            agent,
            Some(&baseline.id),
            &challenger.id,
            "challenger created; active baseline retained",
            None,
        );
        challenger.metrics = merge_prompt_metrics(
            challenger.metrics,
            prompt_decision_metrics(
                PromptDecisionKind::Proposed,
                Some(&baseline.id),
                &challenger.id,
                false,
                "challenger",
                diagnostics.clone(),
            ),
        );
        self.prompt_repo.update_prompt_version(&challenger).await?;

        Ok(PromptProposal {
            agent: agent.to_string(),
            baseline_version_id: baseline.id,
            challenger,
            operator: operator.as_str().to_string(),
            diagnostics,
        })
    }

    /// Run one evolution step: tournament select, mutate, persist.
    pub async fn evolve_step(
        &self,
        agent: &str,
        operator: MutationOperator,
        tournament_size: usize,
        rng: &mut impl RngCore,
    ) -> Result<PromptVersion, PromptEvolutionError> {
        let parent = self
            .tournament_select(agent, tournament_size, rng)
            .await?
            .ok_or_else(|| PromptEvolutionError::EmptyPopulation(agent.to_string()))?;
        self.mutate(&parent.id, operator).await
    }

    /// Tournament selection over versions with known fitness.
    /// Versions without fitness are treated as fitness 0.0.
    pub async fn tournament_select(
        &self,
        agent: &str,
        tournament_size: usize,
        rng: &mut impl RngCore,
    ) -> Result<Option<PromptVersion>, PromptEvolutionError> {
        let population = self
            .prompt_repo
            .list_prompt_versions_by_agent(agent)
            .await?;
        if population.is_empty() {
            return Ok(None);
        }

        let size = tournament_size.min(population.len()).max(1);
        let mut selected: Vec<&PromptVersion> = Vec::with_capacity(size);
        while selected.len() < size {
            let idx = (rng.next_u32() as usize) % population.len();
            let candidate = &population[idx];
            if !selected.contains(&candidate) {
                selected.push(candidate);
            }
        }

        Ok(selected
            .into_iter()
            .max_by(|a, b| {
                let af = a.fitness.unwrap_or(0.0);
                let bf = b.fitness.unwrap_or(0.0);
                af.partial_cmp(&bf).unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned())
    }

    /// Recompute the average reward for a prompt version and persist it.
    pub async fn recompute_fitness(&self, version_id: &str) -> Result<f64, PromptEvolutionError> {
        let experiences = self
            .experience_repo
            .list_experiences_by_prompt_version(version_id)
            .await?;
        let fitness = if experiences.is_empty() {
            0.0
        } else {
            experiences.iter().map(|e| e.reward).sum::<f64>() / experiences.len() as f64
        };

        let mut version = self
            .prompt_repo
            .get_prompt_version(version_id)
            .await?
            .ok_or_else(|| PromptEvolutionError::VersionNotFound(version_id.to_string()))?;
        version.fitness = Some(fitness);
        self.prompt_repo.update_prompt_version(&version).await?;
        Ok(fitness)
    }

    /// Activate a prompt version and deactivate all other versions for the same agent.
    pub async fn activate(&self, version_id: &str) -> Result<(), PromptEvolutionError> {
        let version = self
            .prompt_repo
            .get_prompt_version(version_id)
            .await?
            .ok_or_else(|| PromptEvolutionError::VersionNotFound(version_id.to_string()))?;
        self.prompt_repo
            .set_active_prompt_version(version_id, &version.agent)
            .await?;
        Ok(())
    }

    /// Evaluate a challenger against the active baseline and promote it only on held-out improvement.
    pub async fn evaluate_challenger_with_benchmark(
        &self,
        challenger_id: &str,
        gate: &dyn PromptBenchmarkGate,
    ) -> Result<PromptBenchmarkDecision, PromptEvolutionError> {
        let report = self.benchmark_challenger(challenger_id, gate).await?;
        Ok(PromptBenchmarkDecision {
            accepted: report.accepted,
            reason: report.reason,
            baseline_score: report.baseline_score.unwrap_or_default(),
            challenger_score: report.challenger_score.unwrap_or_default(),
            metadata: report.diagnostics,
        })
    }

    /// Evaluate a challenger and activate it only if held-out and regression gates pass.
    pub async fn benchmark_challenger(
        &self,
        challenger_id: &str,
        gate: &dyn PromptBenchmarkGate,
    ) -> Result<PromptEvolutionDecisionReport, PromptEvolutionError> {
        let challenger = self
            .prompt_repo
            .get_prompt_version(challenger_id)
            .await?
            .ok_or_else(|| PromptEvolutionError::VersionNotFound(challenger_id.to_string()))?;
        let baseline = self
            .prompt_repo
            .get_active_prompt_version(&challenger.agent)
            .await?
            .filter(|version| version.id != challenger.id)
            .ok_or_else(|| PromptEvolutionError::NoActiveBaseline(challenger.agent.clone()))?;

        let gate_decision = gate
            .evaluate(PromptBenchmarkRequest {
                agent: challenger.agent.clone(),
                baseline_prompt_version_id: baseline.id.clone(),
                challenger_prompt_version_id: challenger.id.clone(),
            })
            .await?;
        let decision = enforce_regression_policy(enforce_improvement_policy(gate_decision));
        let kind = if decision.accepted {
            PromptDecisionKind::Promoted
        } else {
            PromptDecisionKind::Rejected
        };
        let regression_passed = regression_passed(&decision.metadata);
        let diagnostics = prompt_decision_diagnostics(
            kind,
            &challenger.agent,
            Some(&baseline.id),
            &challenger.id,
            &decision.reason,
            Some(&decision.metadata),
        );

        let mut evaluated_challenger = challenger;
        evaluated_challenger.fitness = Some(decision.challenger_score);
        evaluated_challenger.active = false;
        evaluated_challenger.metrics = merge_prompt_metrics(
            evaluated_challenger.metrics,
            merge_prompt_metrics(
                prompt_benchmark_metrics(&decision),
                prompt_decision_metrics(
                    kind,
                    Some(&baseline.id),
                    &evaluated_challenger.id,
                    regression_passed,
                    kind.as_str(),
                    diagnostics.clone(),
                ),
            ),
        );
        self.prompt_repo
            .update_prompt_version(&evaluated_challenger)
            .await?;

        if decision.accepted {
            self.prompt_repo
                .set_active_prompt_version(&evaluated_challenger.id, &evaluated_challenger.agent)
                .await?;
        } else {
            self.prompt_repo
                .set_active_prompt_version(&baseline.id, &baseline.agent)
                .await?;
        }

        Ok(PromptEvolutionDecisionReport {
            agent: evaluated_challenger.agent,
            baseline_version_id: Some(baseline.id),
            challenger_version_id: evaluated_challenger.id,
            decision_kind: kind,
            accepted: decision.accepted,
            reason: decision.reason,
            baseline_score: Some(decision.baseline_score),
            challenger_score: Some(decision.challenger_score),
            regression_passed,
            diagnostics,
        })
    }

    /// Promote a version only if a previous benchmark decision accepted it.
    pub async fn promote(
        &self,
        agent: &str,
        version_id: &str,
    ) -> Result<PromptEvolutionDecisionReport, PromptEvolutionError> {
        let version = self
            .prompt_repo
            .get_prompt_version(version_id)
            .await?
            .ok_or_else(|| PromptEvolutionError::VersionNotFound(version_id.to_string()))?;
        if version.agent != agent {
            return Err(PromptEvolutionError::BenchmarkRejected(format!(
                "prompt {version_id} belongs to agent {}, not {agent}",
                version.agent
            )));
        }
        if version.metrics["prompt_decision"]["decision"] != "promoted" {
            return Err(PromptEvolutionError::BenchmarkRejected(
                "promotion requires an accepted benchmark decision".into(),
            ));
        }
        self.prompt_repo
            .set_active_prompt_version(version_id, agent)
            .await?;
        Ok(PromptEvolutionDecisionReport {
            agent: agent.to_string(),
            baseline_version_id: version
                .metrics
                .get("prompt_decision")
                .and_then(|v| v.get("baseline_version_id"))
                .and_then(Value::as_str)
                .map(ToString::to_string),
            challenger_version_id: version_id.to_string(),
            decision_kind: PromptDecisionKind::Promoted,
            accepted: true,
            reason: "previous benchmark decision accepted this prompt".into(),
            baseline_score: version
                .metrics
                .get("prompt_benchmark_gate")
                .and_then(|v| v.get("baseline_score"))
                .and_then(Value::as_f64),
            challenger_score: version
                .metrics
                .get("prompt_benchmark_gate")
                .and_then(|v| v.get("challenger_score"))
                .and_then(Value::as_f64),
            regression_passed: version.metrics["prompt_decision"]["regression_passed"]
                .as_bool()
                .unwrap_or(false),
            diagnostics: version.metrics["prompt_decision"].clone(),
        })
    }

    /// Roll back an agent to an earlier prompt version.
    pub async fn rollback(
        &self,
        agent: &str,
        target_version_id: &str,
    ) -> Result<PromptEvolutionDecisionReport, PromptEvolutionError> {
        let previous = self.prompt_repo.get_active_prompt_version(agent).await?;
        let mut target = self
            .prompt_repo
            .get_prompt_version(target_version_id)
            .await?
            .ok_or_else(|| PromptEvolutionError::VersionNotFound(target_version_id.to_string()))?;
        if target.agent != agent {
            return Err(PromptEvolutionError::VersionNotFound(format!(
                "{target_version_id} for agent {agent}"
            )));
        }
        let mut diagnostics = prompt_decision_diagnostics(
            PromptDecisionKind::RolledBack,
            agent,
            previous.as_ref().map(|version| version.id.as_str()),
            target_version_id,
            "active prompt rolled back by operator",
            None,
        );
        diagnostics["regression_passed"] = Value::Bool(true);
        target.metrics = merge_prompt_metrics(
            target.metrics,
            prompt_decision_metrics(
                PromptDecisionKind::RolledBack,
                previous.as_ref().map(|version| version.id.as_str()),
                target_version_id,
                true,
                PromptDecisionKind::RolledBack.as_str(),
                diagnostics.clone(),
            ),
        );
        self.prompt_repo.update_prompt_version(&target).await?;
        self.prompt_repo
            .set_active_prompt_version(target_version_id, agent)
            .await?;
        Ok(PromptEvolutionDecisionReport {
            agent: agent.to_string(),
            baseline_version_id: previous.map(|version| version.id),
            challenger_version_id: target_version_id.to_string(),
            decision_kind: PromptDecisionKind::RolledBack,
            accepted: true,
            reason: "active prompt rolled back by operator".into(),
            baseline_score: None,
            challenger_score: target.fitness,
            regression_passed: true,
            diagnostics,
        })
    }

    /// Return the active version for an agent, if any.
    pub async fn active_version(
        &self,
        agent: &str,
    ) -> Result<Option<PromptVersion>, PromptEvolutionError> {
        Ok(self.prompt_repo.get_active_prompt_version(agent).await?)
    }

    /// Return the system prompt of the active version for an agent.
    pub async fn system_prompt_for_agent(
        &self,
        agent: &str,
    ) -> Result<Option<String>, PromptEvolutionError> {
        Ok(self.active_version(agent).await?.map(|v| v.system_prompt))
    }

    /// List all versions for an agent.
    pub async fn list_versions(
        &self,
        agent: &str,
    ) -> Result<Vec<PromptVersion>, PromptEvolutionError> {
        Ok(self
            .prompt_repo
            .list_prompt_versions_by_agent(agent)
            .await?)
    }
}

fn enforce_improvement_policy(decision: PromptBenchmarkDecision) -> PromptBenchmarkDecision {
    if decision.accepted && decision.challenger_score > decision.baseline_score {
        return decision;
    }

    PromptBenchmarkDecision {
        accepted: false,
        reason: if decision.challenger_score <= decision.baseline_score {
            format!(
                "{}; no held-out improvement: challenger_score={:.4}, baseline_score={:.4}",
                decision.reason, decision.challenger_score, decision.baseline_score
            )
        } else {
            decision.reason
        },
        ..decision
    }
}

fn enforce_regression_policy(decision: PromptBenchmarkDecision) -> PromptBenchmarkDecision {
    if regression_passed(&decision.metadata) {
        return decision;
    }

    PromptBenchmarkDecision {
        accepted: false,
        reason: format!(
            "{}; regression benchmark is required and must pass",
            decision.reason
        ),
        ..decision
    }
}

fn regression_passed(metadata: &Value) -> bool {
    metadata
        .get("regression")
        .and_then(|value| value.get("passed"))
        .and_then(Value::as_bool)
        == Some(true)
}

fn prompt_benchmark_metrics(decision: &PromptBenchmarkDecision) -> Value {
    let mut metadata = match decision.metadata.clone() {
        Value::Object(map) => map,
        _ => serde_json::Map::new(),
    };
    metadata.insert("accepted".into(), Value::Bool(decision.accepted));
    metadata.insert("reason".into(), Value::String(decision.reason.clone()));
    metadata.insert(
        "baseline_score".into(),
        Value::from(decision.baseline_score),
    );
    metadata.insert(
        "challenger_score".into(),
        Value::from(decision.challenger_score),
    );
    serde_json::json!({ "prompt_benchmark_gate": Value::Object(metadata) })
}

fn prompt_decision_metrics(
    kind: PromptDecisionKind,
    baseline_version_id: Option<&str>,
    challenger_version_id: &str,
    regression_passed: bool,
    stage: &str,
    diagnostics: Value,
) -> Value {
    serde_json::json!({
        "prompt_decision": {
            "decision": kind.as_str(),
            "stage": stage,
            "baseline_version_id": baseline_version_id,
            "challenger_version_id": challenger_version_id,
            "regression_passed": regression_passed,
            "diagnostics": diagnostics
        }
    })
}

fn prompt_decision_diagnostics(
    kind: PromptDecisionKind,
    agent: &str,
    baseline_version_id: Option<&str>,
    challenger_version_id: &str,
    reason: &str,
    gate_metadata: Option<&Value>,
) -> Value {
    serde_json::json!({
        "kind": "prompt_evolution_decision",
        "decision": kind.as_str(),
        "agent": agent,
        "baseline_version_id": baseline_version_id,
        "challenger_version_id": challenger_version_id,
        "reason": reason,
        "regression_passed": gate_metadata.map(regression_passed).unwrap_or(false),
        "gate_metadata": gate_metadata.cloned().unwrap_or(Value::Null),
        "timestamp": Utc::now().timestamp_millis()
    })
}

fn merge_prompt_metrics(left: Value, right: Value) -> Value {
    let mut merged = match left {
        Value::Object(map) => map,
        _ => serde_json::Map::new(),
    };
    if let Value::Object(map) = right {
        merged.extend(map);
    }
    Value::Object(merged)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use async_trait::async_trait;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;
    use crate::models::Experience;

    #[derive(Default)]
    struct TestRepo {
        versions: Mutex<HashMap<String, PromptVersion>>,
        experiences: Mutex<HashMap<String, Vec<Experience>>>,
    }

    #[async_trait]
    impl PromptVersionRepository for TestRepo {
        async fn insert_prompt_version(
            &self,
            version: &PromptVersion,
        ) -> Result<(), PersistenceError> {
            self.versions
                .lock()
                .unwrap()
                .insert(version.id.clone(), version.clone());
            Ok(())
        }
        async fn update_prompt_version(
            &self,
            version: &PromptVersion,
        ) -> Result<(), PersistenceError> {
            self.versions
                .lock()
                .unwrap()
                .insert(version.id.clone(), version.clone());
            Ok(())
        }
        async fn get_prompt_version(
            &self,
            id: &str,
        ) -> Result<Option<PromptVersion>, PersistenceError> {
            Ok(self.versions.lock().unwrap().get(id).cloned())
        }
        async fn list_prompt_versions_by_agent(
            &self,
            agent: &str,
        ) -> Result<Vec<PromptVersion>, PersistenceError> {
            Ok(self
                .versions
                .lock()
                .unwrap()
                .values()
                .filter(|v| v.agent == agent)
                .cloned()
                .collect())
        }
        async fn get_active_prompt_version(
            &self,
            agent: &str,
        ) -> Result<Option<PromptVersion>, PersistenceError> {
            Ok(self
                .versions
                .lock()
                .unwrap()
                .values()
                .find(|v| v.agent == agent && v.active)
                .cloned())
        }
        async fn set_active_prompt_version(
            &self,
            id: &str,
            agent: &str,
        ) -> Result<(), PersistenceError> {
            let mut versions = self.versions.lock().unwrap();
            for v in versions.values_mut() {
                if v.agent == agent {
                    v.active = false;
                }
            }
            if let Some(v) = versions.get_mut(id) {
                v.active = true;
            }
            Ok(())
        }
    }

    #[async_trait]
    impl ExperienceRepository for TestRepo {
        async fn insert_experience(&self, exp: &Experience) -> Result<(), PersistenceError> {
            self.experiences
                .lock()
                .unwrap()
                .entry(exp.task_id.clone())
                .or_default()
                .push(exp.clone());
            Ok(())
        }
        async fn list_experiences_by_task(
            &self,
            task_id: &str,
        ) -> Result<Vec<Experience>, PersistenceError> {
            Ok(self
                .experiences
                .lock()
                .unwrap()
                .get(task_id)
                .cloned()
                .unwrap_or_default())
        }
        async fn list_experiences_by_prompt_version(
            &self,
            prompt_version_id: &str,
        ) -> Result<Vec<Experience>, PersistenceError> {
            Ok(self
                .experiences
                .lock()
                .unwrap()
                .values()
                .flat_map(|v| v.iter())
                .filter(|e| e.prompt_version_id.as_deref() == Some(prompt_version_id))
                .cloned()
                .collect())
        }
    }

    fn make_service() -> PromptEvolutionService<TestRepo, TestRepo> {
        let repo = Arc::new(TestRepo::default());
        PromptEvolutionService::new(repo.clone(), repo)
    }

    #[derive(Clone)]
    struct StaticPromptGate {
        decision: PromptBenchmarkDecision,
    }

    #[async_trait]
    impl PromptBenchmarkGate for StaticPromptGate {
        async fn evaluate(
            &self,
            request: PromptBenchmarkRequest,
        ) -> Result<PromptBenchmarkDecision, PromptEvolutionError> {
            assert_ne!(
                request.baseline_prompt_version_id,
                request.challenger_prompt_version_id
            );
            assert_eq!(request.agent, "coder");
            Ok(self.decision.clone())
        }
    }

    fn prompt_decision(
        accepted: bool,
        baseline_score: f64,
        challenger_score: f64,
    ) -> PromptBenchmarkDecision {
        PromptBenchmarkDecision {
            accepted,
            reason: "held-out benchmark decision".into(),
            baseline_score,
            challenger_score,
            metadata: serde_json::json!({
                "held_out": true,
                "baseline_run_id": "prompt-baseline-run",
                "challenger_run_id": "prompt-challenger-run",
                "baseline_pass_rate": baseline_score,
                "challenger_pass_rate": challenger_score
            }),
        }
    }

    fn prompt_decision_with_regression(
        accepted: bool,
        baseline_score: f64,
        challenger_score: f64,
        regression_passed: bool,
    ) -> PromptBenchmarkDecision {
        let mut decision = prompt_decision(accepted, baseline_score, challenger_score);
        decision.metadata["regression"] = serde_json::json!({
            "suite_id": "prompt-regression-suite",
            "passed": regression_passed,
            "required": true,
            "case_count": 4
        });
        decision
    }

    #[tokio::test]
    async fn seed_creates_initial_active_version() {
        let service = make_service();
        let v = service.seed_agent("coder", "base prompt").await.unwrap();

        assert_eq!(v.agent, "coder");
        assert_eq!(v.system_prompt, "base prompt");
        assert!(v.active);
        assert!(v.parent_id.is_none());

        let active = service.active_version("coder").await.unwrap();
        assert_eq!(active.map(|a| a.id), Some(v.id));
    }

    #[tokio::test]
    async fn mutate_produces_child_with_changed_text_and_parent() {
        let service = make_service();
        let parent = service.seed_agent("coder", "base prompt").await.unwrap();
        let child = service
            .mutate(&parent.id, MutationOperator::Rephrase)
            .await
            .unwrap();

        assert_eq!(child.parent_id, Some(parent.id));
        assert_eq!(child.agent, "coder");
        assert_ne!(child.system_prompt, parent.system_prompt);
        assert!(!child.active);
    }

    #[tokio::test]
    async fn tournament_select_returns_highest_fitness_in_sample() {
        let service = make_service();
        let base = service.seed_agent("coder", "base").await.unwrap();
        let mut versions = vec![base.clone()];
        for (i, op) in MutationOperator::all().iter().enumerate() {
            let child = service.mutate(&base.id, *op).await.unwrap();
            let mut child = child;
            child.fitness = Some((i + 1) as f64);
            service
                .prompt_repo
                .update_prompt_version(&child)
                .await
                .unwrap();
            versions.push(child);
        }

        // Set base fitness low and make the tournament include the whole population
        // so the winner is deterministically the globally fittest version.
        let mut base_low = base;
        base_low.fitness = Some(0.0);
        service
            .prompt_repo
            .update_prompt_version(&base_low)
            .await
            .unwrap();

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let winner = service
            .tournament_select("coder", versions.len(), &mut rng)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(winner.fitness, Some(4.0));
    }

    #[tokio::test]
    async fn recompute_fitness_averages_rewards() {
        let service = make_service();
        let version = service.seed_agent("coder", "base").await.unwrap();

        for reward in [3.0, 5.0] {
            let exp = Experience {
                id: Ulid::new().to_string(),
                task_id: Ulid::new().to_string(),
                project_id: None,
                prompt_version_id: Some(version.id.clone()),
                text: None,
                critic_score: None,
                human_score: None,
                reward,
                comment: None,
                created_at: 0,
            };
            service
                .experience_repo
                .insert_experience(&exp)
                .await
                .unwrap();
        }

        let fitness = service.recompute_fitness(&version.id).await.unwrap();
        assert!((fitness - 4.0).abs() < 0.001);

        let updated = service
            .prompt_repo
            .get_prompt_version(&version.id)
            .await
            .unwrap()
            .unwrap();
        assert!((updated.fitness.unwrap() - 4.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn activate_makes_only_one_active_per_agent() {
        let service = make_service();
        let v1 = service.seed_agent("coder", "base").await.unwrap();
        let v2 = service
            .mutate(&v1.id, MutationOperator::ChangeTone)
            .await
            .unwrap();

        service.activate(&v2.id).await.unwrap();

        let active = service.active_version("coder").await.unwrap().unwrap();
        assert_eq!(active.id, v2.id);

        let v1_loaded = service
            .prompt_repo
            .get_prompt_version(&v1.id)
            .await
            .unwrap()
            .unwrap();
        assert!(!v1_loaded.active);
    }

    #[tokio::test]
    async fn system_prompt_for_agent_returns_active_override() {
        let service = make_service();
        service.seed_agent("coder", "active prompt").await.unwrap();
        let prompt = service.system_prompt_for_agent("coder").await.unwrap();
        assert_eq!(prompt, Some("active prompt".to_string()));
    }

    #[tokio::test]
    async fn benchmark_gate_promotes_challenger_that_improves_on_held_out_set() {
        let service = make_service();
        let baseline = service.seed_agent("coder", "base").await.unwrap();
        let challenger = service
            .mutate(&baseline.id, MutationOperator::AddConstraint)
            .await
            .unwrap();
        let gate = StaticPromptGate {
            decision: prompt_decision_with_regression(true, 0.40, 0.75, true),
        };

        let decision = service
            .evaluate_challenger_with_benchmark(&challenger.id, &gate)
            .await
            .unwrap();
        let active = service.active_version("coder").await.unwrap().unwrap();
        let stored_challenger = service
            .prompt_repo
            .get_prompt_version(&challenger.id)
            .await
            .unwrap()
            .unwrap();

        assert!(decision.accepted, "{}", decision.reason);
        assert_eq!(active.id, challenger.id);
        assert_eq!(stored_challenger.fitness, Some(0.75));
        assert_eq!(
            stored_challenger.metrics["prompt_benchmark_gate"]["held_out"],
            serde_json::Value::Bool(true)
        );
    }

    #[tokio::test]
    async fn benchmark_gate_rejects_challenger_and_keeps_baseline_active_on_degradation() {
        let service = make_service();
        let baseline = service.seed_agent("coder", "base").await.unwrap();
        let challenger = service
            .mutate(&baseline.id, MutationOperator::ChangeTone)
            .await
            .unwrap();
        let gate = StaticPromptGate {
            decision: prompt_decision_with_regression(false, 0.80, 0.25, true),
        };

        let decision = service
            .evaluate_challenger_with_benchmark(&challenger.id, &gate)
            .await
            .unwrap();
        let active = service.active_version("coder").await.unwrap().unwrap();
        let stored_challenger = service
            .prompt_repo
            .get_prompt_version(&challenger.id)
            .await
            .unwrap()
            .unwrap();

        assert!(!decision.accepted);
        assert_eq!(active.id, baseline.id);
        assert!(!stored_challenger.active);
        assert_eq!(
            stored_challenger.metrics["prompt_benchmark_gate"]["accepted"],
            serde_json::Value::Bool(false)
        );
    }

    #[tokio::test]
    async fn benchmark_gate_refuses_promotion_when_challenger_does_not_improve() {
        let service = make_service();
        let baseline = service.seed_agent("coder", "base").await.unwrap();
        let challenger = service
            .mutate(&baseline.id, MutationOperator::InjectExample)
            .await
            .unwrap();
        let gate = StaticPromptGate {
            decision: prompt_decision_with_regression(true, 0.70, 0.70, true),
        };

        let decision = service
            .evaluate_challenger_with_benchmark(&challenger.id, &gate)
            .await
            .unwrap();
        let active = service.active_version("coder").await.unwrap().unwrap();

        assert!(!decision.accepted);
        assert_eq!(active.id, baseline.id);
        assert!(decision.reason.contains("no held-out improvement"));
    }

    #[tokio::test]
    async fn propose_creates_inactive_challenger_from_active_prompt() {
        let service = make_service();
        let baseline = service.seed_agent("coder", "base prompt").await.unwrap();

        let proposal = service
            .propose("coder", MutationOperator::AddConstraint)
            .await
            .unwrap();
        let active = service.active_version("coder").await.unwrap().unwrap();

        assert_eq!(proposal.baseline_version_id, baseline.id);
        assert_eq!(proposal.challenger.agent, "coder");
        assert_eq!(proposal.challenger.parent_id, Some(baseline.id.clone()));
        assert!(!proposal.challenger.active);
        assert_eq!(active.id, baseline.id);
        assert_eq!(
            proposal.challenger.metrics["prompt_decision"]["stage"],
            "challenger"
        );
    }

    #[tokio::test]
    async fn benchmark_rejects_challenger_when_regression_suite_is_missing() {
        let service = make_service();
        let baseline = service.seed_agent("coder", "base").await.unwrap();
        let challenger = service
            .mutate(&baseline.id, MutationOperator::Rephrase)
            .await
            .unwrap();
        let gate = StaticPromptGate {
            decision: prompt_decision(true, 0.40, 0.90),
        };

        let decision = service
            .benchmark_challenger(&challenger.id, &gate)
            .await
            .unwrap();
        let active = service.active_version("coder").await.unwrap().unwrap();

        assert!(!decision.accepted);
        assert_eq!(active.id, baseline.id);
        assert_eq!(decision.decision_kind, PromptDecisionKind::Rejected);
        assert!(decision.reason.contains("regression benchmark is required"));
    }

    #[tokio::test]
    async fn benchmark_promotes_only_after_regression_gate_passes_and_records_diagnostics() {
        let service = make_service();
        let baseline = service.seed_agent("coder", "base").await.unwrap();
        let challenger = service
            .mutate(&baseline.id, MutationOperator::InjectExample)
            .await
            .unwrap();
        let gate = StaticPromptGate {
            decision: prompt_decision_with_regression(true, 0.40, 0.85, true),
        };

        let decision = service
            .benchmark_challenger(&challenger.id, &gate)
            .await
            .unwrap();
        let active = service.active_version("coder").await.unwrap().unwrap();
        let stored = service
            .prompt_repo
            .get_prompt_version(&challenger.id)
            .await
            .unwrap()
            .unwrap();

        assert!(decision.accepted);
        assert_eq!(active.id, challenger.id);
        assert_eq!(decision.decision_kind, PromptDecisionKind::Promoted);
        assert_eq!(decision.diagnostics["decision"], "promoted");
        assert_eq!(
            stored.metrics["prompt_decision"]["baseline_version_id"],
            baseline.id
        );
        assert_eq!(
            stored.metrics["prompt_decision"]["regression_passed"],
            serde_json::Value::Bool(true)
        );
    }

    #[tokio::test]
    async fn rollback_reactivates_target_baseline_and_records_decision() {
        let service = make_service();
        let baseline = service.seed_agent("coder", "base").await.unwrap();
        let challenger = service
            .mutate(&baseline.id, MutationOperator::ChangeTone)
            .await
            .unwrap();
        let gate = StaticPromptGate {
            decision: prompt_decision_with_regression(true, 0.40, 0.85, true),
        };
        service
            .benchmark_challenger(&challenger.id, &gate)
            .await
            .unwrap();

        let decision = service.rollback("coder", &baseline.id).await.unwrap();
        let active = service.active_version("coder").await.unwrap().unwrap();
        let stored_baseline = service
            .prompt_repo
            .get_prompt_version(&baseline.id)
            .await
            .unwrap()
            .unwrap();

        assert_eq!(active.id, baseline.id);
        assert_eq!(decision.decision_kind, PromptDecisionKind::RolledBack);
        assert_eq!(
            decision.diagnostics["regression_passed"],
            serde_json::Value::Bool(true)
        );
        assert_eq!(
            stored_baseline.metrics["prompt_decision"]["decision"],
            "rolled_back"
        );
    }

    #[test]
    fn schema_and_format_failures_route_to_prompt_evolution_before_lora() {
        assert_eq!(
            PromptFailureRouter::route(PromptFailureKind::Schema),
            FailureRoute::PromptEvolution
        );
        assert_eq!(
            PromptFailureRouter::route(PromptFailureKind::Format),
            FailureRoute::PromptEvolution
        );
        assert_eq!(
            PromptFailureRouter::route(PromptFailureKind::Quality),
            FailureRoute::Lora
        );
    }
}
