//! Autonomous evolution routing policy.
//!
//! The policy attributes failures before choosing an improvement module. This
//! prevents Crytex from training LoRA on bad context, schema mistakes, weak
//! critic feedback, or security-policy gaps.

use async_trait::async_trait;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

use crate::bus::Event;
use crate::services::EventService;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EvolutionRole {
    Orchestrator,
    Architect,
    CoderPython,
    CoderRust,
    CoderTs,
    Analyst,
    Researcher,
    Qa,
    Devops,
    Security,
    CriticAnalyst,
    CriticCoder,
    CriticResearcher,
    Summarizer,
}

impl EvolutionRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Orchestrator => "orchestrator",
            Self::Architect => "architect",
            Self::CoderPython => "coder-python",
            Self::CoderRust => "coder-rust",
            Self::CoderTs => "coder-ts",
            Self::Analyst => "analyst",
            Self::Researcher => "researcher",
            Self::Qa => "qa",
            Self::Devops => "devops",
            Self::Security => "security",
            Self::CriticAnalyst => "critic-analyst",
            Self::CriticCoder => "critic-coder",
            Self::CriticResearcher => "critic-researcher",
            Self::Summarizer => "summarizer",
        }
    }

    pub fn all() -> &'static [Self] {
        &[
            Self::Orchestrator,
            Self::Architect,
            Self::CoderPython,
            Self::CoderRust,
            Self::CoderTs,
            Self::Analyst,
            Self::Researcher,
            Self::Qa,
            Self::Devops,
            Self::Security,
            Self::CriticAnalyst,
            Self::CriticCoder,
            Self::CriticResearcher,
            Self::Summarizer,
        ]
    }
}

impl std::str::FromStr for EvolutionRole {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::all()
            .iter()
            .copied()
            .find(|role| role.as_str() == value)
            .ok_or_else(|| format!("unknown evolution role: {value}"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvolutionFailureKind {
    BadContext,
    MissingContext,
    PromptInjectionContext,
    Schema,
    Format,
    RepeatedRoleSkillFailure,
    WeakCriticFeedback,
    SecurityPolicyGap,
    BenchmarkCoverageGap,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvolutionAction {
    RagFix,
    PromptEvolution,
    LoraTraining,
    CriticRoleEvolution,
    SecurityPolicy,
    BenchmarkExpansion,
}

impl EvolutionAction {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RagFix => "rag_fix",
            Self::PromptEvolution => "prompt_evolution",
            Self::LoraTraining => "lora_training",
            Self::CriticRoleEvolution => "critic_role_evolution",
            Self::SecurityPolicy => "security_policy",
            Self::BenchmarkExpansion => "benchmark_expansion",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvolutionObservation {
    pub role: EvolutionRole,
    pub failure_kind: EvolutionFailureKind,
    pub task_id: Option<String>,
    pub trace_id: String,
    pub evidence: Value,
    #[serde(default)]
    pub repeated_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvolutionDecision {
    pub role: EvolutionRole,
    pub action: EvolutionAction,
    pub reason: String,
    pub task_id: Option<String>,
    pub trace_id: String,
    pub diagnostics: Value,
}

impl EvolutionDecision {
    pub fn new(
        observation: &EvolutionObservation,
        action: EvolutionAction,
        reason: impl Into<String>,
    ) -> Self {
        let reason = reason.into();
        Self {
            role: observation.role,
            action,
            reason: reason.clone(),
            task_id: observation.task_id.clone(),
            trace_id: observation.trace_id.clone(),
            diagnostics: serde_json::json!({
                "kind": "autonomous_evolution_decision",
                "role": observation.role.as_str(),
                "failure_kind": observation.failure_kind,
                "action": action,
                "reason": reason,
                "repeated_count": observation.repeated_count,
                "source_evidence": observation.evidence
            }),
        }
    }
}

pub struct AutonomousEvolutionPolicy;

impl AutonomousEvolutionPolicy {
    pub fn decide(observation: &EvolutionObservation) -> EvolutionDecision {
        match observation.failure_kind {
            EvolutionFailureKind::BadContext
            | EvolutionFailureKind::MissingContext
            | EvolutionFailureKind::PromptInjectionContext => EvolutionDecision::new(
                observation,
                EvolutionAction::RagFix,
                "failure is attributed to retrieved/project context, so LoRA training is blocked",
            ),
            EvolutionFailureKind::Schema | EvolutionFailureKind::Format => EvolutionDecision::new(
                observation,
                EvolutionAction::PromptEvolution,
                "schema/format failure is routed to prompt evolution before LoRA",
            ),
            EvolutionFailureKind::RepeatedRoleSkillFailure => EvolutionDecision::new(
                observation,
                EvolutionAction::LoraTraining,
                "repeated role skill failure is attributed to model behavior",
            ),
            EvolutionFailureKind::WeakCriticFeedback => EvolutionDecision::new(
                observation,
                EvolutionAction::CriticRoleEvolution,
                "critic feedback lacks actionable detail, so the critic role evolves first",
            ),
            EvolutionFailureKind::SecurityPolicyGap => EvolutionDecision::new(
                observation,
                EvolutionAction::SecurityPolicy,
                "security/tool-use failure is routed to policy before model training",
            ),
            EvolutionFailureKind::BenchmarkCoverageGap | EvolutionFailureKind::Unknown => {
                EvolutionDecision::new(
                    observation,
                    EvolutionAction::BenchmarkExpansion,
                    "failure attribution is under-specified, so benchmarks expand before training",
                )
            }
        }
    }
}

#[async_trait]
pub trait EvolutionObservationSource: Send + Sync {
    async fn observations(&self, all_roles: bool) -> Vec<EvolutionObservation>;
}

pub struct StaticEvolutionObservationSource {
    observations: Vec<EvolutionObservation>,
}

impl StaticEvolutionObservationSource {
    pub fn new(observations: Vec<EvolutionObservation>) -> Self {
        Self { observations }
    }
}

#[async_trait]
impl EvolutionObservationSource for StaticEvolutionObservationSource {
    async fn observations(&self, all_roles: bool) -> Vec<EvolutionObservation> {
        if all_roles {
            return self.observations.clone();
        }
        self.observations.iter().take(1).cloned().collect()
    }
}

pub struct AutonomousEvolutionService {
    source: Box<dyn EvolutionObservationSource>,
    events: Arc<dyn EventService>,
}

impl AutonomousEvolutionService {
    pub fn new(source: Box<dyn EvolutionObservationSource>, events: Arc<dyn EventService>) -> Self {
        Self { source, events }
    }

    pub async fn run(&self, all_roles: bool) -> Vec<EvolutionDecision> {
        let decisions = self
            .source
            .observations(all_roles)
            .await
            .iter()
            .map(AutonomousEvolutionPolicy::decide)
            .collect::<Vec<_>>();
        for decision in &decisions {
            self.events.publish(Event::RunObserved {
                project_id: String::new(),
                task_id: decision.task_id.clone(),
                trace_id: decision.trace_id.clone(),
                action: "autonomous_evolution_decision".into(),
                metadata: serde_json::json!({
                    "timestamp_ms": Utc::now().timestamp_millis(),
                    "diagnostics": decision.diagnostics
                }),
            });
        }
        decisions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::{EventHandler, EventService};
    use std::sync::Mutex;

    fn observation(
        role: EvolutionRole,
        failure_kind: EvolutionFailureKind,
    ) -> EvolutionObservation {
        EvolutionObservation {
            role,
            failure_kind,
            task_id: Some(format!("task-{}", role.as_str())),
            trace_id: format!("trace-{}", role.as_str()),
            evidence: serde_json::json!({ "source": "test" }),
            repeated_count: 3,
        }
    }

    #[test]
    fn bad_context_routes_to_rag_fix_not_lora() {
        let decision = AutonomousEvolutionPolicy::decide(&observation(
            EvolutionRole::CoderPython,
            EvolutionFailureKind::BadContext,
        ));

        assert_eq!(decision.action, EvolutionAction::RagFix);
        assert_ne!(decision.action, EvolutionAction::LoraTraining);
    }

    #[test]
    fn schema_and_format_route_to_prompt_first() {
        for kind in [EvolutionFailureKind::Schema, EvolutionFailureKind::Format] {
            let decision = AutonomousEvolutionPolicy::decide(&observation(EvolutionRole::Qa, kind));

            assert_eq!(decision.action, EvolutionAction::PromptEvolution);
        }
    }

    #[test]
    fn repeated_role_skill_failure_routes_to_lora() {
        let decision = AutonomousEvolutionPolicy::decide(&observation(
            EvolutionRole::CoderRust,
            EvolutionFailureKind::RepeatedRoleSkillFailure,
        ));

        assert_eq!(decision.action, EvolutionAction::LoraTraining);
    }

    #[test]
    fn weak_critic_routes_to_critic_role_evolution() {
        let decision = AutonomousEvolutionPolicy::decide(&observation(
            EvolutionRole::CriticCoder,
            EvolutionFailureKind::WeakCriticFeedback,
        ));

        assert_eq!(decision.action, EvolutionAction::CriticRoleEvolution);
    }

    #[derive(Default)]
    struct RecordingEvents {
        events: Mutex<Vec<Event>>,
    }

    #[async_trait]
    impl EventService for RecordingEvents {
        fn publish(&self, event: Event) {
            self.events.lock().unwrap().push(event);
        }

        fn subscribe(&self) -> tokio::sync::broadcast::Receiver<Event> {
            let (tx, _rx) = tokio::sync::broadcast::channel(1);
            tx.subscribe()
        }

        async fn start_handler(&self, _handler: Arc<dyn EventHandler>) {}
    }

    #[tokio::test]
    async fn service_persists_every_decision_to_diagnostics_events() {
        let events = Arc::new(RecordingEvents::default());
        let source = Box::new(StaticEvolutionObservationSource::new(vec![
            observation(EvolutionRole::CoderPython, EvolutionFailureKind::BadContext),
            observation(
                EvolutionRole::CoderPython,
                EvolutionFailureKind::RepeatedRoleSkillFailure,
            ),
        ]));
        let service = AutonomousEvolutionService::new(source, events.clone());

        let decisions = service.run(true).await;

        assert_eq!(decisions.len(), 2);
        assert_eq!(
            decisions[0].diagnostics["kind"],
            "autonomous_evolution_decision"
        );
        assert_eq!(events.events.lock().unwrap().len(), 2);
    }
}
