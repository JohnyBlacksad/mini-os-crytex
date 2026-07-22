//! Agent roles used for role-based LoRA routing and workflow specialization.

use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// A well-known agent role in the Crytex workflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentRole {
    /// Decomposes goals and coordinates the workflow graph.
    Orchestrator,
    /// Designs the overall architecture and plan.
    Architect,
    /// Writes and edits code.
    Coder,
    /// Writes Python code.
    CoderPython,
    /// Writes Rust code.
    CoderRust,
    /// Writes TypeScript code.
    CoderTs,
    /// Writes code for other languages.
    CoderEtc,
    /// Performs product or technical analysis.
    Analyst,
    /// Runs tests and verifies behavior.
    Qa,
    /// Handles deployment and operations.
    Devops,
    /// Performs security analysis.
    Security,
    /// Reviews and critiques output.
    Critic,
    /// Reviews analytical work.
    CriticAnalyst,
    /// Reviews code work.
    CriticCoder,
    /// Reviews research work.
    CriticResearcher,
    /// Reviews other artifact kinds.
    CriticEtc,
    /// Gathers information from external sources.
    Researcher,
    /// Condenses context and results.
    Summarizer,
}

impl AgentRole {
    /// All known roles.
    pub const ALL: &[AgentRole] = &[
        AgentRole::Orchestrator,
        AgentRole::Architect,
        AgentRole::Coder,
        AgentRole::CoderPython,
        AgentRole::CoderRust,
        AgentRole::CoderTs,
        AgentRole::CoderEtc,
        AgentRole::Analyst,
        AgentRole::Qa,
        AgentRole::Devops,
        AgentRole::Security,
        AgentRole::Critic,
        AgentRole::CriticAnalyst,
        AgentRole::CriticCoder,
        AgentRole::CriticResearcher,
        AgentRole::CriticEtc,
        AgentRole::Researcher,
        AgentRole::Summarizer,
    ];

    /// Return the canonical snake_case identifier for the role.
    pub const fn as_str(&self) -> &'static str {
        match self {
            AgentRole::Orchestrator => "orchestrator",
            AgentRole::Architect => "architect",
            AgentRole::Coder => "coder",
            AgentRole::CoderPython => "coder-python",
            AgentRole::CoderRust => "coder-rust",
            AgentRole::CoderTs => "coder-ts",
            AgentRole::CoderEtc => "coder-etc",
            AgentRole::Analyst => "analyst",
            AgentRole::Qa => "qa",
            AgentRole::Devops => "devops",
            AgentRole::Security => "security",
            AgentRole::Critic => "critic",
            AgentRole::CriticAnalyst => "critic-analyst",
            AgentRole::CriticCoder => "critic-coder",
            AgentRole::CriticResearcher => "critic-researcher",
            AgentRole::CriticEtc => "critic-etc",
            AgentRole::Researcher => "researcher",
            AgentRole::Summarizer => "summarizer",
        }
    }

    /// Parse a role from its canonical identifier.
    pub fn from_identifier(s: &str) -> Option<Self> {
        match s {
            "orchestrator" => Some(AgentRole::Orchestrator),
            "architect" => Some(AgentRole::Architect),
            "coder" => Some(AgentRole::Coder),
            "coder-python" | "coder_python" | "python" => Some(AgentRole::CoderPython),
            "coder-rust" | "coder_rust" | "rust" => Some(AgentRole::CoderRust),
            "coder-ts" | "coder_ts" | "coder-typescript" | "typescript" | "ts" => {
                Some(AgentRole::CoderTs)
            }
            "coder-etc" | "coder_etc" => Some(AgentRole::CoderEtc),
            "analyst" => Some(AgentRole::Analyst),
            "qa" => Some(AgentRole::Qa),
            "devops" => Some(AgentRole::Devops),
            "security" => Some(AgentRole::Security),
            "critic" => Some(AgentRole::Critic),
            "critic-analyst" | "critic_analyst" => Some(AgentRole::CriticAnalyst),
            "critic-coder" | "critic_coder" => Some(AgentRole::CriticCoder),
            "critic-researcher" | "critic_researcher" => Some(AgentRole::CriticResearcher),
            "critic-etc" | "critic_etc" => Some(AgentRole::CriticEtc),
            "researcher" => Some(AgentRole::Researcher),
            "summarizer" => Some(AgentRole::Summarizer),
            _ => None,
        }
    }

    /// Infer a role from an agent name.
    ///
    /// Unknown agent names return `None`.
    pub fn from_agent(agent: &str) -> Option<Self> {
        Self::from_identifier(agent)
    }
}

impl FromStr for AgentRole {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_identifier(s).ok_or(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_role_maps_from_agent_name() {
        assert_eq!(
            AgentRole::from_agent("architect"),
            Some(AgentRole::Architect)
        );
        assert_eq!(AgentRole::from_agent("coder"), Some(AgentRole::Coder));
        assert_eq!(AgentRole::from_agent("qa"), Some(AgentRole::Qa));
        assert_eq!(AgentRole::from_agent("security"), Some(AgentRole::Security));
        assert_eq!(AgentRole::from_agent("critic"), Some(AgentRole::Critic));
        assert_eq!(
            AgentRole::from_agent("researcher"),
            Some(AgentRole::Researcher)
        );
        assert_eq!(
            AgentRole::from_agent("summarizer"),
            Some(AgentRole::Summarizer)
        );
    }

    #[test]
    fn agent_role_returns_none_for_unknown_agent() {
        assert_eq!(AgentRole::from_agent("unknown"), None);
    }

    #[test]
    fn agent_role_round_trips_through_str() {
        for role in AgentRole::ALL {
            assert_eq!(role.as_str().parse::<AgentRole>().ok(), Some(*role));
        }
    }
}
