//! Agent roles used for role-based LoRA routing and workflow specialization.

use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// A well-known agent role in the Crytex workflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentRole {
    /// Designs the overall architecture and plan.
    Architect,
    /// Writes and edits code.
    Coder,
    /// Runs tests and verifies behavior.
    Qa,
    /// Performs security analysis.
    Security,
    /// Reviews and critiques output.
    Critic,
    /// Gathers information from external sources.
    Researcher,
    /// Condenses context and results.
    Summarizer,
}

impl AgentRole {
    /// All known roles.
    pub const ALL: &[AgentRole] = &[
        AgentRole::Architect,
        AgentRole::Coder,
        AgentRole::Qa,
        AgentRole::Security,
        AgentRole::Critic,
        AgentRole::Researcher,
        AgentRole::Summarizer,
    ];

    /// Return the canonical snake_case identifier for the role.
    pub const fn as_str(&self) -> &'static str {
        match self {
            AgentRole::Architect => "architect",
            AgentRole::Coder => "coder",
            AgentRole::Qa => "qa",
            AgentRole::Security => "security",
            AgentRole::Critic => "critic",
            AgentRole::Researcher => "researcher",
            AgentRole::Summarizer => "summarizer",
        }
    }

    /// Parse a role from its canonical identifier.
    pub fn from_identifier(s: &str) -> Option<Self> {
        match s {
            "architect" => Some(AgentRole::Architect),
            "coder" => Some(AgentRole::Coder),
            "qa" => Some(AgentRole::Qa),
            "security" => Some(AgentRole::Security),
            "critic" => Some(AgentRole::Critic),
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
