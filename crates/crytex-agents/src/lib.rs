#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

pub mod architect;
pub mod coder;
pub mod critic;
pub mod critics;
pub mod prompts;
pub mod qa;
pub mod researcher;
pub mod security;
pub mod summarizer;
pub mod tooling;

pub use crytex_core::services::{Agent, AgentError};

use serde_json::Value;
use std::sync::Arc;

/// Extracts an optional backend id from a task payload.
pub fn extract_backend_id(payload: &Value) -> Option<String> {
    payload
        .get("backend")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// Extracts the model identifier from a task payload.
pub fn extract_model(payload: &Value) -> String {
    payload
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("default")
        .to_string()
}

/// Pool of registered agents.
#[derive(Clone)]
pub struct AgentPool {
    agents: Vec<Arc<dyn Agent>>,
}

impl AgentPool {
    pub fn new() -> Self {
        Self { agents: Vec::new() }
    }

    pub fn register(&mut self, agent: Arc<dyn Agent>) {
        self.agents.push(agent);
    }

    pub fn find(&self, name: &str) -> Option<Arc<dyn Agent>> {
        self.agents.iter().find(|a| a.name() == name).cloned()
    }

    pub fn list(&self) -> Vec<String> {
        self.agents.iter().map(|a| a.name().to_string()).collect()
    }
}

impl Default for AgentPool {
    fn default() -> Self {
        Self::new()
    }
}
