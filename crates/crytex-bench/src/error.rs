use thiserror::Error;

#[derive(Debug, Error)]
pub enum BenchError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to parse golden set: {0}")]
    GoldenSetParse(String),
    #[error("golden set not found: {0}")]
    GoldenSetNotFound(String),
    #[error("scoring error: {0}")]
    Scoring(String),
    #[error("runner error: {0}")]
    Runner(String),
    #[error("harness error: {0}")]
    Harness(String),
    #[error("persistence error: {0}")]
    Persistence(String),
    #[error("A/B test error: {0}")]
    ABTest(String),
    #[error("invalid configuration: {0}")]
    Config(String),
}

impl From<crytex_core::persistence::PersistenceError> for BenchError {
    fn from(e: crytex_core::persistence::PersistenceError) -> Self {
        BenchError::Persistence(e.to_string())
    }
}

impl From<crytex_core::services::TaskError> for BenchError {
    fn from(e: crytex_core::services::TaskError) -> Self {
        BenchError::Runner(e.to_string())
    }
}

impl From<crytex_core::services::AgentServiceError> for BenchError {
    fn from(e: crytex_core::services::AgentServiceError) -> Self {
        BenchError::Runner(e.to_string())
    }
}

impl From<crytex_core::services::SandboxServiceError> for BenchError {
    fn from(e: crytex_core::services::SandboxServiceError) -> Self {
        BenchError::Scoring(e.to_string())
    }
}
