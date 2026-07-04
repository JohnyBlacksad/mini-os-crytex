//! Cross-cutting tool-calling service used by agents.

use async_trait::async_trait;
use serde_json::Value;
use thiserror::Error;

/// Errors returned by the tool execution service.
#[derive(Debug, Error)]
pub enum ToolServiceError {
    #[error("tool not found: {0}")]
    NotFound(String),
    #[error("invalid arguments: {0}")]
    InvalidArgs(String),
    #[error("forbidden: {0}")]
    Forbidden(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("filesystem error at {path}: {source}")]
    FileSystem {
        path: String,
        source: std::io::Error,
    },
    #[error("serialization error: {0}")]
    Serialization(String),
    #[error("search error: {0}")]
    Search(String),
    #[error("process error: exit={exit_code:?}, stderr={stderr}")]
    Process {
        exit_code: Option<i32>,
        stderr: String,
    },
    #[error("process timed out after {0}s")]
    Timeout(u64),
    #[error("git error: {0}")]
    Git(String),
    #[error("embedding error: {0}")]
    Embedding(String),
    #[error("vector store error: {0}")]
    VectorStore(String),
    #[error("sandbox error: {0}")]
    Sandbox(String),
    #[error("tool error: {0}")]
    Other(String),
}

/// Public description of a registered tool.
#[derive(Clone, Debug)]
pub struct ToolDescription {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

/// Service abstraction for agent tool invocation.
#[async_trait]
pub trait ToolService: Send + Sync {
    /// Invoke a tool by name with a JSON argument object.
    async fn invoke(&self, name: &str, args: Value) -> Result<Value, ToolServiceError>;

    /// List all registered tools with their schemas.
    fn list_tools(&self) -> Vec<ToolDescription>;
}
