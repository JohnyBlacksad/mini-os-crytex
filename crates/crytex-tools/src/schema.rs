use std::sync::Arc;

use async_trait::async_trait;
use crytex_core::config::SecurityConfig;
use crytex_core::security::SecurityScanner;
use crytex_core::services::SandboxService;
use serde_json::Value;

use crate::policy::{Capability, PermissionSet};
use crate::sandbox::SandboxError;
use crytex_core::services::SandboxServiceError;
use crytex_sandbox::backends::HostBackend;

/// Errors that can occur when executing a tool.
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("invalid arguments: {0}")]
    InvalidArgs(String),
    #[error("forbidden: {0}")]
    Forbidden(String),
    #[error("sandbox error: {0}")]
    Sandbox(#[from] SandboxError),
    #[error("sandbox service error: {0}")]
    SandboxService(#[from] SandboxServiceError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("git error: {0}")]
    Git(String),
    #[error("process error: exit={exit_code:?}, stderr={stderr}")]
    Process {
        exit_code: Option<i32>,
        stderr: String,
    },
    #[error("process timed out after {0}s")]
    Timeout(u64),
    #[error("tool not found: {0}")]
    NotFound(String),
    #[error("embedding error: {0}")]
    Embedding(String),
    #[error("vector store error: {0}")]
    VectorStore(String),
    #[error("filesystem error at {path}: {source}")]
    FileSystem {
        path: String,
        source: std::io::Error,
    },
    #[error("serialization error: {0}")]
    Serialization(String),
    #[error("search error: {0}")]
    Search(String),
    #[error("{0}")]
    Other(String),
}

/// Result type returned by a tool execution.
pub type ToolResult = Result<Value, ToolError>;

/// JSON-schema-like metadata describing a tool's input shape.
#[derive(Debug, Clone)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    pub required: Vec<String>,
}

impl ToolSchema {
    /// Render the schema in OpenAI-function-calling compatible form.
    pub fn to_openai_function(&self) -> Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        })
    }

    /// Validate that `args` contains all required keys.
    pub fn validate_required(&self, args: &Value) -> Result<(), ToolError> {
        let obj = args
            .as_object()
            .ok_or_else(|| ToolError::InvalidArgs(format!("expected JSON object, got {}", args)))?;
        for key in &self.required {
            if !obj.contains_key(key) {
                return Err(ToolError::InvalidArgs(format!(
                    "missing required argument: {}",
                    key
                )));
            }
        }
        Ok(())
    }
}

/// Execution context supplied to every tool invocation.
#[derive(Clone)]
pub struct ToolContext {
    pub project_root: std::path::PathBuf,
    pub permissions: PermissionSet,
    pub timeout_seconds: u64,
    pub sandbox: Arc<dyn SandboxService>,
    /// Optional security scanner used to inspect tool inputs and file contents.
    pub scanner: Option<Arc<dyn SecurityScanner>>,
    /// Security-related configuration (scanning, wrapping, blocking thresholds).
    pub security_config: SecurityConfig,
}

impl std::fmt::Debug for ToolContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolContext")
            .field("project_root", &self.project_root)
            .field("permissions", &self.permissions)
            .field("timeout_seconds", &self.timeout_seconds)
            .field("sandbox", &"<dyn SandboxService>")
            .finish()
    }
}

impl Default for ToolContext {
    fn default() -> Self {
        Self {
            project_root: std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from(".")),
            permissions: PermissionSet::empty(),
            timeout_seconds: 60,
            sandbox: Arc::new(HostBackend::new()),
            scanner: None,
            security_config: SecurityConfig::default(),
        }
    }
}

/// A single tool callable by an agent.
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn schema(&self) -> ToolSchema;

    /// Capabilities required to invoke this tool.
    fn required_capabilities(&self) -> Capability {
        Capability::empty()
    }

    /// Execute the tool with validated JSON arguments.
    async fn execute(&self, ctx: &ToolContext, args: Value) -> ToolResult;

    /// Check permissions and then execute.
    async fn invoke(&self, ctx: &ToolContext, args: Value) -> ToolResult {
        let required = self.required_capabilities();
        if !ctx.permissions.contains(required) {
            return Err(ToolError::Forbidden(format!(
                "tool {} requires capabilities {:?}, granted {:?}",
                self.name(),
                required,
                ctx.permissions
            )));
        }
        self.execute(ctx, args).await
    }
}

/// Extract a typed string argument from a JSON object.
pub fn require_str(args: &Value, key: &str) -> Result<String, ToolError> {
    args.get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| ToolError::InvalidArgs(format!("{} must be a string", key)))
}

/// Extract an optional string argument.
pub fn optional_str(args: &Value, key: &str) -> Option<String> {
    args.get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// Extract an optional usize argument.
pub fn optional_usize(args: &Value, key: &str) -> Result<Option<usize>, ToolError> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(n)) => n.as_u64().map(|u| Some(u as usize)).ok_or_else(|| {
            ToolError::InvalidArgs(format!("{} must be a non-negative integer", key))
        }),
        Some(other) => Err(ToolError::InvalidArgs(format!(
            "{} must be a non-negative integer, got {}",
            key, other
        ))),
    }
}

/// Convert an arbitrary tool result into a structured JSON value.
pub fn result_ok(value: impl serde::Serialize) -> ToolResult {
    serde_json::to_value(value).map_err(|e| ToolError::Serialization(e.to_string()))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::fs::FsWrite;
    use crate::policy::Capability;

    #[tokio::test]
    async fn runtime_denies_tool_call_exceeding_granted_capabilities() {
        let tool = FsWrite::new();
        let ctx = ToolContext {
            permissions: Capability::READ,
            ..Default::default()
        };
        let args = json!({ "path": "out.txt", "content": "hello" });

        let err = tool.invoke(&ctx, args).await.unwrap_err();

        assert!(
            matches!(err, ToolError::Forbidden(_)),
            "expected Forbidden, got {:?}",
            err
        );
    }
}
