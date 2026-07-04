//! [`ToolService`](crytex_core::services::ToolService) implementation bridging the
//! capability-scoped tool registry into the core service contract.

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use crytex_core::config::SecurityConfig;
use crytex_core::security::SecurityScanner;
use crytex_core::services::{SandboxService, ToolDescription, ToolService, ToolServiceError};
use serde_json::Value;

use crate::ToolRegistry;
use crate::policy::Capability;
use crate::schema::{ToolContext, ToolError};

/// Runtime wrapper around a [`ToolRegistry`] and a fixed execution context.
pub struct ToolServiceImpl {
    registry: ToolRegistry,
    context: ToolContext,
}

impl ToolServiceImpl {
    /// Build a service for `project_root` granting the requested capabilities.
    pub fn new(
        registry: ToolRegistry,
        project_root: PathBuf,
        permissions: Capability,
        sandbox: Arc<dyn SandboxService>,
        scanner: Option<Arc<dyn SecurityScanner>>,
        security_config: SecurityConfig,
    ) -> Self {
        Self {
            registry,
            context: ToolContext {
                project_root,
                permissions,
                timeout_seconds: 60,
                sandbox,
                scanner,
                security_config,
            },
        }
    }

    /// Set a per-invocation timeout.
    pub fn with_timeout(mut self, seconds: u64) -> Self {
        self.context.timeout_seconds = seconds;
        self
    }
}

#[async_trait]
impl ToolService for ToolServiceImpl {
    async fn invoke(&self, name: &str, args: Value) -> Result<Value, ToolServiceError> {
        self.registry
            .invoke(&self.context, name, args)
            .await
            .map_err(Into::into)
    }

    fn list_tools(&self) -> Vec<ToolDescription> {
        self.registry
            .schemas()
            .into_iter()
            .map(|schema| ToolDescription {
                name: schema.name,
                description: schema.description,
                parameters: schema.parameters,
            })
            .collect()
    }
}

/// Decorator that runs a [`SecurityScanner`] on every tool invocation before
/// delegating to the inner [`ToolService`].
pub struct ScanningToolService {
    inner: Arc<dyn ToolService>,
    scanner: Arc<dyn SecurityScanner>,
}

impl ScanningToolService {
    pub fn new(inner: Arc<dyn ToolService>, scanner: Arc<dyn SecurityScanner>) -> Self {
        Self { inner, scanner }
    }
}

#[async_trait]
impl ToolService for ScanningToolService {
    async fn invoke(&self, name: &str, args: Value) -> Result<Value, ToolServiceError> {
        if let Some(finding) = self.scanner.scan_tool_args(name, &args).into_iter().next() {
            return Err(ToolServiceError::Forbidden(format!(
                "security scanner blocked {}: {} - {}",
                name, finding.threat, finding.message
            )));
        }
        self.inner.invoke(name, args).await
    }

    fn list_tools(&self) -> Vec<ToolDescription> {
        self.inner.list_tools()
    }
}

impl From<ToolError> for ToolServiceError {
    fn from(err: ToolError) -> Self {
        match err {
            ToolError::InvalidArgs(msg) => ToolServiceError::InvalidArgs(msg),
            ToolError::Forbidden(msg) => ToolServiceError::Forbidden(msg),
            ToolError::Io(e) => ToolServiceError::Io(e),
            ToolError::Process { exit_code, stderr } => {
                ToolServiceError::Process { exit_code, stderr }
            }
            ToolError::Timeout(seconds) => ToolServiceError::Timeout(seconds),
            ToolError::NotFound(name) => ToolServiceError::NotFound(name),
            ToolError::Git(msg) | ToolError::Other(msg) => ToolServiceError::Git(msg),
            ToolError::Embedding(msg) => ToolServiceError::Embedding(msg),
            ToolError::VectorStore(msg) => ToolServiceError::VectorStore(msg),
            ToolError::Sandbox(err) => ToolServiceError::Sandbox(err.to_string()),
            ToolError::SandboxService(err) => ToolServiceError::Sandbox(err.to_string()),
            ToolError::FileSystem { path, source } => ToolServiceError::FileSystem { path, source },
            ToolError::Serialization(msg) => ToolServiceError::Serialization(msg),
            ToolError::Search(msg) => ToolServiceError::Search(msg),
        }
    }
}
