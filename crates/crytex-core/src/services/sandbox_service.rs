//! Secure task execution service used by agents.

use std::collections::HashMap;
use std::path::PathBuf;

use async_trait::async_trait;
use thiserror::Error;

/// A filesystem mount exposed inside the sandbox.
#[derive(Clone, Debug)]
pub struct SandboxMount {
    pub host_path: PathBuf,
    pub guest_path: PathBuf,
    pub read_only: bool,
}

/// Network policy for a sandboxed execution.
#[derive(Clone, Debug, Default)]
pub enum SandboxNetwork {
    /// No outbound network access.
    #[default]
    Deny,
    /// Allow outbound access (egress controls are backend-specific).
    Allow,
}

/// Resource limits for a sandboxed execution.
#[derive(Clone, Copy, Debug)]
pub struct SandboxResources {
    /// Memory limit in megabytes.
    pub memory_mb: usize,
    /// CPU limit as a fraction of a core (1000 = 1 core).
    pub cpu_shares: u32,
    /// Wall-clock timeout in seconds.
    pub timeout_seconds: u64,
}

impl Default for SandboxResources {
    fn default() -> Self {
        Self {
            memory_mb: 1024,
            cpu_shares: 1024,
            timeout_seconds: 300,
        }
    }
}

/// Request to execute a command inside a sandbox.
#[derive(Clone, Debug)]
pub struct ExecutionRequest {
    /// argv command (no shell).
    pub command: Vec<String>,
    /// Working directory inside the sandbox.
    pub cwd: PathBuf,
    /// Environment variables.
    pub env: HashMap<String, String>,
    /// Host paths to expose inside the sandbox.
    pub mounts: Vec<SandboxMount>,
    /// Network policy.
    pub network: SandboxNetwork,
    /// Resource limits.
    pub resources: SandboxResources,
    /// Container/runtime image to use (backend-specific).
    pub image: Option<String>,
}

impl ExecutionRequest {
    pub fn new(command: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            command: command.into_iter().map(Into::into).collect(),
            cwd: PathBuf::from("."),
            env: HashMap::new(),
            mounts: Vec::new(),
            network: SandboxNetwork::Deny,
            resources: SandboxResources::default(),
            image: None,
        }
    }

    pub fn cwd(mut self, path: impl Into<PathBuf>) -> Self {
        self.cwd = path.into();
        self
    }

    pub fn env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.insert(key.into(), value.into());
        self
    }

    pub fn mount(
        mut self,
        host: impl Into<PathBuf>,
        guest: impl Into<PathBuf>,
        read_only: bool,
    ) -> Self {
        self.mounts.push(SandboxMount {
            host_path: host.into(),
            guest_path: guest.into(),
            read_only,
        });
        self
    }

    pub fn network(mut self, network: SandboxNetwork) -> Self {
        self.network = network;
        self
    }

    pub fn resources(mut self, resources: SandboxResources) -> Self {
        self.resources = resources;
        self
    }

    pub fn image(mut self, image: impl Into<String>) -> Self {
        self.image = Some(image.into());
        self
    }
}

/// Result of a sandboxed execution.
#[derive(Clone, Debug)]
pub struct ExecutionResult {
    pub exit_code: i64,
    pub stdout: String,
    pub stderr: String,
}

/// Errors returned by the sandbox service.
#[derive(Debug, Error)]
pub enum SandboxServiceError {
    #[error("sandbox backend unavailable: {0}")]
    BackendUnavailable(String),
    #[error("sandbox execution failed: {0}")]
    Execution(String),
    #[error("sandbox timed out after {0}s")]
    Timeout(u64),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Service abstraction for secure agent task execution.
#[async_trait]
pub trait SandboxService: Send + Sync {
    /// Execute the request inside an isolated environment.
    async fn execute(
        &self,
        request: ExecutionRequest,
    ) -> Result<ExecutionResult, SandboxServiceError>;

    /// Return true if the backend is available on this host.
    fn available(&self) -> bool;
}
