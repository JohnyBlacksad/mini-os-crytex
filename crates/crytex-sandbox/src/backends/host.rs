//! Host fallback sandbox backend.
//!
//! Runs commands directly on the host OS. This backend provides very little isolation
//! and should only be used when Docker is unavailable. It is intended for local
//! development and trusted environments.

use std::path::{Path, PathBuf};
use std::process::Stdio;

use async_trait::async_trait;
use crytex_core::services::{
    ExecutionRequest, ExecutionResult, SandboxMount, SandboxService, SandboxServiceError,
};
use tokio::process::Command;
use tokio::time::{Duration, timeout};

/// Host-backed sandbox backend.
pub struct HostBackend;

impl HostBackend {
    /// Create a new host backend.
    pub fn new() -> Self {
        Self
    }
}

impl Default for HostBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SandboxService for HostBackend {
    async fn execute(
        &self,
        request: ExecutionRequest,
    ) -> Result<ExecutionResult, SandboxServiceError> {
        if request.command.is_empty() {
            return Err(SandboxServiceError::Execution("empty command".into()));
        }

        let (program, args) = request.command.split_first().unwrap();
        let cwd = map_guest_to_host(&request.mounts, &request.cwd);
        let mut cmd = Command::new(program);
        cmd.args(args)
            .current_dir(&cwd)
            .env_clear()
            .envs(&request.env)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let timeout_duration = Duration::from_secs(request.resources.timeout_seconds);

        match timeout(timeout_duration, cmd.output()).await {
            Ok(Ok(output)) => Ok(ExecutionResult {
                exit_code: output.status.code().unwrap_or(-1) as i64,
                stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            }),
            Ok(Err(e)) => Err(SandboxServiceError::Execution(format!("host command: {e}"))),
            Err(_) => Err(SandboxServiceError::Timeout(
                request.resources.timeout_seconds,
            )),
        }
    }

    fn available(&self) -> bool {
        true
    }
}

fn map_guest_to_host(mounts: &[SandboxMount], guest: &Path) -> PathBuf {
    for mount in mounts {
        if let Ok(rel) = guest.strip_prefix(&mount.guest_path) {
            return mount.host_path.join(rel);
        }
    }
    guest.to_path_buf()
}

#[cfg(test)]
mod tests {
    use crytex_core::services::SandboxResources;

    use super::*;

    fn echo_request() -> ExecutionRequest {
        #[cfg(target_os = "windows")]
        let command = vec![
            "cmd".to_string(),
            "/c".to_string(),
            "echo".to_string(),
            "hello".to_string(),
        ];
        #[cfg(not(target_os = "windows"))]
        let command = vec!["sh".to_string(), "-c".to_string(), "echo hello".to_string()];

        ExecutionRequest::new(command).resources(SandboxResources {
            timeout_seconds: 5,
            ..Default::default()
        })
    }

    fn exit_request(code: i32) -> ExecutionRequest {
        #[cfg(target_os = "windows")]
        let command = vec![
            "cmd".to_string(),
            "/c".to_string(),
            "exit".to_string(),
            code.to_string(),
        ];
        #[cfg(not(target_os = "windows"))]
        let command = vec!["sh".to_string(), "-c".to_string(), format!("exit {code}")];

        ExecutionRequest::new(command).resources(SandboxResources {
            timeout_seconds: 5,
            ..Default::default()
        })
    }

    #[tokio::test]
    async fn host_backend_runs_command_and_captures_stdout() {
        let backend = HostBackend::new();
        let result = backend.execute(echo_request()).await.unwrap();

        assert!(result.exit_code == 0);
        assert!(result.stdout.contains("hello"));
        assert!(result.stderr.is_empty());
    }

    #[tokio::test]
    async fn host_backend_returns_nonzero_exit_code() {
        let backend = HostBackend::new();
        let result = backend.execute(exit_request(42)).await.unwrap();

        assert_eq!(result.exit_code, 42);
    }

    #[tokio::test]
    async fn host_backend_rejects_empty_command() {
        let backend = HostBackend::new();
        let request = ExecutionRequest::new(Vec::<String>::new());

        let err = backend.execute(request).await.unwrap_err();

        assert!(matches!(err, SandboxServiceError::Execution(_)));
    }
}
