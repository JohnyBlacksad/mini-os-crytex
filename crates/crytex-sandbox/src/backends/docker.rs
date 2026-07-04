//! Docker-based sandbox backend for real-world agent tasks.

use std::collections::HashMap;
use std::path::PathBuf;

use async_trait::async_trait;
use bollard::Docker;
use bollard::container::{
    Config, CreateContainerOptions, LogOutput, LogsOptions, RemoveContainerOptions,
    WaitContainerOptions,
};
use bollard::errors::Error as BollardError;
use bollard::models::{HostConfig, Mount, MountTypeEnum};
use crytex_core::services::{
    ExecutionRequest, ExecutionResult, SandboxMount, SandboxNetwork, SandboxResources,
    SandboxService, SandboxServiceError,
};
use futures_util::stream::StreamExt;

const DEFAULT_IMAGE: &str = "crytex/sandbox-rust:latest";
const SANDBOX_USER: &str = "10000:10000";

/// Docker-backed sandbox backend.
///
/// Creates an ephemeral container for each execution, runs the command with hardened
/// security settings, captures stdout/stderr, and removes the container afterwards.
pub struct DockerBackend {
    docker: Docker,
}

impl DockerBackend {
    /// Try to create a backend connected to the local Docker daemon.
    pub fn try_new() -> Result<Self, SandboxServiceError> {
        let docker = Docker::connect_with_local_defaults()
            .map_err(|e| SandboxServiceError::BackendUnavailable(e.to_string()))?;
        Ok(Self { docker })
    }

    /// Verify that Docker is reachable.
    pub async fn ping(&self) -> Result<(), SandboxServiceError> {
        self.docker
            .ping()
            .await
            .map_err(|e| SandboxServiceError::BackendUnavailable(e.to_string()))?;
        Ok(())
    }

    fn to_env(env: &HashMap<String, String>) -> Vec<String> {
        env.iter().map(|(k, v)| format!("{}={}", k, v)).collect()
    }

    fn to_mounts(mounts: &[SandboxMount]) -> Vec<Mount> {
        mounts
            .iter()
            .map(|m| {
                let host_path =
                    std::fs::canonicalize(&m.host_path).unwrap_or_else(|_| m.host_path.clone());
                Mount {
                    typ: Some(MountTypeEnum::BIND),
                    source: Some(host_path.to_string_lossy().to_string()),
                    target: Some(m.guest_path.to_string_lossy().to_string()),
                    read_only: Some(m.read_only),
                    ..Default::default()
                }
            })
            .collect()
    }

    fn memory_bytes(resources: &SandboxResources) -> i64 {
        (resources.memory_mb as i64) * 1024 * 1024
    }

    fn cpu_shares(resources: &SandboxResources) -> i64 {
        resources.cpu_shares as i64
    }
}

#[async_trait]
impl SandboxService for DockerBackend {
    async fn execute(
        &self,
        request: ExecutionRequest,
    ) -> Result<ExecutionResult, SandboxServiceError> {
        if request.command.is_empty() {
            return Err(SandboxServiceError::Execution("empty command".into()));
        }

        let image = request.image.as_deref().unwrap_or(DEFAULT_IMAGE);
        let container_name = format!(
            "crytex-sandbox-{}",
            ulid::Ulid::new().to_string().to_lowercase()
        );

        let cwd = if request.cwd.is_absolute() {
            request.cwd.clone()
        } else {
            PathBuf::from("/workspace").join(&request.cwd)
        };

        let network_mode = match request.network {
            SandboxNetwork::Deny => Some("none".to_string()),
            SandboxNetwork::Allow => None,
        };

        let mut tmpfs = HashMap::new();
        tmpfs.insert("/tmp".to_string(), "noexec,nosuid,size=256m".to_string());

        let host_config = HostConfig {
            mounts: Some(Self::to_mounts(&request.mounts)),
            network_mode,
            cap_drop: Some(vec!["ALL".to_string()]),
            security_opt: Some(vec!["no-new-privileges:true".to_string()]),
            readonly_rootfs: Some(true),
            auto_remove: Some(false),
            memory: Some(Self::memory_bytes(&request.resources)),
            cpu_shares: Some(Self::cpu_shares(&request.resources)),
            tmpfs: Some(tmpfs),
            ..Default::default()
        };

        let config = Config {
            image: Some(image.to_string()),
            cmd: Some(request.command.clone()),
            env: Some(Self::to_env(&request.env)),
            working_dir: Some(cwd.to_string_lossy().to_string()),
            user: Some(SANDBOX_USER.to_string()),
            host_config: Some(host_config),
            ..Default::default()
        };

        let create_options = CreateContainerOptions {
            name: &container_name,
            platform: None,
        };

        let container = self
            .docker
            .create_container(Some(create_options), config)
            .await
            .map_err(|e| SandboxServiceError::Execution(format!("create container: {e}")))?;

        if let Err(start_err) = self
            .docker
            .start_container::<String>(&container.id, None)
            .await
        {
            let _ = self
                .docker
                .remove_container(&container.id, None::<RemoveContainerOptions>)
                .await;
            return Err(SandboxServiceError::Execution(format!(
                "start container: {start_err}"
            )));
        }

        let timeout = std::time::Duration::from_secs(request.resources.timeout_seconds);
        let wait_result = tokio::time::timeout(timeout, async {
            let mut stream = self.docker.wait_container(
                &container.id,
                Some(WaitContainerOptions {
                    condition: "not-running".to_string(),
                }),
            );
            stream.next().await
        })
        .await;

        match wait_result {
            Ok(Some(Ok(response))) => {
                let exit_code = response.status_code;

                let logs_options = LogsOptions::<String> {
                    stdout: true,
                    stderr: true,
                    ..Default::default()
                };

                let mut logs = self.docker.logs(&container.id, Some(logs_options));
                let mut stdout = Vec::new();
                let mut stderr = Vec::new();

                while let Some(log) = logs.next().await {
                    match log {
                        Ok(LogOutput::StdOut { message }) => stdout.extend_from_slice(&message),
                        Ok(LogOutput::StdErr { message }) => stderr.extend_from_slice(&message),
                        Ok(_) => {}
                        Err(e) => {
                            let _ = self
                                .docker
                                .remove_container(&container.id, None::<RemoveContainerOptions>)
                                .await;
                            return Err(SandboxServiceError::Execution(format!("logs: {e}")));
                        }
                    }
                }

                let _ = self
                    .docker
                    .remove_container(&container.id, None::<RemoveContainerOptions>)
                    .await;

                Ok(ExecutionResult {
                    exit_code,
                    stdout: String::from_utf8_lossy(&stdout).to_string(),
                    stderr: String::from_utf8_lossy(&stderr).to_string(),
                })
            }
            Ok(Some(Err(e))) => {
                let exit_code = match &e {
                    BollardError::DockerContainerWaitError { code, .. } => *code,
                    _ => {
                        let _ = self
                            .docker
                            .remove_container(&container.id, None::<RemoveContainerOptions>)
                            .await;
                        return Err(SandboxServiceError::Execution(format!(
                            "wait container: {e}"
                        )));
                    }
                };

                let logs_options = LogsOptions::<String> {
                    stdout: true,
                    stderr: true,
                    ..Default::default()
                };

                let mut logs = self.docker.logs(&container.id, Some(logs_options));
                let mut stdout = Vec::new();
                let mut stderr = Vec::new();

                while let Some(log) = logs.next().await {
                    match log {
                        Ok(LogOutput::StdOut { message }) => stdout.extend_from_slice(&message),
                        Ok(LogOutput::StdErr { message }) => stderr.extend_from_slice(&message),
                        Ok(_) => {}
                        Err(log_err) => {
                            let _ = self
                                .docker
                                .remove_container(&container.id, None::<RemoveContainerOptions>)
                                .await;
                            return Err(SandboxServiceError::Execution(format!("logs: {log_err}")));
                        }
                    }
                }

                let _ = self
                    .docker
                    .remove_container(&container.id, None::<RemoveContainerOptions>)
                    .await;

                Ok(ExecutionResult {
                    exit_code,
                    stdout: String::from_utf8_lossy(&stdout).to_string(),
                    stderr: String::from_utf8_lossy(&stderr).to_string(),
                })
            }
            Ok(None) => {
                let _ = self
                    .docker
                    .remove_container(&container.id, None::<RemoveContainerOptions>)
                    .await;
                Err(SandboxServiceError::Execution(
                    "container produced no wait response".into(),
                ))
            }
            Err(_) => {
                let _ = self
                    .docker
                    .kill_container::<String>(&container.id, None)
                    .await;
                let _ = self
                    .docker
                    .remove_container(&container.id, None::<RemoveContainerOptions>)
                    .await;
                Err(SandboxServiceError::Timeout(
                    request.resources.timeout_seconds,
                ))
            }
        }
    }

    fn available(&self) -> bool {
        // Synchronous check is best-effort; the orchestrator should also call ping().await.
        true
    }
}

#[cfg(test)]
mod tests {
    use bollard::image::CreateImageOptions;

    use super::*;

    async fn ensure_image(docker: &Docker, image: &str) {
        let options = CreateImageOptions::<String> {
            from_image: image.to_string(),
            ..Default::default()
        };
        let mut stream = docker.create_image(Some(options), None, None);
        while let Some(item) = stream.next().await {
            if let Err(e) = item {
                eprintln!("pull warning for {image}: {e}");
            }
        }
    }

    #[tokio::test]
    #[ignore = "requires Docker"]
    async fn docker_backend_runs_alpine_echo() {
        let backend = DockerBackend::try_new().expect("docker connection");
        backend.ping().await.expect("docker ping");
        ensure_image(&backend.docker, "alpine:latest").await;

        let request = ExecutionRequest::new(vec!["echo".to_string(), "docker-hello".to_string()])
            .image("alpine:latest")
            .resources(SandboxResources {
                timeout_seconds: 30,
                ..Default::default()
            });

        let result = backend.execute(request).await.unwrap();

        assert_eq!(result.exit_code, 0, "stderr: {}", result.stderr);
        assert!(result.stdout.contains("docker-hello"));
        assert!(result.stderr.is_empty());
    }

    #[tokio::test]
    #[ignore = "requires Docker"]
    async fn docker_backend_returns_nonzero_exit_code() {
        let backend = DockerBackend::try_new().expect("docker connection");
        backend.ping().await.expect("docker ping");
        ensure_image(&backend.docker, "alpine:latest").await;

        let request = ExecutionRequest::new(vec![
            "sh".to_string(),
            "-c".to_string(),
            "exit 7".to_string(),
        ])
        .image("alpine:latest")
        .resources(SandboxResources {
            timeout_seconds: 30,
            ..Default::default()
        });

        let result = backend.execute(request).await.unwrap();

        assert_eq!(result.exit_code, 7);
    }

    #[tokio::test]
    #[ignore = "requires Docker and crytex/sandbox-rust:latest"]
    async fn docker_backend_runs_cargo_version() {
        let backend = DockerBackend::try_new().expect("docker connection");
        backend.ping().await.expect("docker ping");

        let request = ExecutionRequest::new(vec!["cargo".to_string(), "--version".to_string()])
            .image("crytex/sandbox-rust:latest")
            .resources(SandboxResources {
                timeout_seconds: 60,
                ..Default::default()
            });

        let result = backend.execute(request).await.unwrap();

        assert_eq!(result.exit_code, 0);
        assert!(result.stdout.contains("cargo"));
    }
}
