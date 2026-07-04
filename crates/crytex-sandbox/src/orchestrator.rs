//! Sandbox orchestrator that selects an available backend automatically.

use std::sync::Arc;

use async_trait::async_trait;
use crytex_core::services::{
    ExecutionRequest, ExecutionResult, SandboxService, SandboxServiceError,
};

use crate::backends::{DockerBackend, HostBackend};

/// Orchestrator that prefers Docker and transparently falls back to the host.
pub struct SandboxOrchestrator {
    backend: Arc<dyn SandboxService>,
}

impl SandboxOrchestrator {
    /// Create an orchestrator using the explicitly provided backend.
    pub fn new<B>(backend: B) -> Self
    where
        B: SandboxService + 'static,
    {
        Self {
            backend: Arc::new(backend),
        }
    }

    /// Auto-detect the best available backend:
    /// 1. Docker (if daemon is reachable)
    /// 2. Host fallback
    pub async fn auto() -> Self {
        if let Ok(docker) = DockerBackend::try_new()
            && docker.ping().await.is_ok()
        {
            return Self::new(docker);
        }
        Self::new(HostBackend::new())
    }
}

#[async_trait]
impl SandboxService for SandboxOrchestrator {
    async fn execute(
        &self,
        request: ExecutionRequest,
    ) -> Result<ExecutionResult, SandboxServiceError> {
        self.backend.execute(request).await
    }

    fn available(&self) -> bool {
        self.backend.available()
    }
}

#[cfg(test)]
mod tests {
    use crytex_core::services::{ExecutionRequest, SandboxResources};

    use super::*;

    #[tokio::test]
    async fn orchestrator_auto_selects_available_backend() {
        let docker_available = if let Ok(docker) = DockerBackend::try_new() {
            docker.ping().await.is_ok()
        } else {
            false
        };

        let orchestrator = SandboxOrchestrator::auto().await;
        assert!(orchestrator.available());

        let request = if docker_available {
            ExecutionRequest::new(vec!["echo".to_string(), "auto".to_string()])
                .image("alpine:latest")
        } else {
            #[cfg(target_os = "windows")]
            let command = vec![
                "cmd".to_string(),
                "/c".to_string(),
                "echo".to_string(),
                "auto".to_string(),
            ];
            #[cfg(not(target_os = "windows"))]
            let command = vec!["sh".to_string(), "-c".to_string(), "echo auto".to_string()];
            ExecutionRequest::new(command)
        }
        .resources(SandboxResources {
            timeout_seconds: 5,
            ..Default::default()
        });

        let result = orchestrator.execute(request).await.unwrap();
        assert_eq!(result.exit_code, 0);
        assert!(result.stdout.contains("auto"));
    }
}
