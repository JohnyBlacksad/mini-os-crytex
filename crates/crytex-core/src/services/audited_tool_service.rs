//! Tool-service decorator that persists every tool invocation to the audit log.

use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use serde_json::Value;

use crate::services::{
    AuditEvent, AuditLogService, ToolDescription, ToolService, ToolServiceError,
};

pub struct AuditedToolService {
    inner: Arc<dyn ToolService>,
    audit: Arc<dyn AuditLogService>,
    project_id: Option<String>,
    task_id: String,
    agent: String,
    trace_id: String,
}

impl AuditedToolService {
    pub fn new(
        inner: Arc<dyn ToolService>,
        audit: Arc<dyn AuditLogService>,
        project_id: Option<String>,
        task_id: impl Into<String>,
        agent: impl Into<String>,
        trace_id: impl Into<String>,
    ) -> Self {
        Self {
            inner,
            audit,
            project_id,
            task_id: task_id.into(),
            agent: agent.into(),
            trace_id: trace_id.into(),
        }
    }
}

#[async_trait]
impl ToolService for AuditedToolService {
    async fn invoke(&self, name: &str, args: Value) -> Result<Value, ToolServiceError> {
        let started = Instant::now();
        let result = self.inner.invoke(name, args.clone()).await;
        let audit_result = match &result {
            Ok(value) => serde_json::json!({
                "status": "ok",
                "value": value,
            }),
            Err(error) => serde_json::json!({
                "status": "error",
                "error": error.to_string(),
            }),
        };
        let entry = AuditEvent::ToolCalled {
            task_id: self.task_id.clone(),
            agent: self.agent.clone(),
            tool_name: name.to_string(),
            args,
            result: audit_result,
            duration_ms: started.elapsed().as_millis() as u64,
        }
        .into_entry(self.project_id.clone(), self.trace_id.clone());
        let _ = self.audit.log(entry).await;
        result
    }

    fn list_tools(&self) -> Vec<ToolDescription> {
        self.inner.list_tools()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{AgentLog, AuditLogLevel};
    use crate::services::{AuditError, AuditLogEntry};
    use std::sync::Mutex;

    #[derive(Default)]
    struct RecordingAudit {
        entries: Mutex<Vec<AuditLogEntry>>,
    }

    #[async_trait]
    impl AuditLogService for RecordingAudit {
        async fn log(&self, entry: AuditLogEntry) -> Result<(), AuditError> {
            self.entries.lock().unwrap().push(entry);
            Ok(())
        }

        async fn list_by_task(&self, _task_id: &str) -> Result<Vec<AgentLog>, AuditError> {
            Ok(vec![])
        }

        async fn list_by_project(&self, _project_id: &str) -> Result<Vec<AgentLog>, AuditError> {
            Ok(vec![])
        }
    }

    struct EchoToolService;

    #[async_trait]
    impl ToolService for EchoToolService {
        async fn invoke(&self, name: &str, args: Value) -> Result<Value, ToolServiceError> {
            Ok(serde_json::json!({ "tool": name, "args": args }))
        }

        fn list_tools(&self) -> Vec<ToolDescription> {
            vec![]
        }
    }

    #[tokio::test]
    async fn audited_tool_service_logs_every_tool_call() {
        let audit = Arc::new(RecordingAudit::default());
        let service = AuditedToolService::new(
            Arc::new(EchoToolService),
            audit.clone(),
            Some("p1".into()),
            "t1",
            "coder",
            "trace-1",
        );

        let result = service
            .invoke("fs_read", serde_json::json!({ "path": "src/lib.rs" }))
            .await
            .unwrap();

        assert_eq!(result["tool"], "fs_read");
        let entries = audit.entries.lock().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].action, "tool_called");
        assert_eq!(entries[0].project_id.as_deref(), Some("p1"));
        assert_eq!(entries[0].task_id.as_deref(), Some("t1"));
        assert_eq!(entries[0].agent, "coder");
        assert_eq!(entries[0].level, AuditLogLevel::Info);
        assert_eq!(entries[0].metadata["args"]["path"], "src/lib.rs");
    }
}
