use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use thiserror::Error;
use ulid::Ulid;

use crate::models::{AgentLog, AuditLogLevel};
use crate::persistence::{LogRepository, PersistenceError};

/// Errors that can occur in [`AuditLogService`].
#[derive(Debug, Error)]
pub enum AuditError {
    #[error("persistence error: {0}")]
    Persistence(#[from] PersistenceError),
    #[error("invalid audit entry: {0}")]
    InvalidEntry(String),
}

/// A typed audit event following the schema from the architecture.
#[derive(Debug, Clone)]
pub enum AuditEvent {
    TaskStarted {
        task_id: String,
        agent: String,
        model: Option<String>,
        lora_id: Option<String>,
        prompt_id: Option<String>,
    },
    PromptSent {
        task_id: String,
        agent: String,
        prompt: String,
        token_count: Option<usize>,
        model: String,
    },
    ResponseReceived {
        task_id: String,
        agent: String,
        response: String,
        token_count: Option<usize>,
        finish_reason: Option<String>,
        latency_ms: u64,
    },
    ToolCalled {
        task_id: String,
        agent: String,
        tool_name: String,
        args: Value,
        result: Value,
        duration_ms: u64,
    },
    FileRead {
        task_id: String,
        agent: String,
        path: String,
        bytes_read: usize,
        hash: Option<String>,
    },
    FileWritten {
        task_id: String,
        agent: String,
        path: String,
        diff: Option<String>,
        bytes_written: usize,
    },
    TestRun {
        task_id: String,
        agent: String,
        command: String,
        exit_code: i32,
        stdout: String,
        stderr: String,
    },
    StatusChanged {
        task_id: String,
        agent: String,
        old_status: String,
        new_status: String,
        reason: Option<String>,
    },
    Thinking {
        task_id: String,
        agent: String,
        reasoning: String,
    },
    Error {
        task_id: Option<String>,
        agent: String,
        error: String,
        recoverable: bool,
    },
    HumanIntervention {
        task_id: String,
        agent: String,
        kind: String,
        result: String,
    },
}

impl AuditEvent {
    pub fn action(&self) -> &'static str {
        match self {
            AuditEvent::TaskStarted { .. } => "task_started",
            AuditEvent::PromptSent { .. } => "prompt_sent",
            AuditEvent::ResponseReceived { .. } => "response_received",
            AuditEvent::ToolCalled { .. } => "tool_called",
            AuditEvent::FileRead { .. } => "file_read",
            AuditEvent::FileWritten { .. } => "file_written",
            AuditEvent::TestRun { .. } => "test_run",
            AuditEvent::StatusChanged { .. } => "status_changed",
            AuditEvent::Thinking { .. } => "thinking",
            AuditEvent::Error { .. } => "error",
            AuditEvent::HumanIntervention { .. } => "human_intervention",
        }
    }

    pub fn task_id(&self) -> Option<String> {
        match self {
            AuditEvent::TaskStarted { task_id, .. }
            | AuditEvent::PromptSent { task_id, .. }
            | AuditEvent::ResponseReceived { task_id, .. }
            | AuditEvent::ToolCalled { task_id, .. }
            | AuditEvent::FileRead { task_id, .. }
            | AuditEvent::FileWritten { task_id, .. }
            | AuditEvent::TestRun { task_id, .. }
            | AuditEvent::StatusChanged { task_id, .. }
            | AuditEvent::Thinking { task_id, .. }
            | AuditEvent::HumanIntervention { task_id, .. } => Some(task_id.clone()),
            AuditEvent::Error { task_id, .. } => task_id.clone(),
        }
    }

    pub fn agent(&self) -> String {
        match self {
            AuditEvent::TaskStarted { agent, .. }
            | AuditEvent::PromptSent { agent, .. }
            | AuditEvent::ResponseReceived { agent, .. }
            | AuditEvent::ToolCalled { agent, .. }
            | AuditEvent::FileRead { agent, .. }
            | AuditEvent::FileWritten { agent, .. }
            | AuditEvent::TestRun { agent, .. }
            | AuditEvent::StatusChanged { agent, .. }
            | AuditEvent::Thinking { agent, .. }
            | AuditEvent::Error { agent, .. }
            | AuditEvent::HumanIntervention { agent, .. } => agent.clone(),
        }
    }

    /// Convert the event into a low-level entry. Large payloads are referenced
    /// by key in metadata and will be written to NDJSON by the bulk writer.
    pub fn into_entry(
        self,
        project_id: Option<String>,
        trace_id: impl Into<String>,
    ) -> AuditLogEntry {
        let action = self.action();
        let task_id = self.task_id();
        let agent = self.agent();
        let (message, metadata, level) = match self {
            AuditEvent::TaskStarted {
                model,
                lora_id,
                prompt_id,
                ..
            } => (
                None,
                serde_json::json!({
                    "model": model,
                    "lora_id": lora_id,
                    "prompt_id": prompt_id,
                }),
                AuditLogLevel::Info,
            ),
            AuditEvent::PromptSent {
                prompt,
                token_count,
                model,
                ..
            } => (
                None,
                serde_json::json!({
                    "prompt_ref": prompt,
                    "token_count": token_count,
                    "model": model,
                }),
                AuditLogLevel::Info,
            ),
            AuditEvent::ResponseReceived {
                response,
                token_count,
                finish_reason,
                latency_ms,
                ..
            } => (
                None,
                serde_json::json!({
                    "response_ref": response,
                    "token_count": token_count,
                    "finish_reason": finish_reason,
                    "latency_ms": latency_ms,
                }),
                AuditLogLevel::Info,
            ),
            AuditEvent::ToolCalled {
                tool_name,
                args,
                result,
                duration_ms,
                ..
            } => (
                Some(tool_name),
                serde_json::json!({
                    "args": args,
                    "result_ref": result,
                    "duration_ms": duration_ms,
                }),
                AuditLogLevel::Info,
            ),
            AuditEvent::FileRead {
                path,
                bytes_read,
                hash,
                ..
            } => (
                Some(path),
                serde_json::json!({
                    "bytes_read": bytes_read,
                    "hash": hash,
                }),
                AuditLogLevel::Info,
            ),
            AuditEvent::FileWritten {
                path,
                diff,
                bytes_written,
                ..
            } => (
                Some(path),
                serde_json::json!({
                    "diff_ref": diff,
                    "bytes_written": bytes_written,
                }),
                AuditLogLevel::Info,
            ),
            AuditEvent::TestRun {
                command,
                exit_code,
                stdout,
                stderr,
                ..
            } => (
                Some(command),
                serde_json::json!({
                    "exit_code": exit_code,
                    "stdout_ref": stdout,
                    "stderr_ref": stderr,
                }),
                if exit_code == 0 {
                    AuditLogLevel::Info
                } else {
                    AuditLogLevel::Warn
                },
            ),
            AuditEvent::StatusChanged {
                old_status,
                new_status,
                reason,
                ..
            } => (
                Some(format!("{old_status} -> {new_status}")),
                serde_json::json!({
                    "old_status": old_status,
                    "new_status": new_status,
                    "reason": reason,
                }),
                AuditLogLevel::Info,
            ),
            AuditEvent::Thinking { reasoning, .. } => (
                None,
                serde_json::json!({ "reasoning_ref": reasoning }),
                AuditLogLevel::Debug,
            ),
            AuditEvent::Error {
                error, recoverable, ..
            } => (
                Some(error),
                serde_json::json!({ "recoverable": recoverable }),
                AuditLogLevel::Error,
            ),
            AuditEvent::HumanIntervention { kind, result, .. } => (
                Some(kind),
                serde_json::json!({ "result": result }),
                AuditLogLevel::Info,
            ),
        };

        AuditLogEntry {
            id: Ulid::new().to_string(),
            project_id,
            task_id,
            trace_id: trace_id.into(),
            agent,
            action: action.to_string(),
            message,
            level,
            timestamp: chrono::Utc::now().timestamp_millis(),
            metadata,
        }
    }
}

/// A single audit log entry.
#[derive(Debug, Clone)]
pub struct AuditLogEntry {
    pub id: String,
    pub project_id: Option<String>,
    pub task_id: Option<String>,
    pub trace_id: String,
    pub agent: String,
    pub action: String,
    pub message: Option<String>,
    pub level: AuditLogLevel,
    pub timestamp: i64,
    pub metadata: Value,
}

impl AuditLogEntry {
    pub fn new(agent: impl Into<String>, action: impl Into<String>) -> Self {
        Self {
            id: Ulid::new().to_string(),
            project_id: None,
            task_id: None,
            trace_id: String::new(),
            agent: agent.into(),
            action: action.into(),
            message: None,
            level: AuditLogLevel::Info,
            timestamp: chrono::Utc::now().timestamp_millis(),
            metadata: Value::Null,
        }
    }

    pub fn project_id(mut self, id: impl Into<String>) -> Self {
        self.project_id = Some(id.into());
        self
    }

    pub fn task_id(mut self, id: impl Into<String>) -> Self {
        self.task_id = Some(id.into());
        self
    }

    pub fn trace_id(mut self, id: impl Into<String>) -> Self {
        self.trace_id = id.into();
        self
    }

    pub fn message(mut self, message: impl Into<String>) -> Self {
        self.message = Some(message.into());
        self
    }

    pub fn level(mut self, level: AuditLogLevel) -> Self {
        self.level = level;
        self
    }

    pub fn metadata(mut self, metadata: Value) -> Self {
        self.metadata = metadata;
        self
    }
}

impl From<AuditLogEntry> for AgentLog {
    fn from(entry: AuditLogEntry) -> Self {
        Self {
            id: entry.id,
            project_id: entry.project_id,
            task_id: entry.task_id,
            agent: entry.agent,
            action: entry.action,
            message: entry.message,
            level: entry.level.to_string(),
            timestamp: entry.timestamp,
            metadata: entry.metadata,
        }
    }
}

/// Business-logic service for audit logging.
#[async_trait]
pub trait AuditLogService: Send + Sync {
    /// Persist a single audit entry.
    async fn log(&self, entry: AuditLogEntry) -> Result<(), AuditError>;

    /// List audit logs for a specific task.
    async fn list_by_task(&self, task_id: &str) -> Result<Vec<AgentLog>, AuditError>;

    /// List audit logs for a specific project.
    async fn list_by_project(&self, project_id: &str) -> Result<Vec<AgentLog>, AuditError>;
}

/// Default implementation of [`AuditLogService`].
pub struct AuditLogServiceImpl<R> {
    repo: Arc<R>,
}

impl<R> AuditLogServiceImpl<R> {
    pub fn new(repo: Arc<R>) -> Self {
        Self { repo }
    }
}

#[async_trait]
impl<R> AuditLogService for AuditLogServiceImpl<R>
where
    R: LogRepository + 'static,
{
    async fn log(&self, entry: AuditLogEntry) -> Result<(), AuditError> {
        if entry.agent.is_empty() {
            return Err(AuditError::InvalidEntry("agent must be non-empty".into()));
        }
        if entry.action.is_empty() {
            return Err(AuditError::InvalidEntry("action must be non-empty".into()));
        }
        self.repo.insert_agent_log(&entry.into()).await?;
        Ok(())
    }

    async fn list_by_task(&self, task_id: &str) -> Result<Vec<AgentLog>, AuditError> {
        Ok(self.repo.list_logs_by_task(task_id).await?)
    }

    async fn list_by_project(&self, project_id: &str) -> Result<Vec<AgentLog>, AuditError> {
        Ok(self.repo.list_logs_by_project(project_id).await?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::{LogRepository, PersistenceError};
    use async_trait::async_trait;
    use std::sync::Mutex;

    #[derive(Default)]
    struct MockRepo {
        logs: Mutex<Vec<AgentLog>>,
    }

    #[async_trait]
    impl LogRepository for MockRepo {
        async fn insert_agent_log(&self, log: &AgentLog) -> Result<(), PersistenceError> {
            self.logs.lock().unwrap().push(log.clone());
            Ok(())
        }

        async fn list_logs_by_task(
            &self,
            task_id: &str,
        ) -> Result<Vec<AgentLog>, PersistenceError> {
            Ok(self
                .logs
                .lock()
                .unwrap()
                .iter()
                .filter(|l| l.task_id.as_deref() == Some(task_id))
                .cloned()
                .collect())
        }

        async fn list_logs_by_project(
            &self,
            project_id: &str,
        ) -> Result<Vec<AgentLog>, PersistenceError> {
            Ok(self
                .logs
                .lock()
                .unwrap()
                .iter()
                .filter(|l| l.project_id.as_deref() == Some(project_id))
                .cloned()
                .collect())
        }
    }

    fn service() -> AuditLogServiceImpl<MockRepo> {
        AuditLogServiceImpl::new(Arc::new(MockRepo::default()))
    }

    #[tokio::test]
    async fn log_persists_entry() {
        let svc = service();
        let entry = AuditLogEntry::new("coder", "execute")
            .project_id("proj-1")
            .task_id("task-1")
            .message("generated code");

        svc.log(entry).await.unwrap();

        let logs = svc.list_by_task("task-1").await.unwrap();
        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0].agent, "coder");
        assert_eq!(logs[0].action, "execute");
        assert_eq!(logs[0].project_id, Some("proj-1".into()));
    }

    #[tokio::test]
    async fn list_by_project_filters_correctly() {
        let svc = service();
        svc.log(
            AuditLogEntry::new("coder", "execute")
                .project_id("proj-1")
                .task_id("task-1"),
        )
        .await
        .unwrap();
        svc.log(
            AuditLogEntry::new("qa", "review")
                .project_id("proj-2")
                .task_id("task-2"),
        )
        .await
        .unwrap();

        let logs = svc.list_by_project("proj-1").await.unwrap();
        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0].agent, "coder");
    }

    #[tokio::test]
    async fn log_rejects_empty_agent() {
        let svc = service();
        let entry = AuditLogEntry::new("", "execute");

        let err = svc.log(entry).await.unwrap_err();
        assert!(matches!(err, AuditError::InvalidEntry(_)));
    }

    #[tokio::test]
    async fn log_rejects_empty_action() {
        let svc = service();
        let entry = AuditLogEntry::new("coder", "");

        let err = svc.log(entry).await.unwrap_err();
        assert!(matches!(err, AuditError::InvalidEntry(_)));
    }

    #[test]
    fn task_started_event_maps_to_entry() {
        let event = AuditEvent::TaskStarted {
            task_id: "t1".into(),
            agent: "coder".into(),
            model: Some("qwen-7b".into()),
            lora_id: None,
            prompt_id: Some("pv-1".into()),
        };
        let entry = event.into_entry(Some("p1".into()), "");
        assert_eq!(entry.action, "task_started");
        assert_eq!(entry.agent, "coder");
        assert_eq!(entry.task_id, Some("t1".into()));
        assert_eq!(entry.project_id, Some("p1".into()));
        assert_eq!(entry.metadata["model"], "qwen-7b");
    }

    #[test]
    fn prompt_sent_event_stores_prompt_in_metadata() {
        let event = AuditEvent::PromptSent {
            task_id: "t1".into(),
            agent: "coder".into(),
            prompt: "write tests first".into(),
            token_count: Some(1200),
            model: "qwen-7b".into(),
        };
        let entry = event.into_entry(None, "");
        assert_eq!(entry.action, "prompt_sent");
        assert_eq!(entry.metadata["prompt_ref"], "write tests first");
        assert_eq!(entry.metadata["token_count"], 1200);
    }

    #[test]
    fn test_run_event_sets_warn_level_on_failure() {
        let event = AuditEvent::TestRun {
            task_id: "t1".into(),
            agent: "qa".into(),
            command: "cargo test".into(),
            exit_code: 1,
            stdout: "".into(),
            stderr: "failed".into(),
        };
        let entry = event.into_entry(None, "");
        assert_eq!(entry.action, "test_run");
        assert!(matches!(entry.level, AuditLogLevel::Warn));
        assert_eq!(entry.metadata["exit_code"], 1);
    }

    #[test]
    fn error_event_is_marked_error_level() {
        let event = AuditEvent::Error {
            task_id: Some("t1".into()),
            agent: "coder".into(),
            error: "timeout".into(),
            recoverable: false,
        };
        let entry = event.into_entry(None, "");
        assert_eq!(entry.action, "error");
        assert!(matches!(entry.level, AuditLogLevel::Error));
        assert_eq!(entry.message, Some("timeout".into()));
    }
}
