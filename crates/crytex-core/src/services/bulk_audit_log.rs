//! Bulk audit log writer with NDJSON large-payload offload.
//!
//! Incoming [`AuditLogEntry`] values are sent over a channel; a background task
//! extracts large payloads (prompts, responses, diffs, command output) to
//! per-project NDJSON files and persists only lightweight metadata to SQLite.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::Utc;
use serde_json::Value;
use tokio::fs::{self, OpenOptions};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::{mpsc, oneshot};

use crate::models::AgentLog;
use crate::persistence::LogRepository;
use crate::services::audit_log_service::{AuditError, AuditLogEntry, AuditLogService};

/// Metadata keys whose values are considered large and should be offloaded to NDJSON.
const LARGE_PAYLOAD_KEYS: &[&str] = &[
    "prompt_ref",
    "response_ref",
    "result_ref",
    "diff_ref",
    "stdout_ref",
    "stderr_ref",
    "reasoning_ref",
];

#[derive(Debug)]
enum WriterCommand {
    Entry(Box<AuditLogEntry>),
    Flush(oneshot::Sender<()>),
}

/// Audit-log service that never blocks inference on I/O.
pub struct BulkAuditLogService<R> {
    tx: mpsc::UnboundedSender<WriterCommand>,
    repo: Arc<R>,
    base_dir: PathBuf,
}

impl<R> BulkAuditLogService<R>
where
    R: LogRepository + Send + Sync + 'static,
{
    /// Start a new bulk writer. `base_dir` is the NDJSON log root, e.g. `~/.crytex/logs`.
    pub fn new(repo: Arc<R>, base_dir: impl Into<PathBuf>) -> Self {
        let (tx, mut rx) = mpsc::unbounded_channel::<WriterCommand>();
        let base_dir = base_dir.into();
        let writer_dir = base_dir.clone();
        let writer_repo = repo.clone();

        tokio::spawn(async move {
            while let Some(cmd) = rx.recv().await {
                match cmd {
                    WriterCommand::Entry(mut entry) => {
                        if let Err(e) = write_entry(&writer_dir, &writer_repo, &mut entry).await {
                            tracing::error!(error = %e, "failed to write audit log entry");
                        }
                    }
                    WriterCommand::Flush(ack) => {
                        let _ = ack.send(());
                    }
                }
            }
        });

        Self { tx, repo, base_dir }
    }

    /// Enqueue an entry for async persistence.
    pub fn log(&self, entry: AuditLogEntry) {
        let _ = self.tx.send(WriterCommand::Entry(Box::new(entry)));
    }

    /// Wait until all currently enqueued entries have been persisted.
    pub async fn flush(&self) {
        let (tx, rx) = oneshot::channel();
        let _ = self.tx.send(WriterCommand::Flush(tx));
        let _ = rx.await;
    }

    /// Reconstruct the full audit trace for a task, substituting NDJSON payloads.
    pub async fn replay_task(&self, task_id: &str) -> Result<Vec<AgentLog>, AuditError> {
        let mut logs = self.repo.list_logs_by_task(task_id).await?;
        let refs = collect_ndjson_refs(&self.base_dir, &logs);
        if refs.is_empty() {
            return Ok(logs);
        }

        let payloads = load_ndjson_payloads(&refs).await?;
        substitute_payloads(&mut logs, &payloads);
        Ok(logs)
    }
}

#[async_trait::async_trait]
impl<R> AuditLogService for BulkAuditLogService<R>
where
    R: LogRepository + Send + Sync + 'static,
{
    async fn log(&self, entry: AuditLogEntry) -> Result<(), AuditError> {
        self.log(entry);
        Ok(())
    }

    async fn list_by_task(&self, task_id: &str) -> Result<Vec<AgentLog>, AuditError> {
        Ok(self.repo.list_logs_by_task(task_id).await?)
    }

    async fn list_by_project(&self, project_id: &str) -> Result<Vec<AgentLog>, AuditError> {
        Ok(self.repo.list_logs_by_project(project_id).await?)
    }
}

async fn write_entry<R>(
    base_dir: &Path,
    repo: &Arc<R>,
    entry: &mut AuditLogEntry,
) -> Result<(), AuditError>
where
    R: LogRepository + Send + Sync,
{
    let project_id = entry.project_id.clone().unwrap_or_else(|| "_global".into());
    offload_large_payloads(base_dir, &project_id, entry).await?;
    repo.insert_agent_log(&AgentLog::from(entry.clone()))
        .await
        .map_err(AuditError::Persistence)
}

/// Extract large payload fields from metadata, append them to NDJSON, and replace
/// the metadata values with references.
async fn offload_large_payloads(
    base_dir: &Path,
    project_id: &str,
    entry: &mut AuditLogEntry,
) -> Result<(), AuditError> {
    let obj = match entry.metadata.as_object_mut() {
        Some(obj) => obj,
        None => return Ok(()),
    };

    let mut payloads = Vec::new();
    for key in LARGE_PAYLOAD_KEYS {
        if let Some(value) = obj.remove(*key) {
            let text = match value {
                Value::String(s) => s,
                other => other.to_string(),
            };
            payloads.push((key.to_string(), text));
        }
    }

    if payloads.is_empty() {
        return Ok(());
    }

    let dir = base_dir.join(sanitize_project_id(project_id));
    fs::create_dir_all(&dir).await.map_err(|e| {
        AuditError::InvalidEntry(format!(
            "cannot create log directory {}: {e}",
            dir.display()
        ))
    })?;

    let path = dir.join(format!("{}.ndjson", Utc::now().format("%Y-%m-%d")));
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .await
        .map_err(|e| {
            AuditError::InvalidEntry(format!("cannot open ndjson {}: {e}", path.display()))
        })?;

    for (key, payload) in payloads {
        let line = serde_json::json!({
            "entry_id": entry.id,
            "key": key,
            "payload": payload,
        });
        let mut bytes = serde_json::to_vec(&line)
            .map_err(|e| AuditError::InvalidEntry(format!("cannot serialize ndjson line: {e}")))?;
        bytes.push(b'\n');
        file.write_all(&bytes)
            .await
            .map_err(|e| AuditError::InvalidEntry(format!("cannot write ndjson line: {e}")))?;

        obj.insert(
            key,
            Value::String(format!(
                "ndjson:{project_id}/{}.ndjson#{}",
                Utc::now().format("%Y-%m-%d"),
                entry.id
            )),
        );
    }

    file.flush()
        .await
        .map_err(|e| AuditError::InvalidEntry(format!("cannot flush ndjson: {e}")))?;

    Ok(())
}

fn sanitize_project_id(id: &str) -> String {
    id.replace(['/', '\\', ':'], "_")
}

#[derive(Debug, Clone)]
struct NdjsonRef {
    path: PathBuf,
    entry_id: String,
    key: String,
}

fn collect_ndjson_refs(base_dir: &Path, logs: &[AgentLog]) -> Vec<NdjsonRef> {
    let mut refs = Vec::new();
    for log in logs {
        let Some(obj) = log.metadata.as_object() else {
            continue;
        };
        for (key, value) in obj {
            let Some(text) = value.as_str() else {
                continue;
            };
            if let Some((project_id, date, entry_id)) = parse_ndjson_ref(text) {
                refs.push(NdjsonRef {
                    path: base_dir
                        .join(sanitize_project_id(&project_id))
                        .join(format!("{date}.ndjson")),
                    entry_id,
                    key: key.clone(),
                });
            }
        }
    }
    refs
}

fn parse_ndjson_ref(value: &str) -> Option<(String, String, String)> {
    // Format: ndjson:<project_id>/<date>.ndjson#<entry_id>
    let rest = value.strip_prefix("ndjson:")?;
    let (path, entry_id) = rest.split_once('#')?;
    let (project_id, filename) = path.rsplit_once('/')?;
    let date = filename.strip_suffix(".ndjson")?;
    Some((
        project_id.to_string(),
        date.to_string(),
        entry_id.to_string(),
    ))
}

async fn load_ndjson_payloads(
    refs: &[NdjsonRef],
) -> Result<HashMap<(String, String), String>, AuditError> {
    let mut payloads: HashMap<(String, String), String> = HashMap::new();
    for ref_ in refs {
        let file = fs::File::open(&ref_.path).await.map_err(|e| {
            AuditError::InvalidEntry(format!("cannot open ndjson {}: {e}", ref_.path.display()))
        })?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let Ok(record) = serde_json::from_str::<Value>(&line) else {
                continue;
            };
            let record_entry_id = record.get("entry_id").and_then(|v| v.as_str());
            let record_key = record.get("key").and_then(|v| v.as_str());
            let record_payload = record.get("payload").and_then(|v| v.as_str());
            if record_entry_id == Some(ref_.entry_id.as_str())
                && record_key == Some(ref_.key.as_str())
                && let Some(payload) = record_payload
            {
                payloads.insert(
                    (ref_.entry_id.clone(), ref_.key.clone()),
                    payload.to_string(),
                );
                break;
            }
        }
    }
    Ok(payloads)
}

fn substitute_payloads(logs: &mut [AgentLog], payloads: &HashMap<(String, String), String>) {
    for log in logs {
        let Some(obj) = log.metadata.as_object_mut() else {
            continue;
        };
        for (key, value) in obj.iter_mut() {
            if let Some(text) = value.as_str()
                && parse_ndjson_ref(text).is_some()
                && let Some(payload) = payloads.get(&(log.id.clone(), key.clone()))
            {
                *value = Value::String(payload.clone());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::{LogRepository, PersistenceError};
    use crate::services::audit_log_service::AuditEvent;
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

    #[tokio::test]
    async fn bulk_writer_persists_metadata_to_sqlite() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = Arc::new(MockRepo::default());
        let svc = BulkAuditLogService::new(repo.clone(), tmp.path());

        let event = AuditEvent::TaskStarted {
            task_id: "t1".into(),
            agent: "coder".into(),
            model: Some("qwen".into()),
            lora_id: None,
            prompt_id: None,
        };
        svc.log(event.into_entry(Some("p1".into()), ""));
        svc.flush().await;

        let logs = svc.list_by_task("t1").await.unwrap();
        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0].action, "task_started");
    }

    #[tokio::test]
    async fn bulk_writer_offloads_large_payload_to_ndjson() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = Arc::new(MockRepo::default());
        let svc = BulkAuditLogService::new(repo.clone(), tmp.path());

        let event = AuditEvent::PromptSent {
            task_id: "t1".into(),
            agent: "coder".into(),
            prompt: "this is a large prompt".into(),
            token_count: Some(42),
            model: "qwen".into(),
        };
        svc.log(event.into_entry(Some("p1".into()), ""));
        svc.flush().await;

        let logs = svc.list_by_task("t1").await.unwrap();
        assert_eq!(logs.len(), 1);
        let metadata = &logs[0].metadata;
        assert!(
            metadata["prompt_ref"]
                .as_str()
                .unwrap()
                .starts_with("ndjson:"),
            "expected ndjson reference, got {:?}",
            metadata["prompt_ref"]
        );

        // Verify the NDJSON file exists and contains the payload.
        let ndjson_path = tmp
            .path()
            .join("p1")
            .join(format!("{}.ndjson", Utc::now().format("%Y-%m-%d")));
        let content = tokio::fs::read_to_string(&ndjson_path).await.unwrap();
        assert!(content.contains("this is a large prompt"));
        assert!(content.contains("\"key\":\"prompt_ref\""));
    }

    #[tokio::test]
    async fn replay_task_restores_large_payloads() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = Arc::new(MockRepo::default());
        let svc = BulkAuditLogService::new(repo.clone(), tmp.path());

        let event = AuditEvent::PromptSent {
            task_id: "t1".into(),
            agent: "coder".into(),
            prompt: "original prompt text".into(),
            token_count: Some(42),
            model: "qwen".into(),
        };
        svc.log(event.into_entry(Some("p1".into()), ""));
        svc.flush().await;

        let replayed = svc.replay_task("t1").await.unwrap();
        assert_eq!(replayed.len(), 1);
        assert_eq!(replayed[0].metadata["prompt_ref"], "original prompt text");
    }
}
