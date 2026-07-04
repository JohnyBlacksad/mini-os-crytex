//! Editor event bridge: turns editor events into an up-to-date project snapshot.

use crytex_core::bus::Event;
use crytex_core::models::ProjectSnapshot;
use crytex_core::persistence::ProjectSnapshotRepository;
use crytex_core::services::EventService;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

/// Cursor position inside an open file.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct CursorPosition {
    pub line: u32,
    pub character: u32,
}

/// State of a single open file in the editor.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OpenFile {
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(default)]
    pub cursor: CursorPosition,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub diagnostics: Vec<Value>,
}

/// IDE-specific project state persisted as a `ProjectSnapshot`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IdeProjectState {
    pub project_id: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub open_files: Vec<OpenFile>,
}

impl IdeProjectState {
    pub fn new(project_id: String) -> Self {
        Self {
            project_id,
            open_files: Vec::new(),
        }
    }

    fn upsert_open_file<F>(&mut self, path: &str, updater: F)
    where
        F: FnOnce(&mut OpenFile),
    {
        match self.open_files.iter_mut().find(|f| f.path == path) {
            Some(file) => updater(file),
            None => {
                let mut file = OpenFile {
                    path: path.to_string(),
                    language: None,
                    cursor: CursorPosition::default(),
                    diagnostics: Vec::new(),
                };
                updater(&mut file);
                self.open_files.push(file);
            }
        }
    }

    fn close_file(&mut self, path: &str) {
        self.open_files.retain(|f| f.path != path);
    }
}

/// Bridges editor events to persisted project snapshots.
pub struct EditorBridge {
    handle: JoinHandle<()>,
    states: Arc<Mutex<HashMap<String, IdeProjectState>>>,
}

impl EditorBridge {
    pub async fn start(
        event_service: Arc<dyn EventService>,
        snapshots: Arc<dyn ProjectSnapshotRepository>,
    ) -> Self {
        let states: Arc<Mutex<HashMap<String, IdeProjectState>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let states_clone = states.clone();
        let mut rx = event_service.subscribe();

        let handle = tokio::spawn(async move {
            while let Ok(event) = rx.recv().await {
                Self::handle_event(&event, &states_clone, snapshots.as_ref()).await;
            }
        });

        Self { handle, states }
    }

    /// Get a clone of the current in-memory state for a project.
    pub async fn state(&self, project_id: &str) -> Option<IdeProjectState> {
        self.states.lock().await.get(project_id).cloned()
    }

    /// Stop the background listener.
    pub fn stop(self) {
        self.handle.abort();
    }

    async fn handle_event(
        event: &Event,
        states: &Mutex<HashMap<String, IdeProjectState>>,
        snapshots: &dyn ProjectSnapshotRepository,
    ) {
        let (project_id, changed) = match event {
            Event::FileOpened {
                project_id,
                file_path,
                language,
            } => {
                let mut guard = states.lock().await;
                let state = guard
                    .entry(project_id.clone())
                    .or_insert_with(|| IdeProjectState::new(project_id.clone()));
                state.upsert_open_file(file_path, |f| {
                    f.language.clone_from(language);
                });
                (project_id.clone(), true)
            }
            Event::FileClosed {
                project_id,
                file_path,
            } => {
                let mut guard = states.lock().await;
                if let Some(state) = guard.get_mut(project_id) {
                    state.close_file(file_path);
                }
                (project_id.clone(), true)
            }
            Event::CursorMoved {
                project_id,
                file_path,
                line,
                character,
            } => {
                let mut guard = states.lock().await;
                let state = guard
                    .entry(project_id.clone())
                    .or_insert_with(|| IdeProjectState::new(project_id.clone()));
                state.upsert_open_file(file_path, |f| {
                    f.cursor = CursorPosition {
                        line: *line,
                        character: *character,
                    };
                });
                (project_id.clone(), true)
            }
            Event::DiagnosticsReceived {
                project_id,
                file_path,
                diagnostics,
            } => {
                let mut guard = states.lock().await;
                let state = guard
                    .entry(project_id.clone())
                    .or_insert_with(|| IdeProjectState::new(project_id.clone()));
                state.upsert_open_file(file_path, |f| {
                    f.diagnostics.clone_from(diagnostics);
                });
                (project_id.clone(), true)
            }
            _ => return,
        };

        if changed {
            let state = states.lock().await.get(&project_id).cloned();
            if let Some(state) = state {
                let snapshot_id = format!(
                    "ide-{}-{}",
                    project_id,
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis()
                );
                let snapshot = ProjectSnapshot {
                    id: snapshot_id,
                    project_id: project_id.clone(),
                    name: "ide".to_string(),
                    state_json: match serde_json::to_value(&state) {
                        Ok(v) => v,
                        Err(_) => Value::Null,
                    },
                    created_at: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis() as i64,
                };
                let _ = snapshots.insert_project_snapshot(&snapshot).await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crytex_core::bus::EventBus;
    use crytex_core::persistence::{PersistenceError, ProjectSnapshotRepository};
    use crytex_core::services::{EventService, EventServiceImpl};

    struct CaptureSnapshotRepo {
        snapshots: Mutex<Vec<ProjectSnapshot>>,
    }

    #[async_trait::async_trait]
    impl ProjectSnapshotRepository for CaptureSnapshotRepo {
        async fn insert_project_snapshot(
            &self,
            snapshot: &ProjectSnapshot,
        ) -> Result<(), PersistenceError> {
            self.snapshots.lock().await.push(snapshot.clone());
            Ok(())
        }

        async fn get_project_snapshot(
            &self,
            _id: &str,
        ) -> Result<Option<ProjectSnapshot>, PersistenceError> {
            Ok(None)
        }

        async fn list_project_snapshots(
            &self,
            _project_id: &str,
        ) -> Result<Vec<ProjectSnapshot>, PersistenceError> {
            Ok(vec![])
        }
    }

    #[tokio::test]
    async fn bridge_updates_open_files_in_project_snapshot() {
        let bus = Arc::new(EventBus::new());
        let event_service: Arc<dyn EventService> = Arc::new(EventServiceImpl::new(bus.clone()));
        let repo = Arc::new(CaptureSnapshotRepo {
            snapshots: Mutex::new(Vec::new()),
        });

        let bridge = EditorBridge::start(event_service.clone(), repo.clone()).await;

        event_service.publish(Event::FileOpened {
            project_id: "p1".into(),
            file_path: "src/lib.rs".into(),
            language: Some("rust".into()),
        });
        event_service.publish(Event::CursorMoved {
            project_id: "p1".into(),
            file_path: "src/lib.rs".into(),
            line: 42,
            character: 10,
        });

        // Give the spawned task a moment to process the events.
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        let state = bridge.state("p1").await.expect("state exists");
        assert_eq!(state.open_files.len(), 1);
        assert_eq!(state.open_files[0].path, "src/lib.rs");
        assert_eq!(state.open_files[0].cursor.line, 42);

        let snapshots = repo.snapshots.lock().await;
        assert!(!snapshots.is_empty());
        assert_eq!(snapshots[0].project_id, "p1");

        bridge.stop();
    }
}
