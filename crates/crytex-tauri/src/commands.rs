//! Plain async command functions that map 1-to-1 to planned Tauri IPC calls.

use crytex_core::bus::Event;
use crytex_core::metrics::MetricsService;
use crytex_core::models::{KanbanState, Project, Task};
use crytex_core::persistence::ProjectSnapshotRepository;
use crytex_core::services::{
    AuditLogService, EventService, ProjectError, ProjectService, TaskError, TaskService,
};
use crytex_core::state_export::{ProjectState, StateExportError, export_project_state};
use std::sync::Arc;
use tokio::sync::broadcast;

/// Errors returned by Tauri command handlers.
#[derive(Debug, thiserror::Error)]
pub enum TauriCommandError {
    #[error("project error: {0}")]
    Project(ProjectError),
    #[error("task error: {0}")]
    Task(TaskError),
    #[error("state export error: {0}")]
    StateExport(#[from] StateExportError),
}

impl From<ProjectError> for TauriCommandError {
    fn from(err: ProjectError) -> Self {
        Self::Project(err)
    }
}

impl From<TaskError> for TauriCommandError {
    fn from(err: TaskError) -> Self {
        Self::Task(err)
    }
}

/// Return the full serializable state for a project.
pub async fn get_project_state(
    project_service: Arc<dyn ProjectService>,
    task_service: Arc<dyn TaskService>,
    audit_service: Arc<dyn AuditLogService>,
    snapshot_repo: Arc<dyn ProjectSnapshotRepository>,
    metrics_service: Arc<dyn MetricsService>,
    project_id: &str,
) -> Result<ProjectState, TauriCommandError> {
    Ok(export_project_state(
        project_service,
        task_service,
        audit_service,
        snapshot_repo,
        metrics_service,
        project_id,
    )
    .await?)
}

/// List all projects.
pub async fn list_projects(
    project_service: Arc<dyn ProjectService>,
) -> Result<Vec<Project>, TauriCommandError> {
    Ok(project_service.list().await?)
}

/// Return the Kanban board view for a project.
pub async fn kanban_state(
    project_service: Arc<dyn ProjectService>,
    project_id: &str,
) -> Result<KanbanState, TauriCommandError> {
    Ok(project_service.kanban_state(project_id).await?)
}

/// List tasks belonging to a project.
pub async fn list_tasks(
    task_service: Arc<dyn TaskService>,
    project_id: &str,
) -> Result<Vec<Task>, TauriCommandError> {
    Ok(task_service.list_by_project(project_id).await?)
}

/// Subscribe to the kernel event stream.
pub async fn subscribe_to_events(
    event_service: Arc<dyn EventService>,
) -> Result<broadcast::Receiver<Event>, TauriCommandError> {
    Ok(event_service.subscribe())
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use crytex_core::metrics::{MetricsError, MetricsSnapshot};
    use crytex_core::models::{
        AgentLog, KanbanColumn, KanbanTaskCard, ProjectSnapshot, TaskDependency, TaskStatus,
    };
    use crytex_core::persistence::PersistenceError;
    use crytex_core::services::{
        AuditError, AuditLogEntry, CreateProjectRequest, CreateTaskRequest, EventHandler,
        ProjectService,
    };
    use serde_json::{Value, json};

    fn dummy_project(id: &str) -> Project {
        Project {
            id: id.into(),
            name: "demo".into(),
            root_path: "/tmp/demo".into(),
            created_at: 0,
            updated_at: 0,
            metadata: json!({}),
        }
    }

    struct DummyProjectService {
        project: Project,
    }

    #[async_trait]
    impl ProjectService for DummyProjectService {
        async fn create(
            &self,
            _request: CreateProjectRequest<'_>,
        ) -> Result<Project, ProjectError> {
            Ok(self.project.clone())
        }
        async fn get(&self, _id: &str) -> Result<Option<Project>, ProjectError> {
            Ok(Some(self.project.clone()))
        }
        async fn list(&self) -> Result<Vec<Project>, ProjectError> {
            Ok(vec![self.project.clone()])
        }
        async fn update_metadata(
            &self,
            _id: &str,
            metadata: Value,
        ) -> Result<Project, ProjectError> {
            let mut p = self.project.clone();
            p.metadata = metadata;
            Ok(p)
        }
        async fn kanban_state(&self, project_id: &str) -> Result<KanbanState, ProjectError> {
            Ok(KanbanState {
                project_id: project_id.to_string(),
                columns: vec![KanbanColumn {
                    status: TaskStatus::Pending,
                    title: "pending".into(),
                    tasks: vec![KanbanTaskCard {
                        id: "t1".into(),
                        title: "task".into(),
                        kind: "codegen".into(),
                        status: TaskStatus::Pending,
                        priority: 0,
                        assigned_agent: None,
                    }],
                }],
            })
        }
    }

    struct DummyTaskService;

    #[async_trait]
    impl TaskService for DummyTaskService {
        async fn submit(&self, _request: CreateTaskRequest) -> Result<Task, TaskError> {
            Err(TaskError::NotFound("mock".into()))
        }
        async fn add_dependency(&self, _dep: TaskDependency) -> Result<(), TaskError> {
            Ok(())
        }
        async fn get(&self, _id: &str) -> Result<Option<Task>, TaskError> {
            Ok(None)
        }
        async fn list_by_project(&self, project_id: &str) -> Result<Vec<Task>, TaskError> {
            Ok(vec![Task {
                id: "t1".into(),
                project_id: project_id.to_string(),
                parent_id: None,
                title: "task".into(),
                description: None,
                kind: "codegen".into(),
                status: TaskStatus::Pending,
                assigned_agent: None,
                priority: 0,
                payload: json!({}),
                result: None,
                created_at: 0,
                started_at: None,
                finished_at: None,
                iteration_count: 0,
                priority_score: 0.0,
                critic_score: None,
                human_score: None,
                prompt_version_id: None,
                lora_adapter_id: None,
                trace_id: "trace".into(),
            }])
        }
        async fn list_ready(&self) -> Result<Vec<Task>, TaskError> {
            Ok(vec![])
        }
        async fn set_status(&self, _id: &str, _status: TaskStatus) -> Result<Task, TaskError> {
            Err(TaskError::NotFound("mock".into()))
        }
        async fn cancel(&self, _id: &str) -> Result<Task, TaskError> {
            Err(TaskError::NotFound("mock".into()))
        }
        async fn set_result(&self, _id: &str, _result: Value) -> Result<Task, TaskError> {
            Err(TaskError::NotFound("mock".into()))
        }
        async fn set_critic_score(&self, _id: &str, _score: f64) -> Result<Task, TaskError> {
            Err(TaskError::NotFound("mock".into()))
        }
        async fn set_human_score(&self, _id: &str, _score: f64) -> Result<Task, TaskError> {
            Err(TaskError::NotFound("mock".into()))
        }
        async fn retry(&self, _id: &str, _feedback: Option<&str>) -> Result<Task, TaskError> {
            Err(TaskError::NotFound("mock".into()))
        }
        async fn load_all_tasks(&self) -> Result<Vec<Task>, TaskError> {
            Ok(vec![])
        }
        async fn update_task(&self, _task: &Task) -> Result<(), TaskError> {
            Ok(())
        }
    }

    struct DummyAuditService;

    #[async_trait]
    impl AuditLogService for DummyAuditService {
        async fn log(&self, _entry: AuditLogEntry) -> Result<(), AuditError> {
            Ok(())
        }
        async fn list_by_task(&self, _task_id: &str) -> Result<Vec<AgentLog>, AuditError> {
            Ok(vec![])
        }
        async fn list_by_project(&self, _project_id: &str) -> Result<Vec<AgentLog>, AuditError> {
            Ok(vec![])
        }
    }

    struct DummySnapshotRepo;

    #[async_trait]
    impl ProjectSnapshotRepository for DummySnapshotRepo {
        async fn insert_project_snapshot(
            &self,
            _snapshot: &ProjectSnapshot,
        ) -> Result<(), PersistenceError> {
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

    struct DummyMetricsService;

    #[async_trait]
    impl MetricsService for DummyMetricsService {
        async fn snapshot(&self) -> Result<MetricsSnapshot, MetricsError> {
            Ok(MetricsSnapshot::default())
        }
        async fn record_task_completion(
            &self,
            _latency_ms: u64,
            _success: bool,
        ) -> Result<(), MetricsError> {
            Ok(())
        }
        async fn record_cache_hit(&self) -> Result<(), MetricsError> {
            Ok(())
        }
        async fn record_cache_miss(&self) -> Result<(), MetricsError> {
            Ok(())
        }
        async fn history(
            &self,
            _from: i64,
            _to: i64,
        ) -> Result<Vec<MetricsSnapshot>, MetricsError> {
            Ok(vec![])
        }
    }

    struct DummyEventService;

    #[async_trait]
    impl EventService for DummyEventService {
        fn publish(&self, _event: Event) {}
        fn subscribe(&self) -> broadcast::Receiver<Event> {
            let (tx, _rx) = broadcast::channel(16);
            tx.subscribe()
        }
        async fn start_handler(&self, _handler: Arc<dyn EventHandler>) {}
    }

    #[tokio::test]
    async fn list_projects_returns_projects() {
        let projects = list_projects(Arc::new(DummyProjectService {
            project: dummy_project("p1"),
        }))
        .await
        .unwrap();
        assert_eq!(projects.len(), 1);
        assert_eq!(projects[0].id, "p1");
    }

    #[tokio::test]
    async fn kanban_state_returns_columns() {
        let kanban = kanban_state(
            Arc::new(DummyProjectService {
                project: dummy_project("p1"),
            }),
            "p1",
        )
        .await
        .unwrap();
        assert_eq!(kanban.project_id, "p1");
        assert!(!kanban.columns.is_empty());
    }

    #[tokio::test]
    async fn list_tasks_returns_tasks() {
        let tasks = list_tasks(Arc::new(DummyTaskService), "p1").await.unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].id, "t1");
    }

    #[tokio::test]
    async fn get_project_state_assembles_state() {
        let state = get_project_state(
            Arc::new(DummyProjectService {
                project: dummy_project("p1"),
            }),
            Arc::new(DummyTaskService),
            Arc::new(DummyAuditService),
            Arc::new(DummySnapshotRepo),
            Arc::new(DummyMetricsService),
            "p1",
        )
        .await
        .unwrap();
        assert_eq!(state.project.id, "p1");
        assert!(!state.tasks.is_empty());
    }

    #[tokio::test]
    async fn subscribe_to_events_returns_receiver() {
        let mut rx = subscribe_to_events(Arc::new(DummyEventService))
            .await
            .unwrap();
        assert!(rx.try_recv().is_err());
    }
}
