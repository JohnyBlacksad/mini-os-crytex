//! Build a full project-state export for the Tauri UI and CLI.

use crate::metrics::{MetricsError, MetricsService};
use crate::models::{AgentLog, KanbanState, Project, ProjectSnapshot, Task};
use crate::persistence::ProjectSnapshotRepository;
use crate::services::{
    AuditError, AuditLogService, ProjectError, ProjectService, TaskError, TaskService,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Errors that can occur while exporting project state.
#[derive(Debug, thiserror::Error)]
pub enum StateExportError {
    #[error("project error: {0}")]
    Project(#[from] ProjectError),
    #[error("task error: {0}")]
    Task(#[from] TaskError),
    #[error("audit error: {0}")]
    Audit(#[from] AuditError),
    #[error("persistence error: {0}")]
    Persistence(#[from] crate::persistence::PersistenceError),
    #[error("metrics error: {0}")]
    Metrics(#[from] MetricsError),
}

/// A serializable full project state snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectState {
    pub project: Project,
    pub kanban: KanbanState,
    pub tasks: Vec<Task>,
    pub recent_logs: Vec<AgentLog>,
    pub latest_snapshot: Option<ProjectSnapshot>,
    pub metrics: crate::metrics::MetricsSnapshot,
}

/// Export the current state of a project.
pub async fn export_project_state(
    project_service: Arc<dyn ProjectService>,
    task_service: Arc<dyn TaskService>,
    audit_service: Arc<dyn AuditLogService>,
    snapshot_repo: Arc<dyn ProjectSnapshotRepository>,
    metrics_service: Arc<dyn MetricsService>,
    project_id: &str,
) -> Result<ProjectState, StateExportError> {
    let project = project_service
        .get(project_id)
        .await?
        .ok_or_else(|| ProjectError::NotFound(project_id.to_string()))?;

    let kanban = project_service.kanban_state(project_id).await?;
    let tasks = task_service.list_by_project(project_id).await?;

    let mut recent_logs = audit_service.list_by_project(project_id).await?;
    recent_logs.sort_by_key(|b| std::cmp::Reverse(b.timestamp));
    recent_logs.truncate(50);

    let mut snapshots = snapshot_repo.list_project_snapshots(project_id).await?;
    snapshots.sort_by_key(|b| std::cmp::Reverse(b.created_at));
    let latest_snapshot = snapshots.into_iter().next();

    let metrics = metrics_service.snapshot().await?;

    Ok(ProjectState {
        project,
        kanban,
        tasks,
        recent_logs,
        latest_snapshot,
        metrics,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{MetricsError, MetricsSnapshot};
    use crate::models::{
        KanbanColumn, KanbanTaskCard, ProjectSnapshot, Task, TaskDependency, TaskStatus,
    };
    use crate::persistence::{PersistenceError, ProjectSnapshotRepository};
    use crate::services::{
        AuditLogEntry, AuditLogService, CreateTaskRequest, ProjectService, TaskError, TaskService,
    };
    use async_trait::async_trait;
    use serde_json::{Value, json};

    struct DummyProjectService {
        project: Project,
    }

    #[async_trait]
    impl ProjectService for DummyProjectService {
        async fn create(
            &self,
            _request: crate::services::CreateProjectRequest<'_>,
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
            metadata: serde_json::Value,
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
        async fn list_by_task(
            &self,
            _task_id: &str,
        ) -> Result<Vec<crate::models::AgentLog>, AuditError> {
            Ok(vec![])
        }
        async fn list_by_project(
            &self,
            _project_id: &str,
        ) -> Result<Vec<crate::models::AgentLog>, AuditError> {
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

    #[tokio::test]
    async fn state_export_contains_all_expected_keys() {
        let project = Project {
            id: "p1".into(),
            name: "demo".into(),
            root_path: "/tmp/demo".into(),
            created_at: 0,
            updated_at: 0,
            metadata: json!({}),
        };

        let state = export_project_state(
            Arc::new(DummyProjectService {
                project: project.clone(),
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
        assert!(!state.kanban.columns.is_empty());
        assert!(!state.tasks.is_empty());
        assert!(state.recent_logs.is_empty());
        assert!(state.latest_snapshot.is_none());

        let value = serde_json::to_value(&state).unwrap();
        assert!(value.get("project").is_some());
        assert!(value.get("kanban").is_some());
        assert!(value.get("tasks").is_some());
        assert!(value.get("recent_logs").is_some());
        assert!(value.get("latest_snapshot").is_some());
        assert!(value.get("metrics").is_some());
    }
}
