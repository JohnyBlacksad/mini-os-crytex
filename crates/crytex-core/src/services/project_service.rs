use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};
use thiserror::Error;
use ulid::Ulid;

use crate::models::{KanbanColumn, KanbanState, KanbanTaskCard, Project, TaskStatus};
use crate::persistence::{PersistenceError, ProjectRepository, TaskRepository};

/// Errors that can occur in [`ProjectService`].
#[derive(Debug, Error)]
pub enum ProjectError {
    #[error("project name must be non-empty")]
    InvalidName,
    #[error("project root path must be absolute: {0}")]
    InvalidPath(String),
    #[error("project not found: {0}")]
    NotFound(String),
    #[error("project already exists: {0}")]
    AlreadyExists(String),
    #[error("persistence error: {0}")]
    Persistence(#[from] PersistenceError),
}

/// Request to create a new project.
#[derive(Debug, Clone)]
pub struct CreateProjectRequest<'a> {
    pub name: &'a str,
    pub root_path: &'a Path,
}

/// Business-logic service for managing projects.
#[async_trait]
pub trait ProjectService: Send + Sync {
    /// Create a new project, persist it and return the populated model.
    async fn create(&self, request: CreateProjectRequest<'_>) -> Result<Project, ProjectError>;

    /// Get a project by id.
    async fn get(&self, id: &str) -> Result<Option<Project>, ProjectError>;

    /// List all projects.
    async fn list(&self) -> Result<Vec<Project>, ProjectError>;

    /// Update project metadata (e.g. after taking a snapshot).
    async fn update_metadata(&self, id: &str, metadata: Value) -> Result<Project, ProjectError>;

    /// Return the project's tasks grouped into Kanban columns by status.
    async fn kanban_state(&self, project_id: &str) -> Result<KanbanState, ProjectError>;
}

/// Default implementation of [`ProjectService`].
pub struct ProjectServiceImpl<R> {
    repo: Arc<R>,
}

impl<R> ProjectServiceImpl<R> {
    pub fn new(repo: Arc<R>) -> Self {
        Self { repo }
    }
}

#[async_trait]
impl<R> ProjectService for ProjectServiceImpl<R>
where
    R: ProjectRepository + TaskRepository + 'static,
{
    async fn create(&self, request: CreateProjectRequest<'_>) -> Result<Project, ProjectError> {
        let name = request.name.trim();
        if name.is_empty() {
            return Err(ProjectError::InvalidName);
        }

        let root_path = request.root_path;
        if !root_path.is_absolute() {
            return Err(ProjectError::InvalidPath(root_path.display().to_string()));
        }

        let id = Ulid::new().to_string();
        let now = chrono::Utc::now().timestamp_millis();
        let project = Project {
            id,
            name: name.to_string(),
            root_path: root_path.display().to_string(),
            created_at: now,
            updated_at: now,
            metadata: json!({}),
        };

        self.repo.insert_project(&project).await?;
        Ok(project)
    }

    async fn get(&self, id: &str) -> Result<Option<Project>, ProjectError> {
        Ok(self.repo.get_project(id).await?)
    }

    async fn list(&self) -> Result<Vec<Project>, ProjectError> {
        Ok(self.repo.list_projects().await?)
    }

    async fn update_metadata(&self, id: &str, metadata: Value) -> Result<Project, ProjectError> {
        let mut project = self
            .repo
            .get_project(id)
            .await?
            .ok_or_else(|| ProjectError::NotFound(id.to_string()))?;

        project.metadata = metadata;
        project.updated_at = chrono::Utc::now().timestamp_millis();
        self.repo.insert_project(&project).await?;
        Ok(project)
    }

    async fn kanban_state(&self, project_id: &str) -> Result<KanbanState, ProjectError> {
        let _project = self
            .repo
            .get_project(project_id)
            .await?
            .ok_or_else(|| ProjectError::NotFound(project_id.to_string()))?;

        let tasks = self.repo.list_tasks_by_project(project_id).await?;
        let columns = [
            TaskStatus::Backlog,
            TaskStatus::Pending,
            TaskStatus::InProgress,
            TaskStatus::Review,
            TaskStatus::Completed,
            TaskStatus::Failed,
            TaskStatus::Cancelled,
        ]
        .map(|status| {
            let title = status.as_str().to_string();
            let column_tasks: Vec<KanbanTaskCard> = tasks
                .iter()
                .filter(|t| t.status == status)
                .map(|t| KanbanTaskCard {
                    id: t.id.clone(),
                    title: t.title.clone(),
                    kind: t.kind.clone(),
                    status: t.status.clone(),
                    priority: t.priority,
                    assigned_agent: t.assigned_agent.clone(),
                })
                .collect();
            KanbanColumn {
                status,
                title,
                tasks: column_tasks,
            }
        });

        Ok(KanbanState {
            project_id: project_id.to_string(),
            columns: columns.to_vec(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Task, TaskDependency, TaskStatus};
    use crate::persistence::{ProjectRepository, TaskRepository};
    use async_trait::async_trait;
    use std::collections::HashMap;
    use std::sync::Mutex;

    struct MockRepo {
        projects: Mutex<HashMap<String, Project>>,
        tasks: Mutex<HashMap<String, Vec<Task>>>,
    }

    impl Default for MockRepo {
        fn default() -> Self {
            Self {
                projects: Mutex::new(HashMap::new()),
                tasks: Mutex::new(HashMap::new()),
            }
        }
    }

    #[async_trait]
    impl ProjectRepository for MockRepo {
        async fn insert_project(&self, project: &Project) -> Result<(), PersistenceError> {
            self.projects
                .lock()
                .unwrap()
                .insert(project.id.clone(), project.clone());
            Ok(())
        }

        async fn get_project(&self, id: &str) -> Result<Option<Project>, PersistenceError> {
            Ok(self.projects.lock().unwrap().get(id).cloned())
        }

        async fn list_projects(&self) -> Result<Vec<Project>, PersistenceError> {
            Ok(self.projects.lock().unwrap().values().cloned().collect())
        }
    }

    #[async_trait]
    impl TaskRepository for MockRepo {
        async fn insert_task(&self, task: &Task) -> Result<(), PersistenceError> {
            self.tasks
                .lock()
                .unwrap()
                .entry(task.project_id.clone())
                .or_default()
                .push(task.clone());
            Ok(())
        }

        async fn update_task(&self, _task: &Task) -> Result<(), PersistenceError> {
            Ok(())
        }

        async fn update_task_status(
            &self,
            _id: &str,
            _status: TaskStatus,
            _result: Option<Value>,
        ) -> Result<(), PersistenceError> {
            Ok(())
        }

        async fn get_task(&self, _id: &str) -> Result<Option<Task>, PersistenceError> {
            Ok(None)
        }

        async fn list_tasks_by_project(
            &self,
            project_id: &str,
        ) -> Result<Vec<Task>, PersistenceError> {
            Ok(self
                .tasks
                .lock()
                .unwrap()
                .get(project_id)
                .cloned()
                .unwrap_or_default())
        }

        async fn list_all_tasks(&self) -> Result<Vec<Task>, PersistenceError> {
            Ok(self
                .tasks
                .lock()
                .unwrap()
                .values()
                .flatten()
                .cloned()
                .collect())
        }

        async fn list_ready_tasks(&self) -> Result<Vec<Task>, PersistenceError> {
            Ok(vec![])
        }

        async fn add_dependency(&self, _dep: &TaskDependency) -> Result<(), PersistenceError> {
            Ok(())
        }

        async fn list_dependencies(&self) -> Result<Vec<TaskDependency>, PersistenceError> {
            Ok(vec![])
        }
    }

    fn service() -> ProjectServiceImpl<MockRepo> {
        ProjectServiceImpl::new(Arc::new(MockRepo::default()))
    }

    fn absolute_path(sub: &str) -> &'static Path {
        // Leak a small string to obtain a static path reference for tests.
        let s: &'static str = Box::leak(format!("C:/tmp/{}", sub).into_boxed_str());
        Path::new(s)
    }

    #[tokio::test]
    async fn create_project_succeeds() {
        let svc = service();
        let project = svc
            .create(CreateProjectRequest {
                name: "demo",
                root_path: absolute_path("demo"),
            })
            .await
            .unwrap();

        assert_eq!(project.name, "demo");
        assert!(project.root_path.contains("/tmp/demo"));
        assert!(!project.id.is_empty());
    }

    #[tokio::test]
    async fn create_project_rejects_empty_name() {
        let svc = service();
        let err = svc
            .create(CreateProjectRequest {
                name: "   ",
                root_path: absolute_path("demo"),
            })
            .await
            .unwrap_err();

        assert!(matches!(err, ProjectError::InvalidName));
    }

    #[tokio::test]
    async fn create_project_rejects_relative_path() {
        let svc = service();
        let err = svc
            .create(CreateProjectRequest {
                name: "demo",
                root_path: Path::new("relative/path"),
            })
            .await
            .unwrap_err();

        assert!(matches!(err, ProjectError::InvalidPath(_)));
    }

    #[tokio::test]
    async fn get_project_returns_persisted_project() {
        let svc = service();
        let created = svc
            .create(CreateProjectRequest {
                name: "demo",
                root_path: absolute_path("demo"),
            })
            .await
            .unwrap();

        let fetched = svc.get(&created.id).await.unwrap().unwrap();
        assert_eq!(fetched.id, created.id);
        assert_eq!(fetched.name, "demo");
    }

    #[tokio::test]
    async fn list_projects_returns_all() {
        let svc = service();
        svc.create(CreateProjectRequest {
            name: "a",
            root_path: absolute_path("a"),
        })
        .await
        .unwrap();
        svc.create(CreateProjectRequest {
            name: "b",
            root_path: absolute_path("b"),
        })
        .await
        .unwrap();

        let projects = svc.list().await.unwrap();
        assert_eq!(projects.len(), 2);
    }

    #[tokio::test]
    async fn update_metadata_changes_project() {
        let svc = service();
        let created = svc
            .create(CreateProjectRequest {
                name: "demo",
                root_path: absolute_path("demo"),
            })
            .await
            .unwrap();

        let updated = svc
            .update_metadata(&created.id, json!({ "snapshot": "v1" }))
            .await
            .unwrap();

        assert_eq!(updated.metadata, json!({ "snapshot": "v1" }));
        assert!(updated.updated_at >= created.updated_at);
    }

    #[tokio::test]
    async fn update_metadata_missing_project_fails() {
        let svc = service();
        let err = svc.update_metadata("missing", json!({})).await.unwrap_err();

        assert!(matches!(err, ProjectError::NotFound(_)));
    }

    #[tokio::test]
    async fn kanban_state_groups_tasks_into_columns() {
        let svc = service();
        let project = svc
            .create(CreateProjectRequest {
                name: "demo",
                root_path: absolute_path("demo"),
            })
            .await
            .unwrap();

        let now = chrono::Utc::now().timestamp_millis();
        let pending = Task {
            id: "t1".into(),
            project_id: project.id.clone(),
            parent_id: None,
            title: "pending task".into(),
            description: None,
            kind: "codegen".into(),
            status: TaskStatus::Pending,
            assigned_agent: None,
            priority: 1,
            payload: json!({}),
            result: None,
            created_at: now,
            started_at: None,
            finished_at: None,
            iteration_count: 0,
            priority_score: 0.0,
            critic_score: None,
            human_score: None,
            prompt_version_id: None,
            lora_adapter_id: None,
            trace_id: "trace-1".into(),
        };
        let mut completed = pending.clone();
        completed.id = "t2".into();
        completed.title = "completed task".into();
        completed.status = TaskStatus::Completed;

        svc.repo.insert_task(&pending).await.unwrap();
        svc.repo.insert_task(&completed).await.unwrap();

        let state = svc.kanban_state(&project.id).await.unwrap();
        assert_eq!(state.columns.len(), 7);

        let pending_column = state
            .columns
            .iter()
            .find(|c| c.status == TaskStatus::Pending)
            .unwrap();
        assert_eq!(pending_column.tasks.len(), 1);
        assert_eq!(pending_column.tasks[0].id, "t1");

        let completed_column = state
            .columns
            .iter()
            .find(|c| c.status == TaskStatus::Completed)
            .unwrap();
        assert_eq!(completed_column.tasks.len(), 1);
        assert_eq!(completed_column.tasks[0].id, "t2");
    }
}
