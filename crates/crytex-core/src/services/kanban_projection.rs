use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::models::{Task, TaskDependency, TaskStatus};
use crate::persistence::{PersistenceError, ProjectRepository, TaskRepository};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum KanbanStatus {
    Backlog,
    Ready,
    InProgress,
    Review,
    Remediation,
    Done,
    Failed,
    Blocked,
}

impl KanbanStatus {
    pub fn all() -> [Self; 8] {
        [
            Self::Backlog,
            Self::Ready,
            Self::InProgress,
            Self::Review,
            Self::Remediation,
            Self::Done,
            Self::Failed,
            Self::Blocked,
        ]
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Backlog => "backlog",
            Self::Ready => "ready",
            Self::InProgress => "in_progress",
            Self::Review => "review",
            Self::Remediation => "remediation",
            Self::Done => "done",
            Self::Failed => "failed",
            Self::Blocked => "blocked",
        }
    }
}

impl From<&TaskStatus> for KanbanStatus {
    fn from(status: &TaskStatus) -> Self {
        match status {
            TaskStatus::Backlog => Self::Backlog,
            TaskStatus::Ready | TaskStatus::Pending => Self::Ready,
            TaskStatus::InProgress => Self::InProgress,
            TaskStatus::Review => Self::Review,
            TaskStatus::Remediation => Self::Remediation,
            TaskStatus::Done | TaskStatus::Completed => Self::Done,
            TaskStatus::Failed => Self::Failed,
            TaskStatus::Blocked | TaskStatus::Cancelled => Self::Blocked,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KanbanTaskProjection {
    pub id: String,
    pub title: String,
    pub goal: String,
    pub agent_role: Option<String>,
    pub task_kind: String,
    pub dependency_chain: Vec<String>,
    pub queue_position: usize,
    pub status: KanbanStatus,
    pub critic_comment: Option<String>,
    pub remediation_link: Option<String>,
    pub trace_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KanbanColumnProjection {
    pub status: KanbanStatus,
    pub title: String,
    pub tasks: Vec<KanbanTaskProjection>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KanbanBoardProjection {
    pub project_id: String,
    pub columns: Vec<KanbanColumnProjection>,
    pub tasks: Vec<KanbanTaskProjection>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KanbanMovement {
    pub task_id: String,
    pub goal: String,
    pub agent_role: Option<String>,
    pub task_kind: String,
    pub dependency_chain: Vec<String>,
    pub queue_position: usize,
    pub status: KanbanStatus,
    pub critic_comment: Option<String>,
    pub remediation_link: Option<String>,
    pub trace_id: String,
    pub timestamp: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KanbanHistoryProjection {
    pub project_id: String,
    pub run_id: Option<String>,
    pub movements: Vec<KanbanMovement>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KanbanRunSelector {
    Latest,
    Id(String),
}

#[derive(Debug, Error)]
pub enum KanbanProjectionError {
    #[error("project not found: {0}")]
    ProjectNotFound(String),
    #[error("persistence error: {0}")]
    Persistence(#[from] PersistenceError),
}

pub struct KanbanProjectionService<R> {
    repo: Arc<R>,
}

impl<R> KanbanProjectionService<R> {
    pub fn new(repo: Arc<R>) -> Self {
        Self { repo }
    }
}

impl<R> KanbanProjectionService<R>
where
    R: ProjectRepository + TaskRepository + 'static,
{
    pub async fn show(
        &self,
        project_id: &str,
    ) -> Result<KanbanBoardProjection, KanbanProjectionError> {
        self.ensure_project(project_id).await?;
        let tasks = self.repo.list_tasks_by_project(project_id).await?;
        let deps = self.repo.list_dependencies().await?;
        let cards = build_cards(tasks, deps);
        let columns = KanbanStatus::all()
            .into_iter()
            .map(|status| KanbanColumnProjection {
                status,
                title: status.as_str().to_string(),
                tasks: cards
                    .iter()
                    .filter(|card| card.status == status)
                    .cloned()
                    .collect(),
            })
            .collect();

        Ok(KanbanBoardProjection {
            project_id: project_id.to_string(),
            columns,
            tasks: cards,
        })
    }

    pub async fn history(
        &self,
        project_id: &str,
        selector: KanbanRunSelector,
    ) -> Result<KanbanHistoryProjection, KanbanProjectionError> {
        let board = self.show(project_id).await?;
        let run_id = select_run_id(&board.tasks, selector);
        let mut movements = board
            .tasks
            .into_iter()
            .filter(|card| run_id.as_ref().is_none_or(|id| card.trace_id == *id))
            .map(|card| KanbanMovement {
                timestamp: card.queue_position as i64,
                task_id: card.id,
                goal: card.goal,
                agent_role: card.agent_role,
                task_kind: card.task_kind,
                dependency_chain: card.dependency_chain,
                queue_position: card.queue_position,
                status: card.status,
                critic_comment: card.critic_comment,
                remediation_link: card.remediation_link,
                trace_id: card.trace_id,
            })
            .collect::<Vec<_>>();
        movements.sort_by_key(|movement| movement.queue_position);

        Ok(KanbanHistoryProjection {
            project_id: project_id.to_string(),
            run_id,
            movements,
        })
    }

    async fn ensure_project(&self, project_id: &str) -> Result<(), KanbanProjectionError> {
        self.repo
            .get_project(project_id)
            .await?
            .map(|_| ())
            .ok_or_else(|| KanbanProjectionError::ProjectNotFound(project_id.to_string()))
    }
}

fn build_cards(tasks: Vec<Task>, deps: Vec<TaskDependency>) -> Vec<KanbanTaskProjection> {
    let task_ids = tasks
        .iter()
        .map(|task| task.id.clone())
        .collect::<HashSet<_>>();
    let mut dependency_map: HashMap<String, Vec<String>> = HashMap::new();
    deps.into_iter()
        .filter(|dep| task_ids.contains(&dep.task_id))
        .for_each(|dep| {
            dependency_map
                .entry(dep.task_id)
                .or_default()
                .push(dep.depends_on);
        });

    let mut tasks = tasks;
    tasks.sort_by_key(|task| {
        (
            KanbanStatus::from(&task.status).as_str(),
            task.priority,
            task.created_at,
        )
    });
    tasks
        .into_iter()
        .enumerate()
        .map(|(index, task)| task_card(task, &dependency_map, index + 1))
        .collect()
}

fn task_card(
    task: Task,
    dependency_map: &HashMap<String, Vec<String>>,
    queue_position: usize,
) -> KanbanTaskProjection {
    let goal = task_goal(&task);
    let critic_comment = critic_comment(&task);
    let remediation_link = remediation_link(&task);
    let status = KanbanStatus::from(&task.status);
    KanbanTaskProjection {
        id: task.id.clone(),
        title: task.title,
        goal,
        agent_role: task.assigned_agent,
        task_kind: task.kind,
        dependency_chain: dependency_map.get(&task.id).cloned().unwrap_or_default(),
        queue_position,
        status,
        critic_comment,
        remediation_link,
        trace_id: task.trace_id,
    }
}

fn task_goal(task: &Task) -> String {
    task.payload
        .get("goal")
        .and_then(serde_json::Value::as_str)
        .or(task.description.as_deref())
        .unwrap_or(&task.title)
        .to_string()
}

fn critic_comment(task: &Task) -> Option<String> {
    task.payload
        .pointer("/critic_report/feedback")
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            task.payload
                .get("retry_feedback")
                .and_then(serde_json::Value::as_array)
                .and_then(|entries| entries.last())
                .and_then(|entry| entry.get("comment"))
                .and_then(serde_json::Value::as_str)
        })
        .map(str::to_string)
}

fn remediation_link(task: &Task) -> Option<String> {
    task.payload
        .get("remediation_task_id")
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            task.payload
                .get("remediation_for")
                .and_then(serde_json::Value::as_str)
        })
        .map(str::to_string)
}

fn select_run_id(tasks: &[KanbanTaskProjection], selector: KanbanRunSelector) -> Option<String> {
    match selector {
        KanbanRunSelector::Latest => tasks
            .iter()
            .max_by_key(|task| task.queue_position)
            .map(|task| task.trace_id.clone()),
        KanbanRunSelector::Id(id) => Some(id),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Project, Task, TaskDependency, TaskStatus};
    use crate::persistence::{PersistenceError, ProjectRepository, TaskRepository};
    use async_trait::async_trait;
    use serde_json::{Value, json};
    use std::collections::HashMap;
    use std::sync::Mutex;

    #[derive(Default)]
    struct KanbanRepo {
        projects: Mutex<HashMap<String, Project>>,
        tasks: Mutex<HashMap<String, Task>>,
        deps: Mutex<Vec<TaskDependency>>,
    }

    #[async_trait]
    impl ProjectRepository for KanbanRepo {
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
    impl TaskRepository for KanbanRepo {
        async fn insert_task(&self, task: &Task) -> Result<(), PersistenceError> {
            self.tasks
                .lock()
                .unwrap()
                .insert(task.id.clone(), task.clone());
            Ok(())
        }

        async fn update_task(&self, task: &Task) -> Result<(), PersistenceError> {
            self.insert_task(task).await
        }

        async fn update_task_status(
            &self,
            id: &str,
            status: TaskStatus,
            result: Option<Value>,
        ) -> Result<(), PersistenceError> {
            let mut tasks = self.tasks.lock().unwrap();
            let task = tasks
                .get_mut(id)
                .ok_or_else(|| PersistenceError::Database(id.into()))?;
            task.status = status;
            task.result = result;
            Ok(())
        }

        async fn get_task(&self, id: &str) -> Result<Option<Task>, PersistenceError> {
            Ok(self.tasks.lock().unwrap().get(id).cloned())
        }

        async fn list_tasks_by_project(
            &self,
            project_id: &str,
        ) -> Result<Vec<Task>, PersistenceError> {
            Ok(self
                .tasks
                .lock()
                .unwrap()
                .values()
                .filter(|task| task.project_id == project_id)
                .cloned()
                .collect())
        }

        async fn list_all_tasks(&self) -> Result<Vec<Task>, PersistenceError> {
            Ok(self.tasks.lock().unwrap().values().cloned().collect())
        }

        async fn list_ready_tasks(&self) -> Result<Vec<Task>, PersistenceError> {
            Ok(vec![])
        }

        async fn add_dependency(&self, dep: &TaskDependency) -> Result<(), PersistenceError> {
            self.deps.lock().unwrap().push(dep.clone());
            Ok(())
        }

        async fn list_dependencies(&self) -> Result<Vec<TaskDependency>, PersistenceError> {
            Ok(self.deps.lock().unwrap().clone())
        }
    }

    fn project() -> Project {
        Project {
            id: "project-1".into(),
            name: "Project".into(),
            root_path: "A:/project".into(),
            created_at: 1,
            updated_at: 1,
            metadata: json!({}),
        }
    }

    fn task(id: &str, status: TaskStatus, role: &str, priority: i32) -> Task {
        Task {
            id: id.into(),
            project_id: "project-1".into(),
            parent_id: None,
            title: format!("Task {id}"),
            description: Some(format!("Goal for {id}")),
            kind: "codegen".into(),
            status,
            assigned_agent: Some(role.into()),
            priority,
            created_at: priority as i64,
            started_at: None,
            finished_at: None,
            payload: json!({
                "goal": format!("Goal for {id}"),
                "critic_report": {
                    "feedback": "missing regression evidence"
                },
                "remediation_task_id": "task-remediate"
            }),
            result: None,
            iteration_count: 0,
            priority_score: priority as f64,
            critic_score: None,
            human_score: None,
            prompt_version_id: None,
            lora_adapter_id: None,
            trace_id: "run-latest".into(),
        }
    }

    #[tokio::test]
    async fn kanban_projection_contains_canonical_columns_and_full_task_cards() {
        let repo = std::sync::Arc::new(KanbanRepo::default());
        repo.insert_project(&project()).await.unwrap();
        repo.insert_task(&task("task-a", TaskStatus::Ready, "coder", 10))
            .await
            .unwrap();
        repo.insert_task(&task("task-b", TaskStatus::InProgress, "qa", 20))
            .await
            .unwrap();
        repo.add_dependency(&TaskDependency {
            task_id: "task-b".into(),
            depends_on: "task-a".into(),
            dep_type: "blocks".into(),
        })
        .await
        .unwrap();

        let projection = KanbanProjectionService::new(repo);
        let board = projection.show("project-1").await.unwrap();

        assert_eq!(
            board
                .columns
                .iter()
                .map(|column| column.status)
                .collect::<Vec<_>>(),
            vec![
                KanbanStatus::Backlog,
                KanbanStatus::Ready,
                KanbanStatus::InProgress,
                KanbanStatus::Review,
                KanbanStatus::Remediation,
                KanbanStatus::Done,
                KanbanStatus::Failed,
                KanbanStatus::Blocked,
            ]
        );

        let running = board.tasks.iter().find(|card| card.id == "task-b").unwrap();
        assert_eq!(running.goal, "Goal for task-b");
        assert_eq!(running.agent_role.as_deref(), Some("qa"));
        assert_eq!(running.task_kind, "codegen");
        assert_eq!(running.dependency_chain, vec!["task-a"]);
        assert_eq!(running.queue_position, 1);
        assert_eq!(running.status, KanbanStatus::InProgress);
    }

    #[tokio::test]
    async fn returned_task_exposes_critic_comment_and_remediation_link() {
        let repo = std::sync::Arc::new(KanbanRepo::default());
        repo.insert_project(&project()).await.unwrap();
        repo.insert_task(&task("task-review", TaskStatus::Remediation, "critic", 1))
            .await
            .unwrap();

        let projection = KanbanProjectionService::new(repo);
        let board = projection.show("project-1").await.unwrap();
        let returned = board
            .tasks
            .iter()
            .find(|card| card.id == "task-review")
            .unwrap();

        assert_eq!(
            returned.critic_comment.as_deref(),
            Some("missing regression evidence")
        );
        assert_eq!(returned.remediation_link.as_deref(), Some("task-remediate"));
    }

    #[tokio::test]
    async fn history_latest_run_orders_movements_by_task_time() {
        let repo = std::sync::Arc::new(KanbanRepo::default());
        repo.insert_project(&project()).await.unwrap();
        repo.insert_task(&task("task-a", TaskStatus::Done, "coder", 10))
            .await
            .unwrap();
        repo.insert_task(&task("task-b", TaskStatus::Review, "critic", 20))
            .await
            .unwrap();

        let projection = KanbanProjectionService::new(repo);
        let history = projection
            .history("project-1", KanbanRunSelector::Latest)
            .await
            .unwrap();

        assert_eq!(history.run_id.as_deref(), Some("run-latest"));
        assert_eq!(history.movements.len(), 2);
        assert_eq!(history.movements[0].task_id, "task-a");
        assert_eq!(history.movements[1].status, KanbanStatus::Review);
    }
}
