use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use petgraph::Direction;
use petgraph::graph::{DiGraph, NodeIndex};
use serde_json::Value;
use thiserror::Error;
use tokio::sync::{Mutex, RwLock};
use tokio_util::sync::CancellationToken;
use ulid::Ulid;

use crate::bus::Event;
use crate::models::{AuditLogLevel, Task, TaskDependency, TaskStatus};
use crate::persistence::{PersistenceError, PromptVersionRepository, TaskRepository};
use crate::services::{AuditLogEntry, AuditLogService, EventService};
use crate::tracing::TraceContext;

/// Errors that can occur in [`TaskService`].
#[derive(Debug, Error)]
pub enum TaskError {
    #[error("persistence error: {0}")]
    Persistence(#[from] PersistenceError),
    #[error("task not found: {0}")]
    NotFound(String),
    #[error("invalid status transition from {from} to {to}")]
    InvalidStatusTransition { from: TaskStatus, to: TaskStatus },
    #[error("task already exists: {0}")]
    AlreadyExists(String),
    #[error("dependency cycle detected")]
    CycleDetected,
    #[error("internal graph inconsistency: {0}")]
    Internal(String),
}

/// Request to create a new task.
#[derive(Debug, Clone)]
pub struct CreateTaskRequest {
    pub project_id: String,
    pub parent_id: Option<String>,
    pub title: String,
    pub description: Option<String>,
    pub kind: String,
    pub assigned_agent: Option<String>,
    pub priority: i32,
    pub payload: Value,
    pub trace_id: Option<String>,
}

/// Business-logic service for managing tasks and their dependency graph.
#[async_trait]
pub trait TaskService: Send + Sync {
    /// Submit a new task, persist it, and publish `Event::TaskCreated`.
    async fn submit(&self, request: CreateTaskRequest) -> Result<Task, TaskError>;

    /// Add a dependency edge: `dep.task_id` depends on `dep.depends_on`.
    async fn add_dependency(&self, dep: TaskDependency) -> Result<(), TaskError>;

    /// Get a task by id (from in-memory cache or persistence).
    async fn get(&self, id: &str) -> Result<Option<Task>, TaskError>;

    /// List all tasks for a project.
    async fn list_by_project(&self, project_id: &str) -> Result<Vec<Task>, TaskError>;

    /// List tasks that are pending and have all dependencies completed.
    async fn list_ready(&self) -> Result<Vec<Task>, TaskError>;

    /// Transition a task to a new status, validating the transition.
    async fn set_status(&self, id: &str, status: TaskStatus) -> Result<Task, TaskError>;

    /// Cancel a non-terminal task and emit `Event::TaskCancelled`.
    /// Idempotent for already cancelled tasks.
    async fn cancel(&self, id: &str) -> Result<Task, TaskError>;

    /// Mark a task as completed with a result.
    async fn set_result(&self, id: &str, result: Value) -> Result<Task, TaskError>;

    /// Set the aggregated critic score for a task.
    async fn set_critic_score(&self, id: &str, score: f64) -> Result<Task, TaskError>;

    /// Set the human feedback score for a task.
    async fn set_human_score(&self, id: &str, score: f64) -> Result<Task, TaskError>;

    /// Return a task to Pending for retry, incrementing iteration count and recording feedback.
    async fn retry(&self, id: &str, feedback: Option<&str>) -> Result<Task, TaskError>;

    /// Load all tasks from persistence into the in-memory graph.
    async fn load_all_tasks(&self) -> Result<Vec<Task>, TaskError>;

    /// Persist an updated task and refresh the in-memory cache.
    async fn update_task(&self, task: &Task) -> Result<(), TaskError>;
}

/// Default implementation of [`TaskService`].
pub struct TaskServiceImpl<R> {
    repo: Arc<R>,
    event_service: Arc<dyn EventService>,
    audit: Arc<dyn AuditLogService>,
    tasks: RwLock<HashMap<String, Task>>,
    graph: Mutex<DiGraph<String, ()>>,
    node_map: RwLock<HashMap<String, NodeIndex>>,
    cancel_tokens: Mutex<HashMap<String, CancellationToken>>,
    prompt_repo: Option<Arc<dyn PromptVersionRepository>>,
}

impl<R> TaskServiceImpl<R> {
    pub fn new(
        repo: Arc<R>,
        event_service: Arc<dyn EventService>,
        audit: Arc<dyn AuditLogService>,
    ) -> Self {
        Self {
            repo,
            event_service,
            audit,
            tasks: RwLock::new(HashMap::new()),
            graph: Mutex::new(DiGraph::new()),
            node_map: RwLock::new(HashMap::new()),
            cancel_tokens: Mutex::new(HashMap::new()),
            prompt_repo: None,
        }
    }

    /// Attach an optional prompt version repository so that new tasks are bound
    /// to the active prompt version for their agent kind.
    pub fn with_prompt_repo(mut self, repo: Arc<dyn PromptVersionRepository>) -> Self {
        self.prompt_repo = Some(repo);
        self
    }

    fn agent_for_kind(kind: &str) -> &str {
        match kind {
            "codegen" => "coder",
            "architecture" | "design" => "architect",
            "research" => "researcher",
            "summarization" => "summarizer",
            "qa" => "qa",
            "security" => "security",
            "review" => "critic",
            other => other,
        }
    }

    fn validate_transition(from: TaskStatus, to: TaskStatus) -> bool {
        use TaskStatus::*;
        match (from, to) {
            // Backlog is the intake column.
            (Backlog, Pending) => true,
            (Backlog, Cancelled) => true,

            // Pending task may start, fail fast, or be cancelled.
            (Pending, InProgress) => true,
            (Pending, Completed) => true, // allow immediate completion for trivial tasks
            (Pending, Failed) => true,    // allow immediate failure for invalid tasks
            (Pending, Cancelled) => true,

            // Running task can finish, go to review, fail, or be cancelled.
            (InProgress, Review) => true,
            (InProgress, Completed) => true,
            (InProgress, Failed) => true,
            (InProgress, Cancelled) => true,

            // Review can approve, reject (back to pending for retry), or fail.
            (Review, Completed) => true,
            (Review, Pending) => true, // retry after review rejection
            (Review, Failed) => true,
            (Review, Cancelled) => true,

            // Failed tasks can be retried.
            (Failed, Pending) => true,
            (Failed, Cancelled) => true,

            // Same status is always idempotent.
            (a, b) if a == b => true,
            _ => false,
        }
    }

    fn all_parents_completed(
        graph: &DiGraph<String, ()>,
        tasks: &HashMap<String, Task>,
        node_map: &HashMap<String, NodeIndex>,
        id: &str,
    ) -> Result<bool, TaskError> {
        let Some(&idx) = node_map.get(id) else {
            return Ok(true);
        };
        let parents: Vec<_> = graph.neighbors_directed(idx, Direction::Incoming).collect();
        for &p in &parents {
            let parent_id = graph.node_weight(p).ok_or_else(|| {
                TaskError::Internal(format!("missing graph node for parent index {:?}", p))
            })?;
            if !tasks
                .get(parent_id)
                .is_none_or(|t| matches!(t.status, TaskStatus::Completed))
            {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

#[async_trait]
impl<R> TaskService for TaskServiceImpl<R>
where
    R: TaskRepository + 'static,
{
    async fn submit(&self, request: CreateTaskRequest) -> Result<Task, TaskError> {
        let now = chrono::Utc::now().timestamp_millis();
        let prompt_version_id = if let Some(repo) = &self.prompt_repo {
            let agent = Self::agent_for_kind(&request.kind);
            repo.get_active_prompt_version(agent)
                .await
                .unwrap_or_default()
                .map(|v| v.id)
        } else {
            None
        };
        let task = Task {
            id: Ulid::new().to_string(),
            project_id: request.project_id,
            parent_id: request.parent_id,
            title: request.title,
            description: request.description,
            kind: request.kind,
            status: TaskStatus::Pending,
            assigned_agent: request.assigned_agent,
            priority: request.priority,
            payload: request.payload,
            result: None,
            created_at: now,
            started_at: None,
            finished_at: None,
            iteration_count: 0,
            priority_score: request.priority as f64,
            critic_score: None,
            human_score: None,
            prompt_version_id,
            lora_adapter_id: None,
            trace_id: request
                .trace_id
                .clone()
                .unwrap_or_else(|| TraceContext::new().trace_id),
        };

        self.repo.insert_task(&task).await?;

        {
            let mut graph = self.graph.lock().await;
            let mut node_map = self.node_map.write().await;
            let idx = graph.add_node(task.id.clone());
            node_map.insert(task.id.clone(), idx);
        }

        {
            let mut tasks = self.tasks.write().await;
            tasks.insert(task.id.clone(), task.clone());
        }

        self.event_service.publish(Event::TaskCreated {
            task_id: task.id.clone(),
            project_id: task.project_id.clone(),
        });

        let _ = self
            .audit
            .log(
                AuditLogEntry::new("task_service", "submit")
                    .project_id(&task.project_id)
                    .task_id(&task.id)
                    .trace_id(&task.trace_id)
                    .level(AuditLogLevel::Info)
                    .metadata(serde_json::json!({ "kind": task.kind })),
            )
            .await;

        Ok(task)
    }

    async fn add_dependency(&self, dep: TaskDependency) -> Result<(), TaskError> {
        self.repo.add_dependency(&dep).await?;

        let mut graph = self.graph.lock().await;
        let node_map = self.node_map.read().await;

        if let (Some(&from), Some(&to)) =
            (node_map.get(&dep.depends_on), node_map.get(&dep.task_id))
        {
            let edge = graph.add_edge(from, to, ());
            if petgraph::algo::is_cyclic_directed(&*graph) {
                // Best-effort rollback of the edge; persistence still holds the dependency.
                graph.remove_edge(edge);
                return Err(TaskError::CycleDetected);
            }
        }
        Ok(())
    }

    async fn get(&self, id: &str) -> Result<Option<Task>, TaskError> {
        let tasks = self.tasks.read().await;
        if let Some(task) = tasks.get(id) {
            return Ok(Some(task.clone()));
        }
        drop(tasks);
        Ok(self.repo.get_task(id).await?)
    }

    async fn list_by_project(&self, project_id: &str) -> Result<Vec<Task>, TaskError> {
        Ok(self.repo.list_tasks_by_project(project_id).await?)
    }

    async fn list_ready(&self) -> Result<Vec<Task>, TaskError> {
        let tasks = self.tasks.read().await;
        let graph = self.graph.lock().await;
        let node_map = self.node_map.read().await;

        let mut ready = Vec::new();
        for (id, task) in tasks.iter() {
            if !matches!(task.status, TaskStatus::Pending) {
                continue;
            }
            if Self::all_parents_completed(&graph, &tasks, &node_map, id)? {
                ready.push(task.clone());
            }
        }
        Ok(ready)
    }

    async fn set_status(&self, id: &str, status: TaskStatus) -> Result<Task, TaskError> {
        let mut tasks = self.tasks.write().await;
        let task = tasks
            .get_mut(id)
            .ok_or_else(|| TaskError::NotFound(id.to_string()))?;

        if task.status == status {
            return Ok(task.clone());
        }

        if !Self::validate_transition(task.status.clone(), status.clone()) {
            return Err(TaskError::InvalidStatusTransition {
                from: task.status.clone(),
                to: status,
            });
        }

        task.status = status.clone();
        match status {
            TaskStatus::InProgress => {
                task.started_at = Some(chrono::Utc::now().timestamp_millis());
            }
            TaskStatus::Completed | TaskStatus::Failed | TaskStatus::Cancelled => {
                task.finished_at = Some(chrono::Utc::now().timestamp_millis());
            }
            TaskStatus::Backlog | TaskStatus::Pending | TaskStatus::Review => {
                // Non-terminal statuses do not record a finish time.
            }
        }

        let result = task.result.clone();
        let updated = task.clone();
        drop(tasks);

        self.repo
            .update_task_status(id, status.clone(), result.clone())
            .await?;

        match status {
            TaskStatus::InProgress => {
                self.event_service.publish(Event::TaskStarted {
                    task_id: id.to_string(),
                });
            }
            TaskStatus::Review => {
                self.event_service.publish(Event::TaskReview {
                    task_id: id.to_string(),
                });
            }
            TaskStatus::Completed => {
                self.event_service.publish(Event::TaskCompleted {
                    task_id: id.to_string(),
                    result: result.clone().unwrap_or(Value::Null),
                });
            }
            TaskStatus::Failed => {
                self.event_service.publish(Event::TaskFailed {
                    task_id: id.to_string(),
                    error: "Task failed".to_string(),
                });
            }
            TaskStatus::Cancelled => {
                self.event_service.publish(Event::TaskCancelled {
                    task_id: id.to_string(),
                });
            }
            TaskStatus::Backlog | TaskStatus::Pending => {
                self.event_service.publish(Event::TaskProgress {
                    task_id: id.to_string(),
                    status: status.as_str().to_string(),
                    message: format!("Task moved to {status}"),
                });
            }
        }

        let _ = self
            .audit
            .log(
                AuditLogEntry::new("task_service", "set_status")
                    .project_id(&updated.project_id)
                    .task_id(&updated.id)
                    .trace_id(&updated.trace_id)
                    .level(AuditLogLevel::Info)
                    .metadata(serde_json::json!({ "status": updated.status.as_str() })),
            )
            .await;

        Ok(updated)
    }

    async fn cancel(&self, id: &str) -> Result<Task, TaskError> {
        {
            let tasks = self.tasks.read().await;
            let task = tasks
                .get(id)
                .ok_or_else(|| TaskError::NotFound(id.to_string()))?;
            if task.status == TaskStatus::Cancelled {
                return Ok(task.clone());
            }
            if task.status.is_terminal() {
                return Err(TaskError::InvalidStatusTransition {
                    from: task.status.clone(),
                    to: TaskStatus::Cancelled,
                });
            }
        }

        // Signal any running worker to stop.
        {
            let tokens = self.cancel_tokens.lock().await;
            if let Some(token) = tokens.get(id) {
                token.cancel();
            }
        }

        let updated = self.set_status(id, TaskStatus::Cancelled).await?;
        let _ = self
            .audit
            .log(
                AuditLogEntry::new("task_service", "cancel")
                    .project_id(&updated.project_id)
                    .task_id(&updated.id)
                    .trace_id(&updated.trace_id)
                    .level(AuditLogLevel::Info)
                    .message("task cancelled"),
            )
            .await;
        Ok(updated)
    }

    async fn set_result(&self, id: &str, result: Value) -> Result<Task, TaskError> {
        {
            let mut tasks = self.tasks.write().await;
            let task = tasks
                .get_mut(id)
                .ok_or_else(|| TaskError::NotFound(id.to_string()))?;
            task.result = Some(result.clone());
            task.status = TaskStatus::Completed;
            task.finished_at = Some(chrono::Utc::now().timestamp_millis());
        }

        self.repo
            .update_task_status(id, TaskStatus::Completed, Some(result.clone()))
            .await?;

        self.event_service.publish(Event::TaskCompleted {
            task_id: id.to_string(),
            result: result.clone(),
        });

        let task = self
            .get(id)
            .await?
            .ok_or_else(|| TaskError::NotFound(id.to_string()))?;

        let _ = self
            .audit
            .log(
                AuditLogEntry::new("task_service", "set_result")
                    .project_id(&task.project_id)
                    .task_id(&task.id)
                    .trace_id(&task.trace_id)
                    .level(AuditLogLevel::Info)
                    .message("task completed with result"),
            )
            .await;

        Ok(task)
    }

    async fn set_critic_score(&self, id: &str, score: f64) -> Result<Task, TaskError> {
        let mut task = self
            .get(id)
            .await?
            .ok_or_else(|| TaskError::NotFound(id.to_string()))?;

        task.critic_score = Some(score);
        self.repo.update_task(&task).await?;

        {
            let mut tasks = self.tasks.write().await;
            tasks.insert(id.to_string(), task.clone());
        }

        let _ = self
            .audit
            .log(
                AuditLogEntry::new("task_service", "set_critic_score")
                    .project_id(&task.project_id)
                    .task_id(&task.id)
                    .trace_id(&task.trace_id)
                    .level(AuditLogLevel::Info)
                    .metadata(serde_json::json!({ "critic_score": score })),
            )
            .await;

        Ok(task)
    }

    async fn set_human_score(&self, id: &str, score: f64) -> Result<Task, TaskError> {
        let mut task = self
            .get(id)
            .await?
            .ok_or_else(|| TaskError::NotFound(id.to_string()))?;

        task.human_score = Some(score);
        self.repo.update_task(&task).await?;

        {
            let mut tasks = self.tasks.write().await;
            tasks.insert(id.to_string(), task.clone());
        }

        let _ = self
            .audit
            .log(
                AuditLogEntry::new("task_service", "set_human_score")
                    .project_id(&task.project_id)
                    .task_id(&task.id)
                    .trace_id(&task.trace_id)
                    .level(AuditLogLevel::Info)
                    .metadata(serde_json::json!({ "human_score": score })),
            )
            .await;

        Ok(task)
    }

    async fn retry(&self, id: &str, feedback: Option<&str>) -> Result<Task, TaskError> {
        let mut task = self
            .get(id)
            .await?
            .ok_or_else(|| TaskError::NotFound(id.to_string()))?;

        if task.status.is_terminal() {
            return Err(TaskError::InvalidStatusTransition {
                from: task.status.clone(),
                to: TaskStatus::Pending,
            });
        }

        if !Self::validate_transition(task.status.clone(), TaskStatus::Pending) {
            return Err(TaskError::InvalidStatusTransition {
                from: task.status.clone(),
                to: TaskStatus::Pending,
            });
        }

        task.iteration_count += 1;
        task.status = TaskStatus::Pending;
        task.started_at = None;
        task.finished_at = None;

        if let Some(text) = feedback {
            let entry = serde_json::json!({
                "iteration": task.iteration_count,
                "comment": text,
                "timestamp": chrono::Utc::now().timestamp_millis(),
            });
            if let Some(arr) = task
                .payload
                .get_mut("retry_feedback")
                .and_then(|v| v.as_array_mut())
            {
                arr.push(entry);
            } else {
                task.payload["retry_feedback"] = serde_json::json!([entry]);
            }
        }

        self.repo.update_task(&task).await?;

        {
            let mut tasks = self.tasks.write().await;
            tasks.insert(id.to_string(), task.clone());
        }

        self.event_service.publish(Event::TaskProgress {
            task_id: id.to_string(),
            status: TaskStatus::Pending.as_str().to_string(),
            message: "retry after feedback".to_string(),
        });

        let _ = self
            .audit
            .log(
                AuditLogEntry::new("task_service", "retry")
                    .project_id(&task.project_id)
                    .task_id(&task.id)
                    .trace_id(&task.trace_id)
                    .level(AuditLogLevel::Info)
                    .metadata(serde_json::json!({
                        "iteration_count": task.iteration_count,
                        "feedback": feedback.unwrap_or("")
                    })),
            )
            .await;

        Ok(task)
    }

    async fn load_all_tasks(&self) -> Result<Vec<Task>, TaskError> {
        let db_tasks = self.repo.list_all_tasks().await?;
        self.load_tasks_into_memory(db_tasks).await
    }

    async fn update_task(&self, task: &Task) -> Result<(), TaskError> {
        self.repo.update_task(task).await?;
        let mut tasks = self.tasks.write().await;
        tasks.insert(task.id.clone(), task.clone());
        Ok(())
    }
}

impl<R> TaskServiceImpl<R> {
    async fn load_tasks_into_memory(&self, db_tasks: Vec<Task>) -> Result<Vec<Task>, TaskError> {
        let mut tasks = self.tasks.write().await;
        let mut graph = self.graph.lock().await;
        let mut node_map = self.node_map.write().await;

        for task in db_tasks {
            if !node_map.contains_key(&task.id) {
                let idx = graph.add_node(task.id.clone());
                node_map.insert(task.id.clone(), idx);
            }
            tasks.insert(task.id.clone(), task);
        }
        Ok(tasks.values().cloned().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::{Event, EventBus};
    use crate::models::AgentLog;
    use crate::persistence::{PersistenceError, TaskRepository};
    use crate::services::{AuditError, AuditLogEntry, AuditLogService, EventService};
    use async_trait::async_trait;
    use std::collections::HashMap;
    use std::sync::Mutex;

    #[derive(Default)]
    struct MockRepo {
        tasks: Mutex<HashMap<String, Task>>,
        deps: Mutex<Vec<TaskDependency>>,
    }

    #[async_trait]
    impl TaskRepository for MockRepo {
        async fn insert_task(&self, task: &Task) -> Result<(), PersistenceError> {
            self.tasks
                .lock()
                .unwrap()
                .insert(task.id.clone(), task.clone());
            Ok(())
        }

        async fn update_task(&self, task: &Task) -> Result<(), PersistenceError> {
            self.tasks
                .lock()
                .unwrap()
                .insert(task.id.clone(), task.clone());
            Ok(())
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
                .filter(|t| t.project_id == project_id)
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
    }

    #[derive(Default)]
    struct MockEventService {
        bus: EventBus,
    }

    #[async_trait]
    impl EventService for MockEventService {
        fn publish(&self, event: Event) {
            self.bus.publish(event);
        }
        fn subscribe(&self) -> tokio::sync::broadcast::Receiver<Event> {
            self.bus.subscribe()
        }
        async fn start_handler(&self, _handler: Arc<dyn crate::services::EventHandler>) {}
    }

    #[derive(Default)]
    struct MockAuditService {
        entries: Mutex<Vec<AuditLogEntry>>,
    }

    impl MockAuditService {
        fn entries(&self) -> Vec<AuditLogEntry> {
            self.entries.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl AuditLogService for MockAuditService {
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

    fn service() -> (TaskServiceImpl<MockRepo>, Arc<dyn EventService>) {
        let (svc, event_service, _) = service_with_audit();
        (svc, event_service)
    }

    fn service_with_audit() -> (
        TaskServiceImpl<MockRepo>,
        Arc<dyn EventService>,
        Arc<MockAuditService>,
    ) {
        let event_service: Arc<dyn EventService> = Arc::new(MockEventService::default());
        let audit = Arc::new(MockAuditService::default());
        let svc = TaskServiceImpl::new(
            Arc::new(MockRepo::default()),
            event_service.clone(),
            audit.clone(),
        );
        (svc, event_service, audit)
    }

    fn request() -> CreateTaskRequest {
        CreateTaskRequest {
            project_id: "proj-1".into(),
            parent_id: None,
            title: "do something".into(),
            description: None,
            kind: "codegen".into(),
            assigned_agent: Some("coder".into()),
            priority: 1,
            payload: Value::Null,
            trace_id: None,
        }
    }

    fn sample_task(
        id: &str,
        project_id: &str,
        status: TaskStatus,
        assigned_agent: Option<&str>,
    ) -> Task {
        Task {
            id: id.into(),
            project_id: project_id.into(),
            parent_id: None,
            title: "test".into(),
            description: None,
            kind: "codegen".into(),
            status,
            assigned_agent: assigned_agent.map(|s| s.into()),
            priority: 0,
            payload: Value::Null,
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
            trace_id: "trace-1".into(),
        }
    }

    #[tokio::test]
    async fn submit_task_persists_and_publishes_event() {
        let (svc, bus) = service();
        let mut rx = bus.subscribe();

        let task = svc.submit(request()).await.unwrap();

        assert_eq!(task.project_id, "proj-1");
        assert_eq!(task.status, TaskStatus::Pending);
        assert!(svc.get(&task.id).await.unwrap().is_some());
        assert!(matches!(rx.try_recv(), Ok(Event::TaskCreated { .. })));
    }

    #[tokio::test]
    async fn submit_generates_trace_id_when_not_provided() {
        let (svc, _) = service();
        let task = svc.submit(request()).await.unwrap();
        assert!(!task.trace_id.is_empty());
    }

    #[tokio::test]
    async fn submit_inherits_parent_trace_id() {
        let (svc, _) = service();
        let parent = svc.submit(request()).await.unwrap();

        let mut child_req = request();
        child_req.parent_id = Some(parent.id.clone());
        child_req.trace_id = Some(parent.trace_id.clone());
        let child = svc.submit(child_req).await.unwrap();

        assert_eq!(child.trace_id, parent.trace_id);
    }

    #[tokio::test]
    async fn trace_id_propagates_across_service_boundaries() {
        let (svc, _, audit) = service_with_audit();

        let mut req = request();
        req.trace_id = Some("trace-abc-123".into());
        let task = svc.submit(req).await.unwrap();

        assert_eq!(task.trace_id, "trace-abc-123");
        let entries = audit.entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].trace_id, "trace-abc-123");
    }

    #[tokio::test]
    async fn set_status_in_progress_publishes_started_event() {
        let (svc, bus) = service();
        let mut rx = bus.subscribe();
        let task = svc.submit(request()).await.unwrap();

        let updated = svc
            .set_status(&task.id, TaskStatus::InProgress)
            .await
            .unwrap();

        assert_eq!(updated.status, TaskStatus::InProgress);
        assert!(updated.started_at.is_some());
        assert!(matches!(rx.try_recv(), Ok(Event::TaskCreated { .. })));
        assert!(matches!(rx.try_recv(), Ok(Event::TaskStarted { .. })));
    }

    #[tokio::test]
    async fn set_status_completed_publishes_completed_event() {
        let (svc, bus) = service();
        let mut rx = bus.subscribe();
        let task = svc.submit(request()).await.unwrap();
        svc.set_status(&task.id, TaskStatus::InProgress)
            .await
            .unwrap();

        let updated = svc
            .set_status(&task.id, TaskStatus::Completed)
            .await
            .unwrap();

        assert_eq!(updated.status, TaskStatus::Completed);
        assert!(updated.finished_at.is_some());
        let _ = rx.try_recv(); // TaskCreated
        let _ = rx.try_recv(); // TaskStarted
        assert!(matches!(rx.try_recv(), Ok(Event::TaskCompleted { .. })));
    }

    #[tokio::test]
    async fn invalid_status_transition_is_rejected() {
        let (svc, _bus) = service();
        let task = svc.submit(request()).await.unwrap();
        svc.set_status(&task.id, TaskStatus::Completed)
            .await
            .unwrap();

        let err = svc
            .set_status(&task.id, TaskStatus::Pending)
            .await
            .unwrap_err();

        assert!(matches!(err, TaskError::InvalidStatusTransition { .. }));
    }

    #[tokio::test]
    async fn cancel_emits_cancelled_event() {
        let (svc, bus) = service();
        let mut rx = bus.subscribe();
        let task = svc.submit(request()).await.unwrap();

        let updated = svc.cancel(&task.id).await.unwrap();

        assert_eq!(updated.status, TaskStatus::Cancelled);
        assert!(updated.finished_at.is_some());
        let _ = rx.try_recv(); // TaskCreated
        assert!(matches!(rx.try_recv(), Ok(Event::TaskCancelled { .. })));
    }

    #[tokio::test]
    async fn cancel_is_idempotent() {
        let (svc, _bus) = service();
        let task = svc.submit(request()).await.unwrap();
        svc.cancel(&task.id).await.unwrap();

        let updated = svc.cancel(&task.id).await.unwrap();

        assert_eq!(updated.status, TaskStatus::Cancelled);
    }

    #[tokio::test]
    async fn cancel_terminal_task_is_rejected() {
        let (svc, _bus) = service();
        let task = svc.submit(request()).await.unwrap();
        svc.set_result(&task.id, Value::String("ok".into()))
            .await
            .unwrap();

        let err = svc.cancel(&task.id).await.unwrap_err();
        assert!(matches!(err, TaskError::InvalidStatusTransition { .. }));
    }

    #[tokio::test]
    async fn set_status_is_idempotent() {
        let (svc, _bus) = service();
        let task = svc.submit(request()).await.unwrap();

        let first = svc.set_status(&task.id, TaskStatus::Pending).await.unwrap();
        let second = svc.set_status(&task.id, TaskStatus::Pending).await.unwrap();

        assert_eq!(first.status, TaskStatus::Pending);
        assert_eq!(first.created_at, second.created_at);
    }

    #[tokio::test]
    async fn set_result_marks_completed() {
        let (svc, _bus) = service();
        let task = svc.submit(request()).await.unwrap();

        let updated = svc
            .set_result(&task.id, Value::String("ok".into()))
            .await
            .unwrap();

        assert_eq!(updated.status, TaskStatus::Completed);
        assert_eq!(updated.result, Some(Value::String("ok".into())));
    }

    #[tokio::test]
    async fn dependency_blocks_ready_task() {
        let (svc, _bus) = service();
        let a = svc.submit(request()).await.unwrap();
        let mut b_req = request();
        b_req.title = "b".into();
        let b = svc.submit(b_req).await.unwrap();

        svc.add_dependency(TaskDependency {
            task_id: b.id.clone(),
            depends_on: a.id.clone(),
            dep_type: "blocks".into(),
        })
        .await
        .unwrap();

        let ready = svc.list_ready().await.unwrap();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].id, a.id);

        svc.set_status(&a.id, TaskStatus::Completed).await.unwrap();
        let ready = svc.list_ready().await.unwrap();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].id, b.id);
    }

    #[tokio::test]
    async fn cycle_dependency_is_rejected() {
        let (svc, _bus) = service();
        let a = svc.submit(request()).await.unwrap();
        let mut b_req = request();
        b_req.title = "b".into();
        let b = svc.submit(b_req).await.unwrap();

        svc.add_dependency(TaskDependency {
            task_id: b.id.clone(),
            depends_on: a.id.clone(),
            dep_type: "blocks".into(),
        })
        .await
        .unwrap();

        let err = svc
            .add_dependency(TaskDependency {
                task_id: a.id.clone(),
                depends_on: b.id.clone(),
                dep_type: "blocks".into(),
            })
            .await
            .unwrap_err();

        assert!(matches!(err, TaskError::CycleDetected));
    }

    #[tokio::test]
    async fn load_all_tasks_populates_in_memory_graph() {
        let repo = Arc::new(MockRepo::default());
        let preloaded = sample_task("pre-1", "proj-1", TaskStatus::Pending, None);
        repo.insert_task(&preloaded).await.unwrap();

        let svc = TaskServiceImpl::new(
            repo,
            Arc::new(MockEventService::default()),
            Arc::new(MockAuditService::default()),
        );
        let loaded = svc.load_all_tasks().await.unwrap();

        assert_eq!(loaded.len(), 1);
        assert!(svc.get("pre-1").await.unwrap().is_some());
    }

    #[test]
    fn all_parents_completed_ready_when_no_parents() {
        let graph = DiGraph::<String, ()>::new();
        let tasks = HashMap::new();
        let node_map = HashMap::new();

        let result =
            TaskServiceImpl::<MockRepo>::all_parents_completed(&graph, &tasks, &node_map, "child");
        assert!(result.unwrap());
    }

    #[test]
    fn all_parents_completed_false_when_parent_pending() {
        let mut graph = DiGraph::<String, ()>::new();
        let parent_idx = graph.add_node("parent".into());
        let child_idx = graph.add_node("child".into());
        graph.add_edge(parent_idx, child_idx, ());

        let mut tasks = HashMap::new();
        tasks.insert(
            "parent".into(),
            sample_task("parent", "p", TaskStatus::Pending, None),
        );

        let mut node_map = HashMap::new();
        node_map.insert("parent".into(), parent_idx);
        node_map.insert("child".into(), child_idx);

        let result =
            TaskServiceImpl::<MockRepo>::all_parents_completed(&graph, &tasks, &node_map, "child");
        assert!(!result.unwrap());
    }

    #[test]
    fn all_parents_completed_true_when_parent_completed() {
        let mut graph = DiGraph::<String, ()>::new();
        let parent_idx = graph.add_node("parent".into());
        let child_idx = graph.add_node("child".into());
        graph.add_edge(parent_idx, child_idx, ());

        let mut tasks = HashMap::new();
        tasks.insert(
            "parent".into(),
            sample_task("parent", "p", TaskStatus::Completed, None),
        );

        let mut node_map = HashMap::new();
        node_map.insert("parent".into(), parent_idx);
        node_map.insert("child".into(), child_idx);

        let result =
            TaskServiceImpl::<MockRepo>::all_parents_completed(&graph, &tasks, &node_map, "child");
        assert!(result.unwrap());
    }

    #[tokio::test]
    async fn set_critic_score_persists_score() {
        let (svc, _) = service();
        let task = svc.submit(request()).await.unwrap();

        let updated = svc.set_critic_score(&task.id, 4.2).await.unwrap();
        assert_eq!(updated.critic_score, Some(4.2));

        let loaded = svc.get(&task.id).await.unwrap().unwrap();
        assert_eq!(loaded.critic_score, Some(4.2));
    }

    #[tokio::test]
    async fn retry_increments_iteration_and_moves_to_pending() {
        let (svc, bus) = service();
        let mut task = svc.submit(request()).await.unwrap();
        task.status = TaskStatus::Review;
        svc.repo.update_task(&task).await.unwrap();
        svc.load_all_tasks().await.unwrap();

        let mut rx = bus.subscribe();
        let updated = svc.retry(&task.id, Some("fix tests")).await.unwrap();

        assert_eq!(updated.status, TaskStatus::Pending);
        assert_eq!(updated.iteration_count, 1);
        assert_eq!(updated.payload["retry_feedback"][0]["comment"], "fix tests");
        assert!(matches!(rx.try_recv(), Ok(Event::TaskProgress { .. })));
    }

    #[tokio::test]
    async fn retry_from_completed_is_rejected() {
        let (svc, _) = service();
        let task = svc.submit(request()).await.unwrap();
        svc.set_status(&task.id, TaskStatus::Completed)
            .await
            .unwrap();

        let err = svc.retry(&task.id, None).await.unwrap_err();
        assert!(matches!(err, TaskError::InvalidStatusTransition { .. }));
    }
}
