//! Ready-task scheduler.
//!
//! Selects the next batch of tasks that can be executed right now, ordered by
//! priority and creation time.

use std::sync::Arc;

use async_trait::async_trait;
use thiserror::Error;

use crate::models::Task;
use crate::services::{TaskError, TaskService};

/// Errors returned by the scheduler.
#[derive(Debug, Error)]
pub enum SchedulerError {
    #[error("task service error: {0}")]
    TaskService(#[from] TaskError),
}

/// Selects the next batch of ready tasks.
#[async_trait]
pub trait Scheduler: Send + Sync {
    /// Return up to `limit` tasks that are pending and have all dependencies
    /// satisfied, ordered by descending `priority_score` and ascending
    /// `created_at`.
    async fn next_batch(&self, limit: usize) -> Result<Vec<Task>, SchedulerError>;
}

/// Default scheduler implementation.
pub struct SchedulerImpl {
    task_service: Arc<dyn TaskService>,
}

impl SchedulerImpl {
    /// Create a scheduler backed by the given task service.
    pub fn new(task_service: Arc<dyn TaskService>) -> Self {
        Self { task_service }
    }
}

#[async_trait]
impl Scheduler for SchedulerImpl {
    async fn next_batch(&self, limit: usize) -> Result<Vec<Task>, SchedulerError> {
        let mut ready = self.task_service.list_ready().await?;
        ready.sort_by(|a, b| {
            b.priority_score
                .partial_cmp(&a.priority_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.created_at.cmp(&b.created_at))
        });
        Ok(ready.into_iter().take(limit).collect())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;

    use crate::models::Task;
    use crate::persistence::MemoryTaskRepository;
    use crate::services::{CreateTaskRequest, TaskService, TaskServiceImpl};
    use crate::services::{
        audit_log_service::{AuditError, AuditLogEntry, AuditLogService},
        event_service::EventServiceImpl,
    };

    use super::*;

    struct NoopAudit;

    #[async_trait]
    impl AuditLogService for NoopAudit {
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

    fn task_service() -> Arc<TaskServiceImpl<MemoryTaskRepository>> {
        let repo = Arc::new(MemoryTaskRepository::new());
        let events = Arc::new(EventServiceImpl::new(Arc::new(crate::EventBus::new())));
        Arc::new(TaskServiceImpl::new(repo, events, Arc::new(NoopAudit)))
    }

    async fn submit(
        svc: &TaskServiceImpl<MemoryTaskRepository>,
        title: &str,
        priority: i32,
    ) -> Task {
        svc.submit(CreateTaskRequest {
            project_id: "p1".to_string(),
            parent_id: None,
            title: title.to_string(),
            description: None,
            kind: "codegen".to_string(),
            assigned_agent: None,
            priority,
            payload: serde_json::json!({}),
            trace_id: None,
        })
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn scheduler_returns_ready_tasks_ordered_by_priority() {
        let svc = task_service();
        let _low = submit(svc.as_ref(), "low", 1).await;
        let _mid = submit(svc.as_ref(), "mid", 3).await;
        let high = submit(svc.as_ref(), "high", 5).await;

        let scheduler = SchedulerImpl::new(svc);
        let batch = scheduler.next_batch(2).await.unwrap();

        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].id, high.id);
        assert!(batch[0].priority_score >= batch[1].priority_score);
    }

    #[tokio::test]
    async fn scheduler_respects_limit() {
        let svc = task_service();
        for i in 0..5 {
            submit(svc.as_ref(), &format!("t{i}"), i).await;
        }

        let scheduler = SchedulerImpl::new(svc);
        let batch = scheduler.next_batch(2).await.unwrap();

        assert_eq!(batch.len(), 2);
    }

    #[tokio::test]
    async fn scheduler_skips_blocked_tasks() {
        let svc = task_service();
        let a = submit(svc.as_ref(), "a", 1).await;
        let b = svc
            .submit(CreateTaskRequest {
                project_id: "p1".to_string(),
                parent_id: None,
                title: "b".to_string(),
                description: None,
                kind: "codegen".to_string(),
                assigned_agent: None,
                priority: 0,
                payload: serde_json::json!({}),
                trace_id: None,
            })
            .await
            .unwrap();
        svc.add_dependency(crate::models::TaskDependency {
            task_id: b.id.clone(),
            depends_on: a.id.clone(),
            dep_type: "serial".to_string(),
        })
        .await
        .unwrap();

        let scheduler = SchedulerImpl::new(svc);
        let batch = scheduler.next_batch(10).await.unwrap();

        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].title, "a");
    }
}
