use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;

use crytex_core::{
    EventBus,
    models::{AgentLog, Task, TaskStatus},
    persistence::MemoryTaskRepository,
    services::{
        AuditError, AuditLogEntry, AuditLogService, CreateTaskRequest, EventService,
        EventServiceImpl, Orchestrator, OrchestratorImpl, Scheduler, SchedulerImpl, TaskHandler,
        TaskService, TaskServiceImpl, WorkerError, WorkerPool,
    },
};

struct NoopAudit;

#[async_trait]
impl AuditLogService for NoopAudit {
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

struct CompleteHandler {
    task_service: Arc<dyn TaskService>,
}

#[async_trait]
impl TaskHandler for CompleteHandler {
    async fn handle(&self, task: Task) -> Result<(), WorkerError> {
        self.task_service
            .set_status(&task.id, TaskStatus::InProgress)
            .await
            .map_err(|e| WorkerError::Handler(e.to_string()))?;
        self.task_service
            .set_result(&task.id, Value::String("done".into()))
            .await
            .map_err(|e| WorkerError::Handler(e.to_string()))?;
        Ok(())
    }
}

fn task_service() -> Arc<dyn TaskService> {
    let repo: Arc<MemoryTaskRepository> = Arc::new(MemoryTaskRepository::new());
    let event_bus = Arc::new(EventBus::new());
    let event_service: Arc<dyn EventService> = Arc::new(EventServiceImpl::new(event_bus));
    let audit: Arc<dyn AuditLogService> = Arc::new(NoopAudit);
    Arc::new(TaskServiceImpl::new(repo, event_service, audit))
}

#[tokio::test]
async fn codegen_task_is_decomposed_and_worker_executes_ready_subtasks() {
    let task_service = task_service();

    let task = task_service
        .submit(CreateTaskRequest {
            project_id: "project-1".to_string(),
            parent_id: None,
            title: "Codegen hello world".to_string(),
            description: Some("Generate code".to_string()),
            kind: "codegen".to_string(),
            assigned_agent: None,
            priority: 1,
            payload: serde_json::json!({"prompt": "write a hello world program"}),
            trace_id: None,
        })
        .await
        .expect("submit codegen task");

    let orchestrator = OrchestratorImpl::new(task_service.clone());
    let subtasks = orchestrator
        .orchestrate(&task)
        .await
        .expect("orchestrate codegen task");
    assert_eq!(
        subtasks.len(),
        5,
        "codegen should decompose into 5 subtasks"
    );

    let scheduler = Arc::new(SchedulerImpl::new(task_service.clone()));
    let ready = scheduler.next_batch(10).await.expect("fetch ready tasks");
    assert!(!ready.is_empty(), "at least one subtask should be ready");

    let pool = Arc::new(WorkerPool::new(2));
    let handler = Arc::new(CompleteHandler {
        task_service: task_service.clone(),
    });

    let worker = tokio::spawn({
        let pool = pool.clone();
        let scheduler = scheduler.clone();
        async move {
            let _ = pool.run(scheduler, handler).await;
        }
    });

    tokio::time::sleep(Duration::from_millis(400)).await;
    pool.shutdown();
    let _ = worker.await;

    let mut completed = 0;
    for sub in &subtasks {
        if let Ok(Some(t)) = task_service.get(&sub.id).await
            && t.status == TaskStatus::Completed
        {
            completed += 1;
        }
    }

    assert!(
        completed >= 1,
        "worker should have completed at least one ready subtask"
    );
}
