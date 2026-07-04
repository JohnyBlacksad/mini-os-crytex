use std::sync::Arc;

use async_trait::async_trait;

use crytex_core::models::TaskStatus;
use crytex_core::persistence::MemoryTaskRepository;
use crytex_core::services::{
    AuditError, AuditLogEntry, AuditLogService, CreateTaskRequest, EventService, EventServiceImpl,
    RecordRewardRequest, RewardService, TaskService, TaskServiceImpl,
};

struct NoopAudit;

#[async_trait]
impl AuditLogService for NoopAudit {
    async fn log(&self, _entry: AuditLogEntry) -> Result<(), AuditError> {
        Ok(())
    }
    async fn list_by_task(
        &self,
        _task_id: &str,
    ) -> Result<Vec<crytex_core::models::AgentLog>, AuditError> {
        Ok(vec![])
    }
    async fn list_by_project(
        &self,
        _project_id: &str,
    ) -> Result<Vec<crytex_core::models::AgentLog>, AuditError> {
        Ok(vec![])
    }
}

fn task_service() -> (
    Arc<dyn TaskService>,
    Arc<dyn crytex_core::persistence::ExperienceRepository>,
) {
    let repo: Arc<MemoryTaskRepository> = Arc::new(MemoryTaskRepository::new());
    let event_bus = Arc::new(crytex_core::EventBus::new());
    let event_service: Arc<dyn EventService> = Arc::new(EventServiceImpl::new(event_bus));
    let audit: Arc<dyn AuditLogService> = Arc::new(NoopAudit);
    let task_service: Arc<dyn TaskService> =
        Arc::new(TaskServiceImpl::new(repo.clone(), event_service, audit));
    (task_service, repo)
}

#[tokio::test]
async fn approve_review_task_completes_it_and_records_experience() {
    let (task_service, repo) = task_service();

    let task = task_service
        .submit(CreateTaskRequest {
            project_id: "project-1".to_string(),
            parent_id: None,
            title: "Codegen hello world".to_string(),
            description: None,
            kind: "codegen".to_string(),
            assigned_agent: None,
            priority: 1,
            payload: serde_json::json!({"prompt": "write a hello world program"}),
            trace_id: None,
        })
        .await
        .unwrap();

    task_service
        .set_status(&task.id, TaskStatus::InProgress)
        .await
        .unwrap();
    task_service
        .set_status(&task.id, TaskStatus::Review)
        .await
        .unwrap();
    task_service.set_critic_score(&task.id, 4.0).await.unwrap();

    let reward_service = RewardService::new(repo.clone());
    let reward = reward_service
        .record(RecordRewardRequest {
            task_id: &task.id,
            project_id: None,
            prompt_version_id: None,
            critic_score: Some(4.0),
            human_score: Some(5.0),
            text: None,
            comment: None,
        })
        .await
        .unwrap();
    assert!((reward - 4.4).abs() < 0.001);

    task_service
        .set_status(&task.id, TaskStatus::Completed)
        .await
        .unwrap();

    let completed = task_service.get(&task.id).await.unwrap().unwrap();
    assert_eq!(completed.status, TaskStatus::Completed);

    let experiences = repo.list_experiences_by_task(&task.id).await.unwrap();
    assert_eq!(experiences.len(), 1);
    assert_eq!(experiences[0].human_score, Some(5.0));
}

#[tokio::test]
async fn reject_review_task_retries_and_records_experience() {
    let (task_service, repo) = task_service();

    let task = task_service
        .submit(CreateTaskRequest {
            project_id: "project-1".to_string(),
            parent_id: None,
            title: "Codegen hello world".to_string(),
            description: None,
            kind: "codegen".to_string(),
            assigned_agent: None,
            priority: 1,
            payload: serde_json::json!({"prompt": "write a hello world program"}),
            trace_id: None,
        })
        .await
        .unwrap();

    task_service
        .set_status(&task.id, TaskStatus::InProgress)
        .await
        .unwrap();
    task_service
        .set_status(&task.id, TaskStatus::Review)
        .await
        .unwrap();
    task_service.set_critic_score(&task.id, 2.0).await.unwrap();

    let reward_service = RewardService::new(repo.clone());
    let reward = reward_service
        .record(RecordRewardRequest {
            task_id: &task.id,
            project_id: None,
            prompt_version_id: None,
            critic_score: Some(2.0),
            human_score: Some(1.0),
            text: None,
            comment: Some("fix tests"),
        })
        .await
        .unwrap();
    assert!((reward - 1.6).abs() < 0.001);

    let retried = task_service
        .retry(&task.id, Some("fix tests"))
        .await
        .unwrap();
    assert_eq!(retried.status, TaskStatus::Pending);
    assert_eq!(retried.iteration_count, 1);

    let experiences = repo.list_experiences_by_task(&task.id).await.unwrap();
    assert_eq!(experiences.len(), 1);
    assert_eq!(experiences[0].human_score, Some(1.0));
}
