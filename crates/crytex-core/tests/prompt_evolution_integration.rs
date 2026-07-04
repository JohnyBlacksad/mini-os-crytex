use std::sync::Arc;

use async_trait::async_trait;
use crytex_core::models::{AgentLog, Task, TaskStatus};
use crytex_core::persistence::{MemoryTaskRepository, PromptVersionRepository};
use crytex_core::services::{
    Agent, AgentError, AgentService, AgentServiceImpl, AuditError, AuditLogEntry, AuditLogService,
    CreateTaskRequest, EventServiceImpl, InferenceService, InferenceServiceError,
    PromptEvolutionService, TaskService, TaskServiceImpl, ToolDescription, ToolService,
    ToolServiceError,
};
use crytex_inference::{InferenceRequest, InferenceResponse, LoRAAdapter, ModelInfo};
use serde_json::Value;

struct MockAudit;

#[async_trait]
impl AuditLogService for MockAudit {
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

struct NullInference;

#[async_trait]
impl InferenceService for NullInference {
    async fn generate(
        &self,
        _request: InferenceRequest,
    ) -> Result<InferenceResponse, InferenceServiceError> {
        Err(InferenceServiceError::NoBackend)
    }
    async fn embed(&self, _text: &str) -> Result<Vec<f32>, InferenceServiceError> {
        Err(InferenceServiceError::NoBackend)
    }
    fn available_backends(&self) -> Vec<crytex_inference::BackendInfo> {
        vec![]
    }
    async fn register_lora(&self, _lora: LoRAAdapter) -> Result<(), InferenceServiceError> {
        Ok(())
    }
    async fn swap_lora(&self, _lora_id: &str) -> Result<(), InferenceServiceError> {
        Ok(())
    }
    async fn list_models(
        &self,
        _backend_id: Option<&str>,
    ) -> Result<Vec<ModelInfo>, InferenceServiceError> {
        Ok(vec![])
    }
}

struct NullTools;

#[async_trait]
impl ToolService for NullTools {
    async fn invoke(&self, _name: &str, _args: Value) -> Result<Value, ToolServiceError> {
        Ok(Value::Null)
    }
    fn list_tools(&self) -> Vec<ToolDescription> {
        vec![]
    }
}

#[derive(Default)]
struct RecordingAgent {
    seen_system_prompt: std::sync::Mutex<Option<String>>,
}

#[async_trait]
impl Agent for RecordingAgent {
    fn name(&self) -> &str {
        "coder"
    }
    fn capabilities(&self) -> Vec<String> {
        vec!["codegen".into()]
    }
    async fn execute(
        &self,
        task: &Task,
        _inference: Arc<dyn InferenceService>,
        _tools: Arc<dyn ToolService>,
    ) -> Result<Value, AgentError> {
        let system = task
            .payload
            .get("system_prompt_override")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        *self.seen_system_prompt.lock().unwrap() = system;
        Ok(Value::String("done".into()))
    }
}

fn make_repos() -> Arc<MemoryTaskRepository> {
    Arc::new(MemoryTaskRepository::new())
}

#[tokio::test]
async fn task_submit_binds_active_prompt_version() {
    let repo = make_repos();
    let event_service = Arc::new(EventServiceImpl::new(
        Arc::new(crytex_core::EventBus::new()),
    ));
    let audit = Arc::new(MockAudit);
    let prompt_service = PromptEvolutionService::new(repo.clone(), repo.clone());
    prompt_service
        .seed_agent("coder", "evolved coder prompt")
        .await
        .unwrap();

    let task_service =
        TaskServiceImpl::new(repo.clone(), event_service, audit).with_prompt_repo(repo.clone());

    let task = task_service
        .submit(CreateTaskRequest {
            project_id: "p1".into(),
            parent_id: None,
            title: "write code".into(),
            description: None,
            kind: "codegen".into(),
            assigned_agent: None,
            priority: 0,
            payload: Value::Null,
            trace_id: None,
        })
        .await
        .unwrap();

    assert!(task.prompt_version_id.is_some());
    let version = repo
        .get_prompt_version(task.prompt_version_id.as_ref().unwrap())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(version.system_prompt, "evolved coder prompt");
}

#[tokio::test]
async fn agent_service_uses_system_prompt_override() {
    let repo = make_repos();
    let audit = Arc::new(MockAudit);
    let prompt_service = PromptEvolutionService::new(repo.clone(), repo.clone());
    let version = prompt_service
        .seed_agent("coder", "override prompt")
        .await
        .unwrap();

    let agent_service = AgentServiceImpl::new(audit).with_prompt_repo(repo.clone());
    let agent = Arc::new(RecordingAgent::default());
    agent_service.register(agent.clone()).await;

    let task = Task {
        id: "t1".into(),
        project_id: "p1".into(),
        parent_id: None,
        title: "task".into(),
        description: None,
        kind: "codegen".into(),
        status: TaskStatus::Pending,
        assigned_agent: None,
        priority: 0,
        created_at: 0,
        started_at: None,
        finished_at: None,
        payload: Value::Null,
        result: None,
        iteration_count: 0,
        priority_score: 0.0,
        critic_score: None,
        human_score: None,
        prompt_version_id: Some(version.id),
        lora_adapter_id: None,
        trace_id: "trace-1".into(),
    };

    agent_service
        .execute(&task, Arc::new(NullInference), Arc::new(NullTools))
        .await
        .unwrap();

    let seen = agent.seen_system_prompt.lock().unwrap().clone();
    assert_eq!(seen, Some("override prompt".to_string()));
}
