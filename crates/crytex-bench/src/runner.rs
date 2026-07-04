use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use crytex_core::models::TaskStatus;
use crytex_core::services::{
    AgentService, AgentWorkflowNodeExecutor, CreateTaskRequest, InferenceService, LoraRouter,
    TaskService, ToolService, WorkflowDefinition, WorkflowEngine, WorkflowState,
};
use serde_json::Value;
use ulid::Ulid;

use crate::error::BenchError;
use crate::models::{BenchmarkCase, BenchmarkVariant};

/// Output produced by running a single benchmark case.
#[derive(Debug, Clone)]
pub struct BenchmarkRunOutput {
    pub task_id: Option<String>,
    pub result: Value,
    pub latency_ms: u64,
    pub token_usage: Option<crytex_inference::TokenUsage>,
}

/// Strategy for executing a benchmark case against a variant.
#[async_trait]
pub trait BenchmarkRunner: Send + Sync {
    async fn run(
        &self,
        case: &BenchmarkCase,
        variant: &BenchmarkVariant,
    ) -> Result<BenchmarkRunOutput, BenchError>;
}

/// Runs a case by submitting a task and invoking [`AgentService::execute`].
#[derive(Clone)]
pub struct AgentBenchmarkRunner {
    project_id: String,
    task_kind: String,
    task_service: Arc<dyn TaskService>,
    agent_service: Arc<dyn AgentService>,
    inference: Arc<dyn InferenceService>,
    tool_service: Arc<dyn ToolService>,
}

impl AgentBenchmarkRunner {
    pub fn new(
        project_id: String,
        task_kind: String,
        task_service: Arc<dyn TaskService>,
        agent_service: Arc<dyn AgentService>,
        inference: Arc<dyn InferenceService>,
        tool_service: Arc<dyn ToolService>,
    ) -> Self {
        Self {
            project_id,
            task_kind,
            task_service,
            agent_service,
            inference,
            tool_service,
        }
    }
}

#[async_trait]
impl BenchmarkRunner for AgentBenchmarkRunner {
    async fn run(
        &self,
        case: &BenchmarkCase,
        variant: &BenchmarkVariant,
    ) -> Result<BenchmarkRunOutput, BenchError> {
        let trace_id = Ulid::new().to_string();
        let mut payload = case.input.clone();
        if let Some(backend_id) = &variant.backend_id {
            payload["backend_id"] = Value::String(backend_id.clone());
        }

        let request = CreateTaskRequest {
            project_id: self.project_id.clone(),
            parent_id: None,
            title: format!("benchmark {}", case.id),
            description: None,
            kind: self.task_kind.clone(),
            assigned_agent: variant.agent_role.clone(),
            priority: 0,
            payload,
            trace_id: Some(trace_id.clone()),
        };

        let mut task = self.task_service.submit(request).await?;
        task.lora_adapter_id = variant.lora_adapter_id.clone();
        task.prompt_version_id = variant.prompt_version_id.clone();
        self.task_service.update_task(&task).await?;

        self.task_service
            .set_status(&task.id, TaskStatus::InProgress)
            .await?;

        let start = Instant::now();
        let result = self
            .agent_service
            .execute(&task, self.inference.clone(), self.tool_service.clone())
            .await;
        let latency_ms = start.elapsed().as_millis() as u64;

        match result {
            Ok(value) => {
                self.task_service
                    .set_result(&task.id, value.clone())
                    .await?;
                Ok(BenchmarkRunOutput {
                    task_id: Some(task.id),
                    result: value,
                    latency_ms,
                    token_usage: None,
                })
            }
            Err(e) => {
                let _ = self
                    .task_service
                    .set_status(&task.id, TaskStatus::Failed)
                    .await;
                Err(BenchError::Runner(e.to_string()))
            }
        }
    }
}

/// Runs a case through a [`WorkflowEngine`].
pub struct WorkflowBenchmarkRunner {
    project_id: String,
    workflow: WorkflowDefinition,
    engine: WorkflowEngine,
}

impl WorkflowBenchmarkRunner {
    pub fn new(
        project_id: String,
        workflow: WorkflowDefinition,
        agent_service: Arc<dyn AgentService>,
        inference: Arc<dyn InferenceService>,
        tool_service: Arc<dyn ToolService>,
        lora_router: Option<Arc<dyn LoraRouter>>,
    ) -> Self {
        let mut executor = AgentWorkflowNodeExecutor::new(agent_service, inference, tool_service);
        if let Some(router) = lora_router {
            executor = executor.with_lora_router(router);
        }
        let engine = WorkflowEngine::new(Arc::new(executor));
        Self {
            project_id,
            workflow,
            engine,
        }
    }
}

#[async_trait]
impl BenchmarkRunner for WorkflowBenchmarkRunner {
    async fn run(
        &self,
        case: &BenchmarkCase,
        _variant: &BenchmarkVariant,
    ) -> Result<BenchmarkRunOutput, BenchError> {
        let trace_id = Ulid::new().to_string();
        let mut state = WorkflowState::Object(serde_json::Map::new());
        if let Value::Object(map) = &mut state {
            map.insert("project_id".into(), Value::String(self.project_id.clone()));
            map.insert("trace_id".into(), Value::String(trace_id));
            map.insert("task".into(), case.input.clone());
        }

        let start = Instant::now();
        let result = self.engine.run(&self.workflow, state).await;
        let latency_ms = start.elapsed().as_millis() as u64;

        match result {
            Ok(workflow_result) => Ok(BenchmarkRunOutput {
                task_id: None,
                result: workflow_result.state,
                latency_ms,
                token_usage: None,
            }),
            Err(e) => Err(BenchError::Runner(e.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use chrono::Utc;
    use crytex_core::models::{Task, TaskStatus};
    use std::collections::HashMap;
    use std::sync::Mutex;

    struct DummyTaskService {
        tasks: Mutex<HashMap<String, Task>>,
    }

    impl DummyTaskService {
        fn new() -> Self {
            Self {
                tasks: Mutex::new(HashMap::new()),
            }
        }
    }

    #[async_trait]
    impl TaskService for DummyTaskService {
        async fn submit(
            &self,
            request: CreateTaskRequest,
        ) -> Result<Task, crytex_core::services::TaskError> {
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
                created_at: Utc::now().timestamp(),
                started_at: None,
                finished_at: None,
                payload: request.payload,
                result: None,
                iteration_count: 0,
                priority_score: 0.0,
                critic_score: None,
                human_score: None,
                prompt_version_id: None,
                lora_adapter_id: None,
                trace_id: request.trace_id.unwrap_or_default(),
            };
            self.tasks
                .lock()
                .unwrap()
                .insert(task.id.clone(), task.clone());
            Ok(task)
        }

        async fn add_dependency(
            &self,
            _dep: crytex_core::models::TaskDependency,
        ) -> Result<(), crytex_core::services::TaskError> {
            Ok(())
        }
        async fn get(&self, id: &str) -> Result<Option<Task>, crytex_core::services::TaskError> {
            Ok(self.tasks.lock().unwrap().get(id).cloned())
        }
        async fn list_by_project(
            &self,
            _project_id: &str,
        ) -> Result<Vec<Task>, crytex_core::services::TaskError> {
            Ok(vec![])
        }
        async fn list_ready(&self) -> Result<Vec<Task>, crytex_core::services::TaskError> {
            Ok(vec![])
        }
        async fn set_status(
            &self,
            id: &str,
            status: TaskStatus,
        ) -> Result<Task, crytex_core::services::TaskError> {
            let mut tasks = self.tasks.lock().unwrap();
            let task = tasks
                .get_mut(id)
                .ok_or_else(|| crytex_core::services::TaskError::NotFound(id.into()))?;
            task.status = status;
            Ok(task.clone())
        }
        async fn cancel(&self, id: &str) -> Result<Task, crytex_core::services::TaskError> {
            self.set_status(id, TaskStatus::Cancelled).await
        }
        async fn set_result(
            &self,
            id: &str,
            result: Value,
        ) -> Result<Task, crytex_core::services::TaskError> {
            let mut tasks = self.tasks.lock().unwrap();
            let task = tasks
                .get_mut(id)
                .ok_or_else(|| crytex_core::services::TaskError::NotFound(id.into()))?;
            task.result = Some(result);
            task.status = TaskStatus::Completed;
            Ok(task.clone())
        }
        async fn set_critic_score(
            &self,
            _id: &str,
            _score: f64,
        ) -> Result<Task, crytex_core::services::TaskError> {
            unimplemented!()
        }
        async fn set_human_score(
            &self,
            _id: &str,
            _score: f64,
        ) -> Result<Task, crytex_core::services::TaskError> {
            unimplemented!()
        }
        async fn retry(
            &self,
            _id: &str,
            _feedback: Option<&str>,
        ) -> Result<Task, crytex_core::services::TaskError> {
            unimplemented!()
        }
        async fn load_all_tasks(&self) -> Result<Vec<Task>, crytex_core::services::TaskError> {
            Ok(vec![])
        }
        async fn update_task(&self, task: &Task) -> Result<(), crytex_core::services::TaskError> {
            self.tasks
                .lock()
                .unwrap()
                .insert(task.id.clone(), task.clone());
            Ok(())
        }
    }

    struct DummyAgentService;

    #[async_trait]
    impl AgentService for DummyAgentService {
        async fn register(&self, _agent: Arc<dyn crytex_core::services::Agent>) {}
        async fn find(&self, _name: &str) -> Option<Arc<dyn crytex_core::services::Agent>> {
            None
        }
        async fn list(&self) -> Vec<String> {
            vec![]
        }
        fn route(&self, task: &Task) -> Option<String> {
            task.assigned_agent.clone()
        }
        async fn execute(
            &self,
            _task: &Task,
            _inference: Arc<dyn InferenceService>,
            _tools: Arc<dyn ToolService>,
        ) -> Result<Value, crytex_core::services::AgentServiceError> {
            Ok(serde_json::json!({"answer": 42}))
        }
    }

    fn make_tool_service() -> Arc<dyn ToolService> {
        struct NoopToolService;
        #[async_trait]
        impl ToolService for NoopToolService {
            async fn invoke(
                &self,
                _name: &str,
                _args: Value,
            ) -> Result<Value, crytex_core::services::ToolServiceError> {
                Ok(Value::Null)
            }
            fn list_tools(&self) -> Vec<crytex_core::services::ToolDescription> {
                vec![]
            }
        }
        Arc::new(NoopToolService)
    }

    struct DummyInferenceService;

    #[async_trait]
    impl InferenceService for DummyInferenceService {
        async fn generate(
            &self,
            _request: crytex_inference::InferenceRequest,
        ) -> Result<crytex_inference::InferenceResponse, crytex_core::services::InferenceServiceError>
        {
            Ok(crytex_inference::InferenceResponse {
                content: "42".into(),
                usage: crytex_inference::TokenUsage {
                    prompt_tokens: 1,
                    completion_tokens: 1,
                    total_tokens: 2,
                },
                finish_reason: "stop".into(),
            })
        }
        async fn embed(
            &self,
            _text: &str,
        ) -> Result<Vec<f32>, crytex_core::services::InferenceServiceError> {
            Ok(vec![])
        }
        fn available_backends(&self) -> Vec<crytex_inference::BackendInfo> {
            vec![]
        }
        async fn register_lora(
            &self,
            _lora: crytex_inference::LoRAAdapter,
        ) -> Result<(), crytex_core::services::InferenceServiceError> {
            Ok(())
        }
        async fn swap_lora(
            &self,
            _lora_id: &str,
        ) -> Result<(), crytex_core::services::InferenceServiceError> {
            Ok(())
        }
        async fn list_models(
            &self,
            _backend_id: Option<&str>,
        ) -> Result<Vec<crytex_inference::ModelInfo>, crytex_core::services::InferenceServiceError>
        {
            Ok(vec![])
        }
    }

    #[tokio::test]
    async fn agent_runner_executes_and_returns_result() {
        let task_service: Arc<dyn TaskService> = Arc::new(DummyTaskService::new());
        let agent_service: Arc<dyn AgentService> = Arc::new(DummyAgentService);
        let inference: Arc<dyn InferenceService> = Arc::new(DummyInferenceService);
        let tools = make_tool_service();
        let runner = AgentBenchmarkRunner::new(
            "p1".into(),
            "benchmark".into(),
            task_service,
            agent_service,
            inference,
            tools,
        );
        let case = BenchmarkCase {
            id: "c1".into(),
            input: serde_json::json!({"prompt": "hello"}),
            expected: Some(serde_json::json!({"answer": 42})),
            tags: vec![],
            metadata: Value::Object(serde_json::Map::new()),
        };
        let variant = BenchmarkVariant::default();
        let output = runner.run(&case, &variant).await.unwrap();
        assert_eq!(output.result, serde_json::json!({"answer": 42}));
        assert!(output.task_id.is_some());
    }
}
