use std::collections::HashMap;
use std::sync::Arc;

use crate::services::InferenceService;
use serde_json::Value;
use thiserror::Error;

use crate::models::{AuditLogLevel, Task, TaskStatus};
use crate::services::{AgentService, AuditLogEntry, AuditLogService, TaskService, ToolService};

/// Errors returned by the critic council.
#[derive(Debug, Error)]
pub enum CriticCouncilError {
    #[error("agent service error: {0}")]
    Agent(String),
    #[error("task service error: {0}")]
    Task(String),
    #[error("audit error: {0}")]
    Audit(String),
    #[error("no critics produced a score")]
    NoScores,
}

/// Runs a panel of specialized critics in parallel and aggregates their scores.
#[derive(Clone)]
pub struct CriticCouncil {
    agent_service: Arc<dyn AgentService>,
    task_service: Arc<dyn TaskService>,
    inference: Arc<dyn InferenceService>,
    tools: Arc<dyn ToolService>,
    audit: Arc<dyn AuditLogService>,
    weights: HashMap<String, f64>,
}

impl CriticCouncil {
    /// Create a council with default critic weights.
    pub fn new(
        agent_service: Arc<dyn AgentService>,
        task_service: Arc<dyn TaskService>,
        inference: Arc<dyn InferenceService>,
        tools: Arc<dyn ToolService>,
        audit: Arc<dyn AuditLogService>,
    ) -> Self {
        let mut weights = HashMap::new();
        weights.insert("code".to_string(), 0.30);
        weights.insert("style".to_string(), 0.20);
        weights.insert("security".to_string(), 0.30);
        weights.insert("test".to_string(), 0.20);
        Self {
            agent_service,
            task_service,
            inference,
            tools,
            audit,
            weights,
        }
    }

    /// Replace the default weights with custom ones.
    pub fn with_weights(mut self, weights: HashMap<String, f64>) -> Self {
        self.weights = weights;
        self
    }

    /// Evaluate `task` with all configured critics and persist the aggregated score.
    pub async fn evaluate(&self, task: &Task) -> Result<f64, CriticCouncilError> {
        let parent_result = task
            .result
            .clone()
            .or_else(|| task.payload.get("parent_result").cloned())
            .unwrap_or(Value::Null);

        let mut futures = Vec::new();
        for dimension in self.weights.keys() {
            let agent_name = format!("critic-{dimension}");
            let critic_task = Task {
                id: format!("{}-crit-{}", task.id, dimension),
                project_id: task.project_id.clone(),
                parent_id: Some(task.id.clone()),
                title: format!("{dimension} review"),
                description: None,
                kind: "review".to_string(),
                status: TaskStatus::Pending,
                assigned_agent: Some(agent_name),
                priority: task.priority,
                payload: serde_json::json!({
                    "prompt": format!("Evaluate {dimension} aspects of the implementation"),
                    "parent_result": parent_result.clone(),
                }),
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
                trace_id: task.trace_id.clone(),
            };

            let agent_service = self.agent_service.clone();
            let inference = self.inference.clone();
            let tools = self.tools.clone();
            let dimension = dimension.clone();

            futures.push(async move {
                let result = agent_service
                    .execute(&critic_task, inference, tools)
                    .await
                    .map_err(|e| CriticCouncilError::Agent(e.to_string()))?;
                let score = result
                    .get("score")
                    .and_then(|v| v.as_f64())
                    .ok_or_else(|| {
                        CriticCouncilError::Agent(format!(
                            "critic-{dimension} did not return a score"
                        ))
                    })?;
                let comment = result
                    .get("comment")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                Ok::<(String, f64, String), CriticCouncilError>((dimension, score, comment))
            });
        }

        let results = futures::future::join_all(futures).await;

        let mut total_weight = 0.0;
        let mut weighted_sum = 0.0;
        for result in results {
            match result {
                Ok((dimension, score, comment)) => {
                    let weight = self.weights.get(&dimension).copied().unwrap_or(0.0);
                    weighted_sum += score * weight;
                    total_weight += weight;

                    let _ = self
                        .audit
                        .log(
                            AuditLogEntry::new("critic_council", "critic_score")
                                .project_id(&task.project_id)
                                .task_id(&task.id)
                                .level(AuditLogLevel::Info)
                                .metadata(serde_json::json!({
                                    "dimension": dimension,
                                    "score": score,
                                    "comment": comment,
                                    "weight": weight,
                                })),
                        )
                        .await;
                }
                Err(e) => {
                    let _ = self
                        .audit
                        .log(
                            AuditLogEntry::new("critic_council", "critic_failed")
                                .project_id(&task.project_id)
                                .task_id(&task.id)
                                .level(AuditLogLevel::Warn)
                                .message(e.to_string()),
                        )
                        .await;
                }
            }
        }

        if total_weight == 0.0 {
            return Err(CriticCouncilError::NoScores);
        }

        let aggregate = weighted_sum / total_weight;
        self.task_service
            .set_critic_score(&task.id, aggregate)
            .await
            .map_err(|e| CriticCouncilError::Task(e.to_string()))?;

        let _ = self
            .audit
            .log(
                AuditLogEntry::new("critic_council", "aggregate_score")
                    .project_id(&task.project_id)
                    .task_id(&task.id)
                    .level(AuditLogLevel::Info)
                    .metadata(serde_json::json!({ "critic_score": aggregate })),
            )
            .await;

        Ok(aggregate)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use async_trait::async_trait;
    use serde_json::Value;

    use crate::models::{Task, TaskStatus};
    use crate::services::{
        Agent, AgentService, AgentServiceError, AuditError, AuditLogEntry, AuditLogService,
        InferenceService, ToolDescription, ToolService, ToolServiceError,
    };

    use super::*;

    struct MockAgentService {
        scores: Mutex<HashMap<String, f64>>,
    }

    #[async_trait]
    impl AgentService for MockAgentService {
        async fn register(&self, _agent: Arc<dyn Agent>) {}
        async fn find(&self, _name: &str) -> Option<Arc<dyn Agent>> {
            None
        }
        async fn list(&self) -> Vec<String> {
            vec![]
        }
        fn route(&self, _task: &Task) -> Option<String> {
            None
        }
        async fn execute(
            &self,
            task: &Task,
            _inference: Arc<dyn InferenceService>,
            _tools: Arc<dyn ToolService>,
        ) -> Result<Value, AgentServiceError> {
            let dim = task
                .assigned_agent
                .as_deref()
                .unwrap_or("")
                .strip_prefix("critic-")
                .unwrap_or("");
            let score = self
                .scores
                .lock()
                .unwrap()
                .get(dim)
                .copied()
                .ok_or_else(|| AgentServiceError::AgentNotFound(format!("critic-{dim}")))?;
            Ok(serde_json::json!({ "dimension": dim, "score": score, "comment": "" }))
        }
    }

    struct MockTaskService {
        scores: Mutex<HashMap<String, f64>>,
    }

    #[async_trait]
    impl TaskService for MockTaskService {
        async fn submit(
            &self,
            _request: crate::services::CreateTaskRequest,
        ) -> Result<Task, crate::services::TaskError> {
            unimplemented!()
        }
        async fn add_dependency(
            &self,
            _dep: crate::models::TaskDependency,
        ) -> Result<(), crate::services::TaskError> {
            unimplemented!()
        }
        async fn get(&self, _id: &str) -> Result<Option<Task>, crate::services::TaskError> {
            unimplemented!()
        }
        async fn list_by_project(
            &self,
            _project_id: &str,
        ) -> Result<Vec<Task>, crate::services::TaskError> {
            unimplemented!()
        }
        async fn list_ready(&self) -> Result<Vec<Task>, crate::services::TaskError> {
            unimplemented!()
        }
        async fn set_status(
            &self,
            _id: &str,
            _status: TaskStatus,
        ) -> Result<Task, crate::services::TaskError> {
            unimplemented!()
        }
        async fn cancel(&self, _id: &str) -> Result<Task, crate::services::TaskError> {
            unimplemented!()
        }
        async fn set_result(
            &self,
            _id: &str,
            _result: Value,
        ) -> Result<Task, crate::services::TaskError> {
            unimplemented!()
        }
        async fn set_critic_score(
            &self,
            id: &str,
            score: f64,
        ) -> Result<Task, crate::services::TaskError> {
            self.scores.lock().unwrap().insert(id.to_string(), score);
            Ok(sample_task(id))
        }
        async fn set_human_score(
            &self,
            _id: &str,
            _score: f64,
        ) -> Result<Task, crate::services::TaskError> {
            unimplemented!()
        }
        async fn retry(
            &self,
            _id: &str,
            _feedback: Option<&str>,
        ) -> Result<Task, crate::services::TaskError> {
            unimplemented!()
        }
        async fn load_all_tasks(&self) -> Result<Vec<Task>, crate::services::TaskError> {
            unimplemented!()
        }
        async fn update_task(&self, _task: &Task) -> Result<(), crate::services::TaskError> {
            unimplemented!()
        }
    }

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

    struct NoopInference;

    #[async_trait]
    impl InferenceService for NoopInference {
        async fn generate(
            &self,
            _request: crytex_inference::InferenceRequest,
        ) -> Result<crytex_inference::InferenceResponse, crate::services::InferenceServiceError>
        {
            unimplemented!()
        }
        async fn embed(
            &self,
            _text: &str,
        ) -> Result<Vec<f32>, crate::services::InferenceServiceError> {
            Ok(vec![])
        }
        fn available_backends(&self) -> Vec<crytex_inference::BackendInfo> {
            vec![]
        }
        async fn register_lora(
            &self,
            _lora: crytex_inference::LoRAAdapter,
        ) -> Result<(), crate::services::InferenceServiceError> {
            Ok(())
        }
        async fn swap_lora(
            &self,
            _lora_id: &str,
        ) -> Result<(), crate::services::InferenceServiceError> {
            Ok(())
        }
        async fn list_models(
            &self,
            _backend_id: Option<&str>,
        ) -> Result<Vec<crytex_inference::ModelInfo>, crate::services::InferenceServiceError>
        {
            Ok(vec![])
        }
    }

    struct NoopTools;

    #[async_trait]
    impl ToolService for NoopTools {
        async fn invoke(&self, _name: &str, _args: Value) -> Result<Value, ToolServiceError> {
            Ok(Value::Null)
        }
        fn list_tools(&self) -> Vec<ToolDescription> {
            vec![]
        }
    }

    fn sample_task(id: &str) -> Task {
        Task {
            id: id.into(),
            project_id: "p1".into(),
            parent_id: None,
            title: "task".into(),
            description: None,
            kind: "review".into(),
            status: TaskStatus::Pending,
            assigned_agent: None,
            priority: 0,
            payload: Value::Null,
            result: Some(serde_json::json!({ "summary": "done" })),
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

    fn council(scores: HashMap<String, f64>) -> (CriticCouncil, Arc<MockTaskService>) {
        let task_service = Arc::new(MockTaskService {
            scores: Mutex::new(HashMap::new()),
        });
        let agent_service = Arc::new(MockAgentService {
            scores: Mutex::new(scores),
        });
        let council = CriticCouncil::new(
            agent_service,
            task_service.clone(),
            Arc::new(NoopInference),
            Arc::new(NoopTools),
            Arc::new(NoopAudit),
        );
        (council, task_service)
    }

    #[tokio::test]
    async fn critic_council_computes_weighted_average() {
        let mut scores = HashMap::new();
        scores.insert("code".to_string(), 5.0);
        scores.insert("style".to_string(), 4.0);
        scores.insert("security".to_string(), 3.0);
        scores.insert("test".to_string(), 2.0);

        let (council, task_service) = council(scores);
        let task = sample_task("t1");
        let aggregate = council.evaluate(&task).await.unwrap();

        // (5*0.3 + 4*0.2 + 3*0.3 + 2*0.2) = 1.5 + 0.8 + 0.9 + 0.4 = 3.6
        assert!((aggregate - 3.6).abs() < 0.001);
        assert!((task_service.scores.lock().unwrap()["t1"] - 3.6).abs() < 0.001);
    }

    #[tokio::test]
    async fn critic_council_ignores_missing_critic() {
        let mut scores = HashMap::new();
        scores.insert("code".to_string(), 5.0);
        scores.insert("style".to_string(), 4.0);
        // security and test missing

        let (council, task_service) = council(scores);
        let task = sample_task("t2");
        let aggregate = council.evaluate(&task).await.unwrap();

        // (5*0.3 + 4*0.2) / (0.3 + 0.2) = 2.3 / 0.5 = 4.6
        assert!((aggregate - 4.6).abs() < 0.001);
        assert!((task_service.scores.lock().unwrap()["t2"] - 4.6).abs() < 0.001);
    }
}
