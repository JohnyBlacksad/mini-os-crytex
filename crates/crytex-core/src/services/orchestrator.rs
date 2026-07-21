//! High-level orchestrator that decomposes tasks into subtasks or executes
//! declarative workflow DAGs.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use thiserror::Error;

use crate::models::{Task, TaskDependency};
use crate::services::{
    Agent, CreateTaskRequest, InferenceService, MemoryWorkflowRepository, TaskError, TaskService,
    ToolService, WorkflowDefinition, WorkflowEdge, WorkflowEngine, WorkflowNode,
    WorkflowNodeExecutor, WorkflowRepository,
};

const MAX_ATOMIC_TITLE_CHARS: usize = 120;
const MAX_ATOMIC_PROMPT_CHARS: usize = 2_000;

/// Errors returned by the orchestrator.
#[derive(Debug, Error)]
pub enum OrchestratorError {
    #[error("task service error: {0}")]
    TaskService(#[from] TaskError),
    #[error("workflow error: {0}")]
    Workflow(#[from] crate::services::WorkflowError),
    #[error("orchestration error: {0}")]
    Internal(String),
}

/// Decomposes a high-level task into concrete subtasks with dependencies.
#[async_trait]
pub trait Orchestrator: Send + Sync {
    /// Analyze `task` and create its subtasks. Returns the newly created tasks.
    async fn orchestrate(&self, task: &Task) -> Result<Vec<Task>, OrchestratorError>;
}

fn agent_node(id: &str, agent: &str) -> WorkflowNode {
    agent_node_io(id, agent, "task", "result")
}

fn agent_node_io(id: &str, agent: &str, input: &str, output: &str) -> WorkflowNode {
    WorkflowNode::Agent {
        id: id.to_string(),
        agent: agent.to_string(),
        task_kind: None,
        input: input.to_string(),
        output: output.to_string(),
        timeout_seconds: None,
        retry: crate::services::WorkflowRetryPolicy::default(),
    }
}

/// Built-in fallback workflow for the `codegen` task kind.
fn default_codegen_workflow() -> WorkflowDefinition {
    WorkflowDefinition {
        id: "codegen".to_string(),
        name: "Default code generation pipeline".to_string(),
        version: "1.0.0".to_string(),
        entry: "architect".to_string(),
        max_concurrency: 1,
        nodes: vec![
            agent_node("architect", "architect"),
            agent_node("coder", "coder"),
            agent_node("qa", "qa"),
            agent_node("security", "security"),
            agent_node("critic", "critic"),
            WorkflowNode::End {
                id: "end".to_string(),
            },
        ],
        edges: vec![
            WorkflowEdge {
                from: "architect".to_string(),
                to: "coder".to_string(),
            },
            WorkflowEdge {
                from: "coder".to_string(),
                to: "qa".to_string(),
            },
            WorkflowEdge {
                from: "qa".to_string(),
                to: "security".to_string(),
            },
            WorkflowEdge {
                from: "security".to_string(),
                to: "critic".to_string(),
            },
            WorkflowEdge {
                from: "critic".to_string(),
                to: "end".to_string(),
            },
        ],
    }
}

/// Built-in workflow for remediation after a critic rejection.
fn default_debug_workflow() -> WorkflowDefinition {
    WorkflowDefinition {
        id: "debug".to_string(),
        name: "Reviewer rejection remediation pipeline".to_string(),
        version: "1.0.0".to_string(),
        entry: "debug".to_string(),
        max_concurrency: 1,
        nodes: vec![
            WorkflowNode::Agent {
                id: "debug".to_string(),
                agent: "coder".to_string(),
                task_kind: Some("debug".to_string()),
                input: "task".to_string(),
                output: "debug_artifact".to_string(),
                timeout_seconds: None,
                retry: crate::services::WorkflowRetryPolicy::default(),
            },
            agent_node_io("fix", "coder", "debug_artifact", "fix_artifact"),
            agent_node_io("qa", "qa", "fix_artifact", "qa_artifact"),
            agent_node_io("critic", "critic", "qa_artifact", "result"),
            WorkflowNode::End {
                id: "end".to_string(),
            },
        ],
        edges: vec![
            WorkflowEdge {
                from: "debug".to_string(),
                to: "fix".to_string(),
            },
            WorkflowEdge {
                from: "fix".to_string(),
                to: "qa".to_string(),
            },
            WorkflowEdge {
                from: "qa".to_string(),
                to: "critic".to_string(),
            },
            WorkflowEdge {
                from: "critic".to_string(),
                to: "end".to_string(),
            },
        ],
    }
}

/// Default orchestrator implementation.
pub struct OrchestratorImpl {
    task_service: Arc<dyn TaskService>,
    planning_agent: Option<Arc<dyn Agent>>,
    inference: Option<Arc<dyn InferenceService>>,
    tools: Option<Arc<dyn ToolService>>,
    workflow_repo: Option<Arc<dyn WorkflowRepository>>,
    workflow_executor: Option<Arc<dyn WorkflowNodeExecutor>>,
}

impl OrchestratorImpl {
    /// Create an orchestrator backed by the given task service.
    ///
    /// The orchestrator is initialized with an in-memory workflow repository
    /// containing the default `codegen` pipeline so that decomposition works
    /// out of the box even when no external workflow directory is configured.
    pub fn new(task_service: Arc<dyn TaskService>) -> Self {
        let repo = Arc::new(MemoryWorkflowRepository::default());
        repo.insert(default_codegen_workflow());
        repo.insert(default_debug_workflow());
        Self {
            task_service,
            planning_agent: None,
            inference: None,
            tools: None,
            workflow_repo: Some(repo),
            workflow_executor: None,
        }
    }

    /// Use an LLM planning agent (e.g. ArchitectAgent) to decompose codegen tasks.
    pub fn with_planning_agent(mut self, agent: Arc<dyn Agent>) -> Self {
        self.planning_agent = Some(agent);
        self
    }

    /// Provide the inference backend required by the planning agent.
    pub fn with_inference(mut self, inference: Arc<dyn InferenceService>) -> Self {
        self.inference = Some(inference);
        self
    }

    /// Provide the tool service required by the planning agent.
    pub fn with_tools(mut self, tools: Arc<dyn ToolService>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Replace the default in-memory workflow repository.
    pub fn with_workflow_repository(mut self, repo: Arc<dyn WorkflowRepository>) -> Self {
        self.workflow_repo = Some(repo);
        self
    }

    /// Provide a workflow node executor.
    ///
    /// When an executor is configured, the orchestrator runs the full workflow
    /// engine for matching task kinds and persists the results instead of only
    /// creating pending subtasks.
    pub fn with_workflow_executor(mut self, executor: Arc<dyn WorkflowNodeExecutor>) -> Self {
        self.workflow_executor = Some(executor);
        self
    }

    fn atomic_title(title: &str) -> String {
        title
            .chars()
            .take(MAX_ATOMIC_TITLE_CHARS)
            .collect::<String>()
    }

    fn atomic_prompt(prompt: Value) -> Value {
        prompt.as_str().map_or(prompt.clone(), |text| {
            Value::String(
                text.chars()
                    .take(MAX_ATOMIC_PROMPT_CHARS)
                    .collect::<String>(),
            )
        })
    }

    fn acceptance_criteria(agent: &str, kind: &str) -> Vec<&'static str> {
        match (agent, kind) {
            ("architect", _) => vec![
                "produce an atomic task plan",
                "declare artifact handoff requirements",
                "include acceptance criteria for downstream agents",
            ],
            ("coder", "debug") => vec![
                "reproduce the critic or human rejection",
                "identify the smallest failing behavior",
                "produce a fix artifact linked to the rejected task",
            ],
            ("coder", _) => vec![
                "change only files needed for this task",
                "include implementation evidence",
                "describe tests or validation performed",
            ],
            ("qa", _) => vec![
                "validate the incoming artifact",
                "record concrete test evidence",
                "flag regressions with reproduction steps",
            ],
            ("security", _) => vec![
                "review security-sensitive changes",
                "record risk level and mitigation",
                "block unsafe or unreviewed behavior",
            ],
            ("critic", _) => vec![
                "return pass or reject decision",
                "include blocking issues when rejecting",
                "explain remediation required for each rejection",
            ],
            _ => vec![
                "complete the assigned atomic scope",
                "return a structured artifact",
                "include validation evidence",
            ],
        }
    }

    fn enrich_stage_payload(
        parent: &Task,
        mut payload: Value,
        node_id: &str,
        agent: &str,
        kind: &str,
        prompt: Value,
        position: usize,
    ) -> Value {
        if let Some(obj) = payload.as_object_mut() {
            obj.insert("prompt".to_string(), Self::atomic_prompt(prompt));
            obj.insert(
                "orchestration_quality".to_string(),
                serde_json::json!({
                    "quality_gate_version": 1,
                    "atomic": true,
                    "node_id": node_id,
                    "role": agent,
                    "kind": kind,
                    "position": position,
                    "max_title_chars": MAX_ATOMIC_TITLE_CHARS,
                    "max_prompt_chars": MAX_ATOMIC_PROMPT_CHARS,
                    "acceptance_criteria": Self::acceptance_criteria(agent, kind),
                    "handoff_contract": {
                        "requires_input_artifact": position > 0 || kind == "debug",
                        "requires_output_artifact": true,
                        "lineage_parent_task_id": parent.id
                    }
                }),
            );
            if kind == "debug" || parent.payload.get("reviewer_task_id").is_some() {
                obj.insert(
                    "critic_report".to_string(),
                    serde_json::json!({
                        "reviewer_task_id": parent.payload.get("reviewer_task_id").cloned(),
                        "feedback": parent.payload.get("feedback").cloned()
                            .or_else(|| parent.payload.pointer("/retry_feedback/0/comment").cloned())
                    }),
                );
            }
        }
        payload
    }

    async fn decompose_codegen(&self, parent: &Task) -> Result<Vec<Task>, OrchestratorError> {
        // Try LLM-driven decomposition first if all required services are present.
        if let (Some(agent), Some(inference), Some(tools)) =
            (&self.planning_agent, &self.inference, &self.tools)
        {
            match agent
                .execute(parent, inference.clone(), tools.clone())
                .await
            {
                Ok(plan) => {
                    if let Some(tasks) = self.create_subtasks_from_plan(parent, &plan).await? {
                        return Ok(tasks);
                    }
                }
                Err(_e) => {
                    // Planning failed; fall back to the configured workflow so
                    // execution is not blocked by a bad plan.
                }
            }
        }

        let workflow = self.load_workflow(&parent.kind).await?;
        if let Some(executor) = &self.workflow_executor {
            self.execute_workflow(parent, &workflow, executor.clone())
                .await
        } else {
            self.materialize_workflow_subtasks(parent, &workflow).await
        }
    }

    /// Load the workflow definition for a task kind.
    async fn load_workflow(&self, kind: &str) -> Result<WorkflowDefinition, OrchestratorError> {
        let repo = self.workflow_repo.as_ref().ok_or_else(|| {
            OrchestratorError::Internal("no workflow repository configured".into())
        })?;
        repo.load(kind).await?.ok_or_else(|| {
            OrchestratorError::Internal(format!("no workflow found for kind {kind}"))
        })
    }

    /// Execute `workflow` for `parent` and persist the results.
    async fn execute_workflow(
        &self,
        parent: &Task,
        workflow: &WorkflowDefinition,
        executor: Arc<dyn WorkflowNodeExecutor>,
    ) -> Result<Vec<Task>, OrchestratorError> {
        let engine = WorkflowEngine::new(executor);
        let mut initial_state = parent.payload.clone();
        if let Some(obj) = initial_state.as_object_mut() {
            obj.insert(
                "project_id".to_string(),
                Value::String(parent.project_id.clone()),
            );
            obj.insert(
                "trace_id".to_string(),
                Value::String(parent.trace_id.clone()),
            );
            obj.insert("task_id".to_string(), Value::String(parent.id.clone()));
        }

        let result = engine.run(workflow, initial_state).await?;
        let persisted = self
            .persist_workflow_result(parent, workflow, &result)
            .await?;
        Ok(persisted)
    }

    /// Create completed child tasks from a workflow result and mark the parent
    /// task as completed with the final aggregated state.
    async fn persist_workflow_result(
        &self,
        parent: &Task,
        workflow: &WorkflowDefinition,
        result: &crate::services::WorkflowResult,
    ) -> Result<Vec<Task>, OrchestratorError> {
        let mut tasks = Vec::new();
        let mut task_id_by_node: HashMap<String, String> = HashMap::new();

        for node in &workflow.nodes {
            let WorkflowNode::Agent { id, .. } = node else {
                continue;
            };
            let agent = node.agent_name().unwrap_or(id);
            let kind = node.task_kind().unwrap_or(id);
            let node_result = result.node_results.get(id).cloned().unwrap_or(Value::Null);
            let task = self
                .task_service
                .submit(CreateTaskRequest {
                    project_id: parent.project_id.clone(),
                    parent_id: Some(parent.id.clone()),
                    title: Self::atomic_title(&format!("{id}: {}", parent.title)),
                    description: parent.description.clone(),
                    kind: kind.to_string(),
                    assigned_agent: Some(agent.to_string()),
                    priority: parent.priority,
                    payload: Self::enrich_stage_payload(
                        parent,
                        parent.payload.clone(),
                        id,
                        agent,
                        kind,
                        parent
                            .payload
                            .get("prompt")
                            .cloned()
                            .unwrap_or(Value::String(parent.title.clone())),
                        tasks.len(),
                    ),
                    trace_id: Some(parent.trace_id.clone()),
                })
                .await?;
            self.task_service.set_result(&task.id, node_result).await?;
            task_id_by_node.insert(id.clone(), task.id.clone());
            tasks.push(task);
        }

        for edge in &workflow.edges {
            if let (Some(dep), Some(target)) = (
                task_id_by_node.get(&edge.from),
                task_id_by_node.get(&edge.to),
            ) {
                self.task_service
                    .add_dependency(TaskDependency {
                        task_id: target.clone(),
                        depends_on: dep.clone(),
                        dep_type: "serial".to_string(),
                    })
                    .await?;
            }
        }

        self.task_service
            .set_result(&parent.id, result.state.clone())
            .await?;
        Ok(tasks)
    }

    /// Materialize a workflow as pending subtasks with dependencies.
    async fn materialize_workflow_subtasks(
        &self,
        parent: &Task,
        workflow: &WorkflowDefinition,
    ) -> Result<Vec<Task>, OrchestratorError> {
        let prompt = parent
            .payload
            .get("prompt")
            .cloned()
            .unwrap_or(Value::String(parent.title.clone()));

        let mut tasks = Vec::new();
        let mut task_id_by_node: HashMap<String, String> = HashMap::new();

        for node in &workflow.nodes {
            let WorkflowNode::Agent { id, .. } = node else {
                continue;
            };
            let agent = node.agent_name().unwrap_or(id);
            let kind = node.task_kind().unwrap_or(id);
            let stage_payload = Self::enrich_stage_payload(
                parent,
                parent.payload.clone(),
                id,
                agent,
                kind,
                prompt.clone(),
                tasks.len(),
            );

            let task = self
                .task_service
                .submit(CreateTaskRequest {
                    project_id: parent.project_id.clone(),
                    parent_id: Some(parent.id.clone()),
                    title: Self::atomic_title(&format!("{id}: {}", parent.title)),
                    description: parent.description.clone(),
                    kind: kind.to_string(),
                    assigned_agent: Some(agent.to_string()),
                    priority: parent.priority,
                    payload: stage_payload,
                    trace_id: Some(parent.trace_id.clone()),
                })
                .await?;
            task_id_by_node.insert(id.clone(), task.id.clone());
            tasks.push(task);
        }

        for edge in &workflow.edges {
            if let (Some(dep), Some(target)) = (
                task_id_by_node.get(&edge.from),
                task_id_by_node.get(&edge.to),
            ) {
                self.task_service
                    .add_dependency(TaskDependency {
                        task_id: target.clone(),
                        depends_on: dep.clone(),
                        dep_type: "serial".to_string(),
                    })
                    .await?;
            }
        }

        Ok(tasks)
    }

    /// Create subtasks from a planning agent result.
    ///
    /// Returns `Ok(Some(tasks))` when the plan contains a valid `subtasks` array,
    /// otherwise returns `Ok(None)` to trigger the fallback pipeline.
    async fn create_subtasks_from_plan(
        &self,
        parent: &Task,
        plan: &Value,
    ) -> Result<Option<Vec<Task>>, OrchestratorError> {
        let subtasks = plan
            .get("plan")
            .and_then(|p| p.get("subtasks"))
            .and_then(|s| s.as_array())
            .filter(|s| !s.is_empty());

        let subtasks = match subtasks {
            Some(s) => s,
            None => return Ok(None),
        };

        let prompt = parent
            .payload
            .get("prompt")
            .cloned()
            .unwrap_or(Value::String(parent.title.clone()));

        let mut tasks = Vec::with_capacity(subtasks.len());
        for subtask in subtasks {
            let kind = subtask
                .get("kind")
                .and_then(|v| v.as_str())
                .unwrap_or("codegen");
            let agent = subtask
                .get("agent")
                .and_then(|v| v.as_str())
                .unwrap_or("coder");
            let title = subtask
                .get("title")
                .and_then(|v| v.as_str())
                .unwrap_or("Task");
            let description = subtask
                .get("description")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let subtask_prompt = subtask
                .get("prompt")
                .and_then(|v| v.as_str())
                .map(|s| Value::String(s.to_string()))
                .unwrap_or_else(|| prompt.clone());

            let mut stage_payload = parent.payload.clone();
            if let Some(obj) = stage_payload.as_object_mut() {
                obj.insert("plan".to_string(), plan.clone());
            }
            let stage_payload = Self::enrich_stage_payload(
                parent,
                stage_payload,
                title,
                agent,
                kind,
                subtask_prompt,
                tasks.len(),
            );

            let task = self
                .task_service
                .submit(CreateTaskRequest {
                    project_id: parent.project_id.clone(),
                    parent_id: Some(parent.id.clone()),
                    title: Self::atomic_title(&format!("{}: {}", title, parent.title)),
                    description: description.or_else(|| parent.description.clone()),
                    kind: kind.to_string(),
                    assigned_agent: Some(agent.to_string()),
                    priority: parent.priority,
                    payload: stage_payload,
                    trace_id: Some(parent.trace_id.clone()),
                })
                .await?;
            tasks.push(task);
        }

        // Serial dependencies: each subtask depends on the previous one.
        for window in tasks.windows(2) {
            self.task_service
                .add_dependency(TaskDependency {
                    task_id: window[1].id.clone(),
                    depends_on: window[0].id.clone(),
                    dep_type: "serial".to_string(),
                })
                .await?;
        }

        Ok(Some(tasks))
    }
}

#[async_trait]
impl Orchestrator for OrchestratorImpl {
    async fn orchestrate(&self, task: &Task) -> Result<Vec<Task>, OrchestratorError> {
        match task.kind.as_str() {
            "codegen" => self.decompose_codegen(task).await,
            "debug" => {
                let workflow = self.load_workflow(&task.kind).await?;
                self.materialize_workflow_subtasks(task, &workflow).await
            }
            _ => Ok(vec![]),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;

    use crate::models::{Task, TaskStatus};
    use crate::persistence::MemoryTaskRepository;
    use crate::services::{Agent, AgentError, TaskService, TaskServiceImpl};
    use crate::services::{InferenceService, ToolDescription, ToolService, ToolServiceError};
    use crate::services::{
        audit_log_service::{AuditError, AuditLogEntry, AuditLogService},
        event_service::EventServiceImpl,
    };
    use serde_json::Value;

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

    fn codegen_task() -> Task {
        Task {
            id: "parent".to_string(),
            project_id: "p1".to_string(),
            parent_id: None,
            title: "landing page".to_string(),
            description: None,
            kind: "codegen".to_string(),
            status: TaskStatus::Pending,
            assigned_agent: None,
            priority: 1,
            created_at: 0,
            started_at: None,
            finished_at: None,
            payload: serde_json::json!({ "prompt": "write a landing page" }),
            result: None,
            iteration_count: 0,
            priority_score: 1.0,
            critic_score: None,
            human_score: None,
            prompt_version_id: None,
            lora_adapter_id: None,
            trace_id: "trace-parent".into(),
        }
    }

    #[tokio::test]
    async fn codegen_decomposes_into_four_tasks_by_default() {
        let svc = task_service();
        let orchestrator = OrchestratorImpl::new(svc.clone());

        let tasks = orchestrator.orchestrate(&codegen_task()).await.unwrap();

        assert_eq!(tasks.len(), 5);
        let kinds: Vec<_> = tasks.iter().map(|t| t.kind.as_str()).collect();
        assert_eq!(
            kinds,
            vec!["architecture", "codegen", "qa", "security", "review"]
        );

        let ready = svc.list_ready().await.unwrap();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].kind, "architecture");
    }

    #[tokio::test]
    async fn codegen_subtasks_inherit_parent_trace_id() {
        let svc = task_service();
        let orchestrator = OrchestratorImpl::new(svc.clone());

        let tasks = orchestrator.orchestrate(&codegen_task()).await.unwrap();
        assert!(!tasks.is_empty());
        for task in &tasks {
            assert_eq!(task.trace_id, "trace-parent");
        }
    }

    #[tokio::test]
    async fn codegen_tasks_have_serial_dependencies() {
        let svc = task_service();
        let orchestrator = OrchestratorImpl::new(svc.clone());

        let tasks = orchestrator.orchestrate(&codegen_task()).await.unwrap();
        assert_eq!(tasks.len(), 5);

        // Complete architect -> coder becomes ready.
        svc.set_status(&tasks[0].id, TaskStatus::Completed)
            .await
            .unwrap();
        let ready = svc.list_ready().await.unwrap();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].kind, "codegen");

        // Complete coder -> qa becomes ready.
        svc.set_status(&tasks[1].id, TaskStatus::Completed)
            .await
            .unwrap();
        let ready = svc.list_ready().await.unwrap();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].kind, "qa");

        // Complete qa -> security becomes ready.
        svc.set_status(&tasks[2].id, TaskStatus::Completed)
            .await
            .unwrap();
        let ready = svc.list_ready().await.unwrap();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].kind, "security");

        // Complete security -> review becomes ready.
        svc.set_status(&tasks[3].id, TaskStatus::Completed)
            .await
            .unwrap();
        let ready = svc.list_ready().await.unwrap();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].kind, "review");
    }

    #[tokio::test]
    async fn codegen_quality_gate_adds_atomic_bounds_and_acceptance_criteria() {
        let svc = task_service();
        let orchestrator = OrchestratorImpl::new(svc);

        let tasks = orchestrator.orchestrate(&codegen_task()).await.unwrap();

        assert_eq!(tasks.len(), 5);
        for task in &tasks {
            assert!(task.title.chars().count() <= MAX_ATOMIC_TITLE_CHARS);
            let quality = &task.payload["orchestration_quality"];
            assert_eq!(quality["quality_gate_version"], 1);
            assert_eq!(quality["atomic"], true);
            assert_eq!(
                quality["role"],
                task.assigned_agent.as_deref().unwrap_or_default()
            );
            assert_eq!(quality["kind"], task.kind);
            assert!(
                quality["acceptance_criteria"]
                    .as_array()
                    .is_some_and(|items| {
                        (2..=8).contains(&items.len())
                            && items.iter().all(|item| item.as_str().is_some())
                    })
            );
            assert_eq!(
                quality["handoff_contract"]["requires_output_artifact"],
                true
            );
            assert_eq!(
                quality["handoff_contract"]["lineage_parent_task_id"],
                "parent"
            );
            assert!(
                task.payload["prompt"]
                    .as_str()
                    .is_some_and(|prompt| prompt.chars().count() <= MAX_ATOMIC_PROMPT_CHARS)
            );
        }
        assert_eq!(
            tasks[0].payload["orchestration_quality"]["handoff_contract"]["requires_input_artifact"],
            false
        );
        assert_eq!(
            tasks[1].payload["orchestration_quality"]["handoff_contract"]["requires_input_artifact"],
            true
        );
    }

    #[test]
    fn debug_workflow_passes_artifact_between_remediation_agents() {
        let workflow = default_debug_workflow();

        let debug = workflow.node("debug").unwrap();
        let fix = workflow.node("fix").unwrap();
        let qa = workflow.node("qa").unwrap();
        let critic = workflow.node("critic").unwrap();

        assert!(matches!(
            debug,
            WorkflowNode::Agent {
                input, output, ..
            } if input == "task" && output == "debug_artifact"
        ));
        assert!(matches!(
            fix,
            WorkflowNode::Agent {
                input, output, ..
            } if input == "debug_artifact" && output == "fix_artifact"
        ));
        assert!(matches!(
            qa,
            WorkflowNode::Agent {
                input, output, ..
            } if input == "fix_artifact" && output == "qa_artifact"
        ));
        assert!(matches!(
            critic,
            WorkflowNode::Agent {
                input, output, ..
            } if input == "qa_artifact" && output == "result"
        ));
    }

    #[tokio::test]
    async fn non_codegen_returns_empty() {
        let svc = task_service();
        let orchestrator = OrchestratorImpl::new(svc);

        let mut task = codegen_task();
        task.kind = "research".to_string();

        let tasks = orchestrator.orchestrate(&task).await.unwrap();
        assert!(tasks.is_empty());
    }

    struct MockPlanningAgent {
        plan: Value,
    }

    #[async_trait]
    impl Agent for MockPlanningAgent {
        fn name(&self) -> &str {
            "architect"
        }
        fn capabilities(&self) -> Vec<String> {
            vec!["planning".into()]
        }
        async fn execute(
            &self,
            _task: &Task,
            _inference: Arc<dyn InferenceService>,
            _tools: Arc<dyn ToolService>,
        ) -> Result<Value, AgentError> {
            Ok(self.plan.clone())
        }
    }

    struct MockInference;

    #[async_trait]
    impl InferenceService for MockInference {
        async fn generate(
            &self,
            _request: crytex_inference::InferenceRequest,
        ) -> Result<crytex_inference::InferenceResponse, crate::services::InferenceServiceError>
        {
            unreachable!("mock planning agent does not call inference")
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

    struct MockToolService;

    #[async_trait]
    impl ToolService for MockToolService {
        async fn invoke(&self, _name: &str, _args: Value) -> Result<Value, ToolServiceError> {
            Ok(Value::Null)
        }
        fn list_tools(&self) -> Vec<ToolDescription> {
            vec![]
        }
    }

    #[tokio::test]
    async fn codegen_uses_planning_agent_when_configured() {
        let svc = task_service();
        let plan = serde_json::json!({
            "plan": {
                "goal": "build landing page",
                "assumptions": [],
                "subtasks": [
                    { "kind": "codegen", "agent": "coder", "title": "Implement", "description": "", "prompt": "write html" },
                    { "kind": "qa", "agent": "qa", "title": "Verify", "description": "", "prompt": "run tests" }
                ]
            },
            "summary": "Two-step plan"
        });
        let orchestrator = OrchestratorImpl::new(svc.clone())
            .with_planning_agent(Arc::new(MockPlanningAgent { plan }))
            .with_inference(Arc::new(MockInference))
            .with_tools(Arc::new(MockToolService));

        let tasks = orchestrator.orchestrate(&codegen_task()).await.unwrap();

        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].kind, "codegen");
        assert_eq!(tasks[0].assigned_agent, Some("coder".to_string()));
        assert_eq!(tasks[1].kind, "qa");
        assert_eq!(tasks[1].assigned_agent, Some("qa".to_string()));

        let ready = svc.list_ready().await.unwrap();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].id, tasks[0].id);
    }

    #[tokio::test]
    async fn planning_agent_subtasks_are_bounded_and_keep_acceptance_criteria() {
        let svc = task_service();
        let long_title = "Implement ".repeat(80);
        let long_prompt = "write validated code ".repeat(300);
        let plan = serde_json::json!({
            "plan": {
                "goal": "build bounded task graph",
                "subtasks": [
                    {
                        "kind": "codegen",
                        "agent": "coder",
                        "title": long_title,
                        "description": "bounded",
                        "prompt": long_prompt
                    }
                ]
            }
        });
        let orchestrator = OrchestratorImpl::new(svc)
            .with_planning_agent(Arc::new(MockPlanningAgent { plan }))
            .with_inference(Arc::new(MockInference))
            .with_tools(Arc::new(MockToolService));

        let tasks = orchestrator.orchestrate(&codegen_task()).await.unwrap();

        assert_eq!(tasks.len(), 1);
        assert!(tasks[0].title.chars().count() <= MAX_ATOMIC_TITLE_CHARS);
        assert!(
            tasks[0].payload["prompt"]
                .as_str()
                .is_some_and(|prompt| prompt.chars().count() <= MAX_ATOMIC_PROMPT_CHARS)
        );
        assert_eq!(
            tasks[0].payload["orchestration_quality"]["acceptance_criteria"][0],
            "change only files needed for this task"
        );
    }

    #[tokio::test]
    async fn codegen_falls_back_to_default_when_planning_agent_returns_no_subtasks() {
        let svc = task_service();
        let plan = serde_json::json!({ "plan": { "subtasks": [] }, "summary": "empty" });
        let orchestrator = OrchestratorImpl::new(svc.clone())
            .with_planning_agent(Arc::new(MockPlanningAgent { plan }))
            .with_inference(Arc::new(MockInference))
            .with_tools(Arc::new(MockToolService));

        let tasks = orchestrator.orchestrate(&codegen_task()).await.unwrap();
        assert_eq!(tasks.len(), 5);
    }

    #[tokio::test]
    async fn debug_remediation_quality_gate_preserves_rejection_feedback() {
        let svc = task_service();
        let mut rejected = codegen_task();
        rejected.id = "critic-review-1".into();
        rejected.kind = "debug".into();
        rejected.assigned_agent = Some("coder".into());
        rejected.payload = serde_json::json!({
            "prompt": "fix rejected implementation",
            "source": "reviewer_rejection",
            "reviewer_task_id": "critic-review-1",
            "feedback": "missing deterministic regression evidence"
        });
        let orchestrator = OrchestratorImpl::new(svc.clone());

        let tasks = orchestrator.orchestrate(&rejected).await.unwrap();

        assert_eq!(tasks.len(), 4);
        assert_eq!(tasks[0].kind, "debug");
        assert_eq!(tasks[0].assigned_agent.as_deref(), Some("coder"));
        assert_eq!(
            tasks[0].payload["critic_report"]["feedback"],
            "missing deterministic regression evidence"
        );
        assert_eq!(
            tasks[0].payload["orchestration_quality"]["acceptance_criteria"][0],
            "reproduce the critic or human rejection"
        );
        assert_eq!(
            tasks[0].payload["orchestration_quality"]["handoff_contract"]["requires_input_artifact"],
            true
        );
        let ready = svc.list_ready().await.unwrap();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].id, tasks[0].id);
    }

    #[derive(Default)]
    struct RecordingExecutor {
        calls: std::sync::Mutex<Vec<String>>,
    }

    #[async_trait]
    impl WorkflowNodeExecutor for RecordingExecutor {
        async fn execute(
            &self,
            node: &WorkflowNode,
            _state: &crate::services::WorkflowState,
        ) -> Result<crate::services::WorkflowState, crate::services::WorkflowError> {
            self.calls.lock().unwrap().push(node.id().to_string());
            Ok(Value::String(format!("result-{}", node.id())))
        }
    }

    #[tokio::test]
    async fn codegen_executes_workflow_and_persists_results_when_executor_configured() {
        let svc = task_service();
        let repo = Arc::new(MemoryWorkflowRepository::default());
        repo.insert(default_codegen_workflow());
        let executor = Arc::new(RecordingExecutor::default());
        let orchestrator = OrchestratorImpl::new(svc.clone())
            .with_workflow_repository(repo)
            .with_workflow_executor(executor.clone());

        let parent = svc
            .submit(CreateTaskRequest {
                project_id: "p1".to_string(),
                parent_id: None,
                title: "landing page".to_string(),
                description: None,
                kind: "codegen".to_string(),
                assigned_agent: None,
                priority: 1,
                payload: serde_json::json!({ "prompt": "write a landing page" }),
                trace_id: None,
            })
            .await
            .unwrap();
        let tasks = orchestrator.orchestrate(&parent).await.unwrap();

        assert_eq!(tasks.len(), 5);
        for task in &tasks {
            let persisted = svc.get(&task.id).await.unwrap().unwrap();
            assert_eq!(persisted.status, TaskStatus::Completed);
            assert!(persisted.result.is_some());
        }

        let parent_task = svc.get(&parent.id).await.unwrap().unwrap();
        assert_eq!(parent_task.status, TaskStatus::Completed);
        assert!(parent_task.result.is_some());

        let calls = executor.calls.lock().unwrap();
        assert_eq!(calls.len(), 5);
        assert!(calls.contains(&"architect".to_string()));
    }
}
