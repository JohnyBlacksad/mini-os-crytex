//! Configurable workflow DAG engine.
//!
//! Workflows are declarative directed acyclic graphs of nodes.  Each node is a
//! unit of work (typically an agent call) that reads from and writes to a
//! shared [`WorkflowState`].  The engine validates the graph, schedules ready
//! nodes, and executes them with bounded concurrency.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::models::{Task, TaskStatus};
use crate::services::{AgentRole, AgentService, InferenceService, LoraRouter, ToolService};
use crate::tracing::TraceContext;

use petgraph::graph::{DiGraph, NodeIndex};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur when working with workflows.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum WorkflowError {
    #[error("workflow has no nodes")]
    EmptyWorkflow,
    #[error("entry node '{0}' not found")]
    EntryNotFound(String),
    #[error("node '{0}' is referenced by an edge but not defined")]
    UnknownNode(String),
    #[error("duplicate node id: {0}")]
    DuplicateNodeId(String),
    #[error("dependency cycle detected in workflow")]
    CycleDetected,
    #[error("invalid condition branch: {0}")]
    InvalidBranch(String),
    #[error("failed to parse workflow: {0}")]
    Parse(String),
    #[error("workflow '{0}' not found")]
    NotFound(String),
    #[error("execution error: {0}")]
    Execution(String),
    #[error("workflow engine error: {0}")]
    Internal(String),
}

/// A declarative workflow definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WorkflowDefinition {
    pub id: String,
    pub name: String,
    pub version: String,
    pub entry: String,
    pub max_concurrency: usize,
    pub nodes: Vec<WorkflowNode>,
    pub edges: Vec<WorkflowEdge>,
}

impl Default for WorkflowDefinition {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: String::new(),
            version: "1.0.0".to_string(),
            entry: String::new(),
            max_concurrency: 4,
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }
}

impl WorkflowDefinition {
    /// Parse a workflow from a TOML string.
    pub fn from_toml(content: &str) -> Result<Self, WorkflowError> {
        toml::from_str(content).map_err(|e| WorkflowError::Parse(e.to_string()))
    }

    /// Serialize the workflow to a TOML string.
    pub fn to_toml(&self) -> Result<String, WorkflowError> {
        toml::to_string_pretty(self).map_err(|e| WorkflowError::Parse(e.to_string()))
    }

    /// Validate the structural correctness of the workflow.
    ///
    /// Checks that:
    /// * the workflow is non-empty,
    /// * the entry node exists,
    /// * all edge endpoints reference defined nodes,
    /// * node ids are unique,
    /// * the graph is acyclic,
    /// * condition branches reference defined nodes.
    pub fn validate(&self) -> Result<(), WorkflowError> {
        if self.nodes.is_empty() {
            return Err(WorkflowError::EmptyWorkflow);
        }

        let mut ids = HashSet::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let id = node.id();
            if !ids.insert(id) {
                return Err(WorkflowError::DuplicateNodeId(id.to_string()));
            }
        }

        if !ids.contains(self.entry.as_str()) {
            return Err(WorkflowError::EntryNotFound(self.entry.clone()));
        }

        for edge in &self.edges {
            if !ids.contains(edge.from.as_str()) {
                return Err(WorkflowError::UnknownNode(edge.from.clone()));
            }
            if !ids.contains(edge.to.as_str()) {
                return Err(WorkflowError::UnknownNode(edge.to.clone()));
            }
        }

        for node in &self.nodes {
            if let WorkflowNode::Condition {
                then_branch,
                else_branch,
                ..
            } = node
            {
                if !ids.contains(then_branch.as_str()) {
                    return Err(WorkflowError::InvalidBranch(format!(
                        "then branch '{then_branch}' not found"
                    )));
                }
                if let Some(branch) = else_branch
                    && !ids.contains(branch.as_str())
                {
                    return Err(WorkflowError::InvalidBranch(format!(
                        "else branch '{branch}' not found"
                    )));
                }
            }
        }

        if self.has_cycle() {
            return Err(WorkflowError::CycleDetected);
        }

        Ok(())
    }

    fn has_cycle(&self) -> bool {
        let mut graph = DiGraph::<&str, ()>::new();
        let mut index_by_id = HashMap::<&str, NodeIndex>::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let idx = graph.add_node(node.id());
            index_by_id.insert(node.id(), idx);
        }
        for edge in &self.edges {
            if let (Some(&from), Some(&to)) = (
                index_by_id.get(edge.from.as_str()),
                index_by_id.get(edge.to.as_str()),
            ) {
                graph.add_edge(from, to, ());
            }
        }
        petgraph::algo::is_cyclic_directed(&graph)
    }

    /// Find a node by id.
    pub fn node(&self, id: &str) -> Option<&WorkflowNode> {
        self.nodes.iter().find(|n| n.id() == id)
    }
}

/// A single node in a workflow.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorkflowNode {
    /// Execute an agent.
    Agent {
        id: String,
        /// Name of the registered agent to execute.
        agent: String,
        /// Optional task kind used for capability selection.  Defaults to a
        /// sensible value derived from `agent`.
        #[serde(default)]
        task_kind: Option<String>,
        /// Key in [`WorkflowState`] to use as the primary prompt/input for the agent.
        #[serde(default = "default_input_key")]
        input: String,
        /// Key in [`WorkflowState`] where the agent result is written.
        #[serde(default = "default_output_key")]
        output: String,
        /// Per-node timeout in seconds.  Falls back to the workflow default.
        #[serde(default)]
        timeout_seconds: Option<u64>,
        /// Per-node retry policy.
        #[serde(default)]
        retry: WorkflowRetryPolicy,
    },
    /// Branch based on a predicate over the workflow state.
    Condition {
        id: String,
        expression: String,
        then_branch: String,
        #[serde(default)]
        else_branch: Option<String>,
    },
    /// Explicit workflow termination.
    End { id: String },
}

fn default_input_key() -> String {
    "task".to_string()
}

fn default_output_key() -> String {
    "result".to_string()
}

impl WorkflowNode {
    /// Return the node id.
    pub fn id(&self) -> &str {
        match self {
            WorkflowNode::Agent { id, .. } => id,
            WorkflowNode::Condition { id, .. } => id,
            WorkflowNode::End { id } => id,
        }
    }

    /// Return the task kind for an agent node, using the explicit override or
    /// deriving it from the agent name.
    pub fn task_kind(&self) -> Option<&str> {
        match self {
            WorkflowNode::Agent {
                task_kind, agent, ..
            } => task_kind
                .as_deref()
                .or_else(|| default_kind_for_agent(agent)),
            _ => None,
        }
    }

    /// Return the agent name for an agent node.
    pub fn agent_name(&self) -> Option<&str> {
        match self {
            WorkflowNode::Agent { agent, .. } => Some(agent),
            _ => None,
        }
    }
}

fn default_kind_for_agent(agent: &str) -> Option<&str> {
    match agent {
        "architect" => Some("architecture"),
        "coder" => Some("codegen"),
        "qa" => Some("qa"),
        "security" => Some("security"),
        "critic" => Some("review"),
        "researcher" => Some("research"),
        "summarizer" => Some("summarization"),
        _ => None,
    }
}

/// A directed edge between two nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowEdge {
    pub from: String,
    pub to: String,
}

/// Retry policy for a single workflow node.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
pub struct WorkflowRetryPolicy {
    pub max_attempts: u32,
    pub delay_seconds: u64,
    pub backoff: BackoffStrategy,
}

impl Default for WorkflowRetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 0,
            delay_seconds: 1,
            backoff: BackoffStrategy::Fixed,
        }
    }
}

/// Backoff strategy used when retrying a failed node.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BackoffStrategy {
    #[default]
    Fixed,
    Linear,
    Exponential,
}

/// Shared state passed between workflow nodes.
pub type WorkflowState = serde_json::Value;

/// Result of executing a workflow.
#[derive(Debug, Clone)]
pub struct WorkflowResult {
    pub state: WorkflowState,
    pub node_results: HashMap<String, WorkflowState>,
}

/// Executes a single workflow node.
#[async_trait::async_trait]
pub trait WorkflowNodeExecutor: Send + Sync {
    /// Execute `node` against the current `state` and return its output.
    async fn execute(
        &self,
        node: &WorkflowNode,
        state: &WorkflowState,
    ) -> Result<WorkflowState, WorkflowError>;
}

/// Engine that runs a validated workflow DAG.
pub struct WorkflowEngine {
    executor: Arc<dyn WorkflowNodeExecutor>,
}

impl WorkflowEngine {
    /// Create a new engine backed by the given node executor.
    pub fn new(executor: Arc<dyn WorkflowNodeExecutor>) -> Self {
        Self { executor }
    }

    /// Execute `workflow` starting from `initial_state`.
    pub async fn run(
        &self,
        workflow: &WorkflowDefinition,
        initial_state: WorkflowState,
    ) -> Result<WorkflowResult, WorkflowError> {
        workflow.validate()?;

        let mut state = initial_state;
        let mut node_results = HashMap::new();

        let incoming = incoming_edges(workflow);
        let outgoing = outgoing_edges(workflow);

        let mut remaining_deps: HashMap<&str, usize> = HashMap::new();
        for node in &workflow.nodes {
            let count = incoming.get(node.id()).map(Vec::len).unwrap_or(0);
            remaining_deps.insert(node.id(), count);
        }

        let mut pending: HashSet<&str> = workflow
            .nodes
            .iter()
            .filter(|n| remaining_deps.get(n.id()).copied().unwrap_or(0) == 0)
            .map(|n| n.id())
            .collect();

        let mut completed: HashSet<String> = HashSet::new();
        let mut skipped: HashSet<String> = HashSet::new();

        let semaphore = Arc::new(tokio::sync::Semaphore::new(workflow.max_concurrency.max(1)));

        while !pending.is_empty() {
            let (conditions, executable): (Vec<&str>, Vec<&str>) = pending
                .iter()
                .copied()
                .partition(|id| matches!(workflow.node(id), Some(WorkflowNode::Condition { .. })));

            for id in conditions {
                let node = workflow
                    .node(id)
                    .ok_or_else(|| WorkflowError::Internal(format!("missing node {id}")))?;
                let branch = evaluate_condition_node(node, &state)?;
                completed.insert(id.to_string());
                pending.remove(id);
                schedule_condition_successors(
                    id,
                    branch.as_deref(),
                    &outgoing,
                    &mut remaining_deps,
                    &mut pending,
                    &mut skipped,
                    &completed,
                );
            }

            if executable.is_empty() && pending.is_empty() {
                break;
            }

            let (ends, agents): (Vec<&str>, Vec<&str>) = executable
                .into_iter()
                .partition(|id| matches!(workflow.node(id), Some(WorkflowNode::End { .. })));

            for id in ends {
                completed.insert(id.to_string());
                pending.remove(id);
                release_successors(
                    id,
                    &outgoing,
                    &mut remaining_deps,
                    &completed,
                    &skipped,
                    &mut pending,
                )?;
            }

            let mut handles = Vec::with_capacity(agents.len());
            for id in agents {
                let node = workflow
                    .node(id)
                    .ok_or_else(|| WorkflowError::Internal(format!("missing node {id}")))?
                    .clone();
                let permit = semaphore
                    .clone()
                    .acquire_owned()
                    .await
                    .map_err(|e| WorkflowError::Internal(format!("semaphore closed: {e}")))?;
                let state = state.clone();
                let executor = self.executor.clone();
                let node_id = id.to_string();
                handles.push(tokio::spawn(async move {
                    let _permit = permit;
                    let output = executor.execute(&node, &state).await?;
                    Ok::<_, WorkflowError>((node_id, output))
                }));
            }

            for handle in handles {
                let (id, output) = handle
                    .await
                    .map_err(|e| WorkflowError::Internal(format!("task join failed: {e}")))??;

                if let WorkflowNode::Agent {
                    output: output_key, ..
                } = workflow
                    .node(id.as_str())
                    .ok_or_else(|| WorkflowError::Internal(format!("missing node {id}")))?
                {
                    state[output_key.clone()] = output.clone();
                }
                node_results.insert(id.clone(), output);
                completed.insert(id.clone());
                pending.remove(id.as_str());

                release_successors(
                    id.as_str(),
                    &outgoing,
                    &mut remaining_deps,
                    &completed,
                    &skipped,
                    &mut pending,
                )?;
            }
        }

        Ok(WorkflowResult {
            state,
            node_results,
        })
    }
}

fn incoming_edges(workflow: &WorkflowDefinition) -> HashMap<&str, Vec<&str>> {
    let mut map: HashMap<&str, Vec<&str>> = HashMap::new();
    for node in &workflow.nodes {
        map.entry(node.id()).or_default();
    }
    for edge in &workflow.edges {
        map.entry(edge.to.as_str())
            .or_default()
            .push(edge.from.as_str());
    }
    map
}

fn outgoing_edges(workflow: &WorkflowDefinition) -> HashMap<&str, Vec<&str>> {
    let mut map: HashMap<&str, Vec<&str>> = HashMap::new();
    for node in &workflow.nodes {
        map.entry(node.id()).or_default();
    }
    for edge in &workflow.edges {
        map.entry(edge.from.as_str())
            .or_default()
            .push(edge.to.as_str());
    }
    map
}

fn evaluate_condition_node(
    node: &WorkflowNode,
    state: &WorkflowState,
) -> Result<Option<String>, WorkflowError> {
    match node {
        WorkflowNode::Condition {
            expression,
            then_branch,
            else_branch,
            ..
        } => {
            let truthy = evaluate_expression(expression, state);
            Ok(if truthy {
                Some(then_branch.clone())
            } else {
                else_branch.clone()
            })
        }
        _ => Err(WorkflowError::Internal(format!(
            "node {} is not a condition",
            node.id()
        ))),
    }
}

fn evaluate_expression(expression: &str, state: &WorkflowState) -> bool {
    let key = expression.trim();
    let key = key.strip_prefix("state.").unwrap_or(key);
    let value = if key.is_empty() {
        state
    } else {
        state.get(key).unwrap_or(&serde_json::Value::Null)
    };
    is_truthy(value)
}

fn is_truthy(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Bool(b) => *b,
        serde_json::Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(false),
        serde_json::Value::String(s) => !s.is_empty(),
        serde_json::Value::Array(a) => !a.is_empty(),
        serde_json::Value::Object(o) => !o.is_empty(),
        serde_json::Value::Null => false,
    }
}

fn schedule_condition_successors<'a>(
    id: &str,
    branch: Option<&str>,
    outgoing: &HashMap<&'a str, Vec<&'a str>>,
    remaining_deps: &mut HashMap<&'a str, usize>,
    pending: &mut HashSet<&'a str>,
    skipped: &mut HashSet<String>,
    completed: &HashSet<String>,
) {
    let targets = outgoing.get(id).cloned().unwrap_or_default();
    for target in targets {
        let selected = branch.map(|b| b == target).unwrap_or(false);
        if selected {
            let remaining = remaining_deps.entry(target).or_insert(0);
            if *remaining > 0 {
                *remaining -= 1;
            }
            if *remaining == 0 && !completed.contains(target) && !skipped.contains(target) {
                pending.insert(target);
            }
        } else {
            mark_skipped(
                target,
                outgoing,
                remaining_deps,
                skipped,
                completed,
                pending,
            );
        }
    }
}

fn mark_skipped<'a>(
    id: &str,
    outgoing: &HashMap<&'a str, Vec<&'a str>>,
    _remaining_deps: &mut HashMap<&'a str, usize>,
    skipped: &mut HashSet<String>,
    _completed: &HashSet<String>,
    pending: &mut HashSet<&'a str>,
) {
    if skipped.insert(id.to_string()) {
        pending.remove(id);
        for target in outgoing.get(id).cloned().unwrap_or_default() {
            mark_skipped(
                target,
                outgoing,
                _remaining_deps,
                skipped,
                _completed,
                pending,
            );
        }
    }
}

fn release_successors<'a>(
    id: &str,
    outgoing: &HashMap<&'a str, Vec<&'a str>>,
    remaining_deps: &mut HashMap<&'a str, usize>,
    completed: &HashSet<String>,
    skipped: &HashSet<String>,
    pending: &mut HashSet<&'a str>,
) -> Result<(), WorkflowError> {
    for target in outgoing.get(id).iter().flat_map(|v| v.iter().copied()) {
        let remaining = remaining_deps
            .get_mut(target)
            .ok_or_else(|| WorkflowError::Internal(format!("missing dep count for {target}")))?;
        if *remaining == 0 {
            continue;
        }
        *remaining -= 1;
        if *remaining == 0 && !completed.contains(target) && !skipped.contains(target) {
            pending.insert(target);
        }
    }
    Ok(())
}

/// Persistence abstraction for workflow definitions.
#[async_trait::async_trait]
pub trait WorkflowRepository: Send + Sync {
    /// Load a workflow by id. Returns `None` if the workflow is not known.
    async fn load(&self, id: &str) -> Result<Option<WorkflowDefinition>, WorkflowError>;
}

/// In-memory workflow repository for tests.
#[derive(Default)]
pub struct MemoryWorkflowRepository {
    workflows: Mutex<HashMap<String, WorkflowDefinition>>,
}

impl MemoryWorkflowRepository {
    /// Store a workflow definition.
    pub fn insert(&self, workflow: WorkflowDefinition) {
        if let Ok(mut guard) = self.workflows.lock() {
            guard.insert(workflow.id.clone(), workflow);
        }
    }
}

#[async_trait::async_trait]
impl WorkflowRepository for MemoryWorkflowRepository {
    async fn load(&self, id: &str) -> Result<Option<WorkflowDefinition>, WorkflowError> {
        Ok(self
            .workflows
            .lock()
            .ok()
            .and_then(|guard| guard.get(id).cloned()))
    }
}

/// File-system workflow repository that loads TOML definitions from a directory.
#[derive(Clone)]
pub struct TomlWorkflowRepository {
    dir: PathBuf,
}

impl TomlWorkflowRepository {
    /// Create a repository rooted at `dir`.
    pub fn new(dir: PathBuf) -> Self {
        Self { dir }
    }

    /// Return the path where a workflow file would be stored.
    pub fn path_for(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{id}.toml"))
    }
}

#[async_trait::async_trait]
impl WorkflowRepository for TomlWorkflowRepository {
    async fn load(&self, id: &str) -> Result<Option<WorkflowDefinition>, WorkflowError> {
        let path = self.path_for(id);
        if !path.exists() {
            return Ok(None);
        }
        let contents = tokio::fs::read_to_string(&path)
            .await
            .map_err(|e| WorkflowError::Internal(format!("failed to read workflow {id}: {e}")))?;
        let workflow = WorkflowDefinition::from_toml(&contents)?;
        if workflow.id != id {
            return Err(WorkflowError::Internal(format!(
                "workflow file {} declares id '{}' but expected '{id}'",
                path.display(),
                workflow.id
            )));
        }
        Ok(Some(workflow))
    }
}

/// Executes agent nodes by delegating to the registered [`AgentService`].
pub struct AgentWorkflowNodeExecutor {
    agent_service: Arc<dyn AgentService>,
    inference: Arc<dyn InferenceService>,
    tool_service: Arc<dyn ToolService>,
    lora_router: Option<Arc<dyn LoraRouter>>,
}

impl AgentWorkflowNodeExecutor {
    /// Create a new executor.
    pub fn new(
        agent_service: Arc<dyn AgentService>,
        inference: Arc<dyn InferenceService>,
        tool_service: Arc<dyn ToolService>,
    ) -> Self {
        Self {
            agent_service,
            inference,
            tool_service,
            lora_router: None,
        }
    }

    /// Attach a role-aware LoRA router so agent nodes can load role-specific adapters.
    pub fn with_lora_router(mut self, router: Arc<dyn LoraRouter>) -> Self {
        self.lora_router = Some(router);
        self
    }

    fn build_task(node: &WorkflowNode, state: &WorkflowState) -> Result<Task, WorkflowError> {
        let WorkflowNode::Agent {
            id, agent, input, ..
        } = node
        else {
            return Err(WorkflowError::Execution(format!(
                "node {} is not an agent",
                node.id()
            )));
        };

        let project_id = state
            .get("project_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let trace_id = state
            .get("trace_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| TraceContext::new().trace_id);

        let session_id = ulid::Ulid::new().to_string();
        let prompt = state.get(input).cloned().unwrap_or_default();
        let payload = serde_json::json!({
            "prompt": prompt,
            "upstream_artifact": state.get(input).cloned().unwrap_or_default(),
            "agent_session": {
                "session_id": session_id,
                "trace_id": trace_id,
                "role": agent,
                "node_id": id,
                "input_key": input,
                "clean_context": true
            }
        });

        let kind = node.task_kind().unwrap_or(agent).to_string();

        Ok(Task {
            id: ulid::Ulid::new().to_string(),
            project_id,
            parent_id: None,
            title: format!("workflow {id}"),
            description: None,
            kind,
            status: TaskStatus::Pending,
            assigned_agent: Some(agent.clone()),
            priority: 0,
            payload,
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
            trace_id,
        })
    }
}

#[async_trait::async_trait]
impl WorkflowNodeExecutor for AgentWorkflowNodeExecutor {
    async fn execute(
        &self,
        node: &WorkflowNode,
        state: &WorkflowState,
    ) -> Result<WorkflowState, WorkflowError> {
        let mut task = Self::build_task(node, state)?;

        if let Some(router) = &self.lora_router
            && let Some(role) = task
                .assigned_agent
                .as_deref()
                .and_then(AgentRole::from_agent)
            && let Some(adapter_id) = router
                .resolve_for_role(role, &task.project_id)
                .await
                .map_err(|e| WorkflowError::Execution(e.to_string()))?
        {
            task.lora_adapter_id = Some(adapter_id);
        }

        let result = self
            .agent_service
            .execute(&task, self.inference.clone(), self.tool_service.clone())
            .await
            .map_err(|e| WorkflowError::Execution(e.to_string()))?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_toml() -> &'static str {
        r#"
id = "codegen"
name = "Code generation"
version = "1.0.0"
entry = "architect"
max_concurrency = 4

[[nodes]]
type = "agent"
id = "architect"
agent = "architect"
input = "task"
output = "design"

[[nodes]]
type = "agent"
id = "coder"
agent = "coder"
input = "design"
output = "patch"

[[nodes]]
type = "end"
id = "done"

[[edges]]
from = "architect"
to = "coder"

[[edges]]
from = "coder"
to = "done"
"#
    }

    #[test]
    fn parse_workflow_from_toml() {
        let wf = WorkflowDefinition::from_toml(sample_toml()).unwrap();
        assert_eq!(wf.id, "codegen");
        assert_eq!(wf.nodes.len(), 3);
        assert_eq!(wf.edges.len(), 2);
    }

    #[test]
    fn validate_accepts_dag() {
        let wf = WorkflowDefinition::from_toml(sample_toml()).unwrap();
        wf.validate().unwrap();
    }

    #[test]
    fn validate_rejects_empty_workflow() {
        let wf = WorkflowDefinition {
            id: "empty".to_string(),
            entry: "start".to_string(),
            ..Default::default()
        };
        assert_eq!(wf.validate(), Err(WorkflowError::EmptyWorkflow));
    }

    #[test]
    fn validate_rejects_missing_entry() {
        let toml = r#"
id = "bad"
entry = "missing"

[[nodes]]
type = "agent"
id = "a"
agent = "a"
"#;
        let wf = WorkflowDefinition::from_toml(toml).unwrap();
        assert_eq!(
            wf.validate(),
            Err(WorkflowError::EntryNotFound("missing".to_string()))
        );
    }

    #[test]
    fn validate_rejects_unknown_edge_target() {
        let toml = r#"
id = "bad"
entry = "a"

[[nodes]]
type = "agent"
id = "a"
agent = "a"

[[edges]]
from = "a"
to = "b"
"#;
        let wf = WorkflowDefinition::from_toml(toml).unwrap();
        assert_eq!(
            wf.validate(),
            Err(WorkflowError::UnknownNode("b".to_string()))
        );
    }

    #[test]
    fn validate_rejects_cycle() {
        let toml = r#"
id = "cycle"
entry = "a"

[[nodes]]
type = "agent"
id = "a"
agent = "a"

[[nodes]]
type = "agent"
id = "b"
agent = "b"

[[edges]]
from = "a"
to = "b"

[[edges]]
from = "b"
to = "a"
"#;
        let wf = WorkflowDefinition::from_toml(toml).unwrap();
        assert_eq!(wf.validate(), Err(WorkflowError::CycleDetected));
    }

    #[test]
    fn validate_rejects_duplicate_node_id() {
        let toml = r#"
id = "dup"
entry = "a"

[[nodes]]
type = "agent"
id = "a"
agent = "a"

[[nodes]]
type = "agent"
id = "a"
agent = "b"
"#;
        let wf = WorkflowDefinition::from_toml(toml).unwrap();
        assert_eq!(
            wf.validate(),
            Err(WorkflowError::DuplicateNodeId("a".to_string()))
        );
    }

    #[test]
    fn agent_node_derives_default_task_kind() {
        let toml = r#"
id = "test"
entry = "architect"

[[nodes]]
type = "agent"
id = "architect"
agent = "architect"
"#;
        let wf = WorkflowDefinition::from_toml(toml).unwrap();
        let node = wf.node("architect").unwrap();
        assert_eq!(node.task_kind(), Some("architecture"));
    }

    #[test]
    fn agent_node_uses_explicit_task_kind() {
        let toml = r#"
id = "test"
entry = "coder"

[[nodes]]
type = "agent"
id = "coder"
agent = "custom_coder"
task_kind = "codegen"
"#;
        let wf = WorkflowDefinition::from_toml(toml).unwrap();
        let node = wf.node("coder").unwrap();
        assert_eq!(node.task_kind(), Some("codegen"));
    }

    #[test]
    fn agent_workflow_task_payload_is_clean_session_with_explicit_artifact() {
        let node = WorkflowNode::Agent {
            id: "coder".to_string(),
            agent: "coder".to_string(),
            task_kind: Some("codegen".to_string()),
            input: "design_artifact".to_string(),
            output: "patch".to_string(),
            timeout_seconds: None,
            retry: WorkflowRetryPolicy::default(),
        };
        let state = serde_json::json!({
            "project_id": "p1",
            "trace_id": "trace-clean",
            "task": "original user goal",
            "design_artifact": {
                "summary": "Use a focused service",
                "files": ["src/lib.rs"]
            },
            "secret_internal_context": "previous agent scratchpad that must not leak",
            "unrelated_previous_output": "do not pass this unless explicitly selected"
        });

        let task = AgentWorkflowNodeExecutor::build_task(&node, &state).unwrap();

        assert_eq!(task.trace_id, "trace-clean");
        assert_eq!(task.assigned_agent.as_deref(), Some("coder"));
        assert_eq!(task.kind, "codegen");
        assert_eq!(task.payload["prompt"], state["design_artifact"]);
        assert_eq!(task.payload["upstream_artifact"], state["design_artifact"]);
        assert_eq!(task.payload["agent_session"]["role"], "coder");
        assert_eq!(
            task.payload["agent_session"]["input_key"],
            "design_artifact"
        );
        assert!(task.payload["agent_session"]["session_id"].is_string());
        assert!(task.payload.get("secret_internal_context").is_none());
        assert!(task.payload.get("unrelated_previous_output").is_none());
        assert!(task.payload.get("task").is_none());
    }

    struct CapturingAgentService {
        seen: Mutex<Option<Task>>,
    }

    #[async_trait::async_trait]
    impl AgentService for CapturingAgentService {
        async fn register(&self, _agent: Arc<dyn crate::services::Agent>) {}

        async fn find(&self, _name: &str) -> Option<Arc<dyn crate::services::Agent>> {
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
            task: &Task,
            _inference: Arc<dyn InferenceService>,
            _tools: Arc<dyn ToolService>,
        ) -> Result<serde_json::Value, crate::services::AgentServiceError> {
            *self.seen.lock().unwrap() = Some(task.clone());
            Ok(serde_json::json!({ "ok": true }))
        }
    }

    struct RoleOnlyLoraRouter;

    #[async_trait::async_trait]
    impl LoraRouter for RoleOnlyLoraRouter {
        async fn resolve(
            &self,
            _task: &Task,
            _project_id: &str,
        ) -> Result<Option<String>, crate::services::LoraRouterError> {
            Ok(None)
        }

        async fn resolve_for_role(
            &self,
            role: AgentRole,
            _project_id: &str,
        ) -> Result<Option<String>, crate::services::LoraRouterError> {
            Ok((role == AgentRole::Coder).then_some("coder-lora-v1".to_string()))
        }
    }

    struct NoopInference;

    #[async_trait::async_trait]
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
            unimplemented!()
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

        fn available_backends(&self) -> Vec<crytex_inference::BackendInfo> {
            vec![]
        }

        async fn list_models(
            &self,
            _backend_id: Option<&str>,
        ) -> Result<Vec<crytex_inference::ModelInfo>, crate::services::InferenceServiceError>
        {
            Ok(vec![])
        }
    }

    struct NoopToolService;

    #[async_trait::async_trait]
    impl ToolService for NoopToolService {
        async fn invoke(
            &self,
            _name: &str,
            _args: serde_json::Value,
        ) -> Result<serde_json::Value, crate::services::ToolServiceError> {
            Ok(serde_json::Value::Null)
        }

        fn list_tools(&self) -> Vec<crate::services::ToolDescription> {
            vec![]
        }
    }

    #[tokio::test]
    async fn agent_workflow_executor_selects_role_lora_for_clean_session() {
        let agent_service = Arc::new(CapturingAgentService {
            seen: Mutex::new(None),
        });
        let executor = AgentWorkflowNodeExecutor::new(
            agent_service.clone(),
            Arc::new(NoopInference),
            Arc::new(NoopToolService),
        )
        .with_lora_router(Arc::new(RoleOnlyLoraRouter));
        let node = WorkflowNode::Agent {
            id: "coder".to_string(),
            agent: "coder".to_string(),
            task_kind: Some("codegen".to_string()),
            input: "design".to_string(),
            output: "patch".to_string(),
            timeout_seconds: None,
            retry: WorkflowRetryPolicy::default(),
        };

        let output = executor
            .execute(
                &node,
                &serde_json::json!({
                    "project_id": "p1",
                    "trace_id": "trace-role-lora",
                    "design": { "artifact": "write minimal patch" },
                    "scratchpad": "must not leak"
                }),
            )
            .await
            .unwrap();

        assert_eq!(output, serde_json::json!({ "ok": true }));
        let task = agent_service.seen.lock().unwrap().clone().unwrap();
        assert_eq!(task.lora_adapter_id.as_deref(), Some("coder-lora-v1"));
        assert_eq!(
            task.payload["upstream_artifact"]["artifact"],
            "write minimal patch"
        );
        assert!(task.payload.get("scratchpad").is_none());
    }

    struct ChainedAgentService {
        seen: Mutex<Vec<Task>>,
    }

    #[async_trait::async_trait]
    impl AgentService for ChainedAgentService {
        async fn register(&self, _agent: Arc<dyn crate::services::Agent>) {}

        async fn find(&self, _name: &str) -> Option<Arc<dyn crate::services::Agent>> {
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
            task: &Task,
            _inference: Arc<dyn InferenceService>,
            _tools: Arc<dyn ToolService>,
        ) -> Result<serde_json::Value, crate::services::AgentServiceError> {
            self.seen.lock().unwrap().push(task.clone());
            Ok(serde_json::json!({
                "agent": task.assigned_agent,
                "artifact": task.payload["upstream_artifact"],
                "session_id": task.payload["agent_session"]["session_id"]
            }))
        }
    }

    #[tokio::test]
    async fn agent_workflow_chain_uses_clean_sessions_and_passes_only_artifacts() {
        let agent_service = Arc::new(ChainedAgentService {
            seen: Mutex::new(Vec::new()),
        });
        let executor = Arc::new(AgentWorkflowNodeExecutor::new(
            agent_service.clone(),
            Arc::new(NoopInference),
            Arc::new(NoopToolService),
        ));
        let engine = WorkflowEngine::new(executor);
        let workflow = WorkflowDefinition {
            id: "agent-chain".to_string(),
            entry: "architect".to_string(),
            max_concurrency: 1,
            nodes: vec![
                WorkflowNode::Agent {
                    id: "architect".to_string(),
                    agent: "architect".to_string(),
                    task_kind: Some("architecture".to_string()),
                    input: "task".to_string(),
                    output: "design_artifact".to_string(),
                    timeout_seconds: None,
                    retry: WorkflowRetryPolicy::default(),
                },
                WorkflowNode::Agent {
                    id: "coder".to_string(),
                    agent: "coder".to_string(),
                    task_kind: Some("codegen".to_string()),
                    input: "design_artifact".to_string(),
                    output: "patch_artifact".to_string(),
                    timeout_seconds: None,
                    retry: WorkflowRetryPolicy::default(),
                },
            ],
            edges: vec![edge("architect", "coder")],
            ..Default::default()
        };

        let result = engine
            .run(
                &workflow,
                serde_json::json!({
                    "project_id": "p1",
                    "trace_id": "trace-chain",
                    "task": "build a token optimizer",
                    "architect_private_scratchpad": "must not leak"
                }),
            )
            .await
            .unwrap();

        let tasks = agent_service.seen.lock().unwrap().clone();
        assert_eq!(tasks.len(), 2);
        assert_ne!(
            tasks[0].payload["agent_session"]["session_id"],
            tasks[1].payload["agent_session"]["session_id"]
        );
        assert_eq!(tasks[0].payload["agent_session"]["trace_id"], "trace-chain");
        assert_eq!(tasks[1].payload["agent_session"]["trace_id"], "trace-chain");
        assert_eq!(
            tasks[1].payload["upstream_artifact"],
            result.state["design_artifact"]
        );
        assert!(
            tasks[1]
                .payload
                .get("architect_private_scratchpad")
                .is_none()
        );
        assert_eq!(
            result.state["patch_artifact"]["artifact"],
            result.state["design_artifact"]
        );
    }

    #[derive(Default)]
    struct RecordingExecutor {
        calls: std::sync::Mutex<Vec<String>>,
    }

    #[async_trait::async_trait]
    impl WorkflowNodeExecutor for RecordingExecutor {
        async fn execute(
            &self,
            node: &WorkflowNode,
            state: &WorkflowState,
        ) -> Result<WorkflowState, WorkflowError> {
            self.calls.lock().unwrap().push(node.id().to_string());
            match node {
                WorkflowNode::Agent { input, .. } => {
                    let input_value = state.get(input).cloned().unwrap_or_default();
                    let input_str = input_value.as_str().unwrap_or("");
                    let value = format!("{}:{}", node.id(), input_str);
                    Ok(serde_json::Value::String(value))
                }
                _ => Err(WorkflowError::Execution(format!(
                    "unexpected node {}",
                    node.id()
                ))),
            }
        }
    }

    fn agent_node(id: &str, input: &str, output: &str) -> WorkflowNode {
        WorkflowNode::Agent {
            id: id.to_string(),
            agent: id.to_string(),
            task_kind: None,
            input: input.to_string(),
            output: output.to_string(),
            timeout_seconds: None,
            retry: WorkflowRetryPolicy::default(),
        }
    }

    fn edge(from: &str, to: &str) -> WorkflowEdge {
        WorkflowEdge {
            from: from.to_string(),
            to: to.to_string(),
        }
    }

    #[tokio::test]
    async fn engine_runs_serial_workflow() {
        let wf = WorkflowDefinition {
            id: "serial".to_string(),
            entry: "a".to_string(),
            max_concurrency: 2,
            nodes: vec![
                agent_node("a", "task", "a_out"),
                agent_node("b", "a_out", "result"),
                WorkflowNode::End {
                    id: "end".to_string(),
                },
            ],
            edges: vec![edge("a", "b"), edge("b", "end")],
            ..Default::default()
        };

        let engine = WorkflowEngine::new(Arc::new(RecordingExecutor::default()));
        let result = engine
            .run(&wf, serde_json::json!({ "task": "hello" }))
            .await
            .unwrap();

        assert_eq!(result.state["result"], "b:a:hello");
        assert!(result.node_results.contains_key("a"));
        assert!(result.node_results.contains_key("b"));
    }

    #[tokio::test]
    async fn engine_runs_parallel_branches() {
        let wf = WorkflowDefinition {
            id: "parallel".to_string(),
            entry: "a".to_string(),
            max_concurrency: 4,
            nodes: vec![
                agent_node("a", "task", "a_out"),
                agent_node("b", "a_out", "b_out"),
                agent_node("c", "a_out", "c_out"),
                WorkflowNode::End {
                    id: "end".to_string(),
                },
            ],
            edges: vec![
                edge("a", "b"),
                edge("a", "c"),
                edge("b", "end"),
                edge("c", "end"),
            ],
            ..Default::default()
        };

        let executor = Arc::new(RecordingExecutor::default());
        let engine = WorkflowEngine::new(executor.clone());
        let result = engine
            .run(&wf, serde_json::json!({ "task": "x" }))
            .await
            .unwrap();

        assert_eq!(result.state["b_out"], "b:a:x");
        assert_eq!(result.state["c_out"], "c:a:x");

        let calls = executor.calls.lock().unwrap().clone();
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0], "a");
        assert!(calls.contains(&"b".to_string()));
        assert!(calls.contains(&"c".to_string()));
    }

    #[tokio::test]
    async fn engine_follows_condition_then_branch() {
        let wf = WorkflowDefinition {
            id: "conditional".to_string(),
            entry: "a".to_string(),
            max_concurrency: 2,
            nodes: vec![
                agent_node("a", "task", "a_out"),
                WorkflowNode::Condition {
                    id: "cond".to_string(),
                    expression: "approved".to_string(),
                    then_branch: "b".to_string(),
                    else_branch: Some("c".to_string()),
                },
                agent_node("b", "a_out", "b_out"),
                agent_node("c", "a_out", "c_out"),
                WorkflowNode::End {
                    id: "end".to_string(),
                },
            ],
            edges: vec![
                edge("a", "cond"),
                edge("cond", "b"),
                edge("cond", "c"),
                edge("b", "end"),
                edge("c", "end"),
            ],
            ..Default::default()
        };

        let executor = Arc::new(RecordingExecutor::default());
        let engine = WorkflowEngine::new(executor.clone());
        let result = engine
            .run(&wf, serde_json::json!({ "task": "x", "approved": true }))
            .await
            .unwrap();

        assert_eq!(result.state["b_out"], "b:a:x");
        assert!(result.state.get("c_out").is_none());

        let calls = executor.calls.lock().unwrap().clone();
        assert!(calls.contains(&"a".to_string()));
        assert!(calls.contains(&"b".to_string()));
        assert!(!calls.contains(&"c".to_string()));
    }

    #[tokio::test]
    async fn memory_repository_loads_inserted_workflow() {
        let repo = MemoryWorkflowRepository::default();
        let wf = WorkflowDefinition {
            id: "demo".to_string(),
            entry: "a".to_string(),
            nodes: vec![agent_node("a", "task", "result")],
            edges: vec![],
            ..Default::default()
        };
        repo.insert(wf.clone());

        let loaded = repo.load("demo").await.unwrap().unwrap();
        assert_eq!(loaded.id, "demo");
    }

    #[tokio::test]
    async fn toml_repository_loads_workflow_from_file() {
        let dir = std::env::temp_dir().join(format!("crytex-wf-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("demo.toml");
        std::fs::write(
            &path,
            r#"
id = "demo"
name = "Demo"
entry = "a"

[[nodes]]
type = "agent"
id = "a"
agent = "a"
"#,
        )
        .unwrap();

        let repo = TomlWorkflowRepository::new(dir.clone());
        let wf = repo.load("demo").await.unwrap().unwrap();
        assert_eq!(wf.id, "demo");
        assert_eq!(wf.nodes.len(), 1);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn toml_repository_returns_none_for_missing_file() {
        let dir = std::env::temp_dir().join(format!("crytex-wf-missing-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let repo = TomlWorkflowRepository::new(dir);
        assert!(repo.load("missing").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn toml_repository_rejects_id_mismatch() {
        let dir = std::env::temp_dir().join(format!("crytex-wf-mismatch-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("demo.toml");
        std::fs::write(
            &path,
            r#"
id = "other"
name = "Other"
entry = "a"

[[nodes]]
type = "agent"
id = "a"
agent = "a"
"#,
        )
        .unwrap();

        let repo = TomlWorkflowRepository::new(dir.clone());
        let err = repo.load("demo").await.unwrap_err();
        assert!(matches!(err, WorkflowError::Internal(_)));

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn engine_follows_condition_else_branch() {
        let wf = WorkflowDefinition {
            id: "conditional".to_string(),
            entry: "a".to_string(),
            max_concurrency: 2,
            nodes: vec![
                agent_node("a", "task", "a_out"),
                WorkflowNode::Condition {
                    id: "cond".to_string(),
                    expression: "approved".to_string(),
                    then_branch: "b".to_string(),
                    else_branch: Some("c".to_string()),
                },
                agent_node("b", "a_out", "b_out"),
                agent_node("c", "a_out", "c_out"),
                WorkflowNode::End {
                    id: "end".to_string(),
                },
            ],
            edges: vec![
                edge("a", "cond"),
                edge("cond", "b"),
                edge("cond", "c"),
                edge("b", "end"),
                edge("c", "end"),
            ],
            ..Default::default()
        };

        let executor = Arc::new(RecordingExecutor::default());
        let engine = WorkflowEngine::new(executor.clone());
        let result = engine
            .run(&wf, serde_json::json!({ "task": "x", "approved": false }))
            .await
            .unwrap();

        assert_eq!(result.state["c_out"], "c:a:x");
        assert!(result.state.get("b_out").is_none());

        let calls = executor.calls.lock().unwrap().clone();
        assert!(calls.contains(&"a".to_string()));
        assert!(calls.contains(&"c".to_string()));
        assert!(!calls.contains(&"b".to_string()));
    }
}
