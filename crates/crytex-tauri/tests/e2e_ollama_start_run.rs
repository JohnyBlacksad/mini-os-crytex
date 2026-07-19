use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use crytex_agents::Agent;
use crytex_agents::coder::CoderAgent;
use crytex_agents::critic::CriticAgent;
use crytex_core::bus::Event;
use crytex_core::config::SecurityConfig;
use crytex_core::models::Task;
use crytex_core::services::{
    AgentServiceError, InferenceService, InferenceServiceImpl, ToolDescription, ToolService,
    ToolServiceError,
};
use crytex_inference::BackendRegistry;
use crytex_inference_ollama::OllamaBackend;
use crytex_sandbox::backends::HostBackend;
use crytex_tauri::app_state::CrytexAppState;
use crytex_tauri::commands::{
    CreateProjectCommand, ExportRunDiagnosticsCommand, InferenceTaskExecutor, PlanDecisionCommand,
    SearchProjectContextCommand, StartRunCommand, SubmitGoalCommand, SubmitTaskCommand,
    TaskExecutor, TaskReviewDecisionCommand, TauriCommandError,
};
use crytex_tools::{Capability, ToolServiceImpl, TypedToolRegistry};
use serde_json::Value;
use serde_json::json;
use tokio::sync::broadcast;

const DEFAULT_OLLAMA_URL: &str = "http://127.0.0.1:11434";
const DEFAULT_E2E_MODEL: &str = "qwen3.5:9b";

fn drain_events(rx: &mut broadcast::Receiver<Event>) -> Vec<Event> {
    let mut events = Vec::new();
    loop {
        match rx.try_recv() {
            Ok(event) => events.push(event),
            Err(broadcast::error::TryRecvError::Empty) => break,
            Err(broadcast::error::TryRecvError::Lagged(_)) => continue,
            Err(broadcast::error::TryRecvError::Closed) => break,
        }
    }
    events
}

#[tokio::test]
async fn real_runtime_smoke_reports_production_agent_chain() {
    let expected_model =
        std::env::var("CRYTEX_E2E_OLLAMA_MODEL").unwrap_or_else(|_| DEFAULT_E2E_MODEL.to_string());
    let report = run_real_runtime_smoke_report().await;

    assert_eq!(report.model, expected_model);
    assert_eq!(report.review_agent.as_deref(), Some("critic"));
    assert!(report.completed_agents.contains(&"architect".to_string()));
    assert!(report.completed_agents.contains(&"coder".to_string()));
    assert!(report.completed_agents.contains(&"qa".to_string()));
    assert!(report.completed_agents.contains(&"security".to_string()));
    assert!(report.traced_actions.contains(&"llm_request".to_string()));
    assert!(
        report
            .live_run_actions
            .contains(&"human_review_ready".to_string())
    );
    assert!(report.human_reward_recorded);

    println!(
        "CRYTEX_REAL_RUNTIME_SMOKE_REPORT {}",
        serde_json::to_string_pretty(&report).unwrap()
    );
}

#[derive(Debug, serde::Serialize)]
struct RealRuntimeSmokeReport {
    model: String,
    project_id: String,
    goal_id: String,
    run_id: String,
    review_task_id: String,
    review_agent: Option<String>,
    completed_agents: Vec<String>,
    traced_actions: Vec<String>,
    live_run_actions: Vec<String>,
    rag_context_sent_to_model: bool,
    human_reward_recorded: bool,
}

async fn run_real_runtime_smoke_report() -> RealRuntimeSmokeReport {
    let ollama_url =
        std::env::var("CRYTEX_E2E_OLLAMA_URL").unwrap_or_else(|_| DEFAULT_OLLAMA_URL.to_string());
    let model =
        std::env::var("CRYTEX_E2E_OLLAMA_MODEL").unwrap_or_else(|_| DEFAULT_E2E_MODEL.to_string());
    ensure_model_available(&ollama_url, &model).await;

    let temp_dir = tempfile::tempdir().expect("temp dir should be created");
    let project_root = temp_dir.path().join("project-real-runtime-smoke");
    std::fs::create_dir_all(project_root.join("docs")).expect("docs dir should be created");
    std::fs::write(
        project_root.join("docs/payment-retry.md"),
        "# Payment Retry Smoke Path\n\nREAL_RUNTIME_SMOKE_RAG_MARKER documents the payment retry requirements for the real runtime smoke test.\n",
    )
    .expect("RAG document should be written");

    let db_path = temp_dir.path().join("crytex-real-runtime-smoke.db");
    let state =
        CrytexAppState::new_sqlite_with_ollama_agent_executor(&db_path, ollama_url, model.clone())
            .await
            .expect("state should initialize with production Ollama agent executor");
    let project = state
        .create_project(CreateProjectCommand {
            name: "Real Runtime Smoke".into(),
            root_path: project_root.display().to_string(),
        })
        .await
        .expect("project should be created and indexed");

    let plan = state
        .submit_goal(SubmitGoalCommand {
            project_id: project.id.clone(),
            goal: "Use the payment retry smoke context, make the coder create docs/real-runtime-smoke-report.md with fs_write, and run the generated agent chain to critic human review. The coder final JSON must include files_changed with docs/real-runtime-smoke-report.md.".into(),
            context: json!({
                "smoke": "real_runtime_smoke_reports_production_agent_chain",
                "expected": "orchestrator creates tasks, coder writes docs/real-runtime-smoke-report.md, agents execute, critic gates, human approval records reward"
            }),
            trace_id: Some("trace-real-runtime-smoke".into()),
        })
        .await
        .expect("goal should be planned by orchestrator");
    assert_eq!(plan.goal.status.as_str(), "review");
    assert_eq!(plan.generated_tasks.len(), 5);

    state
        .approve_plan(PlanDecisionCommand {
            goal_task_id: plan.goal.id.clone(),
            comment: Some("approve real runtime smoke".into()),
        })
        .await
        .expect("plan should be approved");

    let mut live_events = state
        .subscribe_to_events()
        .await
        .expect("smoke should subscribe to live backend events");
    let run = state
        .start_run(StartRunCommand {
            project_id: project.id.clone(),
            max_steps: 10,
        })
        .await
        .expect("real runtime smoke should execute generated chain");

    let run_diagnostics = state
        .export_run_diagnostics(ExportRunDiagnosticsCommand {
            project_id: project.id.clone(),
            run_id: run.run_id.clone(),
            trace_id: Some("trace-real-runtime-smoke".into()),
        })
        .await
        .expect("real runtime smoke diagnostics should export");
    assert_eq!(
        run.review_tasks.len(),
        1,
        "artifact handoff rejections: {:?}",
        run_diagnostics.artifact_handoff_rejections
    );
    assert!(run.remaining_ready_tasks.is_empty());
    let review_task = run.review_tasks[0].clone();
    assert_eq!(review_task.assigned_agent.as_deref(), Some("critic"));
    assert_eq!(review_task.status.as_str(), "review");
    assert_eq!(
        review_task.result.as_ref().unwrap()["source"],
        "agent_service"
    );

    let live_events = drain_events(&mut live_events);
    let live_run_actions = live_events
        .iter()
        .filter_map(|event| match event {
            Event::RunObserved {
                project_id,
                trace_id,
                action,
                ..
            } if project_id == &project.id && trace_id == "trace-real-runtime-smoke" => {
                Some(action.clone())
            }
            _ => None,
        })
        .collect::<Vec<_>>();

    let exported = state
        .get_project_state(&project.id)
        .await
        .expect("project state should export smoke evidence");

    let completed_agents = ["architect", "coder", "qa", "security"]
        .iter()
        .filter_map(|agent| {
            exported
                .tasks
                .iter()
                .find(|task| {
                    task.parent_id.as_deref() == Some(plan.goal.id.as_str())
                        && task.assigned_agent.as_deref() == Some(*agent)
                })
                .map(|task| {
                    assert_eq!(task.status.as_str(), "completed", "{agent} should complete");
                    assert_eq!(task.result.as_ref().unwrap()["source"], "agent_service");
                    (*agent).to_string()
                })
        })
        .collect::<Vec<_>>();

    let traced_logs = exported
        .recent_logs
        .iter()
        .filter(|log| {
            log.metadata
                .get("trace_id")
                .and_then(|trace| trace.as_str())
                == Some("trace-real-runtime-smoke")
        })
        .collect::<Vec<_>>();
    let traced_actions = traced_logs
        .iter()
        .map(|log| log.action.clone())
        .collect::<Vec<_>>();
    let rag_context_sent_to_model = traced_logs.iter().any(|log| {
        log.action == "llm_request"
            && log.metadata.get("messages").is_some_and(|messages| {
                messages
                    .to_string()
                    .contains("REAL_RUNTIME_SMOKE_RAG_MARKER")
            })
    });

    let approved = state
        .approve_task_review(TaskReviewDecisionCommand {
            task_id: review_task.id.clone(),
            comment: Some("real runtime smoke accepted".into()),
        })
        .await
        .expect("human approval should complete review gate");
    assert_eq!(approved.task.status.as_str(), "completed");
    assert_eq!(approved.task.human_score, Some(1.0));

    let exported_after_approval = state
        .get_project_state(&project.id)
        .await
        .expect("project state should export approval evidence");
    let human_reward_recorded = exported_after_approval.recent_logs.iter().any(|log| {
        log.action == "human_review_approved"
            && log.task_id.as_deref() == Some(review_task.id.as_str())
            && log.metadata["human_score"] == 1.0
            && log.metadata["reward"].as_f64().is_some()
    });

    let diagnostics = state
        .export_run_diagnostics(ExportRunDiagnosticsCommand {
            project_id: project.id.clone(),
            run_id: run.run_id.clone(),
            trace_id: Some("trace-real-runtime-smoke".into()),
        })
        .await
        .expect("run diagnostics should export smoke evidence");
    assert_eq!(diagnostics.project_id, project.id);
    assert_eq!(diagnostics.run_id, run.run_id);
    assert_eq!(
        diagnostics.runtime.active_backend.as_deref(),
        Some("ollama")
    );
    assert_eq!(
        diagnostics.runtime.active_model.as_deref(),
        Some(model.as_str())
    );
    assert_eq!(
        diagnostics.trace_ids,
        vec!["trace-real-runtime-smoke".to_string()]
    );
    assert!(diagnostics.review_task_ids.contains(&review_task.id));
    assert!(
        diagnostics
            .tasks
            .iter()
            .any(|task| task.agent.as_deref() == Some("critic"))
    );
    assert!(diagnostics.rag_context_sent_to_model);
    assert!(diagnostics.human_reward_recorded);
    assert!(
        diagnostics
            .events
            .iter()
            .any(|event| event.action == "human_review_approved")
    );

    state.shutdown_project_watchers().await;

    RealRuntimeSmokeReport {
        model,
        project_id: project.id,
        goal_id: plan.goal.id,
        run_id: run.run_id,
        review_task_id: review_task.id,
        review_agent: review_task.assigned_agent,
        completed_agents,
        traced_actions,
        live_run_actions,
        rag_context_sent_to_model,
        human_reward_recorded,
    }
}

#[tokio::test]
async fn start_run_executes_ready_task_with_real_ollama_model() {
    let ollama_url =
        std::env::var("CRYTEX_E2E_OLLAMA_URL").unwrap_or_else(|_| DEFAULT_OLLAMA_URL.to_string());
    let model =
        std::env::var("CRYTEX_E2E_OLLAMA_MODEL").unwrap_or_else(|_| DEFAULT_E2E_MODEL.to_string());
    ensure_model_available(&ollama_url, &model).await;

    let backend = Arc::new(OllamaBackend::new(&ollama_url, &model));
    let executor = Arc::new(InferenceTaskExecutor::new(backend, model.clone()));
    let dir = tempfile::tempdir().expect("temp dir should be created");
    let db_path = dir.path().join("crytex-tauri-e2e.db");
    let state = CrytexAppState::new_sqlite_with_executor(&db_path, executor)
        .await
        .expect("state should initialize");

    let project_root = dir.path().join("project");
    std::fs::create_dir_all(&project_root).expect("project root should be created");
    let project = state
        .create_project(CreateProjectCommand {
            name: "Ollama E2E".into(),
            root_path: project_root.display().to_string(),
        })
        .await
        .expect("project should be created");

    let task = state
        .submit_task(SubmitTaskCommand {
            project_id: project.id.clone(),
            parent_id: None,
            title: "Reply with one short Crytex smoke result sentence".into(),
            description: Some("Prove that Tauri start_run called the local model.".into()),
            kind: "codegen".into(),
            assigned_agent: Some("coder".into()),
            priority: 10,
            payload: json!({
                "prompt": "Return exactly one short sentence. Include the word Crytex."
            }),
            trace_id: Some("trace-tauri-ollama-e2e".into()),
        })
        .await
        .expect("task should be submitted");

    let run = state
        .start_run(StartRunCommand {
            project_id: project.id.clone(),
            max_steps: 1,
        })
        .await
        .expect("run should execute through Ollama");

    assert_eq!(run.review_tasks.len(), 1);
    assert_eq!(run.review_tasks[0].id, task.id);
    assert_eq!(run.review_tasks[0].status.as_str(), "review");
    assert_eq!(
        run.review_tasks[0].result.as_ref().unwrap()["source"],
        "ollama_inference"
    );
    assert_eq!(
        run.review_tasks[0].result.as_ref().unwrap()["contract_repair"]["strategy"],
        "raw_content_to_typed_agent_result"
    );
    assert!(
        !run.review_tasks[0].result.as_ref().unwrap()["agent_result"]["summary"]
            .as_str()
            .unwrap_or_default()
            .trim()
            .is_empty(),
        "typed model artifact summary must be non-empty"
    );
    assert!(
        !run.review_tasks[0].result.as_ref().unwrap()["agent_result"]["evidence"]["content"]
            .as_str()
            .unwrap_or_default()
            .trim()
            .is_empty(),
        "typed model artifact must preserve raw model evidence"
    );

    println!(
        "CRYTEX_TAURI_E2E_RESULT model={model} task_id={} summary={}",
        task.id,
        run.review_tasks[0].result.as_ref().unwrap()["agent_result"]["summary"]
            .as_str()
            .unwrap_or_default()
            .trim()
    );

    state.shutdown_project_watchers().await;
}

#[tokio::test]
async fn goal_plan_approval_starts_first_generated_task_with_real_ollama_model() {
    let ollama_url =
        std::env::var("CRYTEX_E2E_OLLAMA_URL").unwrap_or_else(|_| DEFAULT_OLLAMA_URL.to_string());
    let model =
        std::env::var("CRYTEX_E2E_OLLAMA_MODEL").unwrap_or_else(|_| DEFAULT_E2E_MODEL.to_string());
    ensure_model_available(&ollama_url, &model).await;

    let temp_dir = tempfile::tempdir().expect("temp dir should be created");
    let db_path = temp_dir.path().join("crytex-tauri-goal-agent-e2e.db");
    let state =
        CrytexAppState::new_sqlite_with_ollama_agent_executor(&db_path, ollama_url, model.clone())
            .await
            .expect("state should initialize with Ollama agent executor");

    let project_root = temp_dir.path().join("project-goal");
    std::fs::create_dir_all(project_root.join("docs")).expect("project docs should be created");
    let project = state
        .create_project(CreateProjectCommand {
            name: "Goal Ollama E2E".into(),
            root_path: project_root.display().to_string(),
        })
        .await
        .expect("project should be created");

    let plan = state
        .submit_goal(SubmitGoalCommand {
            project_id: project.id.clone(),
            goal: "Prove the goal-first Crytex run path. The coder final JSON must include files_changed with docs/goal-agent-report.md and summary containing CRYTEX_GOAL_AGENT_REPORT_OK. Run the generated chain to critic human review.".into(),
            context: json!({
                "e2e": "goal_plan_approval_start_run",
                "expected_behavior": "generated task result goes to human review",
                "required_file": "docs/goal-agent-report.md",
                "required_marker": "CRYTEX_GOAL_AGENT_REPORT_OK"
            }),
            trace_id: Some("trace-goal-ollama-e2e".into()),
        })
        .await
        .expect("goal should be planned");

    assert_eq!(plan.goal.status.as_str(), "review");
    assert_eq!(plan.generated_tasks.len(), 5);
    assert!(
        plan.generated_tasks
            .iter()
            .all(|task| task.status.as_str() == "backlog")
    );

    let approved = state
        .approve_plan(PlanDecisionCommand {
            goal_task_id: plan.goal.id.clone(),
            comment: Some("approved for real Ollama e2e".into()),
        })
        .await
        .expect("plan should be approved");

    assert_eq!(approved.goal.status.as_str(), "completed");
    assert!(
        approved
            .generated_tasks
            .iter()
            .all(|task| task.status.as_str() == "pending")
    );

    let run = state
        .start_run(StartRunCommand {
            project_id: project.id.clone(),
            max_steps: 10,
        })
        .await
        .expect("run should execute generated chain through Ollama");

    let exported_after_run = state
        .get_project_state(&project.id)
        .await
        .expect("project state should export generated chain after run");
    if run.review_tasks.len() != 1 {
        let task_statuses = exported_after_run
            .tasks
            .iter()
            .map(|task| {
                json!({
                    "id": task.id,
                    "parent_id": task.parent_id,
                    "agent": task.assigned_agent,
                    "kind": task.kind,
                    "status": task.status.as_str(),
                    "source": task.result.as_ref().and_then(|result| result["source"].as_str()),
                    "priority_score": task.priority_score,
                    "payload_source": task.payload["source"],
                })
            })
            .collect::<Vec<_>>();
        panic!(
            "expected exactly one review task, got {}. remaining_ready_tasks={}, tasks={}",
            run.review_tasks.len(),
            run.remaining_ready_tasks.len(),
            serde_json::to_string_pretty(&task_statuses).unwrap()
        );
    }
    assert_eq!(run.review_tasks.len(), 1);
    assert_eq!(run.review_tasks[0].status.as_str(), "review");
    assert_eq!(
        run.review_tasks[0].assigned_agent.as_deref(),
        Some("critic")
    );
    assert_eq!(
        run.review_tasks[0].result.as_ref().unwrap()["source"],
        "agent_service"
    );
    let review_agent_result = &run.review_tasks[0].result.as_ref().unwrap()["agent_result"];
    assert!(
        !review_agent_result["review_decision"]
            .as_str()
            .or_else(|| review_agent_result["summary"].as_str())
            .unwrap_or_default()
            .trim()
            .is_empty(),
        "critic agent result must include a non-empty review decision or summary"
    );
    assert!(
        run.remaining_ready_tasks.is_empty(),
        "the run should stop at the critic human-review gate"
    );

    for agent in ["architect", "coder", "qa", "security"] {
        let task = exported_after_run
            .tasks
            .iter()
            .find(|task| {
                task.parent_id.as_deref() == Some(plan.goal.id.as_str())
                    && task.assigned_agent.as_deref() == Some(agent)
            })
            .unwrap_or_else(|| panic!("{agent} generated task should exist"));
        assert_eq!(
            task.status.as_str(),
            "completed",
            "{agent} should auto-complete before critic review"
        );
        assert_eq!(
            task.result.as_ref().unwrap()["source"],
            "agent_service",
            "{agent} should have real Ollama output"
        );
    }

    let review_decision = state
        .approve_task_review(TaskReviewDecisionCommand {
            task_id: run.review_tasks[0].id.clone(),
            comment: Some("critic result accepted by e2e".into()),
        })
        .await
        .expect("reviewed task should be approved");
    assert_eq!(review_decision.task.status.as_str(), "completed");
    assert!(review_decision.ready_tasks.is_empty());

    println!(
        "CRYTEX_TAURI_GOAL_E2E_RESULT model={} goal_id={} critic_task_id={} critic_result={}",
        model, plan.goal.id, run.review_tasks[0].id, review_agent_result
    );

    state.shutdown_project_watchers().await;
}

#[tokio::test]
async fn agent_executor_uses_real_ollama_tool_loop_to_read_requirements_write_report_and_audit_trace()
 {
    let ollama_url =
        std::env::var("CRYTEX_E2E_OLLAMA_URL").unwrap_or_else(|_| DEFAULT_OLLAMA_URL.to_string());
    let model =
        std::env::var("CRYTEX_E2E_OLLAMA_MODEL").unwrap_or_else(|_| DEFAULT_E2E_MODEL.to_string());
    ensure_model_available(&ollama_url, &model).await;

    let dir = tempfile::tempdir().expect("temp dir should be created");
    let db_path = dir.path().join("crytex-tauri-agent-e2e.db");
    let state = CrytexAppState::new_sqlite_with_ollama_agent_executor(
        &db_path,
        ollama_url.clone(),
        model.clone(),
    )
    .await
    .expect("state should initialize with Ollama agent executor");

    let project_root = dir.path().join("project-agent");
    std::fs::create_dir_all(&project_root).expect("project root should be created");
    std::fs::write(
        project_root.join("requirements.md"),
        "# Crytex Mini Task\n\n- Verify RAG readiness\n- Verify LoRA evolution readiness\n- Verify Observe trace visibility\n",
    )
    .expect("requirements file should be created");
    let project = state
        .create_project(CreateProjectCommand {
            name: "Agent Tool Loop Ollama E2E".into(),
            root_path: project_root.display().to_string(),
        })
        .await
        .expect("project should be created");

    let task = state
        .submit_task(SubmitTaskCommand {
            project_id: project.id.clone(),
            parent_id: None,
            title: "Read requirements and create Crytex agent report".into(),
            description: Some(
                "Use the real agent tool loop to inspect an input file and write a report.".into(),
            ),
            kind: "codegen".into(),
            assigned_agent: Some("coder".into()),
            priority: 10,
            payload: json!({
                "system_prompt_override": r###"You are a deterministic Crytex agent E2E tester.
Use the registered tools. First respond only with this JSON tool call:
{"tool":"fs_read","args":{"path":"requirements.md"}}
After receiving the fs_read observation, respond only with this JSON tool call:
{"tool":"fs_write","args":{"path":"crytex-agent-report.md","content":"# Crytex Agent Report\n\n## Summary\nRead requirements.md and produced a concrete execution report.\n\n## Checked Requirements\n- RAG readiness\n- LoRA evolution readiness\n- Observe trace visibility\n\n## Result\nCRYTEX_AGENT_REPORT_OK"}}
After receiving the fs_write observation, respond only with a final JSON object containing files_changed, test_results, and summary."###,
                "prompt": "Read requirements.md, then create crytex-agent-report.md summarizing the requirements. Do not merely describe the work."
            }),
            trace_id: Some("trace-agent-tool-loop-ollama-e2e".into()),
        })
        .await
        .expect("task should be submitted");

    let run = state
        .start_run(StartRunCommand {
            project_id: project.id.clone(),
            max_steps: 1,
        })
        .await
        .expect("run should execute through agent service and Ollama");

    let report_file = project_root.join("crytex-agent-report.md");
    let report_content = std::fs::read_to_string(&report_file)
        .expect("agent should create report file via fs_write");

    assert_eq!(run.review_tasks.len(), 1);
    assert_eq!(run.review_tasks[0].id, task.id);
    assert_eq!(run.review_tasks[0].status.as_str(), "review");
    assert_eq!(
        run.review_tasks[0].result.as_ref().unwrap()["source"],
        "agent_service"
    );
    assert!(report_content.contains("CRYTEX_AGENT_REPORT_OK"));
    assert!(report_content.contains("RAG readiness"));
    assert!(report_content.contains("LoRA evolution readiness"));
    assert!(report_content.contains("Observe trace visibility"));
    assert!(
        run.review_tasks[0].result.as_ref().unwrap()["agent_result"]["files_changed"]
            .as_array()
            .is_some_and(|files| files.iter().any(|file| {
                file.get("path").and_then(|path| path.as_str()) == Some("crytex-agent-report.md")
            })),
        "agent result should record the fs_write file"
    );

    let exported = state
        .get_project_state(&project.id)
        .await
        .expect("project state should export recent logs");
    let traced_actions = exported
        .recent_logs
        .iter()
        .filter(|log| {
            log.metadata
                .get("trace_id")
                .and_then(|trace| trace.as_str())
                == Some("trace-agent-tool-loop-ollama-e2e")
        })
        .map(|log| log.action.as_str())
        .collect::<Vec<_>>();
    assert!(
        traced_actions.contains(&"llm_request"),
        "trace should include LLM requests, got {traced_actions:?}"
    );
    assert!(
        traced_actions.contains(&"llm_response"),
        "trace should include LLM responses, got {traced_actions:?}"
    );
    assert!(
        traced_actions.contains(&"tool_call"),
        "trace should include tool calls, got {traced_actions:?}"
    );
    assert!(
        traced_actions.contains(&"tool_result"),
        "trace should include tool results, got {traced_actions:?}"
    );
    assert!(
        exported.recent_logs.iter().any(|log| {
            log.action == "tool_call"
                && log.metadata["tool_name"] == "fs_read"
                && log.metadata["trace_id"] == "trace-agent-tool-loop-ollama-e2e"
        }),
        "trace should prove that the agent read requirements.md"
    );
    assert!(
        exported.recent_logs.iter().any(|log| {
            log.action == "tool_call"
                && log.metadata["tool_name"] == "fs_write"
                && log.metadata["trace_id"] == "trace-agent-tool-loop-ollama-e2e"
        }),
        "trace should prove that the agent wrote the report"
    );

    println!(
        "CRYTEX_TAURI_AGENT_TOOL_E2E_RESULT model={model} task_id={} file={} content={} traced_actions={:?}",
        task.id,
        report_file.display(),
        report_content,
        traced_actions
    );

    state.shutdown_project_watchers().await;
}

#[tokio::test]
async fn real_ollama_agent_run_receives_indexed_rag_context() {
    let ollama_url =
        std::env::var("CRYTEX_E2E_OLLAMA_URL").unwrap_or_else(|_| DEFAULT_OLLAMA_URL.to_string());
    let model =
        std::env::var("CRYTEX_E2E_OLLAMA_MODEL").unwrap_or_else(|_| DEFAULT_E2E_MODEL.to_string());
    ensure_model_available(&ollama_url, &model).await;

    let dir = tempfile::tempdir().expect("temp dir should be created");
    let db_path = dir.path().join("crytex-tauri-real-rag-e2e.db");
    let state = CrytexAppState::new_sqlite_with_ollama_agent_executor(
        &db_path,
        ollama_url.clone(),
        model.clone(),
    )
    .await
    .expect("state should initialize with Ollama agent executor");

    let project_root = dir.path().join("project-real-rag");
    std::fs::create_dir_all(project_root.join("docs")).expect("docs dir should be created");
    std::fs::write(
        project_root.join("docs/payment-retry.md"),
        "# Payment Retry Adapter\n\nRAG_REAL_OLLAMA_CONTEXT_MARKER is the required verification phrase for the agent response.\n",
    )
    .expect("rag document should be written");

    let project = state
        .create_project(CreateProjectCommand {
            name: "Real Ollama RAG E2E".into(),
            root_path: project_root.display().to_string(),
        })
        .await
        .expect("project should be created and indexed");

    let task = state
        .submit_task(SubmitTaskCommand {
            project_id: project.id.clone(),
            parent_id: None,
            title: "Implement payment retry adapter".into(),
            description: Some("Use relevant project context before answering.".into()),
            kind: "codegen".into(),
            assigned_agent: Some("coder".into()),
            priority: 10,
            payload: json!({
                "backend": "ollama",
                "model": model,
                "system_prompt_override": "You are a deterministic Crytex RAG E2E agent. Return exactly one valid JSON object and no markdown. Read the Relevant Context section in the user message. Extract the single uppercase verification token from that context and put it in the summary field. If no verification token exists in Relevant Context, use MISSING_RAG_CONTEXT. Do not call tools.",
                "prompt": "Answer using only the relevant project context. The task text intentionally omits the verification phrase. Return {\"files_changed\":[\"rag://retrieved-context\"],\"test_results\":\"not run in RAG e2e\",\"summary\":\"<verification token from Relevant Context>\"}."
            }),
            trace_id: Some("trace-real-ollama-rag-e2e".into()),
        })
        .await
        .expect("task should be submitted");

    let retrieved = state
        .search_project_context(SearchProjectContextCommand {
            project_id: project.id.clone(),
            query: "payment retry adapter verification phrase".to_string(),
            limit: 5,
        })
        .await
        .expect("indexed RAG context should be searchable before agent execution");
    assert!(
        retrieved.hits.iter().any(|hit| hit
            .text
            .as_deref()
            .is_some_and(|text| text.contains("RAG_REAL_OLLAMA_CONTEXT_MARKER"))),
        "search_project_context should retrieve indexed marker before agent execution, got {:?}",
        retrieved.hits
    );

    let run = state
        .start_run(StartRunCommand {
            project_id: project.id.clone(),
            max_steps: 1,
        })
        .await
        .expect("run should execute through real Ollama agent");

    assert_eq!(run.review_tasks.len(), 1);
    assert_eq!(run.review_tasks[0].id, task.id);
    assert_eq!(run.review_tasks[0].status.as_str(), "review");
    assert_eq!(
        run.review_tasks[0].result.as_ref().unwrap()["source"],
        "agent_service"
    );

    let exported = state
        .get_project_state(&project.id)
        .await
        .expect("project state should export LLM audit trace");
    let rag_request = exported.recent_logs.iter().find(|log| {
        log.action == "llm_request"
            && log.metadata["trace_id"] == "trace-real-ollama-rag-e2e"
            && log.metadata["messages"]
                .to_string()
                .contains("RAG_REAL_OLLAMA_CONTEXT_MARKER")
    });
    assert!(
        rag_request.is_some(),
        "llm_request trace should contain indexed RAG marker in sent messages"
    );
    let diagnostics = state
        .export_run_diagnostics(ExportRunDiagnosticsCommand {
            project_id: project.id.clone(),
            run_id: run.run_id.clone(),
            trace_id: Some("trace-real-ollama-rag-e2e".into()),
        })
        .await
        .expect("run diagnostics should export RAG evidence");
    let rag_evidence = diagnostics
        .events
        .iter()
        .find(|event| event.action == "rag_context_assembled")
        .expect("diagnostics should include RAG context evidence event");
    assert_eq!(
        rag_evidence.metadata["trace_id"],
        "trace-real-ollama-rag-e2e"
    );
    assert_eq!(rag_evidence.metadata["rerank_applied"], false);
    assert!(
        rag_evidence.metadata["chunks"]
            .as_array()
            .is_some_and(|chunks| chunks.iter().any(|chunk| {
                chunk["relative_path"] == "docs/payment-retry.md"
                    && chunk["text_preview"]
                        .as_str()
                        .is_some_and(|text| text.contains("RAG_REAL_OLLAMA_CONTEXT_MARKER"))
                    && chunk["score"].as_f64().is_some_and(|score| score > 0.0)
            })),
        "diagnostics should expose retrieved RAG chunk evidence, got {:?}",
        rag_evidence.metadata
    );

    let summary = run.review_tasks[0].result.as_ref().unwrap()["agent_result"]["summary"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    state.shutdown_project_watchers().await;

    assert_eq!(
        summary, "RAG_REAL_OLLAMA_CONTEXT_MARKER",
        "real model output should prove it saw the indexed RAG context"
    );

    println!(
        "CRYTEX_TAURI_REAL_RAG_E2E_RESULT model={} task_id={} summary={} rag_trace_seen={} rag_evidence_chunks={}",
        model,
        task.id,
        summary,
        rag_request.is_some(),
        rag_evidence.metadata["chunks"]
            .as_array()
            .map_or(0, Vec::len)
    );
}

#[tokio::test]
async fn critic_rejection_from_real_ollama_creates_remediation_chain() {
    let state = ollama_critic_state().await;
    let project_root = state.temp_dir.path().join("project-critic-remediation");
    std::fs::create_dir_all(&project_root).expect("project root should be created");
    let project = state
        .app
        .create_project(CreateProjectCommand {
            name: "Critic Remediation Ollama E2E".into(),
            root_path: project_root.display().to_string(),
        })
        .await
        .expect("project should be created");

    let reviewer = state
        .app
        .submit_task(SubmitTaskCommand {
            project_id: project.id.clone(),
            parent_id: None,
            title: "Review intentionally incomplete Crytex artifact".into(),
            description: Some("Force the real critic model to reject with structured feedback.".into()),
            kind: "review".into(),
            assigned_agent: Some("critic".into()),
            priority: 20,
            payload: json!({
                "model": state.model,
                "backend": "ollama",
                "prompt": "CRYTEX_FORCE_REJECT: Review this artifact and reject it with target_task_id \"manual-coder-task\".",
                "parent_result": {
                    "task_id": "manual-coder-task",
                    "agent": "coder",
                    "summary": "The implementation intentionally omits the required Crytex report and contains no tests."
                },
                "system_prompt_override": r#"You are a deterministic Crytex critic E2E reviewer.
Return exactly one valid JSON object and no markdown.
If the user task contains CRYTEX_FORCE_REJECT, return:
{"score":1.0,"review_decision":"reject","target_task_id":"manual-coder-task","failure_type":"missing_requirement","blocking_issues":[{"severity":"high","reason":"Required report is missing","evidence":"The reviewed artifact says it omits the required Crytex report.","expected":"Create the Crytex report and add verification evidence."}],"feedback":"Create the missing Crytex report and add verification evidence.","comments":["missing required report"]}
Otherwise return:
{"score":4.5,"review_decision":"pass","target_task_id":null,"failure_type":null,"blocking_issues":[],"feedback":"Remediation is acceptable.","comments":[]}"#
            }),
            trace_id: Some("trace-real-critic-remediation-e2e".into()),
        })
        .await
        .expect("review task should be submitted");

    let run = state
        .app
        .start_run(StartRunCommand {
            project_id: project.id.clone(),
            max_steps: 8,
        })
        .await
        .expect("run should execute real critic and remediation chain");

    let exported = state
        .app
        .get_project_state(&project.id)
        .await
        .expect("project state should export remediation chain");
    let task_diagnostics = exported
        .tasks
        .iter()
        .map(|task| {
            json!({
                "id": task.id,
                "parent_id": task.parent_id,
                "kind": task.kind,
                "agent": task.assigned_agent,
                "status": task.status.as_str(),
                "source": task.payload.get("source"),
                "review_decision": task
                    .result
                    .as_ref()
                    .and_then(|result| result.pointer("/agent_result/review_decision"))
                    .or_else(|| task
                        .result
                        .as_ref()
                        .and_then(|result| result.pointer("/agent_result/agent_result/review_decision")))
                    .or_else(|| task.result.as_ref().and_then(|result| result.pointer("/review_decision"))),
            })
        })
        .collect::<Vec<_>>();
    if run.review_tasks.len() != 1 {
        state.app.shutdown_project_watchers().await;
    }
    assert_eq!(
        run.review_tasks.len(),
        1,
        "expected final human review after remediation; remaining_ready={}; tasks={}",
        run.remaining_ready_tasks.len(),
        serde_json::to_string_pretty(&task_diagnostics).unwrap()
    );
    assert_eq!(
        run.review_tasks[0].assigned_agent.as_deref(),
        Some("critic")
    );
    assert_eq!(
        run.review_tasks[0].result.as_ref().unwrap()["agent_result"]["review_decision"],
        "pass"
    );

    let original_reviewer = exported
        .tasks
        .iter()
        .find(|task| task.id == reviewer.id)
        .expect("original reviewer should exist");
    assert_eq!(original_reviewer.status.as_str(), "failed");
    assert_eq!(
        original_reviewer.result.as_ref().unwrap()["agent_result"]["review_decision"],
        "reject"
    );
    assert_eq!(
        original_reviewer.result.as_ref().unwrap()["agent_result"]["feedback"],
        "Create the missing Crytex report and add verification evidence."
    );

    let remediation_parent = exported
        .tasks
        .iter()
        .find(|task| {
            task.kind == "debug"
                && task.payload["source"] == "reviewer_rejection"
                && task.payload["reviewer_task_id"] == reviewer.id
        })
        .expect("remediation parent should be created");
    assert_eq!(remediation_parent.status.as_str(), "completed");
    assert_eq!(
        remediation_parent.payload["critic_report"]["failure_type"],
        "missing_requirement"
    );

    let remediation_tasks = exported
        .tasks
        .iter()
        .filter(|task| task.parent_id.as_deref() == Some(remediation_parent.id.as_str()))
        .collect::<Vec<_>>();
    assert_eq!(remediation_tasks.len(), 4);
    assert!(remediation_tasks.iter().any(|task| {
        task.kind == "debug"
            && task.assigned_agent.as_deref() == Some("coder")
            && task.status.as_str() == "completed"
    }));
    assert!(remediation_tasks.iter().any(|task| {
        task.kind == "codegen"
            && task.assigned_agent.as_deref() == Some("coder")
            && task.status.as_str() == "completed"
    }));
    assert!(remediation_tasks.iter().any(|task| {
        task.kind == "qa"
            && task.assigned_agent.as_deref() == Some("qa")
            && task.status.as_str() == "completed"
    }));

    let traced_logs = exported
        .recent_logs
        .iter()
        .filter(|log| {
            log.metadata
                .get("trace_id")
                .and_then(|trace| trace.as_str())
                == Some("trace-real-critic-remediation-e2e")
        })
        .collect::<Vec<_>>();
    let traced_actions = traced_logs
        .iter()
        .map(|log| log.action.as_str())
        .collect::<Vec<_>>();
    for action in [
        "run_started",
        "task_execution_started",
        "task_execution_finished",
        "critic_rejected",
        "remediation_plan_created",
        "human_review_ready",
    ] {
        assert!(
            traced_actions.contains(&action),
            "trace should include {action}, got {traced_actions:?}"
        );
    }
    assert!(
        traced_logs.iter().any(|log| {
            log.action == "critic_rejected"
                && log.task_id.as_deref() == Some(reviewer.id.as_str())
                && log.metadata["feedback"]
                    == "Create the missing Crytex report and add verification evidence."
                && log.metadata["failure_type"] == "missing_requirement"
        }),
        "critic rejection log should expose structured feedback and failure_type"
    );
    assert!(
        traced_logs.iter().any(|log| {
            log.action == "remediation_plan_created"
                && log.task_id.as_deref() == Some(remediation_parent.id.as_str())
                && log.metadata["reviewer_task_id"] == reviewer.id
                && log
                    .metadata
                    .get("generated_task_ids")
                    .and_then(|ids| ids.as_array())
                    .is_some_and(|ids| ids.len() == 4)
        }),
        "remediation log should expose parent and generated task ids"
    );
    assert!(
        traced_logs.iter().any(|log| {
            log.action == "human_review_ready"
                && log.task_id.as_deref() == Some(run.review_tasks[0].id.as_str())
                && log
                    .metadata
                    .get("review_task_ids")
                    .and_then(|ids| ids.as_array())
                    .is_some_and(|ids| ids.iter().any(|id| id == &run.review_tasks[0].id))
        }),
        "final critic pass should be visible as a human review gate"
    );

    println!(
        "CRYTEX_TAURI_REAL_CRITIC_REMEDIATION_E2E_RESULT model={} original_reviewer={} remediation_parent={} final_critic={} feedback={} traced_actions={:?}",
        state.model,
        reviewer.id,
        remediation_parent.id,
        run.review_tasks[0].id,
        original_reviewer.result.as_ref().unwrap()["agent_result"]["feedback"]
            .as_str()
            .unwrap_or_default(),
        traced_actions
    );

    state.app.shutdown_project_watchers().await;
}

#[tokio::test]
async fn full_happy_path_real_ollama_writes_remediates_and_records_human_reward() {
    let ollama_url =
        std::env::var("CRYTEX_E2E_OLLAMA_URL").unwrap_or_else(|_| DEFAULT_OLLAMA_URL.to_string());
    let model =
        std::env::var("CRYTEX_E2E_OLLAMA_MODEL").unwrap_or_else(|_| DEFAULT_E2E_MODEL.to_string());
    ensure_model_available(&ollama_url, &model).await;

    let temp_dir = tempfile::tempdir().expect("temp dir should be created");
    let project_root = temp_dir.path().join("project-full-happy-path");
    std::fs::create_dir_all(&project_root).expect("project root should be created");
    std::fs::write(
        project_root.join("requirements.md"),
        "# Crytex Vertical E2E\n\n- RAG evidence\n- Observe trace evidence\n- LoRA evolution evidence\n",
    )
    .expect("requirements should be created");

    let backend = Arc::new(OllamaBackend::new(&ollama_url, &model));
    let mut registry = BackendRegistry::new("ollama");
    registry.register("ollama", backend);
    let inference = Arc::new(InferenceServiceImpl::new(
        Arc::new(registry),
        Some("ollama".to_string()),
    ));
    let executor = Arc::new(RealCoderRealCriticVerticalExecutor {
        coder: CoderAgent::new(),
        critic: CriticAgent::new(),
        inference,
        model: model.clone(),
        project_root: project_root.clone(),
    });
    let db_path = temp_dir.path().join("crytex-tauri-full-happy-path-e2e.db");
    let state = CrytexAppState::new_sqlite_with_executor(&db_path, executor)
        .await
        .expect("state should initialize");
    let project = state
        .create_project(CreateProjectCommand {
            name: "Full Happy Path Ollama E2E".into(),
            root_path: project_root.display().to_string(),
        })
        .await
        .expect("project should be created");

    let plan = state
        .submit_goal(SubmitGoalCommand {
            project_id: project.id.clone(),
            goal: "Create a Crytex vertical report from requirements.md. The final report must contain RAG evidence, Observe trace evidence, and LoRA evolution evidence.".into(),
            context: json!({
                "e2e": "full_happy_path_real_ollama",
                "required_file": "crytex-vertical-report.md"
            }),
            trace_id: Some("trace-full-happy-path-ollama-e2e".into()),
        })
        .await
        .expect("goal should be planned");
    state
        .approve_plan(PlanDecisionCommand {
            goal_task_id: plan.goal.id.clone(),
            comment: Some("approved full vertical e2e".into()),
        })
        .await
        .expect("plan should be approved");

    let run = state
        .start_run(StartRunCommand {
            project_id: project.id.clone(),
            max_steps: 14,
        })
        .await
        .expect("run should complete remediation and stop at human review");

    assert_eq!(run.review_tasks.len(), 1);
    assert_eq!(
        run.review_tasks[0].assigned_agent.as_deref(),
        Some("critic")
    );
    assert_eq!(
        run.review_tasks[0].result.as_ref().unwrap()["agent_result"]["review_decision"],
        "pass"
    );

    let report_path = project_root.join("crytex-vertical-report.md");
    let report = std::fs::read_to_string(&report_path)
        .expect("remediation coder should leave the final report on disk");
    assert!(report.contains("CRYTEX_VERTICAL_FIXED"));
    assert!(report.contains("RAG evidence"));
    assert!(report.contains("Observe trace evidence"));
    assert!(report.contains("LoRA evolution evidence"));

    let approved = state
        .approve_task_review(TaskReviewDecisionCommand {
            task_id: run.review_tasks[0].id.clone(),
            comment: Some("full happy path accepted".into()),
        })
        .await
        .expect("human should approve final critic result");
    assert_eq!(approved.task.status.as_str(), "completed");
    assert_eq!(approved.task.human_score, Some(1.0));

    let exported = state
        .get_project_state(&project.id)
        .await
        .expect("project state should export full happy path");
    let original_rejection = exported.tasks.iter().any(|task| {
        task.assigned_agent.as_deref() == Some("critic")
            && task.status.as_str() == "failed"
            && task
                .result
                .as_ref()
                .is_some_and(|result| result["agent_result"]["review_decision"] == "reject")
    });
    assert!(
        original_rejection,
        "first critic must reject the incomplete artifact"
    );

    let traced_actions = exported
        .recent_logs
        .iter()
        .filter(|log| {
            log.metadata
                .get("trace_id")
                .and_then(|trace| trace.as_str())
                == Some("trace-full-happy-path-ollama-e2e")
        })
        .map(|log| log.action.as_str())
        .collect::<Vec<_>>();
    for action in [
        "run_started",
        "critic_rejected",
        "remediation_plan_created",
        "human_review_ready",
        "human_review_approved",
    ] {
        assert!(
            traced_actions.contains(&action),
            "trace should include {action}, got {traced_actions:?}"
        );
    }

    println!(
        "CRYTEX_TAURI_FULL_HAPPY_PATH_E2E_RESULT model={model} final_critic={} report={} traced_actions={:?}",
        run.review_tasks[0].id, report, traced_actions
    );

    state.shutdown_project_watchers().await;
}

#[tokio::test]
async fn production_ollama_agent_executor_runs_goal_chain_with_rag_trace_and_human_reward() {
    let ollama_url =
        std::env::var("CRYTEX_E2E_OLLAMA_URL").unwrap_or_else(|_| DEFAULT_OLLAMA_URL.to_string());
    let model =
        std::env::var("CRYTEX_E2E_OLLAMA_MODEL").unwrap_or_else(|_| DEFAULT_E2E_MODEL.to_string());
    ensure_model_available(&ollama_url, &model).await;

    let temp_dir = tempfile::tempdir().expect("temp dir should be created");
    let project_root = temp_dir.path().join("project-production-agent");
    std::fs::create_dir_all(project_root.join("docs")).expect("docs dir should be created");
    std::fs::write(
        project_root.join("docs/payment-retry.md"),
        "# Payment Retry Production Path\n\nPRODUCTION_AGENT_RAG_MARKER documents the payment retry requirements for the production agent executor.\n",
    )
    .expect("RAG document should be written");

    let db_path = temp_dir.path().join("crytex-tauri-production-agent-e2e.db");
    let state = CrytexAppState::new_sqlite_with_ollama_agent_executor(
        &db_path,
        ollama_url.clone(),
        model.clone(),
    )
    .await
    .expect("state should initialize with production Ollama agent executor");
    let project = state
        .create_project(CreateProjectCommand {
            name: "Production Agent Path Ollama E2E".into(),
            root_path: project_root.display().to_string(),
        })
        .await
        .expect("project should be created and indexed");

    let plan = state
        .submit_goal(SubmitGoalCommand {
            project_id: project.id.clone(),
            goal: "Use the payment retry production context, make the coder create docs/production-agent-report.md with fs_write, and prove the production agent executor can run the generated chain to human review. The coder final JSON must include files_changed with docs/production-agent-report.md and summary containing PRODUCTION_AGENT_RAG_MARKER.".into(),
            context: json!({
                "e2e": "production_ollama_agent_executor",
                "expected_behavior": "default orchestrator materializes the chain and production agent executor runs it",
                "required_file": "docs/production-agent-report.md",
                "required_marker": "PRODUCTION_AGENT_RAG_MARKER"
            }),
            trace_id: Some("trace-production-agent-ollama-e2e".into()),
        })
        .await
        .expect("goal should be planned by production orchestrator");
    assert_eq!(plan.goal.status.as_str(), "review");
    assert_eq!(plan.generated_tasks.len(), 5);

    state
        .approve_plan(PlanDecisionCommand {
            goal_task_id: plan.goal.id.clone(),
            comment: Some("approve production agent executor e2e".into()),
        })
        .await
        .expect("plan should be approved");

    let mut live_events = state
        .subscribe_to_events()
        .await
        .expect("production e2e should subscribe to live backend events");
    let run = state
        .start_run(StartRunCommand {
            project_id: project.id.clone(),
            max_steps: 10,
        })
        .await
        .expect("production agent executor should run generated chain");
    let live_events = drain_events(&mut live_events);
    let live_run_actions = live_events
        .iter()
        .filter_map(|event| match event {
            Event::RunObserved {
                project_id,
                trace_id,
                action,
                ..
            } if project_id == &project.id && trace_id == "trace-production-agent-ollama-e2e" => {
                Some(action.as_str())
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    for action in [
        "run_started",
        "task_execution_started",
        "task_execution_finished",
        "human_review_ready",
    ] {
        assert!(
            live_run_actions.contains(&action),
            "live RunObserved stream should include {action}, got {live_run_actions:?}"
        );
    }

    let exported = state
        .get_project_state(&project.id)
        .await
        .expect("project state should export production run evidence");
    let task_diagnostics = exported
        .tasks
        .iter()
        .map(|task| {
            json!({
                "id": task.id,
                "title": task.title,
                "kind": task.kind,
                "agent": task.assigned_agent,
                "status": task.status.as_str(),
                "parent": task.parent_id,
                "review_decision": task
                    .result
                    .as_ref()
                    .and_then(|result| result.pointer("/agent_result/review_decision"))
                    .or_else(|| task
                        .result
                        .as_ref()
                        .and_then(|result| result.pointer("/agent_result/agent_result/review_decision")))
                    .or_else(|| task
                        .result
                        .as_ref()
                        .and_then(|result| result.pointer("/review_decision"))),
            })
        })
        .collect::<Vec<_>>();

    let has_single_review_task = run.review_tasks.len() == 1;
    if !has_single_review_task {
        state.shutdown_project_watchers().await;
    }
    assert!(
        has_single_review_task,
        "production run should stop at exactly one human-review task; remaining_ready={}; tasks={}",
        run.remaining_ready_tasks.len(),
        serde_json::to_string_pretty(&task_diagnostics).unwrap()
    );
    assert_eq!(
        run.review_tasks[0].assigned_agent.as_deref(),
        Some("critic")
    );
    assert_eq!(run.review_tasks[0].status.as_str(), "review");
    assert_eq!(
        run.review_tasks[0].result.as_ref().unwrap()["source"],
        "agent_service"
    );
    assert!(
        run.remaining_ready_tasks.is_empty(),
        "production run should stop at the critic human-review gate"
    );

    for agent in ["architect", "coder", "qa", "security"] {
        let task = exported
            .tasks
            .iter()
            .find(|task| {
                task.parent_id.as_deref() == Some(plan.goal.id.as_str())
                    && task.assigned_agent.as_deref() == Some(agent)
            })
            .unwrap_or_else(|| panic!("{agent} generated task should exist"));
        assert_eq!(
            task.status.as_str(),
            "completed",
            "{agent} should complete before critic review"
        );
        assert_eq!(
            task.result.as_ref().unwrap()["source"],
            "agent_service",
            "{agent} should execute through the production agent service"
        );
    }

    let initial_critic = exported
        .tasks
        .iter()
        .find(|task| {
            task.parent_id.as_deref() == Some(plan.goal.id.as_str())
                && task.assigned_agent.as_deref() == Some("critic")
        })
        .expect("initial critic task should be exported");
    assert!(
        initial_critic
            .payload
            .get("parent_result")
            .and_then(|value| value.as_array())
            .is_some_and(|artifacts| artifacts.len() >= 4),
        "initial critic should receive upstream artifacts from the generated chain"
    );

    let review_gate = exported
        .tasks
        .iter()
        .find(|task| task.id == run.review_tasks[0].id)
        .expect("critic review gate should be exported");
    assert!(
        review_gate
            .payload
            .get("parent_result")
            .and_then(|value| value.as_array())
            .is_some_and(|artifacts| artifacts.len() >= 3),
        "critic review gate should receive upstream artifacts before human review"
    );

    let traced_logs = exported
        .recent_logs
        .iter()
        .filter(|log| {
            log.metadata
                .get("trace_id")
                .and_then(|trace| trace.as_str())
                == Some("trace-production-agent-ollama-e2e")
        })
        .collect::<Vec<_>>();
    let traced_actions = traced_logs
        .iter()
        .map(|log| log.action.as_str())
        .collect::<Vec<_>>();
    for action in [
        "run_started",
        "task_execution_started",
        "task_execution_finished",
        "llm_request",
        "llm_response",
        "human_review_ready",
    ] {
        assert!(
            traced_actions.contains(&action),
            "trace should include {action}, got {traced_actions:?}"
        );
    }
    assert!(
        traced_logs.iter().any(|log| {
            log.action == "llm_request"
                && log.metadata.get("messages").is_some_and(|messages| {
                    messages.to_string().contains("PRODUCTION_AGENT_RAG_MARKER")
                })
        }),
        "production agent executor should send indexed RAG context to Ollama"
    );

    let approved = state
        .approve_task_review(TaskReviewDecisionCommand {
            task_id: run.review_tasks[0].id.clone(),
            comment: Some("production agent executor accepted".into()),
        })
        .await
        .expect("human should approve production review gate");
    assert_eq!(approved.task.status.as_str(), "completed");
    assert_eq!(approved.task.human_score, Some(1.0));

    let exported_after_approval = state
        .get_project_state(&project.id)
        .await
        .expect("project state should export human reward evidence");
    assert!(
        exported_after_approval.recent_logs.iter().any(|log| {
            log.action == "human_review_approved"
                && log.task_id.as_deref() == Some(run.review_tasks[0].id.as_str())
                && log.metadata["human_score"] == 1.0
                && log.metadata["reward"].as_f64().is_some()
        }),
        "human approval should emit observable reward/evolution evidence"
    );

    state.shutdown_project_watchers().await;

    println!(
        "CRYTEX_TAURI_PRODUCTION_AGENT_E2E_RESULT model={model} critic_task={} traced_actions={:?} live_run_actions={:?}",
        run.review_tasks[0].id, traced_actions, live_run_actions
    );
}

struct OllamaState {
    app: CrytexAppState,
    model: String,
    temp_dir: tempfile::TempDir,
}

async fn ollama_critic_state() -> OllamaState {
    let ollama_url =
        std::env::var("CRYTEX_E2E_OLLAMA_URL").unwrap_or_else(|_| DEFAULT_OLLAMA_URL.to_string());
    let model =
        std::env::var("CRYTEX_E2E_OLLAMA_MODEL").unwrap_or_else(|_| DEFAULT_E2E_MODEL.to_string());
    ensure_model_available(&ollama_url, &model).await;

    let backend = Arc::new(OllamaBackend::new(&ollama_url, &model));
    let mut registry = BackendRegistry::new("ollama");
    registry.register("ollama", backend);
    let inference = Arc::new(InferenceServiceImpl::new(
        Arc::new(registry),
        Some("ollama".to_string()),
    ));
    let executor = Arc::new(RealCriticDeterministicWorkerExecutor {
        critic: CriticAgent::new(),
        inference,
        model: model.clone(),
    });
    let temp_dir = tempfile::tempdir().expect("temp dir should be created");
    let db_path = temp_dir.path().join("crytex-tauri-critic-e2e.db");
    let app = CrytexAppState::new_sqlite_with_executor(&db_path, executor)
        .await
        .expect("state should initialize");

    OllamaState {
        app,
        model,
        temp_dir,
    }
}

struct RealCriticDeterministicWorkerExecutor {
    critic: CriticAgent,
    inference: Arc<dyn InferenceService>,
    model: String,
}

struct RealCoderRealCriticVerticalExecutor {
    coder: CoderAgent,
    critic: CriticAgent,
    inference: Arc<dyn InferenceService>,
    model: String,
    project_root: std::path::PathBuf,
}

#[async_trait]
impl TaskExecutor for RealCoderRealCriticVerticalExecutor {
    async fn execute(&self, task: &Task, run_id: &str) -> Result<Value, TauriCommandError> {
        if task.assigned_agent.as_deref() == Some("coder") {
            let mut coder_task = task.clone();
            coder_task.payload["model"] = Value::String(self.model.clone());
            coder_task.payload["backend"] = Value::String("ollama".to_string());
            coder_task.payload["system_prompt_override"] =
                Value::String(vertical_coder_prompt(task));
            let agent_result = self
                .coder
                .execute(
                    &coder_task,
                    self.inference.clone(),
                    vertical_tool_service(&self.project_root),
                )
                .await
                .map_err(AgentServiceError::Agent)?;
            return Ok(json!({
                "source": "real_ollama_coder",
                "run_id": run_id,
                "task_id": task.id,
                "agent": task.assigned_agent,
                "agent_result": agent_result
            }));
        }

        if task.assigned_agent.as_deref() == Some("critic") {
            let report =
                std::fs::read_to_string(self.project_root.join("crytex-vertical-report.md"))
                    .unwrap_or_default();
            let mut critic_task = task.clone();
            critic_task.payload["model"] = Value::String(self.model.clone());
            critic_task.payload["backend"] = Value::String("ollama".to_string());
            critic_task.payload["parent_result"] = json!({
                "report_path": "crytex-vertical-report.md",
                "report_content": report,
            });
            critic_task.payload["system_prompt_override"] = Value::String(vertical_critic_prompt());
            let agent_result = self
                .critic
                .execute(
                    &critic_task,
                    self.inference.clone(),
                    Arc::new(NoopToolService),
                )
                .await
                .map_err(AgentServiceError::Agent)?;
            return Ok(json!({
                "source": "real_ollama_critic",
                "run_id": run_id,
                "task_id": task.id,
                "agent": task.assigned_agent,
                "agent_result": agent_result
            }));
        }

        let upstream_count = task
            .payload
            .get("upstream_artifacts")
            .and_then(|value| value.as_array())
            .map(Vec::len)
            .unwrap_or_default();
        Ok(json!({
            "source": "deterministic_vertical_worker",
            "run_id": run_id,
            "task_id": task.id,
            "agent": task.assigned_agent,
            "agent_result": {
                "summary": format!("{} completed vertical support step", task.kind),
                "upstream_count": upstream_count
            }
        }))
    }
}

fn vertical_tool_service(project_root: &std::path::Path) -> Arc<dyn ToolService> {
    Arc::new(ToolServiceImpl::new(
        TypedToolRegistry::new().with_default_coding_tools().build(),
        project_root.to_path_buf(),
        Capability::all(),
        Arc::new(HostBackend::new()),
        None,
        SecurityConfig::default(),
    ))
}

fn vertical_coder_prompt(task: &Task) -> String {
    if task.payload["source"] == "reviewer_rejection" || task.kind == "debug" {
        return r###"You are a deterministic Crytex remediation E2E coder.
Use the registered tools. First respond only with this JSON tool call:
{"tool":"fs_read","args":{"path":"requirements.md"}}
After receiving the fs_read observation, respond only with this JSON tool call:
{"tool":"fs_write","args":{"path":"crytex-vertical-report.md","content":"# Crytex Vertical Report\n\n## RAG evidence\nRequirements were read from requirements.md.\n\n## Observe trace evidence\nThe run produced traceable runner, critic, remediation, and human-review events.\n\n## LoRA evolution evidence\nHuman approval records reward data for later prompt and LoRA evolution.\n\nCRYTEX_VERTICAL_FIXED"}}
After receiving the fs_write observation, respond only with a final JSON object containing files_changed, test_results, and summary."###.to_string();
    }

    r###"You are a deterministic Crytex initial E2E coder.
Use the registered tools. First respond only with this JSON tool call:
{"tool":"fs_read","args":{"path":"requirements.md"}}
After receiving the fs_read observation, respond only with this JSON tool call:
{"tool":"fs_write","args":{"path":"crytex-vertical-report.md","content":"# Crytex Vertical Report\n\n## RAG evidence\nRequirements were read from requirements.md.\n\n## Observe trace evidence\nThe run produced visible trace events.\n\n## Missing\nLoRA evolution evidence is intentionally missing in this first attempt."}}
After receiving the fs_write observation, respond only with a final JSON object containing files_changed, test_results, and summary."###.to_string()
}

fn vertical_critic_prompt() -> String {
    r#"You are a deterministic Crytex vertical E2E critic.
Return exactly one valid JSON object and no markdown.
If the reviewed report_content contains CRYTEX_VERTICAL_FIXED, return:
{"score":4.8,"review_decision":"pass","target_task_id":null,"failure_type":null,"blocking_issues":[],"feedback":"Final remediation satisfies the vertical happy path.","comments":["accepted"]}
Otherwise return:
{"score":1.0,"review_decision":"reject","target_task_id":null,"failure_type":"missing_requirement","blocking_issues":[{"severity":"high","reason":"LoRA evolution evidence is missing","evidence":"The report does not contain CRYTEX_VERTICAL_FIXED.","expected":"Add LoRA evolution evidence and explicit verification marker."}],"feedback":"Add LoRA evolution evidence and explicit verification marker.","comments":["missing LoRA evolution evidence"]}"#.to_string()
}

#[async_trait]
impl TaskExecutor for RealCriticDeterministicWorkerExecutor {
    async fn execute(&self, task: &Task, run_id: &str) -> Result<Value, TauriCommandError> {
        if task.assigned_agent.as_deref() == Some("critic") {
            let mut critic_task = task.clone();
            critic_task.payload["model"] = Value::String(self.model.clone());
            critic_task.payload["backend"] = Value::String("ollama".to_string());
            if critic_task.payload.get("system_prompt_override").is_none() {
                critic_task.payload["system_prompt_override"] =
                    Value::String(remediation_final_critic_prompt());
            }
            let agent_result = self
                .critic
                .execute(
                    &critic_task,
                    self.inference.clone(),
                    Arc::new(NoopToolService),
                )
                .await
                .map_err(AgentServiceError::Agent)?;
            return Ok(json!({
                "source": "real_ollama_critic",
                "run_id": run_id,
                "task_id": task.id,
                "agent": task.assigned_agent,
                "agent_result": agent_result
            }));
        }

        let upstream_count = task
            .payload
            .get("upstream_artifacts")
            .and_then(|value| value.as_array())
            .map(Vec::len)
            .unwrap_or_default();
        let agent_result = if task.assigned_agent.as_deref() == Some("coder") {
            json!({
                "summary": format!("{} completed remediation step", task.kind),
                "files_changed": ["crytex-remediation-report.md"],
                "test_results": "deterministic remediation worker passed",
                "upstream_count": upstream_count
            })
        } else {
            json!({
                "summary": format!("{} completed remediation step", task.kind),
                "upstream_count": upstream_count
            })
        };
        Ok(json!({
            "source": "deterministic_remediation_worker",
            "run_id": run_id,
            "task_id": task.id,
            "agent": task.assigned_agent,
            "agent_result": agent_result
        }))
    }
}

fn remediation_final_critic_prompt() -> String {
    r#"You are a deterministic Crytex remediation E2E final critic.
Return exactly one valid JSON object and no markdown.
The remediation chain has completed the debug, codegen, QA, and security steps.
Return:
{"score":4.5,"review_decision":"pass","target_task_id":null,"failure_type":null,"blocking_issues":[],"feedback":"Remediation is acceptable.","comments":["remediation chain completed"]}"#.to_string()
}

struct NoopToolService;

#[async_trait]
impl ToolService for NoopToolService {
    async fn invoke(&self, name: &str, _args: Value) -> Result<Value, ToolServiceError> {
        Err(ToolServiceError::NotFound(name.to_string()))
    }

    fn list_tools(&self) -> Vec<ToolDescription> {
        vec![]
    }
}

async fn ensure_model_available(ollama_url: &str, model: &str) {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(900))
        .build()
        .expect("reqwest client should build");

    let tags = client
        .get(format!("{ollama_url}/api/tags"))
        .send()
        .await
        .expect("Ollama must be running and reachable")
        .json::<serde_json::Value>()
        .await
        .expect("Ollama tags response should be JSON");

    let model_is_cached = tags
        .get("models")
        .and_then(|models| models.as_array())
        .is_some_and(|models| {
            models.iter().any(|entry| {
                entry.get("name").and_then(|name| name.as_str()) == Some(model)
                    || entry.get("model").and_then(|name| name.as_str()) == Some(model)
            })
        });

    if model_is_cached {
        return;
    }

    let response = client
        .post(format!("{ollama_url}/api/pull"))
        .json(&json!({ "name": model, "stream": false }))
        .send()
        .await
        .expect("Ollama must be running and reachable");

    assert!(
        response.status().is_success(),
        "Ollama pull failed for {model}: status={} body={}",
        response.status(),
        response.text().await.unwrap_or_default()
    );
}
