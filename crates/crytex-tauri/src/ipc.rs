//! Tauri IPC facade for the desktop frontend.

use crate::app_state::CrytexAppState;
use crate::commands::{
    AddManagedModelCommand, BackendE2eMatrixCommand, BackendE2eMatrixReport, CreateProjectCommand,
    DownloadManagedModelCommand, EvaluatePromptChallengerCommand, EvaluatePromptChallengerResponse,
    ExportRunDiagnosticsCommand, GoalPlanResponse, ManagedModelRecord,
    ManagedModelRuntimeProofReport, ManagedModelsResponse, OllamaModelsResponse,
    PlanDecisionCommand, PlanDecisionResponse, ProveManagedModelRuntimeCommand,
    RunDiagnosticsReport, RuntimeStatus, SearchProjectContextCommand, SearchProjectContextResponse,
    SetActiveManagedModelCommand, SetActiveOllamaModelCommand, SetTaskStatusCommand,
    StartRunCommand, StartRunResponse, SubmitGoalCommand, SubmitTaskCommand,
    TaskReviewDecisionCommand, TaskReviewDecisionResponse, TauriCommandError,
    TrainLoraAdapterCommand, TrainLoraAdapterResponse,
};
use crytex_core::bus::Event;
use crytex_core::models::{KanbanState, Project, Task};
use crytex_core::state_export::ProjectState;
use serde::Serialize;
use serde_json::Value;
use tauri::State;

/// Serializable error shape returned to the TypeScript frontend.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct IpcError {
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct UiBackendEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    pub project_id: Option<String>,
    pub payload: Value,
}

impl From<TauriCommandError> for IpcError {
    fn from(error: TauriCommandError) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}

impl UiBackendEvent {
    pub fn from_domain(event: Event) -> Self {
        let project_id = event_project_id(&event).map(ToOwned::to_owned);
        let value = serde_json::to_value(event).unwrap_or(Value::Null);
        let (event_type, payload) = match value {
            Value::Object(fields) if fields.len() == 1 => fields
                .into_iter()
                .next()
                .unwrap_or_else(|| ("Unknown".to_string(), Value::Null)),
            payload => ("Unknown".to_string(), payload),
        };
        Self {
            event_type,
            project_id,
            payload,
        }
    }
}

pub fn event_project_id(event: &Event) -> Option<&str> {
    match event {
        Event::TaskCreated { project_id, .. }
        | Event::ProjectOpened { project_id }
        | Event::FileOpened { project_id, .. }
        | Event::FileClosed { project_id, .. }
        | Event::CursorMoved { project_id, .. }
        | Event::DiagnosticsReceived { project_id, .. }
        | Event::ProjectContextUpdated { project_id, .. }
        | Event::LoraSwapped { project_id, .. }
        | Event::RunObserved { project_id, .. } => Some(project_id),
        _ => None,
    }
}

#[tauri::command]
pub async fn list_projects(state: State<'_, CrytexAppState>) -> Result<Vec<Project>, IpcError> {
    state.list_projects().await.map_err(Into::into)
}

#[tauri::command]
pub async fn create_project(
    state: State<'_, CrytexAppState>,
    request: CreateProjectCommand,
) -> Result<Project, IpcError> {
    state.create_project(request).await.map_err(Into::into)
}

#[tauri::command]
pub async fn kanban_state(
    state: State<'_, CrytexAppState>,
    project_id: String,
) -> Result<KanbanState, IpcError> {
    state.kanban_state(&project_id).await.map_err(Into::into)
}

#[tauri::command]
pub async fn list_tasks(
    state: State<'_, CrytexAppState>,
    project_id: String,
) -> Result<Vec<Task>, IpcError> {
    state.list_tasks(&project_id).await.map_err(Into::into)
}

#[tauri::command]
pub async fn submit_task(
    state: State<'_, CrytexAppState>,
    request: SubmitTaskCommand,
) -> Result<Task, IpcError> {
    state.submit_task(request).await.map_err(Into::into)
}

#[tauri::command]
pub async fn submit_goal(
    state: State<'_, CrytexAppState>,
    request: SubmitGoalCommand,
) -> Result<GoalPlanResponse, IpcError> {
    state.submit_goal(request).await.map_err(Into::into)
}

#[tauri::command]
pub async fn set_task_status(
    state: State<'_, CrytexAppState>,
    request: SetTaskStatusCommand,
) -> Result<Task, IpcError> {
    state.set_task_status(request).await.map_err(Into::into)
}

#[tauri::command]
pub async fn approve_plan(
    state: State<'_, CrytexAppState>,
    request: PlanDecisionCommand,
) -> Result<PlanDecisionResponse, IpcError> {
    state.approve_plan(request).await.map_err(Into::into)
}

#[tauri::command]
pub async fn reject_plan(
    state: State<'_, CrytexAppState>,
    request: PlanDecisionCommand,
) -> Result<PlanDecisionResponse, IpcError> {
    state.reject_plan(request).await.map_err(Into::into)
}

#[tauri::command]
pub async fn approve_task_review(
    state: State<'_, CrytexAppState>,
    request: TaskReviewDecisionCommand,
) -> Result<TaskReviewDecisionResponse, IpcError> {
    state.approve_task_review(request).await.map_err(Into::into)
}

#[tauri::command]
pub async fn reject_task_review(
    state: State<'_, CrytexAppState>,
    request: TaskReviewDecisionCommand,
) -> Result<TaskReviewDecisionResponse, IpcError> {
    state.reject_task_review(request).await.map_err(Into::into)
}

#[tauri::command]
pub async fn start_run(
    state: State<'_, CrytexAppState>,
    request: StartRunCommand,
) -> Result<StartRunResponse, IpcError> {
    state.start_run(request).await.map_err(Into::into)
}

#[tauri::command]
pub async fn get_project_state(
    state: State<'_, CrytexAppState>,
    project_id: String,
) -> Result<ProjectState, IpcError> {
    state
        .get_project_state(&project_id)
        .await
        .map_err(Into::into)
}

#[tauri::command]
pub async fn export_run_diagnostics(
    state: State<'_, CrytexAppState>,
    request: ExportRunDiagnosticsCommand,
) -> Result<RunDiagnosticsReport, IpcError> {
    state
        .export_run_diagnostics(request)
        .await
        .map_err(Into::into)
}

#[tauri::command]
pub async fn run_backend_e2e_matrix(
    state: State<'_, CrytexAppState>,
    request: BackendE2eMatrixCommand,
) -> Result<BackendE2eMatrixReport, IpcError> {
    state
        .run_backend_e2e_matrix(request)
        .await
        .map_err(Into::into)
}

#[tauri::command]
pub async fn search_project_context(
    state: State<'_, CrytexAppState>,
    request: SearchProjectContextCommand,
) -> Result<SearchProjectContextResponse, IpcError> {
    state
        .search_project_context(request)
        .await
        .map_err(Into::into)
}

#[tauri::command]
pub async fn runtime_status(state: State<'_, CrytexAppState>) -> Result<RuntimeStatus, IpcError> {
    state.runtime_status().await.map_err(Into::into)
}

#[tauri::command]
pub async fn list_ollama_models(
    state: State<'_, CrytexAppState>,
) -> Result<OllamaModelsResponse, IpcError> {
    state.list_ollama_models().await.map_err(Into::into)
}

#[tauri::command]
pub async fn list_managed_models(
    state: State<'_, CrytexAppState>,
) -> Result<ManagedModelsResponse, IpcError> {
    state.list_managed_models().await.map_err(Into::into)
}

#[tauri::command]
pub async fn download_managed_model(
    state: State<'_, CrytexAppState>,
    request: DownloadManagedModelCommand,
) -> Result<ManagedModelRecord, IpcError> {
    state
        .download_managed_model(request)
        .await
        .map_err(Into::into)
}

#[tauri::command]
pub async fn add_managed_model(
    state: State<'_, CrytexAppState>,
    request: AddManagedModelCommand,
) -> Result<ManagedModelRecord, IpcError> {
    state.add_managed_model(request).await.map_err(Into::into)
}

#[tauri::command]
pub async fn set_active_ollama_model(
    state: State<'_, CrytexAppState>,
    request: SetActiveOllamaModelCommand,
) -> Result<RuntimeStatus, IpcError> {
    state
        .set_active_ollama_model_from_command(request)
        .await
        .map_err(Into::into)
}

#[tauri::command]
pub async fn set_active_managed_model(
    state: State<'_, CrytexAppState>,
    request: SetActiveManagedModelCommand,
) -> Result<RuntimeStatus, IpcError> {
    state
        .set_active_managed_model(request)
        .await
        .map_err(Into::into)
}

#[tauri::command]
pub async fn prove_managed_model_runtime(
    state: State<'_, CrytexAppState>,
    request: ProveManagedModelRuntimeCommand,
) -> Result<ManagedModelRuntimeProofReport, IpcError> {
    state
        .prove_managed_model_runtime(request)
        .await
        .map_err(Into::into)
}

#[tauri::command]
pub async fn train_lora_adapter(
    state: State<'_, CrytexAppState>,
    request: TrainLoraAdapterCommand,
) -> Result<TrainLoraAdapterResponse, IpcError> {
    state.train_lora_adapter(request).await.map_err(Into::into)
}

#[tauri::command]
pub async fn evaluate_prompt_challenger(
    state: State<'_, CrytexAppState>,
    request: EvaluatePromptChallengerCommand,
) -> Result<EvaluatePromptChallengerResponse, IpcError> {
    state
        .evaluate_prompt_challenger(request)
        .await
        .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn ipc_error_serializes_message_for_frontend() {
        let error = IpcError::from(TauriCommandError::Bootstrap("missing app data".into()));
        let value = serde_json::to_value(error).unwrap();

        assert_eq!(
            value,
            json!({
                "message": "bootstrap error: missing app data"
            })
        );
    }

    #[test]
    fn event_project_id_extracts_project_scoped_events() {
        assert_eq!(
            event_project_id(&crytex_core::bus::Event::TaskCreated {
                task_id: "task-1".into(),
                project_id: "project-1".into(),
            }),
            Some("project-1")
        );
        assert_eq!(
            event_project_id(&crytex_core::bus::Event::TaskStarted {
                task_id: "task-1".into(),
            }),
            None
        );
    }

    #[test]
    fn ui_backend_event_wraps_domain_event_for_frontend() {
        let event = UiBackendEvent::from_domain(crytex_core::bus::Event::TaskCreated {
            task_id: "task-1".into(),
            project_id: "project-1".into(),
        });

        let value = serde_json::to_value(event).unwrap();

        assert_eq!(
            value,
            json!({
                "type": "TaskCreated",
                "project_id": "project-1",
                "payload": {
                    "task_id": "task-1",
                    "project_id": "project-1"
                }
            })
        );
    }
}
