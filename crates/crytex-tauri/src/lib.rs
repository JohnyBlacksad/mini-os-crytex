//! Tauri IPC command scaffold for the Crytex desktop UI.
//!
//! This crate exposes plain async functions that mirror the commands the
//! Tauri frontend will invoke.  The functions intentionally depend on the
//! core service traits rather than the Tauri runtime so they can be unit
//! tested and reused outside of the Tauri process.

pub mod app_state;
pub mod commands;
pub mod ipc;

use app_state::CrytexAppState;
use tokio::sync::broadcast;

use tauri::{Emitter, Manager};

pub use commands::TauriCommandError;

pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            let app_data = app.path().app_data_dir()?;
            std::fs::create_dir_all(&app_data)?;
            let db_path = app_data.join("crytex-ui.db");
            let handle = app.handle().clone();

            tauri::async_runtime::block_on(async move {
                let state = CrytexAppState::new_sqlite(&db_path)
                    .await
                    .map_err(|err| std::io::Error::other(err.to_string()))?;
                let mut event_rx = state
                    .subscribe_to_events()
                    .await
                    .map_err(|err| std::io::Error::other(err.to_string()))?;
                let event_handle = handle.clone();
                tauri::async_runtime::spawn(async move {
                    loop {
                        match event_rx.recv().await {
                            Ok(event) => {
                                let _ = event_handle.emit(
                                    "crytex://event",
                                    ipc::UiBackendEvent::from_domain(event),
                                );
                            }
                            Err(broadcast::error::RecvError::Lagged(_)) => continue,
                            Err(broadcast::error::RecvError::Closed) => break,
                        }
                    }
                });
                handle.manage(state);
                Ok::<(), std::io::Error>(())
            })?;

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            ipc::list_projects,
            ipc::create_project,
            ipc::kanban_state,
            ipc::list_tasks,
            ipc::submit_task,
            ipc::submit_goal,
            ipc::set_task_status,
            ipc::approve_plan,
            ipc::reject_plan,
            ipc::approve_task_review,
            ipc::reject_task_review,
            ipc::start_run,
            ipc::get_project_state,
            ipc::export_run_diagnostics,
            ipc::run_backend_e2e_matrix,
            ipc::search_project_context,
            ipc::runtime_status,
            ipc::list_ollama_models,
            ipc::list_managed_models,
            ipc::download_managed_model,
            ipc::add_managed_model,
            ipc::set_active_ollama_model,
            ipc::set_active_managed_model
        ])
        .run(tauri::generate_context!())
        .expect("failed to run Crytex desktop UI");
}
