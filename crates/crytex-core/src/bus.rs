use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Event {
    TaskCreated {
        task_id: String,
        project_id: String,
    },
    TaskStarted {
        task_id: String,
    },
    TaskProgress {
        task_id: String,
        status: String,
        message: String,
    },
    TaskMoved {
        task_id: String,
        project_id: String,
        from: Option<String>,
        to: String,
        trace_id: String,
        timestamp: i64,
    },
    TaskReview {
        task_id: String,
    },
    TaskCompleted {
        task_id: String,
        result: serde_json::Value,
    },
    TaskFailed {
        task_id: String,
        error: String,
    },
    TaskCancelled {
        task_id: String,
    },
    AgentThinking {
        task_id: String,
        agent: String,
        message: String,
    },
    LoraSwapped {
        project_id: String,
        lora_id: String,
    },
    ProjectOpened {
        project_id: String,
    },
    FileOpened {
        project_id: String,
        file_path: String,
        language: Option<String>,
    },
    FileClosed {
        project_id: String,
        file_path: String,
    },
    CursorMoved {
        project_id: String,
        file_path: String,
        line: u32,
        character: u32,
    },
    DiagnosticsReceived {
        project_id: String,
        file_path: String,
        diagnostics: Vec<serde_json::Value>,
    },
    ProjectContextUpdated {
        project_id: String,
        snapshot_id: String,
    },
    ModelDownloadProgress {
        model_id: String,
        progress: f32,
    },
    RuntimeSelected {
        backend: String,
        model_id: String,
        model_path: Option<String>,
        endpoint_url: Option<String>,
        context_size: Option<usize>,
        gpu_layers: Option<usize>,
        quantization: Option<String>,
    },
    RunObserved {
        project_id: String,
        task_id: Option<String>,
        trace_id: String,
        action: String,
        metadata: serde_json::Value,
    },
    Alert {
        severity: crate::services::AlertSeverity,
        message: String,
        metric: String,
        value: String,
        threshold: String,
    },
    MetricsSnapshot {
        snapshot: crate::metrics::MetricsSnapshot,
    },
    BenchmarkRunCompleted {
        run_id: String,
        name: String,
        pass_rate: f64,
    },
    ABTestCompleted {
        report_id: String,
        baseline_run_id: String,
        challenger_run_id: String,
        winner: String,
    },
}

pub struct EventBus {
    sender: broadcast::Sender<Event>,
}

impl EventBus {
    pub fn new() -> Self {
        let (sender, _) = broadcast::channel(256);
        Self { sender }
    }

    pub fn publish(&self, event: Event) {
        let _ = self.sender.send(event);
    }

    pub fn subscribe(&self) -> broadcast::Receiver<Event> {
        self.sender.subscribe()
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}
