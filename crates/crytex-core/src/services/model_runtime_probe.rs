//! Runtime probe reports for selected models.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use std::{fs, io};

use crytex_inference::{InferenceRequest, Message};
use serde::{Deserialize, Serialize};

use crate::services::hardware::DeviceKind;
use crate::services::model_compatibility::{
    CompatibilityStatus, ModelCompatibilityPlan, ModelCompatibilityPlanner, RuntimeFeatureSet,
};
use crate::services::model_manager::ManagedModel;
use crate::services::{InferenceService, InferenceServiceError};
use crate::tracing::TraceContext;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProbeStageName {
    Metadata,
    Compatibility,
    Generation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProbeStageStatus {
    Passed,
    Failed,
    Skipped,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProbeStageReport {
    pub name: ProbeStageName,
    pub status: ProbeStageStatus,
    pub message: String,
    pub duration_ms: u128,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRuntimeProbeReport {
    pub trace_id: String,
    pub model_id: String,
    pub backend_id: Option<String>,
    pub compatibility: ModelCompatibilityPlan,
    pub stages: Vec<ProbeStageReport>,
    pub generated_preview: Option<String>,
    pub passed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRuntimeProbeRequest {
    pub backend_id: Option<String>,
    pub model_name: String,
    pub trace_id: Option<String>,
    pub max_tokens: usize,
    pub lora_adapter_id: Option<String>,
}

impl ModelRuntimeProbeRequest {
    pub fn smoke(model_name: impl Into<String>) -> Self {
        Self {
            backend_id: None,
            model_name: model_name.into(),
            trace_id: None,
            max_tokens: 16,
            lora_adapter_id: None,
        }
    }
}

pub struct ModelRuntimeProbe {
    inference: Arc<dyn InferenceService>,
}

impl ModelRuntimeProbe {
    pub fn new(inference: Arc<dyn InferenceService>) -> Self {
        Self { inference }
    }

    pub async fn probe(
        &self,
        model: &ManagedModel,
        device: &DeviceKind,
        runtime: &RuntimeFeatureSet,
        request: ModelRuntimeProbeRequest,
    ) -> ModelRuntimeProbeReport {
        let trace_id = request
            .trace_id
            .clone()
            .unwrap_or_else(|| TraceContext::new().trace_id);
        let mut stages = Vec::new();
        let compatibility = ModelCompatibilityPlanner::plan(model, device, runtime);

        stages.push(stage(
            ProbeStageName::Metadata,
            ProbeStageStatus::Passed,
            format!(
                "detected format {:?} with features {:?}",
                compatibility.format, compatibility.features
            ),
            0,
        ));

        if compatibility.status == CompatibilityStatus::Unsupported {
            stages.push(stage(
                ProbeStageName::Compatibility,
                ProbeStageStatus::Failed,
                compatibility.blockers.join("; "),
                0,
            ));
            stages.push(stage(
                ProbeStageName::Generation,
                ProbeStageStatus::Skipped,
                "generation skipped because compatibility check failed",
                0,
            ));
            return ModelRuntimeProbeReport {
                trace_id,
                model_id: model.id.clone(),
                backend_id: request.backend_id,
                compatibility,
                stages,
                generated_preview: None,
                passed: false,
            };
        }

        stages.push(stage(
            ProbeStageName::Compatibility,
            ProbeStageStatus::Passed,
            format!("selected execution strategy {:?}", compatibility.strategy),
            0,
        ));

        let started = Instant::now();
        let generation = self
            .inference
            .generate(smoke_request(&request))
            .await
            .map(|response| response.content);
        let duration_ms = started.elapsed().as_millis();

        match generation {
            Ok(content) if is_expected_smoke_response(&content) => {
                let preview = content.chars().take(512).collect::<String>();
                stages.push(stage(
                    ProbeStageName::Generation,
                    ProbeStageStatus::Passed,
                    "smoke generation returned expected sentinel",
                    duration_ms,
                ));
                ModelRuntimeProbeReport {
                    trace_id,
                    model_id: model.id.clone(),
                    backend_id: request.backend_id,
                    compatibility,
                    stages,
                    generated_preview: Some(preview),
                    passed: true,
                }
            }
            Ok(content) if !content.trim().is_empty() => failed_generation_report(
                trace_id,
                model,
                request.backend_id,
                compatibility,
                stages,
                format!(
                    "smoke generation missed expected sentinel CRYTEX_PROBE_OK: {}",
                    content.chars().take(120).collect::<String>()
                ),
                duration_ms,
            ),
            Ok(_) => failed_generation_report(
                trace_id,
                model,
                request.backend_id,
                compatibility,
                stages,
                "smoke generation returned empty output".to_string(),
                duration_ms,
            ),
            Err(error) => failed_generation_report(
                trace_id,
                model,
                request.backend_id,
                compatibility,
                stages,
                generation_error_message(error),
                duration_ms,
            ),
        }
    }
}

fn is_expected_smoke_response(content: &str) -> bool {
    content
        .trim()
        .trim_matches(|ch: char| ch == '"' || ch == '\'' || ch == '`')
        .eq("CRYTEX_PROBE_OK")
}

fn smoke_request(request: &ModelRuntimeProbeRequest) -> InferenceRequest {
    InferenceRequest {
        backend_id: request.backend_id.clone(),
        model: request.model_name.clone(),
        messages: vec![Message {
            role: "user".into(),
            content: "Reply with exactly: CRYTEX_PROBE_OK".into(),
        }],
        system_prompt: Some("You are running a short runtime smoke test.".into()),
        temperature: Some(0.0),
        max_tokens: Some(request.max_tokens.min(32).max(1)),
        lora_adapter_id: request.lora_adapter_id.clone(),
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeMatrixEntryRequest {
    pub label: String,
    pub model: ManagedModel,
    pub backend_id: Option<String>,
    pub model_name: String,
    pub lora_adapter_id: Option<String>,
    pub max_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct ModelRuntimeMatrixRequest {
    pub trace_id: Option<String>,
    pub entries: Vec<RuntimeMatrixEntryRequest>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeMatrixEntryReport {
    pub label: String,
    pub lora_adapter_id: Option<String>,
    pub report: ModelRuntimeProbeReport,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRuntimeMatrixReport {
    pub trace_id: String,
    pub entries: Vec<RuntimeMatrixEntryReport>,
    pub passed: bool,
}

pub struct ModelRuntimeMatrixProbe {
    probe: ModelRuntimeProbe,
}

impl ModelRuntimeMatrixProbe {
    pub fn new(inference: Arc<dyn InferenceService>) -> Self {
        Self {
            probe: ModelRuntimeProbe::new(inference),
        }
    }

    pub async fn probe(
        &self,
        device: &DeviceKind,
        runtime: &RuntimeFeatureSet,
        request: ModelRuntimeMatrixRequest,
    ) -> ModelRuntimeMatrixReport {
        let trace_id = request
            .trace_id
            .clone()
            .unwrap_or_else(|| TraceContext::new().trace_id);
        let mut entries = Vec::new();

        for (index, entry) in request.entries.into_iter().enumerate() {
            let report = self
                .probe
                .probe(
                    &entry.model,
                    device,
                    runtime,
                    ModelRuntimeProbeRequest {
                        backend_id: entry.backend_id,
                        model_name: entry.model_name,
                        trace_id: Some(format!("{trace_id}:{}", entry.label)),
                        max_tokens: entry.max_tokens,
                        lora_adapter_id: entry.lora_adapter_id.clone(),
                    },
                )
                .await;
            entries.push(RuntimeMatrixEntryReport {
                label: non_empty_label(entry.label, index),
                lora_adapter_id: entry.lora_adapter_id,
                report,
            });
        }

        let passed = entries.iter().all(|entry| entry.report.passed);
        ModelRuntimeMatrixReport {
            trace_id,
            entries,
            passed,
        }
    }
}

fn non_empty_label(label: String, index: usize) -> String {
    if label.trim().is_empty() {
        format!("entry-{index}")
    } else {
        label
    }
}

pub struct RuntimeMatrixReportWriter;

impl RuntimeMatrixReportWriter {
    pub fn write_pretty_json(
        report: &ModelRuntimeMatrixReport,
        report_dir: impl AsRef<Path>,
    ) -> Result<PathBuf, io::Error> {
        let report_dir = report_dir.as_ref();
        fs::create_dir_all(report_dir)?;
        let path = report_dir.join(format!(
            "runtime-matrix-{}.json",
            sanitize_report_file_stem(&report.trace_id)
        ));
        let json = serde_json::to_string_pretty(report)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        fs::write(&path, json)?;
        Ok(path)
    }
}

fn sanitize_report_file_stem(value: &str) -> String {
    let sanitized = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
                ch
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_string();

    if sanitized.is_empty() {
        "untraced".to_string()
    } else {
        sanitized
    }
}

fn failed_generation_report(
    trace_id: String,
    model: &ManagedModel,
    backend_id: Option<String>,
    compatibility: ModelCompatibilityPlan,
    mut stages: Vec<ProbeStageReport>,
    message: String,
    duration_ms: u128,
) -> ModelRuntimeProbeReport {
    stages.push(stage(
        ProbeStageName::Generation,
        ProbeStageStatus::Failed,
        message,
        duration_ms,
    ));
    ModelRuntimeProbeReport {
        trace_id,
        model_id: model.id.clone(),
        backend_id,
        compatibility,
        stages,
        generated_preview: None,
        passed: false,
    }
}

fn generation_error_message(error: InferenceServiceError) -> String {
    format!("smoke generation failed: {error}")
}

fn stage(
    name: ProbeStageName,
    status: ProbeStageStatus,
    message: impl Into<String>,
    duration_ms: u128,
) -> ProbeStageReport {
    ProbeStageReport {
        name,
        status,
        message: message.into(),
        duration_ms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use crytex_inference::{
        BackendInfo, InferenceError, InferenceResponse, LoRAAdapter, ModelInfo, TokenUsage,
    };
    use std::sync::Mutex;
    use tempfile::tempdir;

    use crate::config::BackendKind;
    use crate::services::model_manager::ModelStatus;

    #[tokio::test]
    async fn unsupported_compatibility_skips_generation_and_reports_blocker() {
        let inference = Arc::new(RecordingInference::with_response("CRYTEX_PROBE_OK"));
        let probe = ModelRuntimeProbe::new(inference.clone());
        let model = model("tiny-random/qwen3-next-moe");
        let runtime = RuntimeFeatureSet {
            cuda_available: true,
            metal_available: false,
            gdn_cuda_available: false,
            cuda_unquantized_moe_fallback_available: true,
        };

        let report = probe
            .probe(
                &model,
                &cuda_device(),
                &runtime,
                ModelRuntimeProbeRequest {
                    backend_id: Some("mistralrs".into()),
                    model_name: "tiny-random/qwen3-next-moe".into(),
                    trace_id: Some("trace-probe".into()),
                    max_tokens: 8,
                    lora_adapter_id: None,
                },
            )
            .await;

        assert!(!report.passed);
        assert_eq!(report.trace_id, "trace-probe");
        assert_eq!(inference.requests.lock().unwrap().len(), 0);
        assert!(report.stages.iter().any(|stage| {
            stage.name == ProbeStageName::Compatibility
                && stage.status == ProbeStageStatus::Failed
                && stage.message.contains("GDN CUDA kernel")
        }));
        assert!(report.stages.iter().any(|stage| {
            stage.name == ProbeStageName::Generation && stage.status == ProbeStageStatus::Skipped
        }));
    }

    #[tokio::test]
    async fn supported_model_runs_short_generation_smoke_with_traceable_report() {
        let inference = Arc::new(RecordingInference::with_response("CRYTEX_PROBE_OK"));
        let probe = ModelRuntimeProbe::new(inference.clone());
        let model = model("tiny-random/qwen3-next-moe");

        let report = probe
            .probe(
                &model,
                &cuda_device(),
                &RuntimeFeatureSet::fully_enabled_cuda(),
                ModelRuntimeProbeRequest {
                    backend_id: Some("mistralrs".into()),
                    model_name: "tiny-random/qwen3-next-moe".into(),
                    trace_id: Some("trace-ok".into()),
                    max_tokens: 128,
                    lora_adapter_id: None,
                },
            )
            .await;

        let requests = inference.requests.lock().unwrap();
        assert!(report.passed);
        assert_eq!(report.trace_id, "trace-ok");
        assert_eq!(report.generated_preview.as_deref(), Some("CRYTEX_PROBE_OK"));
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].backend_id.as_deref(), Some("mistralrs"));
        assert_eq!(requests[0].max_tokens, Some(32));
        assert!(report.stages.iter().any(|stage| {
            stage.name == ProbeStageName::Generation && stage.status == ProbeStageStatus::Passed
        }));
    }

    #[tokio::test]
    async fn probe_passes_lora_adapter_id_to_generation_request() {
        let inference = Arc::new(RecordingInference::with_response("CRYTEX_PROBE_OK"));
        let probe = ModelRuntimeProbe::new(inference.clone());
        let model = model("tiny-coder");

        let report = probe
            .probe(
                &model,
                &cuda_device(),
                &RuntimeFeatureSet::fully_enabled_cuda(),
                ModelRuntimeProbeRequest {
                    backend_id: Some("mistralrs".into()),
                    model_name: "tiny-coder".into(),
                    trace_id: Some("trace-lora".into()),
                    max_tokens: 16,
                    lora_adapter_id: Some("coder-lora-v1".into()),
                },
            )
            .await;

        let requests = inference.requests.lock().unwrap();
        assert!(report.passed);
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].lora_adapter_id.as_deref(), Some("coder-lora-v1"));
    }

    #[tokio::test]
    async fn runtime_matrix_runs_baseline_and_lora_variants_as_separate_probe_entries() {
        let inference = Arc::new(RecordingInference::with_response("CRYTEX_PROBE_OK"));
        let matrix = ModelRuntimeMatrixProbe::new(inference.clone());
        let model = model("tiny-coder");

        let report = matrix
            .probe(
                &cuda_device(),
                &RuntimeFeatureSet::fully_enabled_cuda(),
                ModelRuntimeMatrixRequest {
                    trace_id: Some("trace-matrix".into()),
                    entries: vec![
                        RuntimeMatrixEntryRequest {
                            label: "baseline".into(),
                            model: model.clone(),
                            backend_id: Some("mistralrs".into()),
                            model_name: "tiny-coder".into(),
                            lora_adapter_id: None,
                            max_tokens: 16,
                        },
                        RuntimeMatrixEntryRequest {
                            label: "coder-lora".into(),
                            model: model.clone(),
                            backend_id: Some("mistralrs".into()),
                            model_name: "tiny-coder".into(),
                            lora_adapter_id: Some("coder-lora-v1".into()),
                            max_tokens: 16,
                        },
                    ],
                },
            )
            .await;

        let requests = inference.requests.lock().unwrap();
        assert!(report.passed);
        assert_eq!(report.trace_id, "trace-matrix");
        assert_eq!(report.entries.len(), 2);
        assert_eq!(report.entries[0].label, "baseline");
        assert_eq!(report.entries[1].label, "coder-lora");
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].lora_adapter_id, None);
        assert_eq!(requests[1].lora_adapter_id.as_deref(), Some("coder-lora-v1"));
        assert!(report.entries.iter().all(|entry| entry.report.passed));
    }

    #[tokio::test]
    async fn generation_failure_is_reported_as_failed_probe() {
        let inference = Arc::new(RecordingInference::with_error());
        let probe = ModelRuntimeProbe::new(inference);
        let model = model("tiny-random/qwen3-next-moe");

        let report = probe
            .probe(
                &model,
                &cuda_device(),
                &RuntimeFeatureSet::fully_enabled_cuda(),
                ModelRuntimeProbeRequest::smoke("tiny-random/qwen3-next-moe"),
            )
            .await;

        assert!(!report.passed);
        assert!(report.stages.iter().any(|stage| {
            stage.name == ProbeStageName::Generation
                && stage.status == ProbeStageStatus::Failed
                && stage.message.contains("backend exploded")
        }));
    }

    #[tokio::test]
    async fn unexpected_smoke_content_fails_probe_instead_of_passing_non_empty_output() {
        let inference = Arc::new(RecordingInference::with_response("hello from a model"));
        let probe = ModelRuntimeProbe::new(inference);
        let model = model("tiny-coder");

        let report = probe
            .probe(
                &model,
                &cuda_device(),
                &RuntimeFeatureSet::fully_enabled_cuda(),
                ModelRuntimeProbeRequest::smoke("tiny-coder"),
            )
            .await;

        assert!(!report.passed);
        assert_eq!(report.generated_preview, None);
        assert!(report.stages.iter().any(|stage| {
            stage.name == ProbeStageName::Generation
                && stage.status == ProbeStageStatus::Failed
                && stage.message.contains("missed expected sentinel")
        }));
    }

    #[test]
    fn runtime_matrix_report_writer_persists_pretty_json_artifact_with_safe_trace_filename() {
        let report_dir = tempdir().unwrap();
        let report = ModelRuntimeMatrixReport {
            trace_id: "trace/runtime matrix:01".into(),
            entries: vec![RuntimeMatrixEntryReport {
                label: "ollama:baseline".into(),
                lora_adapter_id: None,
                report: ModelRuntimeProbeReport {
                    trace_id: "trace/runtime matrix:01:ollama:baseline".into(),
                    model_id: "ollama-qwen".into(),
                    backend_id: Some("ollama".into()),
                    compatibility: ModelCompatibilityPlanner::plan(
                        &model("ollama-qwen"),
                        &cuda_device(),
                        &RuntimeFeatureSet::fully_enabled_cuda(),
                    ),
                    stages: vec![],
                    generated_preview: Some("CRYTEX_PROBE_OK".into()),
                    passed: true,
                },
            }],
            passed: true,
        };

        let path = RuntimeMatrixReportWriter::write_pretty_json(&report, report_dir.path()).unwrap();
        let json = std::fs::read_to_string(&path).unwrap();

        assert_eq!(
            path.file_name().and_then(|name| name.to_str()),
            Some("runtime-matrix-trace-runtime-matrix-01.json")
        );
        assert!(json.contains("\"trace_id\": \"trace/runtime matrix:01\""));
        assert!(json.contains("\"label\": \"ollama:baseline\""));
    }

    struct RecordingInference {
        requests: Mutex<Vec<InferenceRequest>>,
        response: RecordingResponse,
    }

    enum RecordingResponse {
        Success(String),
        Error,
    }

    impl RecordingInference {
        fn with_response(response: &str) -> Self {
            Self {
                requests: Mutex::new(Vec::new()),
                response: RecordingResponse::Success(response.into()),
            }
        }

        fn with_error() -> Self {
            Self {
                requests: Mutex::new(Vec::new()),
                response: RecordingResponse::Error,
            }
        }
    }

    #[async_trait]
    impl InferenceService for RecordingInference {
        async fn generate(
            &self,
            request: InferenceRequest,
        ) -> Result<InferenceResponse, InferenceServiceError> {
            self.requests.lock().unwrap().push(request);
            match &self.response {
                RecordingResponse::Success(content) => Ok(InferenceResponse {
                    content: content.clone(),
                    usage: TokenUsage {
                        prompt_tokens: 1,
                        completion_tokens: 1,
                        total_tokens: 2,
                    },
                    finish_reason: "stop".into(),
                }),
                RecordingResponse::Error => Err(InferenceServiceError::Inference(
                    InferenceError::GenerationFailed("backend exploded".into()),
                )),
            }
        }

        async fn embed(&self, _text: &str) -> Result<Vec<f32>, InferenceServiceError> {
            Ok(vec![])
        }

        fn available_backends(&self) -> Vec<BackendInfo> {
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

    fn model(id: &str) -> ManagedModel {
        ManagedModel {
            id: id.into(),
            name: id.into(),
            repo: None,
            filename: None,
            local_path: None,
            quantization: None,
            preferred_backend: BackendKind::MistralRs,
            params_b: None,
            status: ModelStatus::Available,
        }
    }

    fn cuda_device() -> DeviceKind {
        DeviceKind::Cuda {
            name: "RTX 5080".into(),
            vram_mb: 16_303,
            driver_version: "596.36".into(),
        }
    }
}
