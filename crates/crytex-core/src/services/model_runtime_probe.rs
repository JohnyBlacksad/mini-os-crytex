//! Runtime probe reports for selected models.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::{fs, io};

use crytex_inference::{BackendCapabilityReport, InferenceError, InferenceRequest, Message};
use serde::{Deserialize, Serialize};

use crate::services::hardware::DeviceKind;
use crate::services::model_compatibility::{
    CompatibilityStatus, ModelCompatibilityPlan, ModelCompatibilityPlanner, ModelSupportStatus,
    RuntimeFeatureSet,
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
    pub backend_capability: Option<BackendCapabilityReport>,
    pub compatibility: ModelCompatibilityPlan,
    pub stages: Vec<ProbeStageReport>,
    pub failure_reasons: Vec<String>,
    pub generated_preview: Option<String>,
    pub passed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRuntimeProbeRequest {
    pub backend_id: Option<String>,
    pub model_name: String,
    pub trace_id: Option<String>,
    pub max_tokens: usize,
    pub timeout_seconds: Option<u64>,
    pub lora_adapter_id: Option<String>,
}

impl ModelRuntimeProbeRequest {
    pub fn smoke(model_name: impl Into<String>) -> Self {
        Self {
            backend_id: None,
            model_name: model_name.into(),
            trace_id: None,
            max_tokens: 16,
            timeout_seconds: None,
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
        let backend_capability = self.backend_capability(request.backend_id.as_deref());

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
            let failure_reasons = compatibility_failure_reasons(&compatibility, &[]);
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
                backend_capability,
                compatibility,
                stages,
                failure_reasons,
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
        let generate = self.inference.generate(smoke_request(&request));
        let generation = match request.timeout_seconds {
            Some(seconds) => tokio::time::timeout(Duration::from_secs(seconds), generate)
                .await
                .map_err(|_| {
                    InferenceServiceError::Inference(InferenceError::GenerationFailed(format!(
                        "generation timed out after {seconds}s"
                    )))
                })
                .and_then(|result| result)
                .map(|response| response.content),
            None => generate.await.map(|response| response.content),
        };
        let duration_ms = started.elapsed().as_millis();

        match generation {
            Ok(content) if is_expected_smoke_response(&content) => {
                let preview = content.chars().take(512).collect::<String>();
                let failure_reasons = compatibility_failure_reasons(&compatibility, &[]);
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
                    backend_capability,
                    compatibility,
                    stages,
                    failure_reasons,
                    generated_preview: Some(preview),
                    passed: true,
                }
            }
            Ok(content) if !content.trim().is_empty() => passed_generation_report(
                SuccessfulProbeContext::new(
                    trace_id,
                    model,
                    request.backend_id,
                    backend_capability,
                    compatibility,
                    stages,
                ),
                format!(
                    "smoke generation missed expected sentinel CRYTEX_PROBE_OK: {}",
                    content.chars().take(120).collect::<String>()
                ),
                duration_ms,
            )
            .with_generated_preview(content.chars().take(512).collect::<String>()),
            Ok(_) => failed_generation_report(
                FailedProbeContext::new(
                    trace_id,
                    model,
                    request.backend_id,
                    backend_capability,
                    compatibility,
                    stages,
                ),
                "smoke generation returned empty output".to_string(),
                duration_ms,
            ),
            Err(error) => failed_generation_report(
                FailedProbeContext::new(
                    trace_id,
                    model,
                    request.backend_id,
                    backend_capability,
                    compatibility,
                    stages,
                ),
                generation_error_message(error),
                duration_ms,
            ),
        }
    }

    fn backend_capability(&self, backend_id: Option<&str>) -> Option<BackendCapabilityReport> {
        self.inference
            .available_backends()
            .into_iter()
            .find(|backend| backend_id.is_none_or(|id| backend.id == id))
            .map(|backend| backend.capability_report())
    }
}

fn is_expected_smoke_response(content: &str) -> bool {
    let normalized = content
        .trim()
        .trim_matches(|ch: char| ch == '"' || ch == '\'' || ch == '`')
        .to_ascii_uppercase();
    normalized
        .split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
        .any(|token| token == "CRYTEX_PROBE_OK")
}

fn smoke_request(request: &ModelRuntimeProbeRequest) -> InferenceRequest {
    InferenceRequest {
        backend_id: request.backend_id.clone(),
        model: request.model_name.clone(),
        messages: vec![Message {
            role: "user".into(),
            content: "No preamble. No explanation. Output exactly one token: CRYTEX_PROBE_OK"
                .into(),
        }],
        system_prompt: Some(
            "You are a deterministic runtime probe. The only valid response is CRYTEX_PROBE_OK."
                .into(),
        ),
        temperature: Some(0.0),
        max_tokens: Some(request.max_tokens.clamp(1, 32)),
        lora_adapter_id: request.lora_adapter_id.clone(),
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeMatrixEntryRequest {
    pub label: String,
    pub model: ManagedModel,
    pub device: Option<DeviceKind>,
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
    pub summary: RuntimeMatrixSummary,
    pub passed: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeMatrixSummary {
    pub supported: usize,
    pub partial: usize,
    pub unsupported: usize,
    pub failed: usize,
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
            let entry_device = entry.device.as_ref().unwrap_or(device);
            let report = self
                .probe
                .probe(
                    &entry.model,
                    entry_device,
                    runtime,
                    ModelRuntimeProbeRequest {
                        backend_id: entry.backend_id,
                        model_name: entry.model_name,
                        trace_id: Some(format!("{trace_id}:{}", entry.label)),
                        max_tokens: entry.max_tokens,
                        timeout_seconds: None,
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

        let summary = RuntimeMatrixSummary::from_entries(&entries);
        let passed = entries.iter().all(|entry| entry.report.passed);
        ModelRuntimeMatrixReport {
            trace_id,
            entries,
            summary,
            passed,
        }
    }
}

impl RuntimeMatrixSummary {
    fn from_entries(entries: &[RuntimeMatrixEntryReport]) -> Self {
        entries.iter().fold(Self::default(), |mut summary, entry| {
            match entry.report.compatibility.support_status {
                ModelSupportStatus::Supported => summary.supported += 1,
                ModelSupportStatus::Partial => summary.partial += 1,
                ModelSupportStatus::Unsupported => {
                    summary.unsupported += 1;
                }
            }
            if !entry.report.passed {
                summary.failed += 1;
            }
            summary
        })
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

struct SuccessfulProbeContext {
    trace_id: String,
    model_id: String,
    backend_id: Option<String>,
    backend_capability: Option<BackendCapabilityReport>,
    compatibility: ModelCompatibilityPlan,
    stages: Vec<ProbeStageReport>,
}

impl SuccessfulProbeContext {
    fn new(
        trace_id: String,
        model: &ManagedModel,
        backend_id: Option<String>,
        backend_capability: Option<BackendCapabilityReport>,
        compatibility: ModelCompatibilityPlan,
        stages: Vec<ProbeStageReport>,
    ) -> Self {
        Self {
            trace_id,
            model_id: model.id.clone(),
            backend_id,
            backend_capability,
            compatibility,
            stages,
        }
    }
}

type FailedProbeContext = SuccessfulProbeContext;

fn passed_generation_report(
    mut context: SuccessfulProbeContext,
    message: String,
    duration_ms: u128,
) -> ModelRuntimeProbeReport {
    context.stages.push(stage(
        ProbeStageName::Generation,
        ProbeStageStatus::Passed,
        message,
        duration_ms,
    ));
    let failure_reasons = compatibility_failure_reasons(&context.compatibility, &[]);
    ModelRuntimeProbeReport {
        trace_id: context.trace_id,
        model_id: context.model_id,
        backend_id: context.backend_id,
        backend_capability: context.backend_capability,
        compatibility: context.compatibility,
        stages: context.stages,
        failure_reasons,
        generated_preview: None,
        passed: true,
    }
}

fn failed_generation_report(
    mut context: FailedProbeContext,
    message: String,
    duration_ms: u128,
) -> ModelRuntimeProbeReport {
    let failure_reasons =
        compatibility_failure_reasons(&context.compatibility, std::slice::from_ref(&message));
    context.stages.push(stage(
        ProbeStageName::Generation,
        ProbeStageStatus::Failed,
        message,
        duration_ms,
    ));
    ModelRuntimeProbeReport {
        trace_id: context.trace_id,
        model_id: context.model_id,
        backend_id: context.backend_id,
        backend_capability: context.backend_capability,
        compatibility: context.compatibility,
        stages: context.stages,
        failure_reasons,
        generated_preview: None,
        passed: false,
    }
}

impl ModelRuntimeProbeReport {
    fn with_generated_preview(mut self, preview: String) -> Self {
        self.generated_preview = Some(preview);
        self
    }
}

fn generation_error_message(error: InferenceServiceError) -> String {
    format!("smoke generation failed: {error}")
}

fn compatibility_failure_reasons(
    compatibility: &ModelCompatibilityPlan,
    runtime_failures: &[String],
) -> Vec<String> {
    compatibility
        .failure_reasons
        .iter()
        .cloned()
        .chain(runtime_failures.iter().cloned())
        .collect()
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
                    timeout_seconds: None,
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
        assert_eq!(
            report.compatibility.support_status,
            ModelSupportStatus::Unsupported
        );
        assert!(
            report
                .failure_reasons
                .iter()
                .any(|reason| reason.contains("GDN CUDA kernel")),
            "expected GDN kernel blocker, got {:?}",
            report.failure_reasons
        );
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
                    timeout_seconds: None,
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
        assert!(requests[0].messages[0].content.contains("No preamble"));
        assert!(report.stages.iter().any(|stage| {
            stage.name == ProbeStageName::Generation && stage.status == ProbeStageStatus::Passed
        }));
        assert_eq!(
            report.compatibility.support_status,
            ModelSupportStatus::Supported
        );
        assert!(report.failure_reasons.is_empty());
    }

    #[tokio::test]
    async fn wrapped_case_variant_sentinel_counts_as_successful_smoke_generation() {
        let inference = Arc::new(RecordingInference::with_response(
            "Certainly! Here is the exact CRYTEx_PROBE_OK response:",
        ));
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

        assert!(report.passed);
        assert!(
            report
                .generated_preview
                .as_deref()
                .is_some_and(|preview| preview.contains("CRYTEx_PROBE_OK"))
        );
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
                    timeout_seconds: None,
                    lora_adapter_id: Some("coder-lora-v1".into()),
                },
            )
            .await;

        let requests = inference.requests.lock().unwrap();
        assert!(report.passed);
        assert_eq!(requests.len(), 1);
        assert_eq!(
            requests[0].lora_adapter_id.as_deref(),
            Some("coder-lora-v1")
        );
    }

    #[tokio::test]
    async fn probe_fails_generation_when_backend_exceeds_timeout() {
        let inference = Arc::new(RecordingInference::with_delay(50));
        let probe = ModelRuntimeProbe::new(inference);
        let model = model("tiny-coder");

        let report = probe
            .probe(
                &model,
                &cuda_device(),
                &RuntimeFeatureSet::fully_enabled_cuda(),
                ModelRuntimeProbeRequest {
                    backend_id: Some("mistralrs".into()),
                    model_name: "tiny-coder".into(),
                    trace_id: Some("trace-timeout".into()),
                    max_tokens: 16,
                    timeout_seconds: Some(0),
                    lora_adapter_id: None,
                },
            )
            .await;

        assert!(!report.passed);
        assert!(report.stages.iter().any(|stage| {
            stage.name == ProbeStageName::Generation
                && stage.status == ProbeStageStatus::Failed
                && stage.message.contains("timed out")
        }));
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
                            device: None,
                            backend_id: Some("mistralrs".into()),
                            model_name: "tiny-coder".into(),
                            lora_adapter_id: None,
                            max_tokens: 16,
                        },
                        RuntimeMatrixEntryRequest {
                            label: "coder-lora".into(),
                            model: model.clone(),
                            device: None,
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
        assert_eq!(
            report.entries[1]
                .report
                .backend_capability
                .as_ref()
                .map(|capability| (capability.lora, capability.hot_swap)),
            Some((true, true))
        );
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].lora_adapter_id, None);
        assert_eq!(
            requests[1].lora_adapter_id.as_deref(),
            Some("coder-lora-v1")
        );
        assert!(report.entries.iter().all(|entry| entry.report.passed));
        assert_eq!(
            report.summary,
            RuntimeMatrixSummary {
                supported: 2,
                partial: 0,
                unsupported: 0,
                failed: 0,
            }
        );
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
        assert!(
            report
                .failure_reasons
                .iter()
                .any(|reason| reason.contains("backend exploded")),
            "expected backend failure reason, got {:?}",
            report.failure_reasons
        );
    }

    #[tokio::test]
    async fn runtime_matrix_summarizes_supported_partial_unsupported_and_failed_paths() {
        let inference = Arc::new(RecordingInference::with_error_for("dense-fails"));
        let matrix = ModelRuntimeMatrixProbe::new(inference.clone());

        let report = matrix
            .probe(
                &DeviceKind::Cpu,
                &RuntimeFeatureSet {
                    cuda_available: true,
                    metal_available: false,
                    gdn_cuda_available: false,
                    cuda_unquantized_moe_fallback_available: true,
                },
                ModelRuntimeMatrixRequest {
                    trace_id: Some("trace-runtime-proof".into()),
                    entries: vec![
                        RuntimeMatrixEntryRequest {
                            label: "cpu-dense".into(),
                            model: model("tiny-coder"),
                            device: Some(DeviceKind::Cpu),
                            backend_id: Some("mistralrs".into()),
                            model_name: "tiny-coder".into(),
                            lora_adapter_id: None,
                            max_tokens: 16,
                        },
                        RuntimeMatrixEntryRequest {
                            label: "cpu-moe-gdn-partial".into(),
                            model: model("tiny-random/qwen3-next-moe"),
                            device: Some(DeviceKind::Cpu),
                            backend_id: Some("mistralrs".into()),
                            model_name: "tiny-random/qwen3-next-moe".into(),
                            lora_adapter_id: None,
                            max_tokens: 16,
                        },
                        RuntimeMatrixEntryRequest {
                            label: "gpu-gdn-unsupported".into(),
                            model: model("tiny-random/qwen3-next-gdn"),
                            device: Some(cuda_device()),
                            backend_id: Some("mistralrs".into()),
                            model_name: "tiny-random/qwen3-next-gdn".into(),
                            lora_adapter_id: None,
                            max_tokens: 16,
                        },
                        RuntimeMatrixEntryRequest {
                            label: "dense-fails".into(),
                            model: model("dense-fails"),
                            device: Some(DeviceKind::Cpu),
                            backend_id: Some("mistralrs".into()),
                            model_name: "dense-fails".into(),
                            lora_adapter_id: None,
                            max_tokens: 16,
                        },
                    ],
                },
            )
            .await;

        assert!(!report.passed);
        assert_eq!(
            report.summary,
            RuntimeMatrixSummary {
                supported: 2,
                partial: 1,
                unsupported: 1,
                failed: 2,
            }
        );
        assert_eq!(
            support_status_for(&report, "cpu-moe-gdn-partial"),
            Some(ModelSupportStatus::Partial)
        );
        assert_eq!(
            support_status_for(&report, "gpu-gdn-unsupported"),
            Some(ModelSupportStatus::Unsupported)
        );
        assert!(
            report
                .entries
                .iter()
                .find(|entry| entry.label == "cpu-moe-gdn-partial")
                .is_some_and(|entry| entry
                    .report
                    .failure_reasons
                    .iter()
                    .any(|reason| reason.contains("CPU MoE/GDN execution")))
        );
        assert!(
            report
                .entries
                .iter()
                .find(|entry| entry.label == "gpu-gdn-unsupported")
                .is_some_and(|entry| entry
                    .report
                    .failure_reasons
                    .iter()
                    .any(|reason| reason.contains("GDN CUDA kernel")))
        );
    }

    #[tokio::test]
    async fn unexpected_smoke_content_passes_probe_with_sentinel_miss_diagnostic() {
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

        assert!(report.passed);
        assert_eq!(
            report.generated_preview.as_deref(),
            Some("hello from a model")
        );
        assert!(report.stages.iter().any(|stage| {
            stage.name == ProbeStageName::Generation
                && stage.status == ProbeStageStatus::Passed
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
                    backend_capability: None,
                    compatibility: ModelCompatibilityPlanner::plan(
                        &model("ollama-qwen"),
                        &cuda_device(),
                        &RuntimeFeatureSet::fully_enabled_cuda(),
                    ),
                    stages: vec![],
                    failure_reasons: vec![],
                    generated_preview: Some("CRYTEX_PROBE_OK".into()),
                    passed: true,
                },
            }],
            summary: RuntimeMatrixSummary {
                supported: 1,
                partial: 0,
                unsupported: 0,
                failed: 0,
            },
            passed: true,
        };

        let path =
            RuntimeMatrixReportWriter::write_pretty_json(&report, report_dir.path()).unwrap();
        let json = std::fs::read_to_string(&path).unwrap();

        assert_eq!(
            path.file_name().and_then(|name| name.to_str()),
            Some("runtime-matrix-trace-runtime-matrix-01.json")
        );
        assert!(json.contains("\"trace_id\": \"trace/runtime matrix:01\""));
        assert!(json.contains("\"label\": \"ollama:baseline\""));
        assert!(json.contains("\"summary\""));
        assert!(json.contains("\"supported\": 1"));
    }

    struct RecordingInference {
        requests: Mutex<Vec<InferenceRequest>>,
        response: RecordingResponse,
    }

    enum RecordingResponse {
        Success(String),
        Error,
        ErrorFor(String),
        Delay(u64),
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

        fn with_error_for(model_name: &str) -> Self {
            Self {
                requests: Mutex::new(Vec::new()),
                response: RecordingResponse::ErrorFor(model_name.into()),
            }
        }

        fn with_delay(delay_ms: u64) -> Self {
            Self {
                requests: Mutex::new(Vec::new()),
                response: RecordingResponse::Delay(delay_ms),
            }
        }
    }

    #[async_trait]
    impl InferenceService for RecordingInference {
        async fn generate(
            &self,
            request: InferenceRequest,
        ) -> Result<InferenceResponse, InferenceServiceError> {
            let request_model = request.model.clone();
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
                RecordingResponse::ErrorFor(model_name) if request_model == *model_name => Err(
                    InferenceServiceError::Inference(InferenceError::GenerationFailed(format!(
                        "backend exploded for {}",
                        request_model
                    ))),
                ),
                RecordingResponse::ErrorFor(_) => Ok(InferenceResponse {
                    content: "CRYTEX_PROBE_OK".into(),
                    usage: TokenUsage {
                        prompt_tokens: 1,
                        completion_tokens: 1,
                        total_tokens: 2,
                    },
                    finish_reason: "stop".into(),
                }),
                RecordingResponse::Delay(delay_ms) => {
                    tokio::time::sleep(Duration::from_millis(*delay_ms)).await;
                    Ok(InferenceResponse {
                        content: "CRYTEX_PROBE_OK".into(),
                        usage: TokenUsage {
                            prompt_tokens: 1,
                            completion_tokens: 1,
                            total_tokens: 2,
                        },
                        finish_reason: "stop".into(),
                    })
                }
            }
        }

        async fn embed(&self, _text: &str) -> Result<Vec<f32>, InferenceServiceError> {
            Ok(vec![])
        }

        fn available_backends(&self) -> Vec<BackendInfo> {
            vec![BackendInfo {
                id: "mistralrs".into(),
                name: "mistral.rs".into(),
                capabilities: vec![
                    "generate".into(),
                    "chat".into(),
                    "lora".into(),
                    "hot_swap".into(),
                ],
            }]
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

    fn support_status_for(
        report: &ModelRuntimeMatrixReport,
        label: &str,
    ) -> Option<ModelSupportStatus> {
        report
            .entries
            .iter()
            .find(|entry| entry.label == label)
            .map(|entry| entry.report.compatibility.support_status)
    }
}
