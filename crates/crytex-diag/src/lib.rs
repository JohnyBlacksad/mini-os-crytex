use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use crytex_inference::{BackendInfo, InferenceError, InferenceManager, InferenceRequest, Message};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum DiagBackendKind {
    Ollama,
    OpenAiCompatible,
    Anthropic,
    MistralRs,
    Onnx,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagBackendConfig {
    pub id: String,
    pub kind: DiagBackendKind,
    pub model: String,
    pub url: Option<String>,
    pub api_key: Option<String>,
    #[serde(default)]
    pub headers: HashMap<String, String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiagInferenceConfig {
    pub default_backend: Option<String>,
    #[serde(default)]
    pub backends: Vec<DiagBackendConfig>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiagPathsConfig {
    #[serde(default = "default_data_dir")]
    pub data_dir: PathBuf,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiagConfig {
    #[serde(default)]
    pub inference: DiagInferenceConfig,
    #[serde(default)]
    pub paths: DiagPathsConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BackendSummary {
    pub id: String,
    pub kind: DiagBackendKind,
    pub model: String,
    pub url: Option<String>,
    pub is_default: bool,
    pub supported_by_lightweight_diag: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BackendsReport {
    pub default_backend: Option<String>,
    pub backends: Vec<BackendSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DoctorCheckStatus {
    Passed,
    Failed,
    Warning,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DoctorCheck {
    pub name: String,
    pub status: DoctorCheckStatus,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DoctorReport {
    pub passed: bool,
    pub checks: Vec<DoctorCheck>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DoctorOptions {
    pub require_gpu: bool,
}

impl DiagConfig {
    pub fn load() -> Self {
        std::fs::read_to_string(default_config_path())
            .ok()
            .and_then(|contents| toml::from_str(&contents).ok())
            .unwrap_or_default()
    }

    pub fn backend(&self, id: &str) -> Option<&DiagBackendConfig> {
        self.inference
            .backends
            .iter()
            .find(|backend| backend.id == id)
    }
}

pub fn list_backends(config: &DiagConfig) -> BackendsReport {
    BackendsReport {
        default_backend: config.inference.default_backend.clone(),
        backends: config
            .inference
            .backends
            .iter()
            .map(|backend| BackendSummary {
                id: backend.id.clone(),
                kind: backend.kind,
                model: backend.model.clone(),
                url: backend.url.clone(),
                is_default: config.inference.default_backend.as_deref() == Some(&backend.id),
                supported_by_lightweight_diag: is_lightweight_supported_backend(backend.kind),
            })
            .collect(),
    }
}

pub fn doctor(config: &DiagConfig, requested_backend_ids: &[String]) -> DoctorReport {
    let backend_ids = configured_backend_ids(config, requested_backend_ids);
    let mut checks = Vec::new();

    checks.push(if config.inference.backends.is_empty() {
        doctor_check(
            "backends_configured",
            DoctorCheckStatus::Failed,
            "no inference backends are configured",
        )
    } else {
        doctor_check(
            "backends_configured",
            DoctorCheckStatus::Passed,
            format!(
                "{} inference backend(s) configured",
                config.inference.backends.len()
            ),
        )
    });

    checks.push(if backend_ids.is_empty() {
        doctor_check(
            "backend_selection",
            DoctorCheckStatus::Failed,
            "no backend was requested and no default backend is configured",
        )
    } else {
        doctor_check(
            "backend_selection",
            DoctorCheckStatus::Passed,
            format!("selected backend(s): {}", backend_ids.join(", ")),
        )
    });

    checks.push(match config.inference.default_backend.as_deref() {
        Some(default_backend) if config.backend(default_backend).is_some() => doctor_check(
            "default_backend",
            DoctorCheckStatus::Passed,
            format!("default backend is configured: {default_backend}"),
        ),
        Some(default_backend) => doctor_check(
            "default_backend",
            DoctorCheckStatus::Failed,
            format!("default backend {default_backend} is not configured in backend list"),
        ),
        None => doctor_check(
            "default_backend",
            DoctorCheckStatus::Warning,
            "default backend is not configured; pass --backend explicitly",
        ),
    });

    for backend_id in &backend_ids {
        match config.backend(backend_id) {
            Some(backend) if is_lightweight_supported_backend(backend.kind) => checks.push(
                doctor_check(
                    format!("backend_supported:{backend_id}"),
                    DoctorCheckStatus::Passed,
                    format!("backend {backend_id} can be probed by crytex-diag"),
                ),
            ),
            Some(backend) => checks.push(doctor_check(
                format!("backend_supported:{backend_id}"),
                DoctorCheckStatus::Failed,
                format!(
                    "backend {backend_id} has kind {:?}, which is not supported by lightweight diagnostics",
                    backend.kind
                ),
            )),
            None => checks.push(doctor_check(
                format!("backend_exists:{backend_id}"),
                DoctorCheckStatus::Failed,
                format!("backend {backend_id} is not configured"),
            )),
        }
    }

    checks.push(cuda_toolchain_doctor_check(
        detect_cuda_toolchain_presence(),
        false,
    ));

    let passed = checks
        .iter()
        .all(|check| check.status != DoctorCheckStatus::Failed);
    DoctorReport { passed, checks }
}

pub async fn doctor_with_api_checks(
    config: &DiagConfig,
    requested_backend_ids: &[String],
) -> DoctorReport {
    doctor_with_api_checks_and_options(config, requested_backend_ids, DoctorOptions::default())
        .await
}

pub async fn doctor_with_api_checks_and_options(
    config: &DiagConfig,
    requested_backend_ids: &[String],
    options: DoctorOptions,
) -> DoctorReport {
    let mut report = doctor(config, requested_backend_ids);
    if let Some(check) = report
        .checks
        .iter_mut()
        .find(|check| check.name == "cuda_toolchain_preflight")
    {
        *check = cuda_toolchain_doctor_check(detect_cuda_toolchain_presence(), options.require_gpu);
    }
    let backend_ids = configured_backend_ids(config, requested_backend_ids);

    for backend_id in backend_ids {
        if let Some(backend) = config.backend(&backend_id)
            && backend.kind == DiagBackendKind::Ollama
        {
            report.checks.push(check_ollama_tags_api(backend).await);
            report
                .checks
                .push(check_ollama_ps_api(backend, options.require_gpu).await);
        }
    }

    report.passed = report
        .checks
        .iter()
        .all(|check| check.status != DoctorCheckStatus::Failed);
    report
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CudaToolchainPresence {
    nvidia_smi: bool,
    nvcc: bool,
    runtime_library: bool,
}

fn detect_cuda_toolchain_presence() -> CudaToolchainPresence {
    CudaToolchainPresence {
        nvidia_smi: command_succeeds("nvidia-smi", &["--query-gpu=name", "--format=csv,noheader"]),
        nvcc: command_succeeds("nvcc", &["--version"]),
        runtime_library: cuda_runtime_library_visible(),
    }
}

fn command_succeeds(program: &str, args: &[&str]) -> bool {
    std::process::Command::new(program)
        .args(args)
        .output()
        .is_ok_and(|output| output.status.success())
}

fn cuda_runtime_library_visible() -> bool {
    std::env::var_os("PATH")
        .into_iter()
        .flat_map(|paths| std::env::split_paths(&paths).collect::<Vec<_>>())
        .any(|path| {
            [
                "cudart64_12.dll",
                "cudart64_11.dll",
                "libcudart.so",
                "libcudart.dylib",
            ]
            .iter()
            .any(|name| path.join(name).exists())
        })
}

fn cuda_toolchain_doctor_check(presence: CudaToolchainPresence, require_gpu: bool) -> DoctorCheck {
    let toolchain_visible = presence.nvcc || presence.runtime_library;
    match (presence.nvidia_smi, toolchain_visible, require_gpu) {
        (true, true, _) => doctor_check(
            "cuda_toolchain_preflight",
            DoctorCheckStatus::Passed,
            "CUDA GPU and compiler/runtime library are visible",
        ),
        (true, false, _) => doctor_check(
            "cuda_toolchain_preflight",
            DoctorCheckStatus::Warning,
            "nvidia-smi is available, but nvcc/CUDA runtime library was not detected in PATH",
        ),
        (false, _, true) => doctor_check(
            "cuda_toolchain_preflight",
            DoctorCheckStatus::Failed,
            "CUDA GPU was required, but nvidia-smi is not available",
        ),
        (false, true, false) => doctor_check(
            "cuda_toolchain_preflight",
            DoctorCheckStatus::Warning,
            "CUDA runtime/compiler is visible, but nvidia-smi is not available",
        ),
        (false, false, false) => doctor_check(
            "cuda_toolchain_preflight",
            DoctorCheckStatus::Warning,
            "CUDA is not visible; local GPU backends will degrade to CPU or unsupported status",
        ),
    }
}

async fn check_ollama_ps_api(backend: &DiagBackendConfig, require_gpu: bool) -> DoctorCheck {
    let url = backend
        .url
        .clone()
        .unwrap_or_else(|| "http://localhost:11434".to_string())
        .trim_end_matches('/')
        .to_string();
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
    {
        Ok(client) => client,
        Err(error) => {
            return doctor_check(
                format!("ollama_runtime_placement:{}", backend.id),
                DoctorCheckStatus::Failed,
                format!("failed to build HTTP client for Ollama: {error}"),
            );
        }
    };

    let response = match client.get(format!("{url}/api/ps")).send().await {
        Ok(response) => response,
        Err(error) => {
            return doctor_check(
                format!("ollama_runtime_placement:{}", backend.id),
                DoctorCheckStatus::Failed,
                format!("Ollama /api/ps request failed: {error}"),
            );
        }
    };

    if !response.status().is_success() {
        return doctor_check(
            format!("ollama_runtime_placement:{}", backend.id),
            DoctorCheckStatus::Failed,
            format!("Ollama /api/ps returned HTTP {}", response.status()),
        );
    }

    match response.json::<serde_json::Value>().await {
        Ok(ps) => ollama_ps_model_placement_check(&backend.id, &backend.model, &ps, require_gpu),
        Err(error) => doctor_check(
            format!("ollama_runtime_placement:{}", backend.id),
            DoctorCheckStatus::Failed,
            format!("failed to parse Ollama /api/ps JSON: {error}"),
        ),
    }
}

async fn check_ollama_tags_api(backend: &DiagBackendConfig) -> DoctorCheck {
    let url = backend
        .url
        .clone()
        .unwrap_or_else(|| "http://localhost:11434".to_string())
        .trim_end_matches('/')
        .to_string();
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
    {
        Ok(client) => client,
        Err(error) => {
            return doctor_check(
                format!("ollama_tags_api:{}", backend.id),
                DoctorCheckStatus::Failed,
                format!("failed to build HTTP client for Ollama: {error}"),
            );
        }
    };

    let response = match client.get(format!("{url}/api/tags")).send().await {
        Ok(response) => response,
        Err(error) => {
            return doctor_check(
                format!("ollama_tags_api:{}", backend.id),
                DoctorCheckStatus::Failed,
                format!("Ollama /api/tags request failed: {error}"),
            );
        }
    };

    if !response.status().is_success() {
        return doctor_check(
            format!("ollama_tags_api:{}", backend.id),
            DoctorCheckStatus::Failed,
            format!("Ollama /api/tags returned HTTP {}", response.status()),
        );
    }

    let tags = match response.json::<serde_json::Value>().await {
        Ok(tags) => tags,
        Err(error) => {
            return doctor_check(
                format!("ollama_tags_api:{}", backend.id),
                DoctorCheckStatus::Failed,
                format!("failed to parse Ollama /api/tags JSON: {error}"),
            );
        }
    };
    let model_names = ollama_tags_model_names(&tags);

    if model_names.iter().any(|model| model == &backend.model) {
        doctor_check(
            format!("ollama_model_present:{}", backend.id),
            DoctorCheckStatus::Passed,
            format!("Ollama model {} is available locally", backend.model),
        )
    } else {
        doctor_check(
            format!("ollama_model_present:{}", backend.id),
            DoctorCheckStatus::Failed,
            format!(
                "Ollama model {} is not present locally; available models: {}",
                backend.model,
                model_names.join(", ")
            ),
        )
    }
}

fn doctor_check(
    name: impl Into<String>,
    status: DoctorCheckStatus,
    message: impl Into<String>,
) -> DoctorCheck {
    DoctorCheck {
        name: name.into(),
        status,
        message: message.into(),
    }
}

fn is_lightweight_supported_backend(kind: DiagBackendKind) -> bool {
    matches!(
        kind,
        DiagBackendKind::Ollama
            | DiagBackendKind::OpenAiCompatible
            | DiagBackendKind::Anthropic
            | DiagBackendKind::Custom
    )
}

fn ollama_tags_model_names(tags: &serde_json::Value) -> Vec<String> {
    tags.get("models")
        .and_then(|models| models.as_array())
        .into_iter()
        .flatten()
        .filter_map(|entry| {
            entry
                .get("name")
                .or_else(|| entry.get("model"))
                .and_then(|name| name.as_str())
                .map(str::to_string)
        })
        .collect()
}

fn ollama_ps_model_placement_check(
    backend_id: &str,
    model: &str,
    ps: &serde_json::Value,
    require_gpu: bool,
) -> DoctorCheck {
    let placement = ollama_runtime_placement_report(backend_id, model, ps);

    match placement.kind {
        RuntimePlacementKind::Gpu => {
            let size_vram = placement.size_vram_bytes.unwrap_or(0);
            let size = placement.size_bytes.unwrap_or(0);
            doctor_check(
                format!("ollama_runtime_placement:{backend_id}"),
                DoctorCheckStatus::Passed,
                format!(
                    "Ollama model {model} is loaded with {size_vram} bytes in VRAM out of {size} bytes"
                ),
            )
        }
        RuntimePlacementKind::Cpu if require_gpu => doctor_check(
            format!("ollama_runtime_placement:{backend_id}"),
            DoctorCheckStatus::Failed,
            format!("Ollama model {model} is loaded on CPU only; --require-gpu was set"),
        ),
        RuntimePlacementKind::Cpu => doctor_check(
            format!("ollama_runtime_placement:{backend_id}"),
            DoctorCheckStatus::Warning,
            format!("Ollama model {model} is loaded without reported VRAM usage"),
        ),
        RuntimePlacementKind::NotLoaded => doctor_check(
            format!("ollama_runtime_placement:{backend_id}"),
            DoctorCheckStatus::Warning,
            format!(
                "Ollama model {model} is not currently loaded; run a probe before checking GPU placement"
            ),
        ),
        RuntimePlacementKind::Unknown => doctor_check(
            format!("ollama_runtime_placement:{backend_id}"),
            DoctorCheckStatus::Warning,
            placement.message,
        ),
    }
}

fn ollama_runtime_placement_report(
    backend_id: &str,
    model: &str,
    ps: &serde_json::Value,
) -> RuntimePlacementReport {
    let Some(entry) = ps
        .get("models")
        .and_then(|models| models.as_array())
        .into_iter()
        .flatten()
        .find(|entry| ollama_model_entry_matches(entry, model))
    else {
        return RuntimePlacementReport {
            backend_id: backend_id.to_string(),
            model: model.to_string(),
            kind: RuntimePlacementKind::NotLoaded,
            size_bytes: None,
            size_vram_bytes: None,
            message: format!("Ollama model {model} is not currently loaded"),
        };
    };

    let size_vram = entry.get("size_vram").and_then(|value| value.as_u64());
    let size = entry.get("size").and_then(|value| value.as_u64());

    match size_vram {
        Some(value) => RuntimePlacementReport {
            backend_id: backend_id.to_string(),
            model: model.to_string(),
            kind: if value > 0 {
                RuntimePlacementKind::Gpu
            } else {
                RuntimePlacementKind::Cpu
            },
            size_bytes: size,
            size_vram_bytes: Some(value),
            message: if value > 0 {
                format!("Ollama model {model} reports VRAM usage")
            } else {
                format!("Ollama model {model} reports no VRAM usage")
            },
        },
        None => RuntimePlacementReport {
            backend_id: backend_id.to_string(),
            model: model.to_string(),
            kind: RuntimePlacementKind::Unknown,
            size_bytes: size,
            size_vram_bytes: None,
            message: format!("Ollama model {model} did not report size_vram"),
        },
    }
}

fn ollama_model_entry_matches(entry: &serde_json::Value, model: &str) -> bool {
    entry
        .get("name")
        .or_else(|| entry.get("model"))
        .and_then(|name| name.as_str())
        == Some(model)
}

fn default_config_path() -> PathBuf {
    dirs::config_dir()
        .map(|dir| dir.join("crytex").join("config.toml"))
        .unwrap_or_else(|| PathBuf::from(".crytex").join("config.toml"))
}

fn default_data_dir() -> PathBuf {
    dirs::data_dir()
        .map(|dir| dir.join("crytex"))
        .unwrap_or_else(|| PathBuf::from(".crytex"))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeMatrixEntryRequest {
    pub label: String,
    pub model_id: String,
    pub backend_id: String,
    pub model_name: String,
    pub lora_adapter_id: Option<String>,
    pub max_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProbeStageName {
    Metadata,
    Compatibility,
    Generation,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProbeStageStatus {
    Passed,
    Failed,
    Skipped,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProbeStageReport {
    pub name: ProbeStageName,
    pub status: ProbeStageStatus,
    pub message: String,
    pub duration_ms: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompatibilityReport {
    pub format: String,
    pub features: Vec<String>,
    pub strategy: String,
    pub status: String,
    pub support_status: String,
    pub actions: Vec<String>,
    pub warnings: Vec<String>,
    pub blockers: Vec<String>,
    pub failure_reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RuntimePlacementKind {
    Gpu,
    Cpu,
    NotLoaded,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimePlacementReport {
    pub backend_id: String,
    pub model: String,
    pub kind: RuntimePlacementKind,
    pub size_bytes: Option<u64>,
    pub size_vram_bytes: Option<u64>,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeProbeReport {
    pub trace_id: String,
    pub model_id: String,
    pub backend_id: String,
    pub backend_capability: Option<BackendCapabilitySnapshot>,
    pub compatibility: CompatibilityReport,
    pub stages: Vec<ProbeStageReport>,
    pub failure_reasons: Vec<String>,
    pub runtime_placement: Option<RuntimePlacementReport>,
    pub generated_preview: Option<String>,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BackendCapabilitySnapshot {
    pub id: String,
    pub name: String,
    pub generate: bool,
    pub chat: bool,
    pub embed: bool,
    pub rerank: bool,
    pub lora: bool,
    pub hot_swap: bool,
}

impl BackendCapabilitySnapshot {
    fn from_info(info: BackendInfo) -> Self {
        let report = info.capability_report();
        Self {
            id: report.id,
            name: report.name,
            generate: report.generate,
            chat: report.chat,
            embed: report.embed,
            rerank: report.rerank,
            lora: report.lora,
            hot_swap: report.hot_swap,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeMatrixEntryReport {
    pub label: String,
    pub lora_adapter_id: Option<String>,
    pub report: RuntimeProbeReport,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeMatrixReport {
    pub trace_id: String,
    pub entries: Vec<RuntimeMatrixEntryReport>,
    pub summary: RuntimeMatrixSummary,
    pub passed: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeMatrixSummary {
    pub supported: usize,
    pub partial: usize,
    pub unsupported: usize,
    pub failed: usize,
}

pub fn build_runtime_matrix_entries(
    backend_ids: &[String],
    lora_adapter_ids: &[String],
    model_id: &str,
    model_name: &str,
    max_tokens: usize,
) -> Vec<RuntimeMatrixEntryRequest> {
    let lora_variants = std::iter::once(None)
        .chain(lora_adapter_ids.iter().cloned().map(Some))
        .collect::<Vec<_>>();

    let mut entries = Vec::new();
    for backend_id in backend_ids {
        for lora_adapter_id in &lora_variants {
            let variant = lora_adapter_id.as_deref().unwrap_or("baseline");
            entries.push(RuntimeMatrixEntryRequest {
                label: format!("{backend_id}:{variant}"),
                model_id: model_id.to_string(),
                backend_id: backend_id.clone(),
                model_name: model_name.to_string(),
                lora_adapter_id: lora_adapter_id.clone(),
                max_tokens,
            });
        }
    }
    entries
}

pub fn configured_backend_ids(config: &DiagConfig, requested: &[String]) -> Vec<String> {
    if requested.is_empty() {
        config
            .inference
            .default_backend
            .as_ref()
            .map(|backend| vec![backend.clone()])
            .unwrap_or_default()
    } else {
        requested.to_vec()
    }
}

pub fn runtime_model_name(
    config: &DiagConfig,
    backend_id: &str,
    requested_model: Option<&str>,
    fallback_model_id: &str,
) -> String {
    requested_model
        .map(str::to_string)
        .or_else(|| {
            config
                .backend(backend_id)
                .map(|backend| backend.model.clone())
        })
        .unwrap_or_else(|| fallback_model_id.to_string())
}

pub async fn run_runtime_matrix(
    trace_id: String,
    entries: Vec<RuntimeMatrixEntryRequest>,
    backends: &HashMap<String, Arc<dyn InferenceManager>>,
) -> RuntimeMatrixReport {
    run_runtime_matrix_with_config(trace_id, entries, backends, None).await
}

pub async fn run_runtime_matrix_with_config(
    trace_id: String,
    entries: Vec<RuntimeMatrixEntryRequest>,
    backends: &HashMap<String, Arc<dyn InferenceManager>>,
    config: Option<&DiagConfig>,
) -> RuntimeMatrixReport {
    let mut reports = Vec::new();
    for entry in entries {
        let report = run_probe(&trace_id, &entry, backends, config).await;
        reports.push(RuntimeMatrixEntryReport {
            label: entry.label,
            lora_adapter_id: entry.lora_adapter_id,
            report,
        });
    }
    let passed = reports.iter().all(|entry| entry.report.passed);
    let summary = RuntimeMatrixSummary::from_entries(&reports);
    RuntimeMatrixReport {
        trace_id,
        entries: reports,
        summary,
        passed,
    }
}

impl RuntimeMatrixSummary {
    fn from_entries(entries: &[RuntimeMatrixEntryReport]) -> Self {
        entries.iter().fold(Self::default(), |mut summary, entry| {
            match entry.report.compatibility.support_status.as_str() {
                "supported" => summary.supported += 1,
                "partial" => summary.partial += 1,
                "unsupported" => summary.unsupported += 1,
                _ => summary.unsupported += 1,
            }
            if !entry.report.passed {
                summary.failed += 1;
            }
            summary
        })
    }
}

async fn run_probe(
    trace_id: &str,
    entry: &RuntimeMatrixEntryRequest,
    backends: &HashMap<String, Arc<dyn InferenceManager>>,
    config: Option<&DiagConfig>,
) -> RuntimeProbeReport {
    let entry_trace_id = format!("{trace_id}:{}", entry.label);
    let compatibility = remote_compatibility();
    let mut stages = vec![
        stage(
            ProbeStageName::Metadata,
            ProbeStageStatus::Passed,
            "detected format Unknown with features [Dense]",
            0,
        ),
        stage(
            ProbeStageName::Compatibility,
            ProbeStageStatus::Passed,
            "selected execution strategy Remote",
            0,
        ),
    ];
    let backend_capability = backends
        .get(&entry.backend_id)
        .and_then(|backend| backend_capability_snapshot(backend.as_ref(), &entry.backend_id));

    let Some(backend) = backends.get(&entry.backend_id) else {
        let message = format!("backend {} is not registered", entry.backend_id);
        let compatibility = unsupported_compatibility(message.clone());
        stages.push(stage(
            ProbeStageName::Generation,
            ProbeStageStatus::Failed,
            message,
            0,
        ));
        let failure_reasons = compatibility_failure_reasons(&compatibility, &[]);
        return RuntimeProbeReport {
            trace_id: entry_trace_id,
            model_id: entry.model_id.clone(),
            backend_id: entry.backend_id.clone(),
            backend_capability,
            compatibility,
            stages,
            failure_reasons,
            runtime_placement: None,
            generated_preview: None,
            passed: false,
        };
    };

    let started = Instant::now();
    let generation = backend.generate(smoke_request(entry)).await;
    let duration_ms = started.elapsed().as_millis();

    match generation {
        Ok(response) if is_expected_smoke_response(&response.content) => {
            let preview = response.content.chars().take(512).collect::<String>();
            stages.push(stage(
                ProbeStageName::Generation,
                ProbeStageStatus::Passed,
                "smoke generation returned expected sentinel",
                duration_ms,
            ));
            let runtime_placement = runtime_placement_after_generation(config, entry).await;
            let failure_reasons = compatibility_failure_reasons(&compatibility, &[]);
            RuntimeProbeReport {
                trace_id: entry_trace_id,
                model_id: entry.model_id.clone(),
                backend_id: entry.backend_id.clone(),
                backend_capability,
                compatibility,
                stages,
                failure_reasons,
                runtime_placement,
                generated_preview: Some(preview),
                passed: true,
            }
        }
        Ok(response) => failed_probe(
            entry,
            entry_trace_id,
            compatibility,
            stages,
            backend_capability,
            format!(
                "smoke generation missed expected sentinel CRYTEX_PROBE_OK: {}",
                response.content.chars().take(120).collect::<String>()
            ),
            duration_ms,
        ),
        Err(error) => failed_probe(
            entry,
            entry_trace_id,
            compatibility,
            stages,
            backend_capability,
            format!("smoke generation failed: {error}"),
            duration_ms,
        ),
    }
}

fn failed_probe(
    entry: &RuntimeMatrixEntryRequest,
    trace_id: String,
    compatibility: CompatibilityReport,
    mut stages: Vec<ProbeStageReport>,
    backend_capability: Option<BackendCapabilitySnapshot>,
    message: String,
    duration_ms: u128,
) -> RuntimeProbeReport {
    let failure_reasons =
        compatibility_failure_reasons(&compatibility, std::slice::from_ref(&message));
    stages.push(stage(
        ProbeStageName::Generation,
        ProbeStageStatus::Failed,
        message,
        duration_ms,
    ));
    RuntimeProbeReport {
        trace_id,
        model_id: entry.model_id.clone(),
        backend_id: entry.backend_id.clone(),
        backend_capability,
        compatibility,
        stages,
        failure_reasons,
        runtime_placement: None,
        generated_preview: None,
        passed: false,
    }
}

fn backend_capability_snapshot(
    backend: &dyn InferenceManager,
    backend_id: &str,
) -> Option<BackendCapabilitySnapshot> {
    backend
        .available_backends()
        .into_iter()
        .find(|info| info.id == backend_id)
        .map(BackendCapabilitySnapshot::from_info)
}

async fn runtime_placement_after_generation(
    config: Option<&DiagConfig>,
    entry: &RuntimeMatrixEntryRequest,
) -> Option<RuntimePlacementReport> {
    let backend = config.and_then(|config| config.backend(&entry.backend_id))?;
    if backend.kind != DiagBackendKind::Ollama {
        return None;
    }
    fetch_ollama_runtime_placement(backend).await.ok()
}

async fn fetch_ollama_runtime_placement(
    backend: &DiagBackendConfig,
) -> Result<RuntimePlacementReport, String> {
    let url = backend
        .url
        .clone()
        .unwrap_or_else(|| "http://localhost:11434".to_string())
        .trim_end_matches('/')
        .to_string();
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .map_err(|error| format!("failed to build HTTP client for Ollama: {error}"))?;
    let response = client
        .get(format!("{url}/api/ps"))
        .send()
        .await
        .map_err(|error| format!("Ollama /api/ps request failed: {error}"))?;

    if !response.status().is_success() {
        return Err(format!(
            "Ollama /api/ps returned HTTP {}",
            response.status()
        ));
    }

    response
        .json::<serde_json::Value>()
        .await
        .map(|ps| ollama_runtime_placement_report(&backend.id, &backend.model, &ps))
        .map_err(|error| format!("failed to parse Ollama /api/ps JSON: {error}"))
}

fn smoke_request(entry: &RuntimeMatrixEntryRequest) -> InferenceRequest {
    InferenceRequest {
        backend_id: Some(entry.backend_id.clone()),
        model: entry.model_name.clone(),
        messages: vec![Message {
            role: "user".into(),
            content: "Reply with exactly: CRYTEX_PROBE_OK".into(),
        }],
        system_prompt: Some("You are running a short runtime smoke test.".into()),
        temperature: Some(0.0),
        max_tokens: Some(entry.max_tokens.clamp(1, 32)),
        lora_adapter_id: entry.lora_adapter_id.clone(),
    }
}

fn is_expected_smoke_response(content: &str) -> bool {
    content
        .trim()
        .trim_matches(|ch: char| ch == '"' || ch == '\'' || ch == '`')
        .eq("CRYTEX_PROBE_OK")
}

fn remote_compatibility() -> CompatibilityReport {
    CompatibilityReport {
        format: "unknown".to_string(),
        features: vec!["dense".to_string()],
        strategy: "remote".to_string(),
        status: "ready".to_string(),
        support_status: "supported".to_string(),
        actions: vec!["route to configured non-mistral backend".to_string()],
        warnings: vec![],
        blockers: vec![],
        failure_reasons: vec![],
    }
}

fn unsupported_compatibility(reason: String) -> CompatibilityReport {
    CompatibilityReport {
        format: "unknown".to_string(),
        features: vec!["dense".to_string()],
        strategy: "remote".to_string(),
        status: "unsupported".to_string(),
        support_status: "unsupported".to_string(),
        actions: vec!["register or configure the requested backend before probing".to_string()],
        warnings: vec![],
        blockers: vec![reason.clone()],
        failure_reasons: vec![reason],
    }
}

fn compatibility_failure_reasons(
    compatibility: &CompatibilityReport,
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

pub fn create_remote_backends(
    config: &DiagConfig,
    backend_ids: &[String],
) -> Result<HashMap<String, Arc<dyn InferenceManager>>, InferenceError> {
    let mut backends = HashMap::new();
    for backend_id in backend_ids {
        let backend_config = config.backend(backend_id).ok_or_else(|| {
            InferenceError::BackendNotAvailable(format!("backend {backend_id} is not configured"))
        })?;
        backends.insert(
            backend_config.id.clone(),
            create_remote_backend(backend_config)?,
        );
    }
    Ok(backends)
}

fn create_remote_backend(
    config: &DiagBackendConfig,
) -> Result<Arc<dyn InferenceManager>, InferenceError> {
    match config.kind {
        DiagBackendKind::Ollama => {
            let url = config
                .url
                .clone()
                .unwrap_or_else(|| "http://localhost:11434".to_string());
            Ok(Arc::new(crytex_inference_ollama::OllamaBackend::new(
                url,
                &config.model,
            )))
        }
        DiagBackendKind::OpenAiCompatible => {
            let url = config
                .url
                .clone()
                .unwrap_or_else(|| "https://api.openai.com/v1".to_string());
            Ok(Arc::new(crytex_inference_openai::OpenAiBackend::new(
                url,
                &config.model,
                config.api_key.clone(),
            )))
        }
        DiagBackendKind::Anthropic => {
            let url = config
                .url
                .clone()
                .unwrap_or_else(|| "https://api.anthropic.com/v1".to_string());
            let api_key = config.api_key.clone().ok_or_else(|| {
                InferenceError::GenerationFailed("Anthropic API key is required".to_string())
            })?;
            Ok(Arc::new(crytex_inference_anthropic::AnthropicBackend::new(
                url,
                &config.model,
                api_key,
            )))
        }
        DiagBackendKind::Custom => {
            let url = config
                .url
                .clone()
                .unwrap_or_else(|| "http://localhost:8000/v1".to_string());
            Ok(Arc::new(
                crytex_inference_openai::OpenAiBackend::new(
                    url,
                    &config.model,
                    config.api_key.clone(),
                )
                .with_headers(config.headers.clone()),
            ))
        }
        DiagBackendKind::MistralRs | DiagBackendKind::Onnx => {
            Err(InferenceError::BackendNotAvailable(
                "crytex-diag lightweight mode supports remote generation backends only".to_string(),
            ))
        }
    }
}

pub fn write_report_pretty_json(
    report: &RuntimeMatrixReport,
    report_dir: impl AsRef<Path>,
) -> Result<PathBuf, std::io::Error> {
    let report_dir = report_dir.as_ref();
    std::fs::create_dir_all(report_dir)?;
    let path = report_dir.join(format!(
        "runtime-matrix-{}.json",
        sanitize_report_file_stem(&report.trace_id)
    ));
    let json = serde_json::to_string_pretty(report)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    std::fs::write(&path, json)?;
    Ok(path)
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

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use crytex_inference::{InferenceResponse, LoRAAdapter, ModelInfo, TokenUsage};

    #[test]
    fn matrix_entries_include_baseline_and_each_lora_for_every_backend() {
        let entries = build_runtime_matrix_entries(
            &["ollama".into(), "openai".into()],
            &["coder-lora".into(), "critic-lora".into()],
            "diag-model",
            "qwen3.5:9b",
            16,
        );

        let labels = entries
            .iter()
            .map(|entry| entry.label.as_str())
            .collect::<Vec<_>>();

        assert_eq!(
            labels,
            vec![
                "ollama:baseline",
                "ollama:coder-lora",
                "ollama:critic-lora",
                "openai:baseline",
                "openai:coder-lora",
                "openai:critic-lora",
            ]
        );
        assert_eq!(entries[0].lora_adapter_id, None);
        assert_eq!(entries[1].lora_adapter_id.as_deref(), Some("coder-lora"));
        assert!(entries.iter().all(|entry| entry.max_tokens == 16));
    }

    #[tokio::test]
    async fn runtime_matrix_reports_backend_lora_hot_swap_capability() {
        let backend = Arc::new(StaticBackend::new(
            "mistralrs",
            vec!["generate", "chat", "lora", "hot_swap"],
        ));
        let backends = HashMap::from([(
            "mistralrs".to_string(),
            backend as Arc<dyn InferenceManager>,
        )]);
        let entries = build_runtime_matrix_entries(
            &["mistralrs".into()],
            &["coder-lora".into()],
            "diag-model",
            "tiny-coder",
            16,
        );

        let report = run_runtime_matrix("trace-diag-lora".into(), entries, &backends).await;

        assert!(report.passed);
        assert_eq!(report.entries.len(), 2);
        assert_eq!(
            report.summary,
            RuntimeMatrixSummary {
                supported: 2,
                partial: 0,
                unsupported: 0,
                failed: 0,
            }
        );
        assert_eq!(
            report.entries[1].lora_adapter_id.as_deref(),
            Some("coder-lora")
        );
        assert_eq!(
            report.entries[1]
                .report
                .backend_capability
                .as_ref()
                .map(|capability| (capability.lora, capability.hot_swap)),
            Some((true, true))
        );
    }

    #[tokio::test]
    async fn runtime_matrix_json_report_persists_backend_lora_hot_swap_capability() {
        let backend = Arc::new(StaticBackend::new(
            "mistralrs",
            vec!["generate", "chat", "lora", "hot_swap"],
        ));
        let backends = HashMap::from([(
            "mistralrs".to_string(),
            backend as Arc<dyn InferenceManager>,
        )]);
        let entries = build_runtime_matrix_entries(
            &["mistralrs".into()],
            &["coder-lora".into()],
            "diag-model",
            "tiny-coder",
            16,
        );
        let report = run_runtime_matrix("trace-diag-lora-json".into(), entries, &backends).await;
        let temp_dir = tempfile::tempdir().unwrap();

        let report_path = write_report_pretty_json(&report, temp_dir.path()).unwrap();
        let report_json = std::fs::read_to_string(report_path).unwrap();
        let report_value = serde_json::from_str::<serde_json::Value>(&report_json).unwrap();
        let lora_entry = &report_value["entries"][1];

        assert_eq!(lora_entry["lora_adapter_id"], "coder-lora");
        assert_eq!(lora_entry["report"]["backend_capability"]["lora"], true);
        assert_eq!(lora_entry["report"]["backend_capability"]["hot_swap"], true);
        assert_eq!(report_value["summary"]["supported"], 2);
        assert_eq!(
            lora_entry["report"]["compatibility"]["support_status"],
            "supported"
        );
        assert!(
            lora_entry["report"]["failure_reasons"]
                .as_array()
                .is_some_and(Vec::is_empty)
        );
    }

    #[tokio::test]
    async fn runtime_matrix_reports_missing_backend_as_unsupported_with_failure_reason() {
        let backends = HashMap::new();
        let entries =
            build_runtime_matrix_entries(&["missing".into()], &[], "diag-model", "tiny-coder", 16);

        let report = run_runtime_matrix("trace-diag-missing".into(), entries, &backends).await;

        assert!(!report.passed);
        assert_eq!(
            report.summary,
            RuntimeMatrixSummary {
                supported: 0,
                partial: 0,
                unsupported: 1,
                failed: 1,
            }
        );
        let probe = &report.entries[0].report;
        assert_eq!(probe.compatibility.support_status, "unsupported");
        assert!(
            probe
                .failure_reasons
                .iter()
                .any(|reason| reason.contains("backend missing is not registered")),
            "expected missing backend failure reason, got {:?}",
            probe.failure_reasons
        );
    }

    #[test]
    fn partial_config_deserializes_only_diagnostic_fields() {
        let config = toml::from_str::<DiagConfig>(
            r#"
            [inference]
            default_backend = "ollama"

            [[inference.backends]]
            id = "ollama"
            kind = "ollama"
            model = "qwen3.5:9b"
            url = "http://127.0.0.1:11434"
            supports_lora = false

            [paths]
            data_dir = "B:/crytex-data"
            "#,
        )
        .unwrap();

        assert_eq!(configured_backend_ids(&config, &[]), vec!["ollama"]);
        assert_eq!(
            runtime_model_name(&config, "ollama", None, "fallback"),
            "qwen3.5:9b"
        );
        assert_eq!(config.paths.data_dir, PathBuf::from("B:/crytex-data"));
    }

    #[test]
    fn list_backends_marks_default_and_lightweight_support() {
        let config = diag_config_with_backends();

        let report = list_backends(&config);

        assert_eq!(report.default_backend.as_deref(), Some("ollama"));
        assert_eq!(report.backends.len(), 2);
        assert!(report.backends[0].is_default);
        assert!(report.backends[0].supported_by_lightweight_diag);
        assert!(!report.backends[1].supported_by_lightweight_diag);
    }

    #[test]
    fn doctor_fails_when_selected_backend_is_not_supported_by_lightweight_diag() {
        let config = diag_config_with_backends();

        let report = doctor(&config, &["local".into()]);

        assert!(!report.passed);
        assert!(report.checks.iter().any(|check| {
            check.name == "backend_supported:local"
                && check.status == DoctorCheckStatus::Failed
                && check.message.contains("not supported")
        }));
    }

    #[test]
    fn doctor_warns_when_default_backend_is_missing_but_explicit_backend_is_valid() {
        let mut config = diag_config_with_backends();
        config.inference.default_backend = None;

        let report = doctor(&config, &["ollama".into()]);

        assert!(report.passed);
        assert!(report.checks.iter().any(|check| {
            check.name == "default_backend"
                && check.status == DoctorCheckStatus::Warning
                && check.message.contains("not configured")
        }));
    }

    #[test]
    fn cuda_preflight_reports_typed_failure_when_gpu_is_required_and_missing() {
        let check = cuda_toolchain_doctor_check(
            CudaToolchainPresence {
                nvidia_smi: false,
                nvcc: false,
                runtime_library: false,
            },
            true,
        );

        assert_eq!(check.name, "cuda_toolchain_preflight");
        assert_eq!(check.status, DoctorCheckStatus::Failed);
        assert!(check.message.contains("required"));
    }

    #[test]
    fn cuda_preflight_warns_without_failing_when_gpu_is_optional() {
        let check = cuda_toolchain_doctor_check(
            CudaToolchainPresence {
                nvidia_smi: false,
                nvcc: false,
                runtime_library: false,
            },
            false,
        );

        assert_eq!(check.status, DoctorCheckStatus::Warning);
        assert!(check.message.contains("degrade"));
    }

    #[test]
    fn ollama_tags_model_names_include_name_and_model_fields() {
        let tags = serde_json::json!({
            "models": [
                { "name": "qwen3.5:9b" },
                { "model": "nomic-embed-text:latest" }
            ]
        });

        let names = ollama_tags_model_names(&tags);

        assert_eq!(names, vec!["qwen3.5:9b", "nomic-embed-text:latest"]);
    }

    struct StaticBackend {
        id: String,
        capabilities: Vec<String>,
    }

    impl StaticBackend {
        fn new(id: &str, capabilities: Vec<&str>) -> Self {
            Self {
                id: id.into(),
                capabilities: capabilities.into_iter().map(str::to_string).collect(),
            }
        }
    }

    #[async_trait]
    impl InferenceManager for StaticBackend {
        async fn generate(
            &self,
            _request: InferenceRequest,
        ) -> Result<InferenceResponse, InferenceError> {
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

        async fn embed(&self, _text: &str) -> Result<Vec<f32>, InferenceError> {
            Ok(vec![])
        }

        async fn register_lora(&self, _lora: LoRAAdapter) -> Result<(), InferenceError> {
            Ok(())
        }

        async fn swap_lora(&self, _lora_id: &str) -> Result<(), InferenceError> {
            Ok(())
        }

        fn available_backends(&self) -> Vec<BackendInfo> {
            vec![BackendInfo {
                id: self.id.clone(),
                name: self.id.clone(),
                capabilities: self.capabilities.clone(),
            }]
        }

        async fn list_models(&self) -> Result<Vec<ModelInfo>, InferenceError> {
            Ok(vec![])
        }
    }

    #[test]
    fn ollama_tags_model_names_ignore_malformed_entries() {
        let tags = serde_json::json!({
            "models": [
                { "name": 42 },
                { "size": 123 },
                null,
                { "model": "smollm2:135m" }
            ]
        });

        let names = ollama_tags_model_names(&tags);

        assert_eq!(names, vec!["smollm2:135m"]);
    }

    #[test]
    fn ollama_ps_check_passes_when_loaded_model_uses_vram() {
        let ps = serde_json::json!({
            "models": [
                { "name": "qwen3.5:9b", "model": "qwen3.5:9b", "size": 1000, "size_vram": 1000 }
            ]
        });

        let check = ollama_ps_model_placement_check("ollama", "qwen3.5:9b", &ps, true);

        assert_eq!(check.status, DoctorCheckStatus::Passed);
        assert!(check.message.contains("VRAM"));
    }

    #[test]
    fn ollama_ps_check_fails_require_gpu_when_loaded_model_has_no_vram() {
        let ps = serde_json::json!({
            "models": [
                { "name": "qwen3.5:9b", "model": "qwen3.5:9b", "size": 1000, "size_vram": 0 }
            ]
        });

        let check = ollama_ps_model_placement_check("ollama", "qwen3.5:9b", &ps, true);

        assert_eq!(check.status, DoctorCheckStatus::Failed);
        assert!(check.message.contains("CPU"));
    }

    #[test]
    fn ollama_ps_check_warns_when_model_is_not_loaded() {
        let ps = serde_json::json!({ "models": [] });

        let check = ollama_ps_model_placement_check("ollama", "qwen3.5:9b", &ps, true);

        assert_eq!(check.status, DoctorCheckStatus::Warning);
        assert!(check.message.contains("not currently loaded"));
    }

    #[test]
    fn ollama_runtime_placement_report_detects_gpu() {
        let ps = serde_json::json!({
            "models": [
                { "name": "qwen3.5:9b", "model": "qwen3.5:9b", "size": 1000, "size_vram": 800 }
            ]
        });

        let report = ollama_runtime_placement_report("ollama", "qwen3.5:9b", &ps);

        assert_eq!(report.kind, RuntimePlacementKind::Gpu);
        assert_eq!(report.size_bytes, Some(1000));
        assert_eq!(report.size_vram_bytes, Some(800));
    }

    #[test]
    fn ollama_runtime_placement_report_detects_cpu() {
        let ps = serde_json::json!({
            "models": [
                { "name": "qwen3.5:9b", "model": "qwen3.5:9b", "size": 1000, "size_vram": 0 }
            ]
        });

        let report = ollama_runtime_placement_report("ollama", "qwen3.5:9b", &ps);

        assert_eq!(report.kind, RuntimePlacementKind::Cpu);
        assert_eq!(report.size_bytes, Some(1000));
        assert_eq!(report.size_vram_bytes, Some(0));
    }

    #[test]
    fn ollama_runtime_placement_report_marks_not_loaded() {
        let ps = serde_json::json!({ "models": [] });

        let report = ollama_runtime_placement_report("ollama", "qwen3.5:9b", &ps);

        assert_eq!(report.kind, RuntimePlacementKind::NotLoaded);
        assert_eq!(report.size_bytes, None);
        assert_eq!(report.size_vram_bytes, None);
    }

    fn diag_config_with_backends() -> DiagConfig {
        DiagConfig {
            inference: DiagInferenceConfig {
                default_backend: Some("ollama".into()),
                backends: vec![
                    DiagBackendConfig {
                        id: "ollama".into(),
                        kind: DiagBackendKind::Ollama,
                        model: "qwen3.5:9b".into(),
                        url: Some("http://127.0.0.1:11434".into()),
                        api_key: None,
                        headers: HashMap::new(),
                    },
                    DiagBackendConfig {
                        id: "local".into(),
                        kind: DiagBackendKind::MistralRs,
                        model: "model.gguf".into(),
                        url: None,
                        api_key: None,
                        headers: HashMap::new(),
                    },
                ],
            },
            paths: DiagPathsConfig {
                data_dir: PathBuf::from("B:/crytex-data"),
            },
        }
    }
}
