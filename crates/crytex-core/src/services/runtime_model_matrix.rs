//! Truthful runtime/model capability matrix.
//!
//! This module reports what every backend type can do without requiring the
//! caller to instantiate the backend or load model weights. Runtime probes can
//! then validate a concrete configured model against the same contract.

use serde::{Deserialize, Serialize};

use crate::config::BackendKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeSupportLevel {
    Supported,
    Partial,
    Unsupported,
}

impl RuntimeSupportLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Supported => "supported",
            Self::Partial => "partial",
            Self::Unsupported => "unsupported",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendRuntimeCapability {
    pub backend: BackendKind,
    pub status: RuntimeSupportLevel,
    pub generate: RuntimeSupportLevel,
    pub chat: RuntimeSupportLevel,
    pub embeddings: RuntimeSupportLevel,
    pub rerank: RuntimeSupportLevel,
    pub lora_training: RuntimeSupportLevel,
    pub lora_runtime_application: RuntimeSupportLevel,
    pub lora_hot_swap: RuntimeSupportLevel,
    pub model_listing: RuntimeSupportLevel,
    pub download: RuntimeSupportLevel,
    pub cuda: RuntimeSupportLevel,
    pub reasons: Vec<String>,
    pub references: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeModelMatrixReport {
    pub backends: Vec<BackendRuntimeCapability>,
    pub trtllm_future_module: TrtLlmModuleDisposition,
    pub cuda_preflight: CudaPreflightContract,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrtLlmModuleDisposition {
    pub path: String,
    pub status: RuntimeSupportLevel,
    pub decision: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CudaPreflightContract {
    pub doctor_checks: Vec<String>,
    pub failure_mode: String,
}

pub struct RuntimeModelMatrix;

impl RuntimeModelMatrix {
    pub fn report() -> RuntimeModelMatrixReport {
        RuntimeModelMatrixReport {
            backends: vec![
                ollama_capability(),
                mistral_rs_capability(),
                onnx_capability(),
                openai_compatible_capability(),
                anthropic_capability(),
            ],
            trtllm_future_module: TrtLlmModuleDisposition {
                path: "trash/crytex-inference-trtllm".to_string(),
                status: RuntimeSupportLevel::Unsupported,
                decision: "kept as future optional module; it is outside the production runtime matrix until the crate is moved under crates/ and receives CI/toolchain probes".to_string(),
            },
            cuda_preflight: CudaPreflightContract {
                doctor_checks: vec![
                    "nvidia-smi availability".to_string(),
                    "CUDA driver/runtime compatibility".to_string(),
                    "NVCC or runtime DLL visibility when CUDA features are enabled".to_string(),
                    "local backend GPU layer configuration".to_string(),
                ],
                failure_mode: "doctor must report typed unsupported/partial CUDA status and CPU fallback, not panic".to_string(),
            },
        }
    }
}

fn ollama_capability() -> BackendRuntimeCapability {
    BackendRuntimeCapability {
        backend: BackendKind::Ollama,
        status: RuntimeSupportLevel::Partial,
        generate: RuntimeSupportLevel::Supported,
        chat: RuntimeSupportLevel::Supported,
        embeddings: RuntimeSupportLevel::Supported,
        rerank: RuntimeSupportLevel::Unsupported,
        lora_training: RuntimeSupportLevel::Unsupported,
        lora_runtime_application: RuntimeSupportLevel::Unsupported,
        lora_hot_swap: RuntimeSupportLevel::Unsupported,
        model_listing: RuntimeSupportLevel::Supported,
        download: RuntimeSupportLevel::Supported,
        cuda: RuntimeSupportLevel::Partial,
        reasons: vec![
            "Ollama supports local generation/chat, embeddings, tags/listing, and pull/download through its HTTP API".to_string(),
            "Crytex does not treat Ollama as a runtime LoRA adapter backend; use a baked Ollama model outside Crytex if needed".to_string(),
            "CUDA is managed by Ollama/host installation, so Crytex can preflight connectivity but cannot guarantee GPU placement".to_string(),
        ],
        references: vec!["https://github.com/ollama/ollama/blob/main/docs/api.md".to_string()],
    }
}

fn mistral_rs_capability() -> BackendRuntimeCapability {
    BackendRuntimeCapability {
        backend: BackendKind::MistralRs,
        status: RuntimeSupportLevel::Supported,
        generate: RuntimeSupportLevel::Supported,
        chat: RuntimeSupportLevel::Supported,
        embeddings: RuntimeSupportLevel::Unsupported,
        rerank: RuntimeSupportLevel::Unsupported,
        lora_training: RuntimeSupportLevel::Supported,
        lora_runtime_application: RuntimeSupportLevel::Supported,
        lora_hot_swap: RuntimeSupportLevel::Supported,
        model_listing: RuntimeSupportLevel::Supported,
        download: RuntimeSupportLevel::Supported,
        cuda: RuntimeSupportLevel::Partial,
        reasons: vec![
            "mistral.rs is Crytex's local GGUF text runtime and role LoRA application path".to_string(),
            "Embeddings/rerank stay delegated to ONNX or another dedicated backend".to_string(),
            "CUDA support depends on build features, driver/toolchain, model architecture, and fallback availability".to_string(),
        ],
        references: vec!["https://github.com/EricLBuehler/mistral.rs".to_string()],
    }
}

fn onnx_capability() -> BackendRuntimeCapability {
    BackendRuntimeCapability {
        backend: BackendKind::Onnx,
        status: RuntimeSupportLevel::Partial,
        generate: RuntimeSupportLevel::Unsupported,
        chat: RuntimeSupportLevel::Unsupported,
        embeddings: RuntimeSupportLevel::Supported,
        rerank: RuntimeSupportLevel::Supported,
        lora_training: RuntimeSupportLevel::Unsupported,
        lora_runtime_application: RuntimeSupportLevel::Unsupported,
        lora_hot_swap: RuntimeSupportLevel::Unsupported,
        model_listing: RuntimeSupportLevel::Supported,
        download: RuntimeSupportLevel::Partial,
        cuda: RuntimeSupportLevel::Partial,
        reasons: vec![
            "ONNX is an embedding/rerank runtime in Crytex, not a text generation backend".to_string(),
            "FastEmbed presets can download/cache models; local ONNX directories are caller-provided".to_string(),
            "CUDA depends on ONNX Runtime execution provider availability".to_string(),
        ],
        references: vec![
            "https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html".to_string(),
            "https://github.com/Anush008/fastembed-rs".to_string(),
        ],
    }
}

fn openai_compatible_capability() -> BackendRuntimeCapability {
    BackendRuntimeCapability {
        backend: BackendKind::OpenAiCompatible,
        status: RuntimeSupportLevel::Partial,
        generate: RuntimeSupportLevel::Supported,
        chat: RuntimeSupportLevel::Supported,
        embeddings: RuntimeSupportLevel::Supported,
        rerank: RuntimeSupportLevel::Unsupported,
        lora_training: RuntimeSupportLevel::Unsupported,
        lora_runtime_application: RuntimeSupportLevel::Unsupported,
        lora_hot_swap: RuntimeSupportLevel::Unsupported,
        model_listing: RuntimeSupportLevel::Supported,
        download: RuntimeSupportLevel::Unsupported,
        cuda: RuntimeSupportLevel::Unsupported,
        reasons: vec![
            "OpenAI-compatible HTTP backends support chat/completions, embeddings, and model listing when the provider implements those endpoints".to_string(),
            "Crytex cannot apply runtime LoRA adapters inside remote OpenAI-compatible providers".to_string(),
            "Download and CUDA placement are provider-side concerns".to_string(),
        ],
        references: vec![
            "https://platform.openai.com/docs/api-reference/chat".to_string(),
            "https://platform.openai.com/docs/api-reference/models/list".to_string(),
        ],
    }
}

fn anthropic_capability() -> BackendRuntimeCapability {
    BackendRuntimeCapability {
        backend: BackendKind::Anthropic,
        status: RuntimeSupportLevel::Partial,
        generate: RuntimeSupportLevel::Supported,
        chat: RuntimeSupportLevel::Supported,
        embeddings: RuntimeSupportLevel::Unsupported,
        rerank: RuntimeSupportLevel::Unsupported,
        lora_training: RuntimeSupportLevel::Unsupported,
        lora_runtime_application: RuntimeSupportLevel::Unsupported,
        lora_hot_swap: RuntimeSupportLevel::Unsupported,
        model_listing: RuntimeSupportLevel::Partial,
        download: RuntimeSupportLevel::Unsupported,
        cuda: RuntimeSupportLevel::Unsupported,
        reasons: vec![
            "Anthropic Messages API supports text/chat generation through configured model ids".to_string(),
            "Crytex uses the configured Anthropic model id instead of relying on public model discovery".to_string(),
            "Anthropic does not provide Crytex embeddings, rerank, download, CUDA, or runtime LoRA adapter application".to_string(),
        ],
        references: vec!["https://docs.anthropic.com/en/api/messages".to_string()],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_matrix_reports_truthful_lora_capability_per_backend() {
        let report = RuntimeModelMatrix::report();

        assert_eq!(
            capability(&report, BackendKind::MistralRs).lora_runtime_application,
            RuntimeSupportLevel::Supported
        );
        for backend in [
            BackendKind::Ollama,
            BackendKind::Onnx,
            BackendKind::OpenAiCompatible,
            BackendKind::Anthropic,
        ] {
            assert_eq!(
                capability(&report, backend).lora_runtime_application,
                RuntimeSupportLevel::Unsupported
            );
        }
    }

    #[test]
    fn runtime_matrix_marks_embedding_only_and_remote_backends_as_partial() {
        let report = RuntimeModelMatrix::report();

        assert_eq!(
            capability(&report, BackendKind::Onnx).generate,
            RuntimeSupportLevel::Unsupported
        );
        assert_eq!(
            capability(&report, BackendKind::OpenAiCompatible).status,
            RuntimeSupportLevel::Partial
        );
        assert!(
            capability(&report, BackendKind::Anthropic)
                .reasons
                .iter()
                .any(|reason| reason.contains("configured Anthropic model id"))
        );
    }

    #[test]
    fn runtime_matrix_keeps_trtllm_as_future_optional_module() {
        let report = RuntimeModelMatrix::report();

        assert_eq!(
            report.trtllm_future_module.status,
            RuntimeSupportLevel::Unsupported
        );
        assert!(report.trtllm_future_module.path.contains("trtllm"));
    }

    fn capability(
        report: &RuntimeModelMatrixReport,
        backend: BackendKind,
    ) -> &BackendRuntimeCapability {
        report
            .backends
            .iter()
            .find(|capability| capability.backend == backend)
            .expect("backend must exist in runtime matrix")
    }
}
