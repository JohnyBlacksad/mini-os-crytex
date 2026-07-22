use serde::{Deserialize, Serialize};

use crate::config::{BackendKind, CrytexConfig};

/// Stable module ids used by diagnostics, doctor, backend acceptance, and CLI
/// status output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModuleId {
    Core,
    Storage,
    Agents,
    Inference,
    Rag,
    TokenEconomy,
    Lora,
    PromptEvolution,
    Bench,
    Sandbox,
    Cli,
}

/// Capability status is intentionally non-binary: production backends often
/// degrade cleanly when an optional module is disabled or unavailable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CapabilityStatus {
    Ready,
    Degraded,
    Disabled,
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraitBoundary {
    pub requires: Vec<String>,
    pub provides: Vec<String>,
    pub object_safe: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModuleCapabilityReport {
    pub module: ModuleId,
    pub status: CapabilityStatus,
    pub reason: Option<String>,
    pub required_traits: Vec<String>,
    pub provided_traits: Vec<String>,
    pub object_safe_traits: Vec<String>,
    pub disabled_by: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityAuditReport {
    pub modules: Vec<ModuleCapabilityReport>,
}

impl CapabilityAuditReport {
    pub fn from_config(config: &CrytexConfig) -> Self {
        let module_ids = [
            ModuleId::Core,
            ModuleId::Storage,
            ModuleId::Agents,
            ModuleId::Inference,
            ModuleId::Rag,
            ModuleId::TokenEconomy,
            ModuleId::Lora,
            ModuleId::PromptEvolution,
            ModuleId::Bench,
            ModuleId::Sandbox,
            ModuleId::Cli,
        ];
        Self {
            modules: module_ids
                .into_iter()
                .map(|module| module_capability(config, module))
                .collect(),
        }
    }

    pub fn module(&self, module: ModuleId) -> Option<&ModuleCapabilityReport> {
        self.modules.iter().find(|report| report.module == module)
    }
}

pub fn module_capability(config: &CrytexConfig, module: ModuleId) -> ModuleCapabilityReport {
    let boundary = trait_boundary(module);
    let (status, reason, disabled_by) = module_status(config, module);
    ModuleCapabilityReport {
        module,
        status,
        reason,
        required_traits: boundary.requires,
        provided_traits: boundary.provides,
        object_safe_traits: object_safe_traits(module),
        disabled_by,
    }
}

pub fn trait_boundary(module: ModuleId) -> TraitBoundary {
    match module {
        ModuleId::Core => TraitBoundary {
            requires: vec!["Persistence".into(), "EventService".into()],
            provides: vec![
                "TaskService".into(),
                "ProjectService".into(),
                "Orchestrator".into(),
            ],
            object_safe: true,
        },
        ModuleId::Storage => TraitBoundary {
            requires: vec!["StorageConfig".into()],
            provides: vec![
                "Persistence".into(),
                "VectorStore".into(),
                "MetricsRepository".into(),
            ],
            object_safe: true,
        },
        ModuleId::Agents => TraitBoundary {
            requires: vec![
                "InferenceService".into(),
                "ToolService".into(),
                "ContextAssembler".into(),
            ],
            provides: vec![
                "Agent".into(),
                "AgentService".into(),
                "CriticCouncil".into(),
            ],
            object_safe: true,
        },
        ModuleId::Inference => TraitBoundary {
            requires: vec!["BackendConfig".into()],
            provides: vec!["InferenceManager".into(), "InferenceService".into()],
            object_safe: true,
        },
        ModuleId::Rag => TraitBoundary {
            requires: vec!["Embedder".into(), "VectorStore".into()],
            provides: vec![
                "ProjectIndexer".into(),
                "HybridRetriever".into(),
                "ContextAssembler".into(),
            ],
            object_safe: true,
        },
        ModuleId::TokenEconomy => TraitBoundary {
            requires: vec!["Tokenizer".into(), "TokenEstimator".into()],
            provides: vec![
                "Compressor".into(),
                "CcrStore".into(),
                "CompressionPipeline".into(),
            ],
            object_safe: true,
        },
        ModuleId::Lora => TraitBoundary {
            requires: vec!["LoraTrainer".into(), "LoraBenchmarkGate".into()],
            provides: vec!["LoraEvolutionService".into(), "LoraRouter".into()],
            object_safe: true,
        },
        ModuleId::PromptEvolution => TraitBoundary {
            requires: vec![
                "PromptVersionRepository".into(),
                "PromptBenchmarkGate".into(),
            ],
            provides: vec!["PromptEvolutionService".into()],
            object_safe: true,
        },
        ModuleId::Bench => TraitBoundary {
            requires: vec!["BenchmarkHarness".into(), "Scorer".into()],
            provides: vec![
                "BenchmarkRunner".into(),
                "PromptBenchmarkGate".into(),
                "LoraBenchmarkGate".into(),
            ],
            object_safe: true,
        },
        ModuleId::Sandbox => TraitBoundary {
            requires: vec!["SandboxPolicy".into()],
            provides: vec!["SandboxService".into(), "ToolService".into()],
            object_safe: true,
        },
        ModuleId::Cli => TraitBoundary {
            requires: vec!["CapabilityAuditReport".into(), "CommandHandlers".into()],
            provides: vec!["ProductCli".into(), "ExitCodePolicy".into()],
            object_safe: false,
        },
    }
}

fn module_status(
    config: &CrytexConfig,
    module: ModuleId,
) -> (CapabilityStatus, Option<String>, Option<String>) {
    match module {
        ModuleId::Core | ModuleId::Storage | ModuleId::Agents | ModuleId::Cli => {
            ready("required module")
        }
        ModuleId::Inference => inference_status(config),
        ModuleId::Rag => rag_status(config),
        ModuleId::TokenEconomy => enabled_status(
            config.modules.token_economy,
            "config.modules.token_economy",
            "token budgeting and compression enabled",
        ),
        ModuleId::Lora => enabled_status(
            config.modules.lora,
            "config.modules.lora",
            "LoRA evolution enabled",
        ),
        ModuleId::PromptEvolution => enabled_status(
            config.modules.prompt_evolution,
            "config.modules.prompt_evolution",
            "Prompt Evolution enabled",
        ),
        ModuleId::Bench => enabled_status(
            config.modules.bench,
            "config.modules.bench",
            "benchmark gates enabled",
        ),
        ModuleId::Sandbox => sandbox_status(config),
    }
}

fn inference_status(config: &CrytexConfig) -> (CapabilityStatus, Option<String>, Option<String>) {
    if !config.modules.cloud && config.inference.backends.iter().any(is_cloud_backend) {
        return (
            CapabilityStatus::Degraded,
            Some("cloud backends are configured but cloud module is disabled".into()),
            Some("config.modules.cloud".into()),
        );
    }
    if !config.modules.cuda
        && config
            .inference
            .backends
            .iter()
            .any(|backend| backend.gpu_layers.is_some())
    {
        return (
            CapabilityStatus::Degraded,
            Some("GPU layers requested but CUDA module is disabled".into()),
            Some("config.modules.cuda".into()),
        );
    }
    if config.inference.backends.is_empty() {
        return (
            CapabilityStatus::Degraded,
            Some("no inference backend configured; deterministic/mock paths may still run".into()),
            None,
        );
    }
    ready("inference backend configured")
}

fn rag_status(config: &CrytexConfig) -> (CapabilityStatus, Option<String>, Option<String>) {
    if !config.modules.rag {
        return disabled("config.modules.rag");
    }
    if !config.modules.reranker && config.inference.rerank_backend.is_some() {
        return (
            CapabilityStatus::Degraded,
            Some("rerank backend configured but reranker module is disabled".into()),
            Some("config.modules.reranker".into()),
        );
    }
    if !config.modules.external_vector_db && config.inference.vector_store_url.is_some() {
        return (
            CapabilityStatus::Degraded,
            Some(
                "external vector DB configured but disabled; embedded store should be used".into(),
            ),
            Some("config.modules.external_vector_db".into()),
        );
    }
    ready("RAG enabled")
}

fn sandbox_status(config: &CrytexConfig) -> (CapabilityStatus, Option<String>, Option<String>) {
    if !config.modules.sandbox {
        return disabled("config.modules.sandbox");
    }
    if !config.modules.sandbox_docker {
        return (
            CapabilityStatus::Degraded,
            Some(
                "Docker sandbox backend disabled; host/WASI backends may still be available".into(),
            ),
            Some("config.modules.sandbox_docker".into()),
        );
    }
    ready("sandbox enabled")
}

fn enabled_status(
    enabled: bool,
    disabled_by: &str,
    reason: &str,
) -> (CapabilityStatus, Option<String>, Option<String>) {
    if enabled {
        ready(reason)
    } else {
        disabled(disabled_by)
    }
}

fn ready(reason: &str) -> (CapabilityStatus, Option<String>, Option<String>) {
    (CapabilityStatus::Ready, Some(reason.into()), None)
}

fn disabled(disabled_by: &str) -> (CapabilityStatus, Option<String>, Option<String>) {
    (
        CapabilityStatus::Disabled,
        Some("module disabled by configuration".into()),
        Some(disabled_by.into()),
    )
}

fn is_cloud_backend(backend: &crate::config::BackendConfig) -> bool {
    matches!(
        backend.kind,
        BackendKind::OpenAiCompatible | BackendKind::Anthropic | BackendKind::Custom
    )
}

fn object_safe_traits(module: ModuleId) -> Vec<String> {
    trait_boundary(module)
        .provides
        .into_iter()
        .filter(|name| name != "ProductCli" && name != "ExitCodePolicy")
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{BackendConfig, BackendKind, CrytexConfig};

    #[test]
    fn disabled_reranker_degrades_rag_with_typed_report() {
        let mut config = CrytexConfig::default();
        config.modules.reranker = false;
        config.inference.rerank_backend = Some("onnx-rerank".into());
        config
            .inference
            .backends
            .push(BackendConfig::onnx("onnx-rerank", "bge-reranker"));

        let report = module_capability(&config, ModuleId::Rag);

        assert_eq!(report.status, CapabilityStatus::Degraded);
        assert_eq!(
            report.disabled_by.as_deref(),
            Some("config.modules.reranker")
        );
        assert!(report.provided_traits.contains(&"ContextAssembler".into()));
    }

    #[test]
    fn disabled_lora_returns_disabled_report_without_affecting_core() {
        let mut config = CrytexConfig::default();
        config.modules.lora = false;

        let audit = CapabilityAuditReport::from_config(&config);

        assert_eq!(
            audit.module(ModuleId::Lora).map(|report| report.status),
            Some(CapabilityStatus::Disabled)
        );
        assert_eq!(
            audit.module(ModuleId::Core).map(|report| report.status),
            Some(CapabilityStatus::Ready)
        );
    }

    #[test]
    fn disabled_cloud_degrades_only_cloud_inference_backends() {
        let mut config = CrytexConfig::default();
        config.modules.cloud = false;
        config.inference.backends.push(BackendConfig {
            id: "openai".into(),
            kind: BackendKind::OpenAiCompatible,
            model: "gpt-4.1-mini".into(),
            url: Some("https://api.openai.com/v1".into()),
            api_key: None,
            headers: Default::default(),
            timeout_seconds: None,
            context_size: None,
            gpu_layers: None,
            supports_lora: false,
        });

        let report = module_capability(&config, ModuleId::Inference);

        assert_eq!(report.status, CapabilityStatus::Degraded);
        assert_eq!(report.disabled_by.as_deref(), Some("config.modules.cloud"));
    }

    #[test]
    fn disabled_docker_sandbox_keeps_sandbox_degraded() {
        let mut config = CrytexConfig::default();
        config.modules.sandbox_docker = false;

        let report = module_capability(&config, ModuleId::Sandbox);

        assert_eq!(report.status, CapabilityStatus::Degraded);
        assert_eq!(
            report.disabled_by.as_deref(),
            Some("config.modules.sandbox_docker")
        );
    }

    #[test]
    fn disabled_cuda_degrades_gpu_backend_without_disabling_inference() {
        let mut config = CrytexConfig::default();
        config.modules.cuda = false;
        config.inference.backends.push(BackendConfig::mistral_rs(
            "local",
            "model.gguf",
            Some(4096),
            Some(32),
        ));

        let report = module_capability(&config, ModuleId::Inference);

        assert_eq!(report.status, CapabilityStatus::Degraded);
        assert_eq!(report.disabled_by.as_deref(), Some("config.modules.cuda"));
    }

    #[test]
    fn disabled_external_vector_db_degrades_rag_to_embedded_store() {
        let mut config = CrytexConfig::default();
        config.modules.external_vector_db = false;
        config.inference.vector_store_url = Some("http://localhost:6334".into());

        let report = module_capability(&config, ModuleId::Rag);

        assert_eq!(report.status, CapabilityStatus::Degraded);
        assert_eq!(
            report.disabled_by.as_deref(),
            Some("config.modules.external_vector_db")
        );
    }

    #[test]
    fn plugin_boundaries_are_object_safe_where_dynamic_replacement_is_required() {
        let audit = CapabilityAuditReport::from_config(&CrytexConfig::default());

        for report in audit.modules {
            if report.module != ModuleId::Cli {
                assert!(
                    !report.object_safe_traits.is_empty(),
                    "{:?} should expose object-safe replacement traits",
                    report.module
                );
            }
        }
    }
}
