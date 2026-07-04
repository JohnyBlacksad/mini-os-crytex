use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use thiserror::Error;

use crate::security::Severity;
use crate::services::hybrid::FusionStrategyKind;

/// The kind of inference backend.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    /// Local Ollama HTTP API.
    Ollama,
    /// OpenAI-compatible HTTP API (OpenAI, OpenRouter, vLLM, LM Studio, etc.).
    OpenAiCompatible,
    /// Anthropic Messages API.
    Anthropic,
    /// In-process `mistral.rs` backend for local GGUF / ISQ models.
    MistralRs,
    /// Local ONNX embedding/reranker backend powered by fastembed.
    Onnx,
    /// Custom OpenAI-compatible server with arbitrary headers.
    Custom,
}

/// Configuration for a single inference backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendConfig {
    /// Unique identifier used to select this backend at runtime.
    pub id: String,
    /// The kind of backend.
    pub kind: BackendKind,
    /// Default model for this backend.
    ///
    /// For `MistralRs` this is the path to the GGUF model file or model id. For HTTP
    /// backends it is the model identifier used in API requests.
    pub model: String,
    /// Base URL for the backend. If `None`, a default URL for the kind is used.
    /// For `MistralRs` this field is ignored; the model path is read from `model`.
    pub url: Option<String>,
    /// API key for cloud/remote backends.
    pub api_key: Option<String>,
    /// Custom HTTP headers for this backend.
    #[serde(default)]
    pub headers: HashMap<String, String>,
    /// Request timeout in seconds. If `None`, the global default is used.
    pub timeout_seconds: Option<u64>,
    /// Context size for local backends.
    #[serde(default)]
    pub context_size: Option<usize>,
    /// Number of layers to offload to GPU (for local backends).
    #[serde(default)]
    pub gpu_layers: Option<usize>,
    /// Whether this backend supports dynamic LoRA loading/unloading.
    pub supports_lora: bool,
}

impl BackendConfig {
    /// Convenience constructor for an Ollama backend.
    pub fn ollama(id: impl Into<String>, model: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            kind: BackendKind::Ollama,
            model: model.into(),
            url: Some(url.into()),
            api_key: None,
            headers: HashMap::new(),
            timeout_seconds: None,
            context_size: None,
            gpu_layers: None,
            supports_lora: false,
        }
    }

    /// Convenience constructor for a local `mistral.rs` backend.
    pub fn mistral_rs(
        id: impl Into<String>,
        model: impl Into<String>,
        context_size: Option<usize>,
        gpu_layers: Option<usize>,
    ) -> Self {
        Self {
            id: id.into(),
            kind: BackendKind::MistralRs,
            model: model.into(),
            url: None,
            api_key: None,
            headers: HashMap::new(),
            timeout_seconds: None,
            context_size,
            gpu_layers,
            supports_lora: true,
        }
    }

    /// Convenience constructor for a local ONNX embedding/reranker backend.
    pub fn onnx(id: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            kind: BackendKind::Onnx,
            model: model.into(),
            url: None,
            api_key: None,
            headers: HashMap::new(),
            timeout_seconds: None,
            context_size: None,
            gpu_layers: None,
            supports_lora: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct InferenceConfig {
    /// ID of the backend selected by default.
    pub default_backend: Option<String>,
    /// All configured backends.
    pub backends: Vec<BackendConfig>,
    /// Optional separate backend for embeddings. If `None`, `default_backend` is used.
    pub embedding_backend: Option<String>,
    /// Optional separate backend for reranking. If `None`, reranking is disabled.
    pub rerank_backend: Option<String>,
    /// Enable sparse (BM25) vector indexing and search.
    pub sparse_embedding_enabled: bool,
    /// Language passed to the BM25 tokenizer (e.g. `"english"`).
    pub sparse_embedding_language: Option<String>,
    /// Optional Qdrant gRPC URL (e.g. `http://localhost:6334`).
    /// If `None`, an embedded vector store is used.
    pub vector_store_url: Option<String>,
    /// Optional base directory for the embedded Qdrant Edge vector store.
    /// If `None`, defaults to `<data_dir>/vectors`.
    pub vector_store_path: Option<PathBuf>,
    pub default_temperature: f32,
    pub default_max_tokens: usize,
    pub request_timeout_seconds: u64,
    pub context_token_budget: Option<usize>,
}

/// Hybrid search configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SearchConfig {
    /// Fusion algorithm used to combine dense and sparse search results.
    pub fusion_strategy: FusionStrategyKind,
    /// Smoothing constant for Reciprocal Rank Fusion.
    pub rrf_k: f64,
    /// Number of candidates retrieved from each collection before fusion.
    pub per_collection_limit: usize,
    /// Final number of chunks returned to the context assembler.
    pub final_limit: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            fusion_strategy: FusionStrategyKind::Rrf,
            rrf_k: 60.0,
            per_collection_limit: 20,
            final_limit: 10,
        }
    }
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            default_backend: None,
            backends: Vec::new(),
            embedding_backend: None,
            rerank_backend: None,
            sparse_embedding_enabled: true,
            sparse_embedding_language: Some("english".into()),
            vector_store_url: None,
            vector_store_path: None,
            default_temperature: 0.7,
            default_max_tokens: 4096,
            request_timeout_seconds: 120,
            context_token_budget: Some(4096),
        }
    }
}

impl InferenceConfig {
    /// Finds a backend by id.
    pub fn backend(&self, id: &str) -> Option<&BackendConfig> {
        self.backends.iter().find(|b| b.id == id)
    }

    /// Finds the default backend configuration.
    pub fn default_backend_config(&self) -> Option<&BackendConfig> {
        self.default_backend
            .as_deref()
            .and_then(|id| self.backend(id))
    }
}

/// Incremental indexing configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct IndexingConfig {
    /// Enable file-system watching and incremental re-indexing for projects.
    pub incremental_enabled: bool,
    /// Debounce window for file-system events, in milliseconds.
    pub debounce_ms: u64,
}

impl Default for IndexingConfig {
    fn default() -> Self {
        Self {
            incremental_enabled: true,
            debounce_ms: 500,
        }
    }
}

/// Cache-layer tuning.
///
/// Controls bounded in-memory caching for embeddings and vector search results.
/// Both caches are exact-match, time-bounded, and evict via the underlying
/// cache's replacement policy (W-TinyLFU for `moka`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CacheConfig {
    /// Enable caching of `text -> Vec<f32>` lookups.
    pub embedding_cache_enabled: bool,
    /// Maximum number of cached embedding vectors.
    pub embedding_cache_capacity: u64,
    /// TTL for an embedding cache entry, in seconds.
    pub embedding_cache_ttl_seconds: u64,
    /// Enable caching of `VectorStore::search` results.
    pub vector_search_cache_enabled: bool,
    /// Maximum number of cached vector search results per collection.
    pub vector_search_cache_capacity: u64,
    /// TTL for a vector search cache entry, in seconds.
    pub vector_search_cache_ttl_seconds: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            embedding_cache_enabled: true,
            embedding_cache_capacity: 10_000,
            embedding_cache_ttl_seconds: 3_600,
            vector_search_cache_enabled: true,
            vector_search_cache_capacity: 10_000,
            vector_search_cache_ttl_seconds: 300,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PathsConfig {
    pub data_dir: PathBuf,
    pub db_path: PathBuf,
    pub models_dir: PathBuf,
    pub adapters_dir: PathBuf,
    pub projects_dir: PathBuf,
    pub ccr_dir: PathBuf,
}

impl PathsConfig {
    pub fn from_data_dir(data_dir: impl Into<PathBuf>) -> Self {
        let data_dir = data_dir.into();
        Self {
            db_path: data_dir.join("crytex.db"),
            models_dir: data_dir.join("models"),
            adapters_dir: data_dir.join("adapters"),
            projects_dir: data_dir.join("projects"),
            ccr_dir: data_dir.join("ccr"),
            data_dir,
        }
    }
}

impl Default for PathsConfig {
    fn default() -> Self {
        let data_dir = dirs::data_dir()
            .map(|d| d.join("crytex"))
            .unwrap_or_else(|| PathBuf::from(".crytex"));
        Self::from_data_dir(data_dir)
    }
}

/// Security-related configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SecurityConfig {
    /// Master switch for the security scanner integration.
    pub enabled: bool,
    /// Scan file contents read by `fs_read` for prompt-injection patterns.
    pub scan_file_content: bool,
    /// Wrap file contents that triggered findings with an untrusted-content delimiter.
    pub wrap_untrusted_content: bool,
    /// Block the read entirely when findings reach `severity_threshold`.
    pub block_file_read_on_injection: bool,
    /// Minimum severity that counts as an actionable injection finding.
    pub severity_threshold: Severity,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            scan_file_content: true,
            wrap_untrusted_content: true,
            block_file_read_on_injection: false,
            severity_threshold: Severity::Medium,
        }
    }
}

/// Benchmark and A/B test harness configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BenchmarkConfig {
    pub default_concurrency: usize,
    pub default_timeout_seconds: u64,
    pub golden_sets_dir: PathBuf,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            default_concurrency: 1,
            default_timeout_seconds: 120,
            golden_sets_dir: dirs::data_dir()
                .map(|d| d.join("crytex").join("benchmarks"))
                .unwrap_or_else(|| PathBuf::from(".crytex/benchmarks")),
        }
    }
}

/// Workflow engine configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WorkflowConfig {
    /// Directory containing workflow TOML definitions.
    pub workflows_dir: PathBuf,
    /// Default workflow to use when no workflow matches a task kind.
    pub default_workflow: Option<String>,
}

impl Default for WorkflowConfig {
    fn default() -> Self {
        Self {
            workflows_dir: dirs::data_dir()
                .map(|d| d.join("crytex").join("workflows"))
                .unwrap_or_else(|| PathBuf::from(".crytex/workflows")),
            default_workflow: Some("codegen".to_string()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CrytexConfig {
    pub max_concurrent_tasks: usize,
    pub default_timeout_seconds: u64,
    #[serde(default)]
    pub inference: InferenceConfig,
    #[serde(default)]
    pub paths: PathsConfig,
    #[serde(default)]
    pub cache: CacheConfig,
    #[serde(default)]
    pub security: SecurityConfig,
    #[serde(default)]
    pub workflow: WorkflowConfig,
    #[serde(default)]
    pub benchmark: BenchmarkConfig,
    /// Search and retrieval configuration.
    #[serde(default)]
    pub search: SearchConfig,
    /// Incremental indexing configuration.
    #[serde(default)]
    pub indexing: IndexingConfig,
    /// Static mapping from agent role to active LoRA adapter id.
    #[serde(default)]
    pub role_adapters: HashMap<String, String>,
}

impl Default for CrytexConfig {
    fn default() -> Self {
        Self {
            max_concurrent_tasks: 4,
            default_timeout_seconds: 120,
            inference: InferenceConfig::default(),
            paths: PathsConfig::default(),
            cache: CacheConfig::default(),
            security: SecurityConfig::default(),
            workflow: WorkflowConfig::default(),
            benchmark: BenchmarkConfig::default(),
            search: SearchConfig::default(),
            indexing: IndexingConfig::default(),
            role_adapters: HashMap::new(),
        }
    }
}

/// Errors that can occur while loading a configuration file.
#[derive(Debug, Error)]
pub enum ConfigLoadError {
    #[error("failed to read config file: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to parse config file: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("failed to serialize config: {0}")]
    Serialize(#[from] toml::ser::Error),
}

impl CrytexConfig {
    /// Loads configuration from the default user config path.
    ///
    /// If the file does not exist or cannot be parsed, falls back to the default
    /// configuration so the application can still start.
    pub fn load() -> Self {
        Self::load_from_path(Self::config_path()).unwrap_or_default()
    }

    /// Returns the default path for the user configuration file.
    pub fn config_path() -> PathBuf {
        dirs::config_dir()
            .map(|d| d.join("crytex").join("config.toml"))
            .unwrap_or_else(|| PathBuf::from(".crytex").join("config.toml"))
    }

    /// Loads configuration from the provided path.
    pub fn load_from_path(path: impl AsRef<std::path::Path>) -> Result<Self, ConfigLoadError> {
        let contents = std::fs::read_to_string(path.as_ref())?;
        let config: Self = toml::from_str(&contents)?;
        Ok(config)
    }

    pub fn ensure_dirs(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.paths.data_dir)?;
        std::fs::create_dir_all(&self.paths.models_dir)?;
        std::fs::create_dir_all(&self.paths.adapters_dir)?;
        std::fs::create_dir_all(&self.paths.projects_dir)?;
        std::fs::create_dir_all(&self.paths.ccr_dir)?;
        std::fs::create_dir_all(&self.benchmark.golden_sets_dir)?;
        Ok(())
    }

    /// Saves the configuration to the default user config path.
    pub fn save(&self) -> Result<(), ConfigLoadError> {
        self.save_to_path(Self::config_path())
    }

    /// Saves the configuration to the provided path.
    pub fn save_to_path(&self, path: impl AsRef<std::path::Path>) -> Result<(), ConfigLoadError> {
        let parent = path.as_ref().parent().ok_or_else(|| {
            ConfigLoadError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "config path has no parent directory",
            ))
        })?;
        std::fs::create_dir_all(parent)?;
        let contents = toml::to_string_pretty(self)?;
        std::fs::write(path.as_ref(), contents)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_from_path_reads_custom_values() {
        let dir = std::env::temp_dir().join(format!("crytex-config-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        std::fs::write(
            &path,
            r#"
max_concurrent_tasks = 42
default_timeout_seconds = 60

[inference]
default_backend = "openai"
embedding_backend = "openai"
default_temperature = 0.5
default_max_tokens = 2048
request_timeout_seconds = 60
context_token_budget = 8192

[[inference.backends]]
id = "openai"
kind = "open_ai_compatible"
model = "gpt-4o-mini"
url = "https://api.openai.com/v1"
api_key = "sk-test"
supports_lora = false

[[inference.backends]]
id = "ollama"
kind = "ollama"
model = "qwen3.5:9b"
url = "http://localhost:11435"
supports_lora = false
"#,
        )
        .unwrap();

        let config = CrytexConfig::load_from_path(&path).unwrap();
        assert_eq!(config.max_concurrent_tasks, 42);
        assert_eq!(config.default_timeout_seconds, 60);
        assert_eq!(config.inference.default_backend.as_deref(), Some("openai"));
        assert_eq!(config.inference.backends.len(), 2);
        let openai = config.inference.backend("openai").unwrap();
        assert_eq!(openai.kind, BackendKind::OpenAiCompatible);
        assert_eq!(openai.model, "gpt-4o-mini");
        assert_eq!(config.inference.context_token_budget, Some(8192));

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn load_from_missing_path_returns_error() {
        let result = CrytexConfig::load_from_path("/does/not/exist/config.toml");
        assert!(result.is_err());
    }

    #[test]
    fn load_uses_default_when_config_file_missing() {
        let config = CrytexConfig::load();
        assert_eq!(
            config.max_concurrent_tasks,
            CrytexConfig::default().max_concurrent_tasks
        );
        assert!(config.inference.default_backend.is_none());
        assert!(config.inference.backends.is_empty());
    }

    #[test]
    fn default_config_has_no_backend() {
        let config = InferenceConfig::default();
        assert!(config.default_backend.is_none());
        assert!(config.backends.is_empty());
        assert!(config.rerank_backend.is_none());
    }

    #[test]
    fn backend_kind_onnx_serializes_to_snake_case() {
        let kind = BackendKind::Onnx;
        let serialized = serde_json::to_string(&kind).unwrap();
        assert_eq!(serialized, "\"onnx\"");
        let deserialized: BackendKind = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized, BackendKind::Onnx);
    }

    #[test]
    fn onnx_backend_config_roundtrips() {
        let mut config = InferenceConfig::default();
        config.backends.push(BackendConfig::onnx("local-embed", "nomic-ai/nomic-embed-text-v1.5"));
        config.rerank_backend = Some("local-rerank".to_string());

        let serialized = toml::to_string(&config).unwrap();
        assert!(serialized.contains("kind = \"onnx\""));
        assert!(serialized.contains("rerank_backend = \"local-rerank\""));

        let loaded: InferenceConfig = toml::from_str(&serialized).unwrap();
        assert_eq!(loaded.backends[0].kind, BackendKind::Onnx);
        assert_eq!(loaded.rerank_backend.as_deref(), Some("local-rerank"));
    }

    #[test]
    fn save_and_load_roundtrip() {
        let dir =
            std::env::temp_dir().join(format!("crytex-config-save-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");

        let mut config = InferenceConfig {
            default_backend: Some("openai".to_string()),
            ..Default::default()
        };
        config.backends.push(BackendConfig {
            id: "openai".to_string(),
            kind: BackendKind::OpenAiCompatible,
            model: "gpt-4o-mini".to_string(),
            url: Some("https://api.openai.com/v1".to_string()),
            api_key: Some("sk-test".to_string()),
            headers: std::collections::HashMap::new(),
            timeout_seconds: None,
            context_size: None,
            gpu_layers: None,
            supports_lora: false,
        });
        let crytex = CrytexConfig {
            inference: config,
            cache: CacheConfig::default(),
            ..CrytexConfig::default()
        };

        crytex.save_to_path(&path).unwrap();
        let loaded = CrytexConfig::load_from_path(&path).unwrap();
        assert_eq!(loaded.inference.default_backend.as_deref(), Some("openai"));
        assert_eq!(loaded.inference.backends.len(), 1);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn load_workflow_config() {
        let dir = std::env::temp_dir().join(format!("crytex-config-wf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        std::fs::write(
            &path,
            r#"
max_concurrent_tasks = 2

[workflow]
workflows_dir = "/tmp/workflows"
default_workflow = "research"
"#,
        )
        .unwrap();

        let config = CrytexConfig::load_from_path(&path).unwrap();
        assert_eq!(config.max_concurrent_tasks, 2);
        assert_eq!(
            config.workflow.workflows_dir,
            std::path::PathBuf::from("/tmp/workflows")
        );
        assert_eq!(
            config.workflow.default_workflow,
            Some("research".to_string())
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn default_search_config_uses_rrf() {
        let config = SearchConfig::default();
        assert_eq!(config.fusion_strategy, FusionStrategyKind::Rrf);
        assert_eq!(config.rrf_k, 60.0);
        assert_eq!(config.per_collection_limit, 20);
        assert_eq!(config.final_limit, 10);
    }

    #[test]
    fn default_indexing_config_is_enabled() {
        let config = IndexingConfig::default();
        assert!(config.incremental_enabled);
        assert_eq!(config.debounce_ms, 500);
    }

    #[test]
    fn search_config_roundtrips_through_toml() {
        let config = SearchConfig {
            fusion_strategy: FusionStrategyKind::Dbsf,
            ..Default::default()
        };
        let serialized = toml::to_string(&config).unwrap();
        assert!(serialized.contains("strategy = \"dbsf\""));
        let loaded: SearchConfig = toml::from_str(&serialized).unwrap();
        assert_eq!(loaded.fusion_strategy, FusionStrategyKind::Dbsf);
    }

    #[test]
    fn custom_headers_are_parsed() {
        let dir =
            std::env::temp_dir().join(format!("crytex-config-test-headers-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        std::fs::write(
            &path,
            r#"
[[inference.backends]]
id = "custom"
kind = "custom"
model = "any"
url = "http://localhost:1234"
headers = { "X-Custom" = "value", "Authorization" = "Bearer token" }
supports_lora = false
"#,
        )
        .unwrap();

        let config = CrytexConfig::load_from_path(&path).unwrap();
        let backend = config.inference.backend("custom").unwrap();
        assert_eq!(backend.headers.get("X-Custom").unwrap(), "value");
        assert_eq!(
            backend.headers.get("Authorization").unwrap(),
            "Bearer token"
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
