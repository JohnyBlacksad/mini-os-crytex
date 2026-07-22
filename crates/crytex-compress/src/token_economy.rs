use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::ccr::{CcrStore, CcrStoreError, compute_key};
use crate::message::Message;
use crate::token::{TokenError, TokenEstimator};

/// Errors returned by token-economy services.
#[derive(Debug, thiserror::Error)]
pub enum TokenEconomyError {
    #[error("token estimator failed: {0}")]
    Token(#[from] TokenError),
    #[error("ccr store failed: {0}")]
    Ccr(#[from] CcrStoreError),
    #[error("token profile not found for backend '{backend}' and model '{model}'")]
    MissingProfile { backend: String, model: String },
    #[error("model context window is exhausted by prompt and completion reserve")]
    ExhaustedContext,
}

/// Context-window profile for a concrete backend/model pair.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelTokenProfile {
    pub backend: String,
    pub model: String,
    pub context_window: usize,
    pub safety_margin_tokens: usize,
}

impl ModelTokenProfile {
    /// Build a profile with a conservative 10% prompt-cache/headroom reserve.
    pub fn new(
        backend: impl Into<String>,
        model: impl Into<String>,
        context_window: usize,
    ) -> Self {
        Self {
            backend: backend.into(),
            model: model.into(),
            context_window,
            safety_margin_tokens: context_window / 10,
        }
    }
}

/// Budget allocation produced for one request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenBudgetAllocation {
    pub total_budget: usize,
    pub prompt_tokens: usize,
    pub reserved_completion_tokens: usize,
    pub safety_margin_tokens: usize,
    pub rag_budget: usize,
    pub artifact_budget: usize,
    pub shared_context_budget: usize,
}

/// Plans token headroom per backend/model without depending on inference code.
#[derive(Debug, Clone, Default)]
pub struct TokenBudgetPlanner {
    profiles: HashMap<(String, String), ModelTokenProfile>,
}

impl TokenBudgetPlanner {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_profile(mut self, profile: ModelTokenProfile) -> Self {
        self.profiles
            .insert((profile.backend.clone(), profile.model.clone()), profile);
        self
    }

    pub fn plan(
        &self,
        backend: &str,
        model: &str,
        prompt_tokens: usize,
        completion_tokens: usize,
    ) -> Result<TokenBudgetAllocation, TokenEconomyError> {
        let profile = self
            .profiles
            .get(&(backend.to_string(), model.to_string()))
            .ok_or_else(|| TokenEconomyError::MissingProfile {
                backend: backend.to_string(),
                model: model.to_string(),
            })?;
        let fixed = prompt_tokens
            .saturating_add(completion_tokens)
            .saturating_add(profile.safety_margin_tokens);
        let remaining = profile
            .context_window
            .checked_sub(fixed)
            .ok_or(TokenEconomyError::ExhaustedContext)?;

        Ok(TokenBudgetAllocation {
            total_budget: profile.context_window,
            prompt_tokens,
            reserved_completion_tokens: completion_tokens,
            safety_margin_tokens: profile.safety_margin_tokens,
            rag_budget: remaining / 2,
            artifact_budget: remaining / 3,
            shared_context_budget: remaining.saturating_sub(remaining / 2 + remaining / 3),
        })
    }
}

/// Shared compressed context entry reusable by multiple agents.
#[derive(Debug, Clone)]
pub struct SharedContextEntry {
    pub key: String,
    pub original: String,
    pub compressed: String,
    pub original_tokens: usize,
    pub compressed_tokens: usize,
    pub producers: Vec<String>,
    created_at: Instant,
}

/// Aggregate metrics for shared context reuse.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SharedContextStats {
    pub entries: usize,
    pub total_original_tokens: usize,
    pub total_compressed_tokens: usize,
    pub total_tokens_saved: usize,
    pub cache_hits: usize,
}

/// Agent-shared context cache. It keeps originals local while exposing a small
/// compressed representation to downstream agents.
pub struct SharedContext {
    max_entries: usize,
    ttl: Duration,
    estimator: Arc<dyn TokenEstimator>,
    entries: HashMap<String, SharedContextEntry>,
    order: VecDeque<String>,
    cache_hits: usize,
}

impl SharedContext {
    pub fn new(max_entries: usize, ttl_seconds: u64, estimator: Arc<dyn TokenEstimator>) -> Self {
        Self {
            max_entries,
            ttl: Duration::from_secs(ttl_seconds),
            estimator,
            entries: HashMap::new(),
            order: VecDeque::new(),
            cache_hits: 0,
        }
    }

    pub fn put(
        &mut self,
        key: impl Into<String>,
        content: &str,
        agent: Option<&str>,
    ) -> Result<SharedContextEntry, TokenEconomyError> {
        let key = key.into();
        self.remove_expired();
        let original_tokens = self.estimator.estimate_text(content)?;
        let compressed = compress_preserving_markers(content, &[], original_tokens / 3);
        let compressed_tokens = self.estimator.estimate_text(&compressed)?;
        let producer = agent.map(str::to_string);

        if let Some(existing) = self.entries.get_mut(&key) {
            self.cache_hits += 1;
            if let Some(producer) = producer.filter(|p| !existing.producers.contains(p)) {
                existing.producers.push(producer);
            }
            return Ok(existing.clone());
        }

        let entry = SharedContextEntry {
            key: key.clone(),
            original: content.to_string(),
            compressed,
            original_tokens,
            compressed_tokens,
            producers: producer.into_iter().collect(),
            created_at: Instant::now(),
        };
        self.evict_if_needed();
        self.order.push_back(key.clone());
        self.entries.insert(key, entry.clone());
        Ok(entry)
    }

    pub fn get(&mut self, key: &str, full: bool) -> Option<String> {
        self.remove_expired();
        self.entries.get(key).map(|entry| {
            if full {
                entry.original.clone()
            } else {
                entry.compressed.clone()
            }
        })
    }

    pub fn stats(&self) -> SharedContextStats {
        let (total_original_tokens, total_compressed_tokens) =
            self.entries
                .values()
                .fold((0, 0), |(original, compressed), entry| {
                    (
                        original + entry.original_tokens,
                        compressed + entry.compressed_tokens,
                    )
                });

        SharedContextStats {
            entries: self.entries.len(),
            total_original_tokens,
            total_compressed_tokens,
            total_tokens_saved: total_original_tokens.saturating_sub(total_compressed_tokens)
                + self.reuse_savings(),
            cache_hits: self.cache_hits,
        }
    }

    fn evict_if_needed(&mut self) {
        while self.entries.len() >= self.max_entries {
            self.order
                .pop_front()
                .and_then(|oldest| self.entries.remove(&oldest));
        }
    }

    fn remove_expired(&mut self) {
        let ttl = self.ttl;
        let expired = self
            .entries
            .iter()
            .filter(|(_, entry)| entry.created_at.elapsed() > ttl)
            .map(|(key, _)| key.clone())
            .collect::<Vec<_>>();
        expired.into_iter().for_each(|key| {
            self.entries.remove(&key);
            self.order.retain(|stored| stored != &key);
        });
    }

    fn reuse_savings(&self) -> usize {
        self.entries
            .values()
            .map(|entry| {
                entry
                    .producers
                    .len()
                    .saturating_sub(1)
                    .saturating_mul(entry.compressed_tokens)
            })
            .sum()
    }
}

/// Large artifact category eligible for CCR offload.
#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ArtifactKind {
    Diff,
    Log,
    Report,
    ToolOutput,
}

impl ArtifactKind {
    fn label(self) -> &'static str {
        match self {
            Self::Diff => "diff",
            Self::Log => "log",
            Self::Report => "report",
            Self::ToolOutput => "tool-output",
        }
    }
}

/// Result of compressing/offloading a large artifact.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ArtifactOffloadReport {
    pub kind: ArtifactKind,
    pub marker: String,
    pub ccr_key: Option<String>,
    pub original_tokens: usize,
    pub compressed_tokens: usize,
}

/// Stores large artifacts in CCR and emits compact retrieval markers.
pub struct ArtifactOffload {
    store: Arc<dyn CcrStore>,
    estimator: Arc<dyn TokenEstimator>,
    threshold_tokens: usize,
}

impl ArtifactOffload {
    pub fn new(
        store: Arc<dyn CcrStore>,
        estimator: Arc<dyn TokenEstimator>,
        threshold_tokens: usize,
    ) -> Self {
        Self {
            store,
            estimator,
            threshold_tokens,
        }
    }

    pub fn offload(
        &self,
        kind: ArtifactKind,
        content: &str,
    ) -> Result<ArtifactOffloadReport, TokenEconomyError> {
        let original_tokens = self.estimator.estimate_text(content)?;
        if original_tokens <= self.threshold_tokens {
            return Ok(ArtifactOffloadReport {
                kind,
                marker: content.to_string(),
                ccr_key: None,
                original_tokens,
                compressed_tokens: original_tokens,
            });
        }

        let ccr_key = compute_key(content);
        self.store.put(&ccr_key, content.to_string())?;
        let fact_preview = uppercase_fact_preview(content);
        let marker = format!(
            "[{} compressed from {} tokens; facts: {}; retrieve original: ccr:{}]",
            kind.label(),
            original_tokens,
            fact_preview,
            ccr_key
        );
        let compressed_tokens = self.estimator.estimate_text(&marker)?;

        Ok(ArtifactOffloadReport {
            kind,
            marker,
            ccr_key: Some(ccr_key),
            original_tokens,
            compressed_tokens,
        })
    }
}

/// Metrics emitted by a token-economy run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TokenEconomyMetrics {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub saved_tokens: usize,
    pub compression_ratio: f64,
    pub quality_loss: f64,
}

/// Request optimized by [`TokenEconomyEngine`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TokenEconomyRequest {
    pub backend: String,
    pub model: String,
    pub messages: Vec<Message>,
    pub artifacts: Vec<(ArtifactKind, String)>,
    pub required_facts: Vec<String>,
    pub expected_completion_tokens: usize,
    pub trace_id: String,
}

/// Full result with optimized messages and evidence.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TokenEconomyReport {
    pub trace_id: String,
    pub backend: String,
    pub model: String,
    pub metrics: TokenEconomyMetrics,
    pub optimized_messages: Vec<Message>,
    pub artifact_reports: Vec<ArtifactOffloadReport>,
    pub quality: CompressionQualityReport,
}

/// Production orchestrator for prompt/headroom optimization.
pub struct TokenEconomyEngine {
    estimator: Arc<dyn TokenEstimator>,
    artifact_offload: ArtifactOffload,
    shared_context: Option<SharedContext>,
}

impl TokenEconomyEngine {
    pub fn new(estimator: Arc<dyn TokenEstimator>, store: Arc<dyn CcrStore>) -> Self {
        Self {
            artifact_offload: ArtifactOffload::new(store, estimator.clone(), 256),
            estimator,
            shared_context: None,
        }
    }

    pub fn with_shared_context(mut self, shared_context: SharedContext) -> Self {
        self.shared_context = Some(shared_context);
        self
    }

    pub fn optimize(
        &mut self,
        request: TokenEconomyRequest,
    ) -> Result<TokenEconomyReport, TokenEconomyError> {
        let input_prompt_tokens = self.estimator.estimate_messages(&request.messages)?
            + request
                .artifacts
                .iter()
                .map(|(_, content)| self.estimator.estimate_text(content))
                .try_fold(0, |acc, tokens| tokens.map(|count| acc + count))?;
        let mut optimized_messages = self.compress_messages(&request)?;
        let artifact_reports = request
            .artifacts
            .iter()
            .map(|(kind, content)| self.artifact_offload.offload(*kind, content))
            .collect::<Result<Vec<_>, _>>()?;

        optimized_messages.extend(
            artifact_reports
                .iter()
                .map(|report| Message::user(report.marker.clone())),
        );

        let output_prompt_tokens = self.estimator.estimate_messages(&optimized_messages)?;
        let quality = CompressionQualityBenchmark::new(self.estimator.clone()).run_messages(
            &optimized_messages,
            &request.required_facts,
            input_prompt_tokens,
        );
        let saved_tokens = input_prompt_tokens.saturating_sub(output_prompt_tokens);
        let compression_ratio = ratio(output_prompt_tokens, input_prompt_tokens);

        Ok(TokenEconomyReport {
            trace_id: request.trace_id,
            backend: request.backend,
            model: request.model,
            metrics: TokenEconomyMetrics {
                prompt_tokens: output_prompt_tokens,
                completion_tokens: request.expected_completion_tokens,
                saved_tokens,
                compression_ratio,
                quality_loss: quality.quality_loss,
            },
            optimized_messages,
            artifact_reports,
            quality,
        })
    }

    fn compress_messages(
        &mut self,
        request: &TokenEconomyRequest,
    ) -> Result<Vec<Message>, TokenEconomyError> {
        request
            .messages
            .iter()
            .enumerate()
            .map(|(index, message)| self.compress_message(index, message, &request.required_facts))
            .collect()
    }

    fn compress_message(
        &mut self,
        index: usize,
        message: &Message,
        required_facts: &[String],
    ) -> Result<Message, TokenEconomyError> {
        let tokens = self.estimator.estimate_message(message)?;
        if tokens <= 256 {
            return Ok(message.clone());
        }

        let compressed = compress_preserving_markers(&message.content, required_facts, 128);
        self.shared_context
            .as_mut()
            .map(|context| context.put(format!("message-{index}"), &message.content, None))
            .transpose()?;
        Ok(Message::new(message.role.clone(), compressed))
    }
}

/// Quality benchmark checking whether required facts survive compression.
pub struct CompressionQualityBenchmark {
    estimator: Arc<dyn TokenEstimator>,
}

impl CompressionQualityBenchmark {
    pub fn new(estimator: Arc<dyn TokenEstimator>) -> Self {
        Self { estimator }
    }

    pub fn run(
        &self,
        text: String,
        required_facts: &[String],
        budget: usize,
    ) -> CompressionQualityReport {
        let original_tokens = self.estimator.estimate_text(&text).unwrap_or(text.len());
        let compressed = compress_preserving_markers(&text, required_facts, budget);
        let compressed_tokens = self
            .estimator
            .estimate_text(&compressed)
            .unwrap_or(compressed.len());
        let missing_facts = missing_facts(&compressed, required_facts);

        CompressionQualityReport {
            passed: missing_facts.is_empty() && compressed_tokens <= original_tokens,
            missing_facts,
            compression_ratio: ratio(compressed_tokens, original_tokens),
            quality_loss: quality_loss(&compressed, required_facts),
        }
    }

    fn run_messages(
        &self,
        messages: &[Message],
        required_facts: &[String],
        original_tokens: usize,
    ) -> CompressionQualityReport {
        let compressed_text = messages
            .iter()
            .map(|message| message.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let compressed_tokens = self
            .estimator
            .estimate_messages(messages)
            .unwrap_or(compressed_text.len());
        let missing_facts = missing_facts(&compressed_text, required_facts);

        CompressionQualityReport {
            passed: missing_facts.is_empty() && compressed_tokens <= original_tokens,
            missing_facts,
            compression_ratio: ratio(compressed_tokens, original_tokens),
            quality_loss: quality_loss(&compressed_text, required_facts),
        }
    }
}

/// Result of a required-fact preservation benchmark.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompressionQualityReport {
    pub passed: bool,
    pub missing_facts: Vec<String>,
    pub compression_ratio: f64,
    pub quality_loss: f64,
}

fn compress_preserving_markers(text: &str, required_facts: &[String], budget: usize) -> String {
    let required = required_facts
        .iter()
        .filter(|fact| text.contains(fact.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    let mut selected = required.join("\n");
    let headroom = budget.saturating_mul(4).saturating_sub(selected.len());
    let prefix = text.chars().take(headroom).collect::<String>();

    if selected.is_empty() {
        selected = prefix;
    } else if !prefix.is_empty() {
        selected.push('\n');
        selected.push_str(&prefix);
    }

    selected
}

fn missing_facts(text: &str, required_facts: &[String]) -> Vec<String> {
    required_facts
        .iter()
        .filter(|fact| !text.contains(fact.as_str()))
        .cloned()
        .collect()
}

fn quality_loss(text: &str, required_facts: &[String]) -> f64 {
    if required_facts.is_empty() {
        0.0
    } else {
        missing_facts(text, required_facts).len() as f64 / required_facts.len() as f64
    }
}

fn ratio(part: usize, whole: usize) -> f64 {
    if whole == 0 {
        1.0
    } else {
        part as f64 / whole as f64
    }
}

fn uppercase_fact_preview(text: &str) -> String {
    let facts = text
        .split_whitespace()
        .filter(|word| word.chars().any(|c| c == '_') && word.chars().all(is_fact_char))
        .take(8)
        .collect::<Vec<_>>();

    if facts.is_empty() {
        "none".to_string()
    } else {
        facts.join(",")
    }
}

fn is_fact_char(c: char) -> bool {
    c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_'
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::ccr::{CcrStore, InMemoryCcrStore};
    use crate::message::Message;
    use crate::token::CharTokenEstimator;

    #[test]
    fn token_budget_planner_reserves_model_headroom_per_backend() {
        let planner = super::TokenBudgetPlanner::new()
            .with_profile(super::ModelTokenProfile::new(
                "ollama",
                "qwen3.5:9b",
                32_768,
            ))
            .with_profile(super::ModelTokenProfile::new("openai", "gpt-4o", 128_000));

        let qwen = planner.plan("ollama", "qwen3.5:9b", 2_000, 4_000).unwrap();
        let gpt = planner.plan("openai", "gpt-4o", 2_000, 4_000).unwrap();

        assert!(qwen.rag_budget < gpt.rag_budget);
        assert_eq!(qwen.reserved_completion_tokens, 4_000);
        assert!(qwen.total_budget <= 32_768);
    }

    #[test]
    fn shared_context_deduplicates_rag_between_agents_and_reports_savings() {
        let mut context = super::SharedContext::new(10, 3600, Arc::new(CharTokenEstimator));
        let rag = "RAG_FACT_ALPHA ".repeat(200);

        let first = context
            .put("project-rag", &rag, Some("researcher"))
            .unwrap();
        let second = context.put("project-rag", &rag, Some("coder")).unwrap();

        assert_eq!(first.key, second.key);
        assert!(
            context
                .get("project-rag", false)
                .unwrap()
                .contains("RAG_FACT_ALPHA")
        );
        assert_eq!(context.get("project-rag", true).unwrap(), rag);
        assert!(context.stats().total_tokens_saved > 0);
    }

    #[test]
    fn artifact_offload_stores_large_diffs_logs_reports_and_tool_outputs_in_ccr() {
        let store: Arc<dyn CcrStore> = Arc::new(InMemoryCcrStore::new());
        let offloader =
            super::ArtifactOffload::new(store.clone(), Arc::new(CharTokenEstimator), 64);
        let content = "CRITICAL_FACT_X ".repeat(300);

        for kind in [
            super::ArtifactKind::Diff,
            super::ArtifactKind::Log,
            super::ArtifactKind::Report,
            super::ArtifactKind::ToolOutput,
        ] {
            let compressed = offloader.offload(kind, &content).unwrap();
            assert!(compressed.marker.contains("ccr:"));
            assert!(compressed.compressed_tokens < compressed.original_tokens);
            assert_eq!(
                store.get(&compressed.ccr_key.unwrap()).unwrap(),
                Some(content.clone())
            );
        }
    }

    #[test]
    fn token_economy_engine_reports_saved_tokens_ratio_and_quality_loss() {
        let store: Arc<dyn CcrStore> = Arc::new(InMemoryCcrStore::new());
        let mut engine = super::TokenEconomyEngine::new(Arc::new(CharTokenEstimator), store)
            .with_shared_context(super::SharedContext::new(
                10,
                3600,
                Arc::new(CharTokenEstimator),
            ));
        let report = engine
            .optimize(super::TokenEconomyRequest {
                backend: "ollama".into(),
                model: "qwen3.5:9b".into(),
                messages: vec![
                    Message::system("You are a coder."),
                    Message::user("REQUIRED_FACT_OMEGA ".repeat(200)),
                ],
                artifacts: vec![(super::ArtifactKind::Log, "REQUIRED_FACT_SIGMA ".repeat(300))],
                required_facts: vec!["REQUIRED_FACT_OMEGA".into(), "REQUIRED_FACT_SIGMA".into()],
                expected_completion_tokens: 512,
                trace_id: "trace-token-economy".into(),
            })
            .unwrap();

        assert!(report.metrics.saved_tokens > 0);
        assert!(report.metrics.compression_ratio < 1.0);
        assert_eq!(report.metrics.quality_loss, 0.0);
        assert!(report.metrics.prompt_tokens > 0);
        assert!(
            report
                .optimized_messages
                .iter()
                .any(|m| m.content.contains("ccr:"))
        );
    }

    #[test]
    fn compression_quality_benchmark_preserves_required_facts() {
        let benchmark = super::CompressionQualityBenchmark::new(Arc::new(CharTokenEstimator));
        let report = benchmark.run(
            "Important intro. REQUIRED_FACT_ALPHA. filler ".repeat(200),
            &["REQUIRED_FACT_ALPHA".into()],
            128,
        );

        assert!(report.passed);
        assert_eq!(report.missing_facts, Vec::<String>::new());
        assert!(report.compression_ratio < 1.0);
    }
}
