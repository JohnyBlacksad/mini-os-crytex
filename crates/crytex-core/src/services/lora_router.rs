//! Resolves the LoRA adapter that should be used for a given task.
//!
//! Resolution order:
//! 1. Explicit override in `task.payload["lora"]`.
//! 2. Domain heuristic from [`LoraEvolutionService::select_lora`].
//! 3. Semantic fallback over the `lora_adapters` vector collection.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use crate::models::Task;
use crate::services::{
    AgentRole, Embedder, LoraEvolutionError, LoraEvolutionService,
    vector_store::{SearchOptions, VectorStore},
};

/// Errors returned by [`LoraRouter`].
#[derive(Debug, Error)]
pub enum LoraRouterError {
    #[error("evolution service error: {0}")]
    Evolution(#[from] LoraEvolutionError),
    #[error("invalid lora payload: {0}")]
    InvalidPayload(String),
    #[error("semantic search failed: {0}")]
    Semantic(String),
}

const LORA_ADAPTER_COLLECTION: &str = "lora_adapters";

/// Registry that maps an agent role to the currently active LoRA adapter id.
#[async_trait]
pub trait RoleAdapterRegistry: Send + Sync {
    /// Return the active adapter for `role`, if one is configured.
    fn get(&self, role: AgentRole) -> Option<String>;
    /// Assign `adapter_id` as the active adapter for `role`.
    fn set(&self, role: AgentRole, adapter_id: String);
}

/// In-memory implementation of [`RoleAdapterRegistry`].
#[derive(Default)]
pub struct MemoryRoleAdapterRegistry {
    mapping: std::sync::RwLock<HashMap<AgentRole, String>>,
}

impl MemoryRoleAdapterRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a registry pre-populated with the given mapping.
    pub fn with_mapping(mapping: HashMap<AgentRole, String>) -> Self {
        Self {
            mapping: std::sync::RwLock::new(mapping),
        }
    }
}

impl RoleAdapterRegistry for MemoryRoleAdapterRegistry {
    fn get(&self, role: AgentRole) -> Option<String> {
        self.mapping.read().ok().and_then(|g| g.get(&role).cloned())
    }

    fn set(&self, role: AgentRole, adapter_id: String) {
        if let Ok(mut guard) = self.mapping.write() {
            guard.insert(role, adapter_id);
        }
    }
}

/// Decides which LoRA adapter (if any) should be active for a task.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoraSelection {
    pub adapter_id: String,
    pub role: Option<String>,
    pub source: String,
    pub reason: String,
}

impl LoraSelection {
    fn new(
        adapter_id: String,
        role: Option<AgentRole>,
        source: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            adapter_id,
            role: role.map(|r| r.as_str().to_string()),
            source: source.into(),
            reason: reason.into(),
        }
    }
}

/// Decides which LoRA adapter (if any) should be active for a task.
#[async_trait]
pub trait LoraRouter: Send + Sync {
    /// Resolve the adapter id for `task`.
    async fn resolve(
        &self,
        task: &Task,
        project_id: &str,
    ) -> Result<Option<String>, LoraRouterError>;

    /// Resolve the adapter id for a specific agent `role`.
    async fn resolve_for_role(
        &self,
        role: AgentRole,
        project_id: &str,
    ) -> Result<Option<String>, LoraRouterError>;

    /// Resolve the adapter selection with diagnostics for `task`.
    async fn resolve_selection(
        &self,
        task: &Task,
        project_id: &str,
    ) -> Result<Option<LoraSelection>, LoraRouterError> {
        Ok(self
            .resolve(task, project_id)
            .await?
            .map(|adapter_id| LoraSelection {
                adapter_id,
                role: task
                    .assigned_agent
                    .as_deref()
                    .and_then(AgentRole::from_agent)
                    .map(|role| role.as_str().to_string()),
                source: "legacy_router".to_string(),
                reason: "adapter selected by LoraRouter::resolve".to_string(),
            }))
    }

    /// Resolve the adapter selection with diagnostics for a specific agent `role`.
    async fn resolve_selection_for_role(
        &self,
        role: AgentRole,
        project_id: &str,
    ) -> Result<Option<LoraSelection>, LoraRouterError> {
        Ok(self
            .resolve_for_role(role, project_id)
            .await?
            .map(|adapter_id| LoraSelection {
                adapter_id,
                role: Some(role.as_str().to_string()),
                source: "legacy_role_router".to_string(),
                reason: format!("adapter selected for {} role", role.as_str()),
            }))
    }
}

/// Default implementation of [`LoraRouter`].
pub struct LoraRouterImpl {
    evolution: Arc<dyn LoraEvolutionService>,
    role_registry: Option<Arc<dyn RoleAdapterRegistry>>,
    embedder: Option<Arc<dyn Embedder>>,
    vector_store: Option<Arc<dyn VectorStore>>,
}

impl LoraRouterImpl {
    /// Create a new router backed by the given evolution service.
    pub fn new(evolution: Arc<dyn LoraEvolutionService>) -> Self {
        Self {
            evolution,
            role_registry: None,
            embedder: None,
            vector_store: None,
        }
    }

    /// Provide a role-based adapter registry.
    pub fn with_role_registry(mut self, registry: Arc<dyn RoleAdapterRegistry>) -> Self {
        self.role_registry = Some(registry);
        self
    }

    /// Enable semantic fallback using the given embedder and vector store.
    pub fn with_semantic_fallback(
        mut self,
        embedder: Arc<dyn Embedder>,
        vector_store: Arc<dyn VectorStore>,
    ) -> Self {
        self.embedder = Some(embedder);
        self.vector_store = Some(vector_store);
        self
    }

    fn task_text(task: &Task) -> String {
        let mut text = task.kind.clone();
        text.push(' ');
        text.push_str(&task.title);
        if let Some(description) = &task.description {
            text.push('\n');
            text.push_str(description);
        }
        text
    }

    async fn semantic_fallback(&self, text: &str) -> Result<Option<String>, LoraRouterError> {
        let (embedder, vector_store) = match (&self.embedder, &self.vector_store) {
            (Some(e), Some(v)) => (e, v),
            _ => return Ok(None),
        };

        let dim = embedder
            .dimension()
            .await
            .map_err(|e| LoraRouterError::Semantic(e.to_string()))?;
        let vector = embedder
            .embed(text)
            .await
            .map_err(|e| LoraRouterError::Semantic(e.to_string()))?;

        vector_store
            .create_collection(LORA_ADAPTER_COLLECTION, dim)
            .await
            .map_err(|e| LoraRouterError::Semantic(e.to_string()))?;

        let results = vector_store
            .search(
                LORA_ADAPTER_COLLECTION,
                &vector,
                SearchOptions {
                    limit: 1,
                    filter: None,
                    score_threshold: Some(0.5),
                },
            )
            .await
            .map_err(|e| LoraRouterError::Semantic(e.to_string()))?;

        Ok(results.into_iter().next().and_then(|result| {
            result
                .payload
                .get("adapter_id")
                .and_then(|v| v.as_str())
                .map(String::from)
        }))
    }

    fn explicit_override(payload: &Value) -> Result<Option<String>, LoraRouterError> {
        match payload.get("lora") {
            Some(Value::String(id)) => Ok(Some(id.clone())),
            Some(other) => Err(LoraRouterError::InvalidPayload(other.to_string())),
            None => Ok(None),
        }
    }
}

#[async_trait]
impl LoraRouter for LoraRouterImpl {
    async fn resolve(
        &self,
        task: &Task,
        project_id: &str,
    ) -> Result<Option<String>, LoraRouterError> {
        Ok(self
            .resolve_selection(task, project_id)
            .await?
            .map(|selection| selection.adapter_id))
    }

    async fn resolve_for_role(
        &self,
        role: AgentRole,
        project_id: &str,
    ) -> Result<Option<String>, LoraRouterError> {
        Ok(self
            .resolve_selection_for_role(role, project_id)
            .await?
            .map(|selection| selection.adapter_id))
    }

    async fn resolve_selection(
        &self,
        task: &Task,
        project_id: &str,
    ) -> Result<Option<LoraSelection>, LoraRouterError> {
        let role = task
            .assigned_agent
            .as_deref()
            .and_then(AgentRole::from_agent);

        if let Some(id) = Self::explicit_override(&task.payload)? {
            return Ok(Some(LoraSelection::new(
                id,
                role,
                "explicit_override",
                "task payload requested a specific LoRA adapter",
            )));
        }

        if let Some(role) = role
            && let Some(selection) = self.resolve_selection_for_role(role, project_id).await?
        {
            return Ok(Some(selection));
        }

        if let Some(id) = self
            .evolution
            .select_lora(task, project_id)
            .await
            .map_err(LoraRouterError::Evolution)?
        {
            return Ok(Some(LoraSelection::new(
                id,
                role,
                "task_evolution",
                "evolution service selected the best adapter for this task",
            )));
        }

        Ok(self
            .semantic_fallback(&Self::task_text(task))
            .await?
            .map(|id| {
                LoraSelection::new(
                    id,
                    role,
                    "semantic_fallback",
                    "semantic adapter search matched the task text",
                )
            }))
    }

    async fn resolve_selection_for_role(
        &self,
        role: AgentRole,
        project_id: &str,
    ) -> Result<Option<LoraSelection>, LoraRouterError> {
        if let Some(registry) = &self.role_registry
            && let Some(id) = registry.get(role)
        {
            return Ok(Some(LoraSelection::new(
                id,
                Some(role),
                "role_registry",
                format!("active adapter registered for {} role", role.as_str()),
            )));
        }

        if let Some(id) = self
            .evolution
            .select_lora_by_role(role, project_id)
            .await
            .map_err(LoraRouterError::Evolution)?
        {
            return Ok(Some(LoraSelection::new(
                id,
                Some(role),
                "role_evolution",
                format!(
                    "evolution service selected adapter for {} role",
                    role.as_str()
                ),
            )));
        }

        Ok(self.semantic_fallback(role.as_str()).await?.map(|id| {
            LoraSelection::new(
                id,
                Some(role),
                "semantic_fallback",
                format!("semantic adapter search matched {} role", role.as_str()),
            )
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Task, TaskStatus};
    use crate::services::{
        LoraEvolutionError,
        vector_store::{VectorPoint, VectorStoreError},
    };
    use async_trait::async_trait;
    use serde_json::json;
    use std::collections::HashMap;
    use std::sync::Mutex;

    type CollectionMap = HashMap<String, (usize, HashMap<String, VectorPoint>)>;

    #[derive(Default)]
    struct TestVectorStore {
        collections: Mutex<CollectionMap>,
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }

    #[async_trait]
    impl VectorStore for TestVectorStore {
        async fn create_collection(
            &self,
            collection: &str,
            dim: usize,
        ) -> Result<(), VectorStoreError> {
            self.collections
                .lock()
                .unwrap()
                .entry(collection.to_string())
                .or_insert((dim, HashMap::new()));
            Ok(())
        }
        async fn delete_collection(&self, collection: &str) -> Result<(), VectorStoreError> {
            self.collections.lock().unwrap().remove(collection);
            Ok(())
        }
        async fn upsert(
            &self,
            collection: &str,
            points: Vec<VectorPoint>,
        ) -> Result<(), VectorStoreError> {
            let mut collections = self.collections.lock().unwrap();
            let entry = collections.get_mut(collection).ok_or_else(|| {
                VectorStoreError::Collection(format!("collection {} does not exist", collection))
            })?;
            for point in points {
                if point.vector.len() != entry.0 {
                    return Err(VectorStoreError::DimensionMismatch {
                        expected: entry.0,
                        actual: point.vector.len(),
                    });
                }
                entry.1.insert(point.id.clone(), point);
            }
            Ok(())
        }
        async fn search(
            &self,
            collection: &str,
            vector: &[f32],
            options: SearchOptions,
        ) -> Result<Vec<crate::services::vector_store::SearchResult>, VectorStoreError> {
            let collections = self.collections.lock().unwrap();
            let entry = collections.get(collection).ok_or_else(|| {
                VectorStoreError::Collection(format!("collection {} does not exist", collection))
            })?;
            let mut results: Vec<_> = entry
                .1
                .values()
                .map(|point| crate::services::vector_store::SearchResult {
                    id: point.id.clone(),
                    score: cosine_similarity(vector, &point.vector),
                    payload: point.payload.clone(),
                })
                .filter(|result| {
                    options
                        .score_threshold
                        .is_none_or(|threshold| result.score >= threshold)
                })
                .collect();
            results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            results.truncate(options.limit.max(1));
            Ok(results)
        }
    }

    struct MockEvolution {
        selected: Option<String>,
    }

    #[async_trait]
    impl LoraEvolutionService for MockEvolution {
        async fn collect_golden_example(&self, _task_id: &str) -> Result<(), LoraEvolutionError> {
            unimplemented!()
        }
        async fn collect_counter_example(&self, _task_id: &str) -> Result<(), LoraEvolutionError> {
            unimplemented!()
        }
        async fn should_train(&self, _task_kind: &str) -> Result<bool, LoraEvolutionError> {
            unimplemented!()
        }
        async fn train_and_register(
            &self,
            _task_kind: &str,
        ) -> Result<crate::models::LoraAdapter, LoraEvolutionError> {
            unimplemented!()
        }
        async fn should_train_for_role(
            &self,
            _role: AgentRole,
        ) -> Result<bool, LoraEvolutionError> {
            unimplemented!()
        }
        async fn train_and_register_for_role(
            &self,
            _role: AgentRole,
        ) -> Result<crate::models::LoraAdapter, LoraEvolutionError> {
            unimplemented!()
        }
        async fn select_lora(
            &self,
            _task: &Task,
            _project_id: &str,
        ) -> Result<Option<String>, LoraEvolutionError> {
            Ok(self.selected.clone())
        }
        async fn select_lora_by_role(
            &self,
            _role: AgentRole,
            _project_id: &str,
        ) -> Result<Option<String>, LoraEvolutionError> {
            Ok(self.selected.clone())
        }
    }

    fn task_with_payload(payload: Value) -> Task {
        Task {
            id: "t1".into(),
            project_id: "p1".into(),
            parent_id: None,
            title: "title".into(),
            description: None,
            kind: "codegen".into(),
            status: TaskStatus::Pending,
            assigned_agent: None,
            priority: 0,
            created_at: 0,
            started_at: None,
            finished_at: None,
            payload,
            result: None,
            iteration_count: 0,
            priority_score: 0.0,
            critic_score: None,
            human_score: None,
            prompt_version_id: None,
            lora_adapter_id: None,
            trace_id: "trace".into(),
        }
    }

    #[tokio::test]
    async fn router_uses_explicit_payload_lora() {
        let evolution = Arc::new(MockEvolution { selected: None });
        let router = LoraRouterImpl::new(evolution);

        let task = task_with_payload(json!({"lora": "custom-adapter"}));
        let resolved = router.resolve(&task, "p1").await.unwrap();

        assert_eq!(resolved, Some("custom-adapter".into()));
    }

    #[tokio::test]
    async fn router_falls_back_to_domain_heuristic() {
        let evolution = Arc::new(MockEvolution {
            selected: Some("codegen-v1".into()),
        });
        let router = LoraRouterImpl::new(evolution);

        let task = task_with_payload(json!({}));
        let resolved = router.resolve(&task, "p1").await.unwrap();

        assert_eq!(resolved, Some("codegen-v1".into()));
    }

    #[tokio::test]
    async fn router_returns_none_when_no_adapter_and_no_explicit_override() {
        let evolution = Arc::new(MockEvolution { selected: None });
        let router = LoraRouterImpl::new(evolution);

        let task = task_with_payload(json!({}));
        let resolved = router.resolve(&task, "p1").await.unwrap();

        assert_eq!(resolved, None);
    }

    #[tokio::test]
    async fn router_rejects_non_string_lora_payload() {
        let evolution = Arc::new(MockEvolution { selected: None });
        let router = LoraRouterImpl::new(evolution);

        let task = task_with_payload(json!({"lora": 42}));
        assert!(router.resolve(&task, "p1").await.is_err());
    }

    #[tokio::test]
    async fn registry_returns_configured_role_adapter() {
        let registry = MemoryRoleAdapterRegistry::new();
        registry.set(AgentRole::Coder, "coder-v2".into());

        assert_eq!(registry.get(AgentRole::Coder), Some("coder-v2".into()));
        assert_eq!(registry.get(AgentRole::Architect), None);
    }

    #[tokio::test]
    async fn resolve_for_role_uses_registry() {
        let registry = Arc::new(MemoryRoleAdapterRegistry::new());
        registry.set(AgentRole::Security, "sec-v1".into());
        let router = LoraRouterImpl::new(Arc::new(MockEvolution { selected: None }))
            .with_role_registry(registry);

        let resolved = router
            .resolve_for_role(AgentRole::Security, "p1")
            .await
            .unwrap();

        assert_eq!(resolved, Some("sec-v1".into()));
    }

    #[tokio::test]
    async fn resolve_for_specialized_roles_keeps_distinct_adapters() {
        let registry = Arc::new(MemoryRoleAdapterRegistry::new());
        registry.set(AgentRole::CoderPython, "python-lora-v1".into());
        registry.set(AgentRole::CriticCoder, "critic-coder-lora-v1".into());
        let router = LoraRouterImpl::new(Arc::new(MockEvolution { selected: None }))
            .with_role_registry(registry);

        let coder = router
            .resolve_for_role(AgentRole::CoderPython, "p1")
            .await
            .unwrap();
        let critic = router
            .resolve_for_role(AgentRole::CriticCoder, "p1")
            .await
            .unwrap();

        assert_eq!(coder, Some("python-lora-v1".into()));
        assert_eq!(critic, Some("critic-coder-lora-v1".into()));
        assert_ne!(coder, critic);
    }

    #[tokio::test]
    async fn router_prefers_role_registry_over_domain() {
        let evolution = Arc::new(MockEvolution {
            selected: Some("codegen-v1".into()),
        });
        let registry = Arc::new(MemoryRoleAdapterRegistry::new());
        registry.set(AgentRole::Coder, "coder-role-v1".into());
        let router = LoraRouterImpl::new(evolution).with_role_registry(registry);

        let mut task = task_with_payload(json!({}));
        task.assigned_agent = Some("coder".into());
        let resolved = router.resolve(&task, "p1").await.unwrap();

        assert_eq!(resolved, Some("coder-role-v1".into()));
    }

    async fn index_adapter(
        store: &dyn VectorStore,
        embedder: &crate::services::MockEmbedder,
        id: &str,
        text: &str,
    ) {
        store
            .create_collection(LORA_ADAPTER_COLLECTION, embedder.dimension().await.unwrap())
            .await
            .unwrap();
        let vector = embedder.embed(text).await.unwrap();
        store
            .upsert(
                LORA_ADAPTER_COLLECTION,
                vec![VectorPoint {
                    id: id.into(),
                    vector,
                    payload: json!({"adapter_id": id}),
                }],
            )
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn router_falls_back_to_semantic_search() {
        let embedder = Arc::new(crate::services::MockEmbedder::new(16));
        let vector_store: Arc<dyn VectorStore> = Arc::new(TestVectorStore::default());
        index_adapter(
            vector_store.as_ref(),
            &embedder,
            "codegen-v1",
            "codegen implement function",
        )
        .await;
        index_adapter(
            vector_store.as_ref(),
            &embedder,
            "arch-v1",
            "architecture design system",
        )
        .await;

        let evolution = Arc::new(MockEvolution { selected: None });
        let router = LoraRouterImpl::new(evolution).with_semantic_fallback(embedder, vector_store);

        let task = Task {
            description: Some("write a function".into()),
            ..task_with_payload(json!({}))
        };
        let resolved = router.resolve(&task, "p1").await.unwrap();
        assert_eq!(resolved, Some("codegen-v1".into()));
    }

    #[tokio::test]
    async fn router_prefers_domain_heuristic_over_semantic_search() {
        let embedder = Arc::new(crate::services::MockEmbedder::new(16));
        let vector_store: Arc<dyn VectorStore> = Arc::new(TestVectorStore::default());
        index_adapter(
            vector_store.as_ref(),
            &embedder,
            "semantic-adapter",
            "codegen implement function",
        )
        .await;

        let evolution = Arc::new(MockEvolution {
            selected: Some("domain-adapter".into()),
        });
        let router = LoraRouterImpl::new(evolution).with_semantic_fallback(embedder, vector_store);

        let task = Task {
            description: Some("write a function".into()),
            ..task_with_payload(json!({}))
        };
        let resolved = router.resolve(&task, "p1").await.unwrap();
        assert_eq!(resolved, Some("domain-adapter".into()));
    }

    #[tokio::test]
    async fn semantic_fallback_is_skipped_when_explicit_override_present() {
        let embedder = Arc::new(crate::services::MockEmbedder::new(16));
        let vector_store: Arc<dyn VectorStore> = Arc::new(TestVectorStore::default());
        index_adapter(
            vector_store.as_ref(),
            &embedder,
            "codegen-v1",
            "codegen implement function",
        )
        .await;

        let evolution = Arc::new(MockEvolution { selected: None });
        let router = LoraRouterImpl::new(evolution).with_semantic_fallback(embedder, vector_store);

        let task = task_with_payload(json!({"lora": "explicit-adapter"}));
        let resolved = router.resolve(&task, "p1").await.unwrap();
        assert_eq!(resolved, Some("explicit-adapter".into()));
    }

    #[tokio::test]
    async fn semantic_fallback_falls_back_to_domain_when_no_indexed_adapters() {
        let embedder = Arc::new(crate::services::MockEmbedder::new(16));
        let vector_store: Arc<dyn VectorStore> = Arc::new(TestVectorStore::default());

        let evolution = Arc::new(MockEvolution {
            selected: Some("domain-adapter".into()),
        });
        let router = LoraRouterImpl::new(evolution).with_semantic_fallback(embedder, vector_store);

        let task = task_with_payload(json!({}));
        let resolved = router.resolve(&task, "p1").await.unwrap();
        assert_eq!(resolved, Some("domain-adapter".into()));
    }
}
