use std::collections::HashMap;
use std::sync::Arc;

use thiserror::Error;

use crate::{BackendInfo, InferenceManager};

/// Errors returned by [`BackendRegistry`].
#[derive(Debug, Error)]
pub enum RegistryError {
    #[error("backend not found: {0}")]
    NotFound(String),
}

/// Registry of available inference backends.
///
/// The registry holds a set of named `InferenceManager` implementations and a
/// default backend id. It is the single source of truth for backend lookup at
/// runtime and enables hot-swapping by updating the default id.
#[derive(Clone, Default)]
pub struct BackendRegistry {
    default: String,
    backends: HashMap<String, Arc<dyn InferenceManager>>,
}

impl BackendRegistry {
    /// Creates a new registry with the given default backend id.
    ///
    /// The id does not have to point to a registered backend yet; callers can
    /// register backends afterwards.
    pub fn new(default: impl Into<String>) -> Self {
        Self {
            default: default.into(),
            backends: HashMap::new(),
        }
    }

    /// Registers a backend under the given id.
    pub fn register(&mut self, id: impl Into<String>, backend: Arc<dyn InferenceManager>) {
        self.backends.insert(id.into(), backend);
    }

    /// Returns true if no backends are registered.
    pub fn is_empty(&self) -> bool {
        self.backends.is_empty()
    }

    /// Returns a backend by id.
    pub fn get(&self, id: &str) -> Option<Arc<dyn InferenceManager>> {
        self.backends.get(id).cloned()
    }

    /// Returns the currently configured default backend.
    ///
    /// Returns `None` if the default id is not registered.
    pub fn default(&self) -> Option<Arc<dyn InferenceManager>> {
        self.get(&self.default)
    }

    /// Returns the id of the currently configured default backend.
    pub fn default_id(&self) -> &str {
        &self.default
    }

    /// Lists information for all registered backends.
    pub fn list(&self) -> Vec<BackendInfo> {
        self.backends
            .values()
            .flat_map(|b| b.available_backends())
            .collect()
    }

    /// Changes the default backend to the given id.
    ///
    /// Returns an error if the id is not registered.
    pub fn set_default(&mut self, id: &str) -> Result<(), RegistryError> {
        if !self.backends.contains_key(id) {
            return Err(RegistryError::NotFound(id.to_string()));
        }
        self.default = id.to_string();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        InferenceError, InferenceRequest, InferenceResponse, LoRAAdapter, ModelInfo, TokenUsage,
    };
    use async_trait::async_trait;

    struct DummyBackend {
        id: String,
    }

    #[async_trait]
    impl InferenceManager for DummyBackend {
        async fn generate(
            &self,
            _request: InferenceRequest,
        ) -> Result<InferenceResponse, InferenceError> {
            Ok(InferenceResponse {
                content: format!("from {}", self.id),
                usage: TokenUsage {
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    total_tokens: 0,
                },
                finish_reason: "stop".to_string(),
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
                capabilities: vec!["generate".to_string()],
            }]
        }

        async fn list_models(&self) -> Result<Vec<ModelInfo>, InferenceError> {
            Ok(vec![])
        }
    }

    fn dummy(id: &str) -> Arc<dyn InferenceManager> {
        Arc::new(DummyBackend { id: id.to_string() })
    }

    #[test]
    fn register_and_get_backend() {
        let mut registry = BackendRegistry::new("a");
        registry.register("a", dummy("a"));
        assert!(registry.get("a").is_some());
        assert!(registry.get("b").is_none());
    }

    #[test]
    fn default_backend_returns_registered_backend() {
        let mut registry = BackendRegistry::new("a");
        registry.register("a", dummy("a"));
        assert!(registry.default().is_some());
    }

    #[test]
    fn default_backend_is_none_when_not_registered() {
        let registry = BackendRegistry::new("a");
        assert!(registry.default().is_none());
    }

    #[test]
    fn set_default_switches_backend() {
        let mut registry = BackendRegistry::new("a");
        registry.register("a", dummy("a"));
        registry.register("b", dummy("b"));
        registry.set_default("b").unwrap();
        assert_eq!(registry.default_id(), "b");
    }

    #[test]
    fn set_default_unknown_backend_fails() {
        let mut registry = BackendRegistry::new("a");
        registry.register("a", dummy("a"));
        assert!(registry.set_default("unknown").is_err());
    }
}
