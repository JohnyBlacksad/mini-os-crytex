use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use async_trait::async_trait;

use crate::{
    InferenceBackend,
    GenerationRequest,
    GenerationResponse,
    InferenceError,
    TokenUsage,
    ModelInfo,
    LoRAAdapter
};

pub struct MockBackend {
    pub responses: HashMap<String, String>,
    pub loaded_loras: Arc<Mutex<Vec<String>>>,
}

impl MockBackend {
    pub fn new(responses: HashMap<String, String>) -> Self {
        Self {
            responses,
            loaded_loras: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

#[async_trait]
impl InferenceBackend for MockBackend {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, InferenceError> {
        let models = vec![
            ModelInfo {
                id: "mock-model-1".to_string(),
                name: "Mock model 1".to_string(),
            },
            ModelInfo {
                id: "mock-model-2".to_string(),
                name: "Mock Model 2".to_string(),
            },
        ];
        Ok(models)
    }

    async fn generate(&self, request: GenerationRequest) -> Result<GenerationResponse, InferenceError> {
        if let Some(response_text) = self.responses.get(&request.model) {
            Ok(GenerationResponse {
                content: response_text.clone(),
                usage: TokenUsage {
                    prompt_tokens: 10,
                    completion_tokens: 20,
                    total_tokens: 30,
                }
            })
        } else {
            Err(InferenceError::ModelNotFound(request.model))
        }
    }

    async fn embed(&self, _text: &str) -> Result<Vec<f32>, InferenceError> {
        let fake_vector = vec![0.0; 768];
        Ok(fake_vector)
    }

    async fn load_lora(&self, adapter: &LoRAAdapter) -> Result<(), InferenceError> {
        let mut loras = self.loaded_loras.lock().await;
        loras.push(adapter.id.clone());
        Ok(())
    }

    async fn unload_lora(&self, id: &str) -> Result<(), InferenceError> {
        let mut loras = self.loaded_loras.lock().await;
        if let Some(pos) = loras.iter().position(|x| x == id) {
            loras.remove(pos);
            Ok(())
        } else {
            Err(InferenceError::LoRALoadFailed(id.to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const TEST_LORA_ID: &str = "1";

    fn setup_mock() -> MockBackend {
        let mut responses = HashMap::new();
        responses.insert("test-model".to_string(), "Hello from mock!".to_string());
        MockBackend::new(responses)
    }

    async fn setup_lora() -> MockBackend {
        let backend = setup_mock();
        let lora = LoRAAdapter{
            id: TEST_LORA_ID.to_string(),
            path: "src/loras/lora-model".to_string(),
            base_model: "test-model".to_string(),
        };
        backend.load_lora(&lora).await.unwrap();
        backend
    }

    #[tokio::test]
    async fn test_mock_generate() {
        let backend = setup_mock();

        let req = GenerationRequest {
            model: "test-model".to_string(),
            messages: vec![],
            temperature: None,
            max_tokens: None,
        };

        let res = backend.generate(req).await.unwrap();

        assert_eq!(res.content, "Hello from mock!");
    }

    #[tokio::test]
    async fn test_mock_generate_not_found() {
        let backend = setup_mock();

        let req = GenerationRequest {
            model: "unknown-model".to_string(),
            messages: vec![],
            temperature: None,
            max_tokens: None,
        };

        let res = backend.generate(req).await;

        assert!(matches!(res, Err(InferenceError::ModelNotFound(ref m)) if m == "unknown-model"));
    }

    #[tokio::test]
    async fn test_mock_load_lora() {
        let backend = setup_mock();

        let req = LoRAAdapter {
            id: "1".to_string(),
            path: "rsc/loras/lora-model".to_string(),
            base_model: "test-model".to_string(),
        };

        let res = backend.load_lora(&req).await;
        let loras = backend.loaded_loras.lock().await;

        assert!(res.is_ok());
        assert_eq!(loras.len(), 1);
        assert_eq!(loras[0], "1");
    }

    #[tokio::test]
    async fn test_mock_unload_lora() {
        let backend = setup_lora().await;
        let res = backend.unload_lora(TEST_LORA_ID).await;
        let loras = backend.loaded_loras.lock().await;

        assert!(res.is_ok());
        assert_eq!(loras.len(), 0);
    }

    #[tokio::test]
    async fn test_mock_unload_lora_not_found() {
        let backend = setup_lora().await;
        let res = backend.unload_lora("99").await;

        assert!(matches!(res, Err(InferenceError::LoRALoadFailed(ref m)) if m == "99"));
    }

    #[tokio::test]
    async fn test_mock_embed() {
        let backend = setup_mock();
        let res = backend.embed("Hello World!").await.unwrap();
        assert_eq!(res.len(), 768);
    }

    #[tokio::test]
    async fn test_mock_list_models() {
        let backend = setup_mock();
        let res = backend.list_models().await.unwrap();
        let expect_model = ModelInfo{
            id: "mock-model-1".to_string(),
            name: "Mock model 1".to_string(),
        };

        assert_eq!(res.len(), 2);
        assert!(res.contains(&expect_model));
    }
}