use std::time::Duration;

use crytex_inference::{InferenceManager, InferenceRequest, Message};
use crytex_inference_ollama::OllamaBackend;
use serde_json::json;

const DEFAULT_OLLAMA_URL: &str = "http://127.0.0.1:11434";
const DEFAULT_E2E_MODEL: &str = "smollm2:135m";

#[tokio::test]
#[ignore = "requires local Ollama, network/model cache, and can download a model"]
async fn pulls_or_reuses_model_and_generates_real_task_result() {
    let ollama_url =
        std::env::var("CRYTEX_E2E_OLLAMA_URL").unwrap_or_else(|_| DEFAULT_OLLAMA_URL.to_string());
    let model =
        std::env::var("CRYTEX_E2E_OLLAMA_MODEL").unwrap_or_else(|_| DEFAULT_E2E_MODEL.to_string());

    ensure_model_available(&ollama_url, &model).await;

    let backend = OllamaBackend::new(&ollama_url, &model);
    let response = backend
        .generate(InferenceRequest {
            backend_id: Some("ollama".to_string()),
            model: model.clone(),
            messages: vec![Message {
                role: "user".to_string(),
                content: "Task: reply with one short sentence proving this real local model executed a Crytex E2E smoke test.".to_string(),
            }],
            system_prompt: Some(
                "You are a local model called by Crytex. Answer plainly and briefly.".to_string(),
            ),
            temperature: Some(0.0),
            max_tokens: Some(64),
            lora_adapter_id: None,
        })
        .await
        .expect("Ollama backend should generate a real response");

    assert!(
        !response.content.trim().is_empty(),
        "real model response must not be empty"
    );
    assert!(
        response.usage.total_tokens > 0 || response.usage.completion_tokens > 0,
        "Ollama should report token usage for the real generation"
    );

    println!(
        "CRYTEX_E2E_RESULT model={model} finish_reason={} usage={:?} content={}",
        response.finish_reason,
        response.usage,
        response.content.trim()
    );
}

async fn ensure_model_available(ollama_url: &str, model: &str) {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(900))
        .build()
        .expect("reqwest client should build");

    let tags = client
        .get(format!("{ollama_url}/api/tags"))
        .send()
        .await
        .expect("Ollama must be running and reachable")
        .json::<serde_json::Value>()
        .await
        .expect("Ollama tags response should be JSON");

    let model_is_cached = tags
        .get("models")
        .and_then(|models| models.as_array())
        .is_some_and(|models| {
            models.iter().any(|entry| {
                entry.get("name").and_then(|name| name.as_str()) == Some(model)
                    || entry.get("model").and_then(|name| name.as_str()) == Some(model)
            })
        });

    if model_is_cached {
        return;
    }

    let response = client
        .post(format!("{ollama_url}/api/pull"))
        .json(&json!({ "name": model, "stream": false }))
        .send()
        .await
        .expect("Ollama must be running and reachable");

    assert!(
        response.status().is_success(),
        "Ollama pull failed for {model}: status={} body={}",
        response.status(),
        response.text().await.unwrap_or_default()
    );
}
