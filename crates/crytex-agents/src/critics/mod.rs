pub mod code;
pub mod security;
pub mod style;
pub mod test;

use crate::{
    extract_backend_id, extract_model,
    json::parse_llm_json_value,
    prompts::{
        specialized_critic_system_prompt, specialized_critic_user_prompt, system_prompt_override,
    },
    tooling::generate_with_tools,
};
use crytex_core::models::Task;
use crytex_core::services::{AgentError, InferenceService, ToolService};
use crytex_inference::{InferenceRequest, Message, TokenUsage};
use serde_json::Value;
use std::sync::Arc;

pub(crate) fn parse_critic_score(
    dimension: &str,
    content: &str,
    usage: &TokenUsage,
) -> Result<Value, serde_json::Error> {
    let mut value = parse_llm_json_value(content)?;

    value["dimension"] = Value::String(dimension.to_string());

    if value.get("comment").is_none() {
        value["comment"] = Value::String(String::new());
    }

    if value.get("usage").is_none() {
        value["usage"] = serde_json::json!({
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        });
    }

    Ok(value)
}

pub(crate) async fn execute_specialized_critic(
    dimension: &str,
    focus: &str,
    task: &Task,
    inference: Arc<dyn InferenceService>,
    tools: Arc<dyn ToolService>,
) -> Result<Value, AgentError> {
    let override_prompt = system_prompt_override(&task.payload);
    let system_prompt =
        specialized_critic_system_prompt(dimension, focus, &tools.list_tools(), override_prompt);
    let user_prompt = specialized_critic_user_prompt(&task.payload);

    let request = InferenceRequest {
        backend_id: extract_backend_id(&task.payload),
        model: extract_model(&task.payload),
        messages: vec![
            Message {
                role: "system".to_string(),
                content: system_prompt,
            },
            Message {
                role: "user".to_string(),
                content: user_prompt,
            },
        ],
        system_prompt: None,
        temperature: Some(0.4),
        max_tokens: Some(2048),
        lora_adapter_id: task.lora_adapter_id.clone(),
    };

    let response = generate_with_tools(inference, tools, request)
        .await
        .map_err(|e| AgentError::Execution(e.to_string()))?;

    parse_critic_score(dimension, &response.content, &response.usage)
        .map_err(|e| AgentError::Execution(e.to_string()))
}
