use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A single tool call extracted from an agent response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

/// Parse tool calls from a model response.
///
/// Supports:
/// - A bare JSON object `{ "tool": "name", "args": {...} }`
/// - A JSON array of such objects
/// - JSON wrapped in markdown code fences
pub fn parse_tool_calls(text: &str) -> Result<Vec<ToolCall>, String> {
    // 1. Try to extract JSON from markdown fences.
    let cleaned = if let Some(start) = text.find("```json") {
        text[start + 7..]
            .split_once("```")
            .map(|(inner, _)| inner.trim())
            .unwrap_or(text)
    } else if let Some(start) = text.find("```") {
        text[start + 3..]
            .split_once("```")
            .map(|(inner, _)| inner.trim())
            .unwrap_or(text)
    } else {
        text.trim()
    };

    // 2. Try as JSON array.
    if cleaned.starts_with('[') {
        let values: Vec<Value> = serde_json::from_str(cleaned)
            .map_err(|e| format!("failed to parse tool call array: {}", e))?;
        let mut calls = Vec::with_capacity(values.len());
        for value in values {
            let name = value
                .get("name")
                .or_else(|| value.get("tool"))
                .and_then(|v| v.as_str())
                .ok_or_else(|| "tool call missing name/tool field".to_string())?
                .to_string();
            let arguments = value
                .get("arguments")
                .or_else(|| value.get("args"))
                .cloned()
                .unwrap_or(Value::Object(serde_json::Map::new()));
            calls.push(ToolCall { name, arguments });
        }
        return Ok(calls);
    }

    // 3. Try as single JSON object.
    if cleaned.starts_with('{') {
        // Accept both OpenAI-style {"name": ..., "arguments": ...} and our own
        // {"tool": ..., "args": ...} shape.
        let value: Value = serde_json::from_str(cleaned)
            .map_err(|e| format!("failed to parse tool call object: {}", e))?;

        let name = value
            .get("name")
            .or_else(|| value.get("tool"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| "tool call missing name/tool field".to_string())?
            .to_string();
        let arguments = value
            .get("arguments")
            .or_else(|| value.get("args"))
            .cloned()
            .unwrap_or(Value::Object(serde_json::Map::new()));
        return Ok(vec![ToolCall { name, arguments }]);
    }

    Err("no JSON tool call found in response".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_single_object() {
        let text = r#"{"tool":"fs_read","args":{"path":"src/main.rs"}}"#;
        let calls = parse_tool_calls(text).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "fs_read");
        assert_eq!(calls[0].arguments["path"], "src/main.rs");
    }

    #[test]
    fn parse_array() {
        let text = r#"[{"name":"fs_read","arguments":{"path":"a.rs"}},{"name":"fs_write","arguments":{"path":"b.rs","content":"x"}}]"#;
        let calls = parse_tool_calls(text).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[1].name, "fs_write");
    }

    #[test]
    fn parse_markdown_fenced() {
        let text = "```json\n{\"tool\":\"fs_list\",\"args\":{\"path\":\".\"}}\n```";
        let calls = parse_tool_calls(text).unwrap();
        assert_eq!(calls[0].name, "fs_list");
    }

    #[test]
    fn parse_array_with_tool_and_args() {
        let text = r#"[{"tool":"fs_read","args":{"path":"a.rs"}},{"tool":"fs_write","args":{"path":"b.rs","content":"x"}}]"#;
        let calls = parse_tool_calls(text).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "fs_read");
        assert_eq!(calls[1].name, "fs_write");
        assert_eq!(calls[1].arguments["content"], "x");
    }
}
