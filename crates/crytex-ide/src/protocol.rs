//! Protocol for sending suggestions and diffs from Crytex to an editor plugin.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A single diff hunk.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiffHunk {
    pub old_start: usize,
    pub old_lines: usize,
    pub new_start: usize,
    pub new_lines: usize,
    pub old_lines_text: Vec<String>,
    pub new_lines_text: Vec<String>,
}

/// Action the editor can take on a suggestion.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SuggestionAction {
    /// Show the suggestion inline; user accepts or rejects.
    Show,
    /// Apply the diff immediately.
    Apply,
    /// Open a diff review panel.
    Review,
}

/// One inline suggestion or diff for a file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Suggestion {
    pub file_path: String,
    pub description: String,
    pub action: SuggestionAction,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replacement_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hunks: Option<Vec<DiffHunk>>,
}

/// Request payload from the editor asking for suggestions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InlineSuggestionRequest {
    pub project_id: String,
    pub file_path: String,
    pub prompt: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor_line: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor_column: Option<usize>,
}

/// Response payload sent back to the editor.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InlineSuggestionResponse {
    pub project_id: String,
    pub file_path: String,
    pub suggestions: Vec<Suggestion>,
}

/// Serialize a list of suggestions to a JSON [`Value`].
pub fn serialize_suggestions(
    response: &InlineSuggestionResponse,
) -> Result<Value, serde_json::Error> {
    serde_json::to_value(response)
}

/// Deserialize an editor request from a JSON [`Value`].
pub fn deserialize_request(value: Value) -> Result<InlineSuggestionRequest, serde_json::Error> {
    serde_json::from_value(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_serializes_diff_payload() {
        let response = InlineSuggestionResponse {
            project_id: "p1".into(),
            file_path: "src/lib.rs".into(),
            suggestions: vec![Suggestion {
                file_path: "src/lib.rs".into(),
                description: "Add error handling".into(),
                action: SuggestionAction::Review,
                replacement_text: None,
                hunks: Some(vec![DiffHunk {
                    old_start: 10,
                    old_lines: 1,
                    new_start: 10,
                    new_lines: 3,
                    old_lines_text: vec!["let x = 1;".into()],
                    new_lines_text: vec![
                        "let x = match compute() {".into(),
                        "    Ok(v) => v,".into(),
                        "    Err(e) => return Err(e.into()),".into(),
                    ],
                }]),
            }],
        };

        let value = serialize_suggestions(&response).unwrap();
        assert_eq!(value["project_id"], "p1");
        assert_eq!(value["suggestions"][0]["action"], "review");
        assert_eq!(value["suggestions"][0]["hunks"][0]["old_start"], 10);

        let parsed: InlineSuggestionResponse = serde_json::from_value(value).unwrap();
        assert_eq!(parsed, response);
    }
}
