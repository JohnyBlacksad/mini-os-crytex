use std::sync::LazyLock;

use regex::Regex;

use crate::tree_sitter_detector;
use crate::unidiff_detector;

/// Detected content category for a message or text block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContentType {
    /// Plain unstructured text.
    PlainText,
    /// JSON object or array.
    Json,
    /// Source code (heuristic detection).
    SourceCode,
    /// Log lines (timestamps, levels, stack traces).
    Log,
    /// Unified diff or patch.
    Diff,
    /// Search results (file:line:content).
    SearchResults,
}

impl ContentType {
    pub fn as_str(&self) -> &'static str {
        match self {
            ContentType::PlainText => "text",
            ContentType::Json => "json",
            ContentType::SourceCode => "code",
            ContentType::Log => "log",
            ContentType::Diff => "diff",
            ContentType::SearchResults => "search",
        }
    }
}

static DIFF_HEADER_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(diff --git|--- a/|\+\+\+ b/|@@ -\d+)").unwrap());

static SEARCH_RESULT_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[^\s:]+:\d+:").unwrap());

static LOG_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"^\d{4}[-/]\d{2}[-/]\d{2}|^\[\d{4}-\d{2}-\d{2}|^(INFO|WARN|ERROR|DEBUG|TRACE|FATAL)",
    )
    .unwrap()
});

static CODE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(^\s*(def|class|fn|func|function|import|from|package|using|#include|public|private)\s)|(\{\s*$)|(\}\s*$)").unwrap()
});

/// Detect the content type of a text block.
pub fn detect_content_type(text: &str) -> ContentType {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return ContentType::PlainText;
    }

    let non_blank: Vec<&str> = lines
        .iter()
        .copied()
        .filter(|l| !l.trim().is_empty())
        .collect();
    if non_blank.is_empty() {
        return ContentType::PlainText;
    }

    // JSON: try to parse the whole text as JSON.
    if serde_json::from_str::<serde_json::Value>(text.trim()).is_ok() {
        return ContentType::Json;
    }

    // Diff: use a real unified-diff parser first, then fall back to header regex.
    if unidiff_detector::is_diff(text) {
        return ContentType::Diff;
    }
    let diff_lines = non_blank
        .iter()
        .filter(|l| DIFF_HEADER_RE.is_match(l))
        .count();
    if diff_lines > 0 && diff_lines >= non_blank.len() / 10 {
        return ContentType::Diff;
    }

    // Search results: file:line: patterns.
    let search_lines = non_blank
        .iter()
        .filter(|l| SEARCH_RESULT_RE.is_match(l))
        .count();
    if search_lines > 0 && search_lines >= non_blank.len() / 3 {
        return ContentType::SearchResults;
    }

    // Logs: timestamp/level patterns.
    let log_lines = non_blank.iter().filter(|l| LOG_RE.is_match(l)).count();
    if log_lines > 0 && log_lines >= non_blank.len() / 3 {
        return ContentType::Log;
    }

    // Source code: try real tree-sitter parsers first, then fall back to heuristic patterns.
    if tree_sitter_detector::detect_language(text).is_some() {
        return ContentType::SourceCode;
    }
    let code_lines = non_blank.iter().filter(|l| CODE_RE.is_match(l)).count();
    if code_lines > 0 && code_lines >= non_blank.len().min(10) / 3 {
        return ContentType::SourceCode;
    }

    ContentType::PlainText
}

/// Detect the dominant content type across a slice of messages.
pub fn detect_messages_content_type(messages: &[crate::message::Message]) -> ContentType {
    let mut counts = std::collections::HashMap::new();
    for msg in messages {
        let content = &msg.content;
        // Skip very short messages for detection.
        if content.len() < 40 {
            continue;
        }
        *counts.entry(detect_content_type(content)).or_insert(0usize) += content.len();
    }

    counts
        .into_iter()
        .max_by_key(|(_, len)| *len)
        .map(|(ty, _)| ty)
        .unwrap_or(ContentType::PlainText)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_json() {
        assert_eq!(
            detect_content_type(r#"{"foo": "bar", "items": [1, 2]}"#),
            ContentType::Json
        );
    }

    #[test]
    fn detect_diff() {
        let text = "diff --git a/src/lib.rs b/src/lib.rs\n--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1,5 +1,5 @@\n-foo\n+bar";
        assert_eq!(detect_content_type(text), ContentType::Diff);
    }

    #[test]
    fn detect_log() {
        let text = "2024-01-01 10:00:00 INFO starting\n2024-01-01 10:00:01 ERROR failed";
        assert_eq!(detect_content_type(text), ContentType::Log);
    }

    #[test]
    fn detect_search_results() {
        let text = "src/main.rs:42: fn main() {}\nsrc/lib.rs:10: pub fn foo() {}";
        assert_eq!(detect_content_type(text), ContentType::SearchResults);
    }

    #[test]
    fn detect_code() {
        let text = "fn main() {\n    println!(\"hello\");\n}";
        assert_eq!(detect_content_type(text), ContentType::SourceCode);
    }
}
