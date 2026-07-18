use std::collections::HashSet;
use std::path::Path;

use crate::error::BenchError;
use crate::models::BenchmarkCase;

/// Loader for curated golden datasets.
#[derive(Debug, Clone, Default)]
pub struct GoldenSet;

impl GoldenSet {
    /// Load all cases from `path`.
    ///
    /// Supports `.jsonl` (newline-delimited JSON objects) and `.yaml`/`.yml`.
    pub async fn load(path: impl AsRef<Path>) -> Result<Vec<BenchmarkCase>, BenchError> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(BenchError::GoldenSetNotFound(path.display().to_string()));
        }

        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        match ext.as_str() {
            "jsonl" => Self::load_jsonl(path).await,
            "yaml" | "yml" => Self::load_yaml(path).await,
            other => Err(BenchError::GoldenSetParse(format!(
                "unsupported golden set extension: {other}"
            ))),
        }
    }

    /// Load and validate a golden set before it is used as a benchmark gate.
    pub async fn load_validated(path: impl AsRef<Path>) -> Result<Vec<BenchmarkCase>, BenchError> {
        let cases = Self::load(path).await?;
        Self::validate_cases(&cases)?;
        Ok(cases)
    }

    /// Validate benchmark cases for basic hygiene.
    pub fn validate_cases(cases: &[BenchmarkCase]) -> Result<(), BenchError> {
        let mut ids = HashSet::new();
        for case in cases {
            if case.id.trim().is_empty() {
                return Err(BenchError::GoldenSetParse(
                    "low-information benchmark case: empty id".into(),
                ));
            }
            if !ids.insert(case.id.as_str()) {
                return Err(BenchError::GoldenSetParse(format!(
                    "duplicate benchmark case id: {}",
                    case.id
                )));
            }

            let input_text = canonical_json_text(&case.input);
            let expected_text = case
                .expected
                .as_ref()
                .map(canonical_json_text)
                .unwrap_or_default();
            if token_count(&input_text) < 2 || token_count(&expected_text) < 2 {
                return Err(BenchError::GoldenSetParse(format!(
                    "low-information benchmark case: {}",
                    case.id
                )));
            }
        }
        Ok(())
    }

    /// Reject benchmark sets that overlap strongly with training examples.
    pub fn validate_no_training_leakage(
        cases: &[BenchmarkCase],
        training_texts: &[&str],
        max_similarity: f64,
    ) -> Result<(), BenchError> {
        for case in cases {
            let benchmark_text = format!(
                "{} {}",
                canonical_json_text(&case.input),
                case.expected
                    .as_ref()
                    .map(canonical_json_text)
                    .unwrap_or_default()
            );
            for training_text in training_texts {
                let similarity = jaccard_similarity(&benchmark_text, training_text);
                if similarity >= max_similarity {
                    return Err(BenchError::GoldenSetParse(format!(
                        "benchmark leakage detected for case {}: similarity {:.3}",
                        case.id, similarity
                    )));
                }
            }
        }
        Ok(())
    }

    async fn load_jsonl(path: &Path) -> Result<Vec<BenchmarkCase>, BenchError> {
        let contents = tokio::fs::read_to_string(path).await?;
        let mut cases = Vec::new();
        for (line_no, line) in contents.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let case: BenchmarkCase = serde_json::from_str(line).map_err(|e| {
                BenchError::GoldenSetParse(format!("{}:{line_no}: {e}", path.display()))
            })?;
            cases.push(case);
        }
        Ok(cases)
    }

    async fn load_yaml(path: &Path) -> Result<Vec<BenchmarkCase>, BenchError> {
        let contents = tokio::fs::read_to_string(path).await?;
        let cases: Vec<BenchmarkCase> = serde_yaml::from_str(&contents)
            .map_err(|e| BenchError::GoldenSetParse(format!("{}: {e}", path.display())))?;
        Ok(cases)
    }
}

fn canonical_json_text(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => String::new(),
        serde_json::Value::Bool(value) => value.to_string(),
        serde_json::Value::Number(value) => value.to_string(),
        serde_json::Value::String(value) => value.clone(),
        serde_json::Value::Array(values) => values
            .iter()
            .map(canonical_json_text)
            .collect::<Vec<_>>()
            .join(" "),
        serde_json::Value::Object(map) => map
            .values()
            .map(canonical_json_text)
            .collect::<Vec<_>>()
            .join(" "),
    }
}

fn tokens(text: &str) -> HashSet<String> {
    text.split(|ch: char| !ch.is_alphanumeric() && ch != '_')
        .filter_map(|token| {
            let normalized = token.trim().to_lowercase();
            (normalized.len() >= 2).then_some(normalized)
        })
        .collect()
}

fn token_count(text: &str) -> usize {
    tokens(text).len()
}

fn jaccard_similarity(left: &str, right: &str) -> f64 {
    let left = tokens(left);
    let right = tokens(right);
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let intersection = left.intersection(&right).count();
    let union = left.union(&right).count();
    intersection as f64 / union as f64
}

/// Convenience alias for [`GoldenSet`].
pub type GoldenSetLoader = GoldenSet;

#[cfg(test)]
mod tests {
    use super::*;
    fn temp_dir() -> tempfile::TempDir {
        tempfile::tempdir().expect("tempdir")
    }

    #[tokio::test]
    async fn should_error_for_missing_file() {
        let result = GoldenSet::load("/nonexistent/golden.jsonl").await;
        assert!(matches!(result, Err(BenchError::GoldenSetNotFound(_))));
    }

    #[tokio::test]
    async fn should_load_empty_jsonl() {
        let dir = temp_dir();
        let path = dir.path().join("empty.jsonl");
        tokio::fs::write(&path, "").await.unwrap();
        let cases = GoldenSet::load(&path).await.unwrap();
        assert!(cases.is_empty());
    }

    #[tokio::test]
    async fn should_load_jsonl_cases() {
        let dir = temp_dir();
        let path = dir.path().join("cases.jsonl");
        let data = r#"{"id":"a","input":{"prompt":"hello"},"expected":{"out":"hi"},"tags":["greet"],"metadata":{}}
{"id":"b","input":{"prompt":"bye"},"expected":{"out":"bye"},"tags":[]}"#;
        tokio::fs::write(&path, data).await.unwrap();
        let cases = GoldenSet::load(&path).await.unwrap();
        assert_eq!(cases.len(), 2);
        assert_eq!(cases[0].id, "a");
        assert_eq!(cases[0].tags, vec!["greet"]);
    }

    #[tokio::test]
    async fn should_load_yaml_cases() {
        let dir = temp_dir();
        let path = dir.path().join("cases.yaml");
        let data = r#"
- id: a
  input:
    prompt: hello
  expected:
    out: hi
  tags:
    - greet
- id: b
  input:
    prompt: bye
"#;
        tokio::fs::write(&path, data).await.unwrap();
        let cases = GoldenSet::load(&path).await.unwrap();
        assert_eq!(cases.len(), 2);
        assert_eq!(cases[0].expected, Some(serde_json::json!({"out":"hi"})));
        assert!(cases[1].expected.is_none());
    }

    #[tokio::test]
    async fn should_error_on_invalid_jsonl() {
        let dir = temp_dir();
        let path = dir.path().join("bad.jsonl");
        tokio::fs::write(&path, "not json").await.unwrap();
        let result = GoldenSet::load(&path).await;
        assert!(matches!(result, Err(BenchError::GoldenSetParse(_))));
    }

    #[tokio::test]
    async fn should_reject_duplicate_case_ids() {
        let dir = temp_dir();
        let path = dir.path().join("dupes.jsonl");
        let data = r#"{"id":"same","input":{"prompt":"write add"},"expected":{"out":"fn add"}}
{"id":"same","input":{"prompt":"write sub"},"expected":{"out":"fn sub"}}"#;
        tokio::fs::write(&path, data).await.unwrap();

        let result = GoldenSet::load_validated(&path).await;

        assert!(
            matches!(result, Err(BenchError::GoldenSetParse(message)) if message.contains("duplicate"))
        );
    }

    #[tokio::test]
    async fn should_reject_low_information_cases() {
        let dir = temp_dir();
        let path = dir.path().join("empty.jsonl");
        let data = r#"{"id":"empty","input":{"prompt":""},"expected":{"out":"   "}}"#;
        tokio::fs::write(&path, data).await.unwrap();

        let result = GoldenSet::load_validated(&path).await;

        assert!(
            matches!(result, Err(BenchError::GoldenSetParse(message)) if message.contains("low-information"))
        );
    }

    #[test]
    fn should_detect_training_example_leakage_into_benchmark() {
        let cases = vec![BenchmarkCase {
            id: "leaked".into(),
            input: serde_json::json!({"prompt": "Implement add function"}),
            expected: Some(serde_json::json!({"code": "fn add(a: i32, b: i32) -> i32 { a + b }"})),
            tags: vec![],
            metadata: serde_json::Value::Null,
        }];
        let training_texts = vec!["Implement add function fn add(a: i32, b: i32) -> i32 { a + b }"];

        let result = GoldenSet::validate_no_training_leakage(&cases, &training_texts, 0.8);

        assert!(
            matches!(result, Err(BenchError::GoldenSetParse(message)) if message.contains("leakage"))
        );
    }
}
