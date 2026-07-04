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
}
