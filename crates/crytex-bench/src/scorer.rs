use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use crytex_core::services::{ExecutionRequest, InferenceService, SandboxService};
use serde_json::Value;

use crate::error::BenchError;
use crate::models::BenchmarkCase;

/// A scored outcome for a single benchmark case.
#[derive(Debug, Clone, PartialEq)]
pub struct Score {
    pub passed: bool,
    pub value: f64,
    pub confidence: Option<f64>,
    pub explanation: Option<String>,
    pub metadata: Value,
}

impl Score {
    pub fn pass() -> Self {
        Self {
            passed: true,
            value: 1.0,
            confidence: None,
            explanation: None,
            metadata: Value::Object(serde_json::Map::new()),
        }
    }

    pub fn fail(reason: impl Into<String>) -> Self {
        Self {
            passed: false,
            value: 0.0,
            confidence: None,
            explanation: Some(reason.into()),
            metadata: Value::Object(serde_json::Map::new()),
        }
    }
}

/// A strategy for scoring the output of a benchmark case.
#[async_trait]
pub trait Scorer: Send + Sync {
    async fn score(&self, case: &BenchmarkCase, actual: &Value) -> Result<Score, BenchError>;
}

/// Passes when `actual` exactly equals `expected`.
#[derive(Debug, Clone, Default)]
pub struct ExactMatchScorer;

#[async_trait]
impl Scorer for ExactMatchScorer {
    async fn score(&self, case: &BenchmarkCase, actual: &Value) -> Result<Score, BenchError> {
        let expected = case.expected.as_ref().ok_or_else(|| {
            BenchError::Scoring(format!("case {} has no expected value", case.id))
        })?;

        if expected == actual {
            Ok(Score::pass())
        } else {
            Ok(Score::fail(format!(
                "expected {} but got {}",
                serde_json::to_string(expected).unwrap_or_default(),
                serde_json::to_string(actual).unwrap_or_default()
            )))
        }
    }
}

/// Validates `actual` against a JSON schema stored in `case.metadata.schema`.
#[derive(Debug, Clone, Default)]
pub struct JsonSchemaScorer;

#[async_trait]
impl Scorer for JsonSchemaScorer {
    async fn score(&self, case: &BenchmarkCase, actual: &Value) -> Result<Score, BenchError> {
        let schema = case
            .metadata
            .get("schema")
            .ok_or_else(|| BenchError::Scoring(format!("case {} has no schema", case.id)))?;

        let validator = jsonschema::Validator::new(schema).map_err(|e| {
            BenchError::Scoring(format!("invalid schema for case {}: {e}", case.id))
        })?;

        let errors: Vec<_> = validator.iter_errors(actual).collect();
        if errors.is_empty() {
            Ok(Score::pass())
        } else {
            let messages: Vec<String> = errors.into_iter().map(|e| e.to_string()).collect();
            Ok(Score::fail(messages.join("; ")))
        }
    }
}

/// Writes the actual output to a temp file and runs a sandbox test command.
#[derive(Clone)]
pub struct SandboxTestScorer {
    sandbox: Arc<dyn SandboxService>,
    timeout_seconds: u64,
}

impl SandboxTestScorer {
    pub fn new(sandbox: Arc<dyn SandboxService>) -> Self {
        Self {
            sandbox,
            timeout_seconds: 60,
        }
    }

    pub fn with_timeout(mut self, seconds: u64) -> Self {
        self.timeout_seconds = seconds;
        self
    }
}

#[async_trait]
impl Scorer for SandboxTestScorer {
    async fn score(&self, case: &BenchmarkCase, actual: &Value) -> Result<Score, BenchError> {
        let command = case
            .metadata
            .get("test_command")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                BenchError::Scoring(format!(
                    "case {} missing metadata.test_command array",
                    case.id
                ))
            })?;

        let command: Vec<String> = command
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        if command.is_empty() {
            return Err(BenchError::Scoring(format!(
                "case {} has empty test_command",
                case.id
            )));
        }

        let filename = case
            .metadata
            .get("filename")
            .and_then(|v| v.as_str())
            .unwrap_or("actual.txt");

        let temp_dir = tokio::task::spawn_blocking(tempfile::tempdir)
            .await
            .map_err(|e| BenchError::Scoring(format!("failed to create temp dir: {e}")))?
            .map_err(|e| BenchError::Scoring(format!("failed to create temp dir: {e}")))?;
        let file_path = temp_dir.path().join(filename);
        let contents = actual
            .as_str()
            .map(String::from)
            .unwrap_or_else(|| serde_json::to_string(actual).unwrap_or_default());
        tokio::fs::write(&file_path, contents).await?;

        let request = ExecutionRequest::new(command)
            .cwd(PathBuf::from("/workspace"))
            .mount(temp_dir.path(), PathBuf::from("/workspace"), true)
            .resources(crytex_core::services::SandboxResources {
                memory_mb: 512,
                cpu_shares: 512,
                timeout_seconds: self.timeout_seconds,
            });

        let result = self.sandbox.execute(request).await?;
        if result.exit_code == 0 {
            Ok(Score::pass())
        } else {
            Ok(Score::fail(format!(
                "exit_code={} stdout={} stderr={}",
                result.exit_code, result.stdout, result.stderr
            )))
        }
    }
}

/// Uses an LLM as a judge with a rubric stored in `case.metadata.rubric`.
#[derive(Clone)]
pub struct LlmJudgeScorer {
    inference: Arc<dyn InferenceService>,
    model: String,
    backend_id: Option<String>,
}

impl LlmJudgeScorer {
    pub fn new(
        inference: Arc<dyn InferenceService>,
        model: impl Into<String>,
        backend_id: Option<String>,
    ) -> Self {
        Self {
            inference,
            model: model.into(),
            backend_id,
        }
    }
}

#[async_trait]
impl Scorer for LlmJudgeScorer {
    async fn score(&self, case: &BenchmarkCase, actual: &Value) -> Result<Score, BenchError> {
        let rubric = case
            .metadata
            .get("rubric")
            .and_then(|v| v.as_str())
            .unwrap_or("Evaluate whether the response satisfies the request. Reply PASS or FAIL.");

        let expected = case
            .expected
            .as_ref()
            .map(|v| serde_json::to_string_pretty(v).unwrap_or_default())
            .unwrap_or_else(|| "No expected answer provided.".to_string());

        let user = format!(
            "[Rubric]\n{}\n\n[Expected]\n{}\n\n[Actual]\n{}\n\nReply PASS or FAIL.",
            rubric,
            expected,
            serde_json::to_string_pretty(actual).unwrap_or_default()
        );

        let request = self.inference.chat_request(
            self.backend_id.as_deref(),
            &self.model,
            Some("You are a strict evaluator. Only reply PASS or FAIL."),
            &user,
        );

        let response = self.inference.generate(request).await.map_err(|e| {
            BenchError::Scoring(format!("llm judge failed for case {}: {e}", case.id))
        })?;

        let passed = response.content.to_uppercase().contains("PASS")
            && !response.content.to_uppercase().contains("FAIL");

        let value = if passed { 1.0 } else { 0.0 };
        Ok(Score {
            passed,
            value,
            confidence: None,
            explanation: Some(response.content),
            metadata: Value::Object(serde_json::Map::new()),
        })
    }
}

/// Runs multiple scorers and requires all of them to pass.
#[derive(Clone, Default)]
pub struct CompositeScorer {
    scorers: Vec<Arc<dyn Scorer>>,
}

impl CompositeScorer {
    pub fn new(scorers: Vec<Arc<dyn Scorer>>) -> Self {
        Self { scorers }
    }
}

#[async_trait]
impl Scorer for CompositeScorer {
    async fn score(&self, case: &BenchmarkCase, actual: &Value) -> Result<Score, BenchError> {
        let mut explanations = Vec::new();
        let mut total_value = 0.0;
        let mut count = 0;

        for scorer in &self.scorers {
            let score = scorer.score(case, actual).await?;
            if !score.passed {
                explanations.push(score.explanation.unwrap_or_default());
            }
            total_value += score.value;
            count += 1;
        }

        let passed = explanations.is_empty();
        let value = if count == 0 {
            0.0
        } else {
            total_value / count as f64
        };
        let explanation = if passed {
            None
        } else {
            Some(explanations.join("; "))
        };

        Ok(Score {
            passed,
            value,
            confidence: None,
            explanation,
            metadata: Value::Object(serde_json::Map::new()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use crytex_core::services::{ExecutionResult, SandboxServiceError};

    #[tokio::test]
    async fn exact_match_passes_when_equal() {
        let case = BenchmarkCase {
            id: "1".into(),
            input: Value::Null,
            expected: Some(serde_json::json!({"answer": 42})),
            tags: vec![],
            metadata: Value::Object(serde_json::Map::new()),
        };
        let scorer = ExactMatchScorer;
        let score = scorer
            .score(&case, &serde_json::json!({"answer": 42}))
            .await
            .unwrap();
        assert!(score.passed);
        assert_eq!(score.value, 1.0);
    }

    #[tokio::test]
    async fn exact_match_fails_when_different() {
        let case = BenchmarkCase {
            id: "1".into(),
            input: Value::Null,
            expected: Some(serde_json::json!({"answer": 42})),
            tags: vec![],
            metadata: Value::Object(serde_json::Map::new()),
        };
        let scorer = ExactMatchScorer;
        let score = scorer
            .score(&case, &serde_json::json!({"answer": 43}))
            .await
            .unwrap();
        assert!(!score.passed);
    }

    #[tokio::test]
    async fn json_schema_passes_for_valid_output() {
        let case = BenchmarkCase {
            id: "1".into(),
            input: Value::Null,
            expected: None,
            tags: vec![],
            metadata: serde_json::json!({
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    },
                    "required": ["name"]
                }
            }),
        };
        let scorer = JsonSchemaScorer;
        let score = scorer
            .score(&case, &serde_json::json!({"name": "Ada"}))
            .await
            .unwrap();
        assert!(score.passed);
    }

    #[tokio::test]
    async fn json_schema_fails_for_invalid_output() {
        let case = BenchmarkCase {
            id: "1".into(),
            input: Value::Null,
            expected: None,
            tags: vec![],
            metadata: serde_json::json!({
                "schema": {
                    "type": "object",
                    "properties": {
                        "count": {"type": "integer"}
                    },
                    "required": ["count"]
                }
            }),
        };
        let scorer = JsonSchemaScorer;
        let score = scorer
            .score(&case, &serde_json::json!({"count": "many"}))
            .await
            .unwrap();
        assert!(!score.passed);
    }

    struct DummySandbox {
        should_pass: bool,
    }

    #[async_trait]
    impl SandboxService for DummySandbox {
        async fn execute(
            &self,
            _request: ExecutionRequest,
        ) -> Result<ExecutionResult, SandboxServiceError> {
            if self.should_pass {
                Ok(ExecutionResult {
                    exit_code: 0,
                    stdout: String::new(),
                    stderr: String::new(),
                })
            } else {
                Ok(ExecutionResult {
                    exit_code: 1,
                    stdout: String::new(),
                    stderr: "bad".into(),
                })
            }
        }

        fn available(&self) -> bool {
            true
        }
    }

    #[tokio::test]
    async fn sandbox_scorer_passes_on_zero_exit() {
        let case = BenchmarkCase {
            id: "1".into(),
            input: Value::Null,
            expected: None,
            tags: vec![],
            metadata: serde_json::json!({
                "test_command": ["cat", "actual.txt"],
                "filename": "actual.txt"
            }),
        };
        let scorer = SandboxTestScorer::new(Arc::new(DummySandbox { should_pass: true }));
        let score = scorer
            .score(&case, &Value::String("hello".into()))
            .await
            .unwrap();
        assert!(score.passed);
    }
}
