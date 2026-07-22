//! Strict contracts for artifacts produced by well-known agents.

use serde_json::Value;
use thiserror::Error;

/// Contract validation failure for an agent artifact.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{artifact_kind} contract violation: {reason}")]
pub struct ArtifactContractViolation {
    pub artifact_kind: String,
    pub reason: String,
}

/// Return whether an agent role must emit a typed artifact.
pub fn requires_agent_artifact_contract(agent: Option<&str>) -> bool {
    agent.is_some_and(|agent| {
        matches!(
            agent,
            "orchestrator"
                | "architect"
                | "coder"
                | "analyst"
                | "researcher"
                | "qa"
                | "devops"
                | "security"
                | "critic"
                | "summarizer"
        ) || agent.starts_with("coder-")
            || agent.starts_with("critic-")
    })
}

/// Return the artifact kind expected from a known agent.
pub fn artifact_kind_for_agent(agent: Option<&str>, fallback_kind: &str) -> String {
    match agent {
        Some("orchestrator") => "task_graph_artifact".to_string(),
        Some("architect") => "design_artifact".to_string(),
        Some(agent) if agent == "coder" || agent.starts_with("coder-") => {
            "patch_artifact".to_string()
        }
        Some("analyst") => "analysis_artifact".to_string(),
        Some("researcher") => "research_artifact".to_string(),
        Some("qa") => "test_report_artifact".to_string(),
        Some("devops") => "deployment_artifact".to_string(),
        Some("security") => "security_report_artifact".to_string(),
        Some(agent) if agent == "critic" || agent.starts_with("critic-") => {
            "review_decision".to_string()
        }
        Some("summarizer") => "summary_artifact".to_string(),
        Some(agent) => format!("{agent}_artifact"),
        None => format!("{fallback_kind}_artifact"),
    }
}

/// Extract artifact content from the result shape returned by agents.
pub fn artifact_content_from_result(result: &Value) -> Value {
    result
        .pointer("/agent_result/artifact")
        .or_else(|| result.pointer("/artifact"))
        .or_else(|| result.pointer("/agent_result"))
        .cloned()
        .unwrap_or_else(|| result.clone())
}

/// Validate a complete agent result for the agent role.
pub fn validate_agent_result(
    agent: Option<&str>,
    fallback_kind: &str,
    result: &Value,
) -> Result<(), ArtifactContractViolation> {
    if !requires_agent_artifact_contract(agent) {
        return Ok(());
    }
    let artifact_kind = artifact_kind_for_agent(agent, fallback_kind);
    let content = artifact_content_from_result(result);
    validate_artifact_content(&artifact_kind, &content)
}

/// Validate artifact content for a typed artifact kind.
pub fn validate_artifact_content(
    artifact_kind: &str,
    content: &Value,
) -> Result<(), ArtifactContractViolation> {
    require_object(artifact_kind, content, "content")?;
    match artifact_kind {
        "task_graph_artifact" => validate_task_graph_artifact(artifact_kind, content),
        "design_artifact" => validate_design_artifact(artifact_kind, content),
        "patch_artifact" => validate_patch_artifact(artifact_kind, content),
        "analysis_artifact" => validate_analysis_artifact(artifact_kind, content),
        "research_artifact" => validate_research_artifact(artifact_kind, content),
        "test_report_artifact" => validate_test_report_artifact(artifact_kind, content),
        "deployment_artifact" => validate_deployment_artifact(artifact_kind, content),
        "security_report_artifact" => validate_security_report_artifact(artifact_kind, content),
        "review_decision" => validate_review_decision_artifact(artifact_kind, content),
        "summary_artifact" => validate_summary_artifact(artifact_kind, content),
        _ => Ok(()),
    }
}

fn validate_task_graph_artifact(
    artifact_kind: &str,
    content: &Value,
) -> Result<(), ArtifactContractViolation> {
    require_text_field(artifact_kind, content, "summary")?;
    require_array_field(artifact_kind, content, "tasks")?;
    require_array_field(artifact_kind, content, "dependency_edges")?;
    Ok(())
}

fn validate_design_artifact(
    artifact_kind: &str,
    content: &Value,
) -> Result<(), ArtifactContractViolation> {
    require_text_field(artifact_kind, content, "summary")
        .or_else(|_| require_text_field(artifact_kind, content, "content"))
        .map(|_| ())
}

fn validate_patch_artifact(
    artifact_kind: &str,
    content: &Value,
) -> Result<(), ArtifactContractViolation> {
    require_array_field(artifact_kind, content, "files_changed")?;
    require_text_field(artifact_kind, content, "summary")?;
    Ok(())
}

fn validate_analysis_artifact(
    artifact_kind: &str,
    content: &Value,
) -> Result<(), ArtifactContractViolation> {
    require_text_field(artifact_kind, content, "summary")?;
    require_array_field(artifact_kind, content, "findings")?;
    Ok(())
}

fn validate_research_artifact(
    artifact_kind: &str,
    content: &Value,
) -> Result<(), ArtifactContractViolation> {
    require_text_field(artifact_kind, content, "summary")?;
    require_array_field(artifact_kind, content, "sources")?;
    Ok(())
}

fn validate_test_report_artifact(
    artifact_kind: &str,
    content: &Value,
) -> Result<(), ArtifactContractViolation> {
    require_text_field(artifact_kind, content, "summary")
        .or_else(|_| require_text_field(artifact_kind, content, "test_results"))
        .map(|_| ())
}

fn validate_security_report_artifact(
    artifact_kind: &str,
    content: &Value,
) -> Result<(), ArtifactContractViolation> {
    require_text_field(artifact_kind, content, "summary")
        .or_else(|_| require_text_field(artifact_kind, content, "risk"))
        .map(|_| ())
}

fn validate_deployment_artifact(
    artifact_kind: &str,
    content: &Value,
) -> Result<(), ArtifactContractViolation> {
    require_text_field(artifact_kind, content, "summary")?;
    require_array_field(artifact_kind, content, "commands")?;
    Ok(())
}

fn validate_summary_artifact(
    artifact_kind: &str,
    content: &Value,
) -> Result<(), ArtifactContractViolation> {
    require_text_field(artifact_kind, content, "summary")?;
    require_array_field(artifact_kind, content, "key_points")?;
    Ok(())
}

fn validate_review_decision_artifact(
    artifact_kind: &str,
    content: &Value,
) -> Result<(), ArtifactContractViolation> {
    let decision = require_text_field(artifact_kind, content, "decision")
        .or_else(|_| require_text_field(artifact_kind, content, "review_decision"))?;
    require_text_field(artifact_kind, content, "reason")
        .or_else(|_| require_text_field(artifact_kind, content, "summary"))?;
    require_text_field(artifact_kind, content, "target_task")
        .or_else(|_| require_text_field(artifact_kind, content, "target_task_id"))?;
    require_object(
        artifact_kind,
        content.get("remediation_proposal").unwrap_or(&Value::Null),
        "remediation_proposal",
    )?;
    if matches!(decision, "reject" | "request_changes") {
        require_array_field(artifact_kind, content, "blocking_issues")?;
    }
    Ok(())
}

fn require_object(
    artifact_kind: &str,
    value: &Value,
    field: &str,
) -> Result<(), ArtifactContractViolation> {
    value
        .as_object()
        .map(|_| ())
        .ok_or_else(|| violation(artifact_kind, format!("{field} must be an object")))
}

fn require_text_field<'a>(
    artifact_kind: &str,
    content: &'a Value,
    field: &str,
) -> Result<&'a str, ArtifactContractViolation> {
    content
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| {
            violation(
                artifact_kind,
                format!("artifact content requires non-empty string field `{field}`"),
            )
        })
}

fn require_array_field<'a>(
    artifact_kind: &str,
    content: &'a Value,
    field: &str,
) -> Result<&'a Vec<Value>, ArtifactContractViolation> {
    content
        .get(field)
        .and_then(Value::as_array)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            violation(
                artifact_kind,
                format!("artifact content requires non-empty array field `{field}`"),
            )
        })
}

fn violation(artifact_kind: &str, reason: String) -> ArtifactContractViolation {
    ArtifactContractViolation {
        artifact_kind: artifact_kind.to_string(),
        reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn coder_contract_requires_files_changed_and_summary() {
        let result = json!({
            "agent_result": {
                "summary": "patched retry behavior"
            }
        });

        let err = validate_agent_result(Some("coder"), "codegen", &result).unwrap_err();

        assert_eq!(err.artifact_kind, "patch_artifact");
        assert!(err.reason.contains("files_changed"));
    }

    #[test]
    fn critic_reject_contract_requires_blocking_issues() {
        let result = json!({
            "agent_result": {
                "decision": "request_changes",
                "reason": "needs work",
                "target_task": "task-1",
                "remediation_proposal": {"assigned_agent": "coder"},
                "summary": "needs work"
            }
        });

        let err = validate_agent_result(Some("critic"), "review", &result).unwrap_err();

        assert_eq!(err.artifact_kind, "review_decision");
        assert!(err.reason.contains("blocking_issues"));
    }

    #[test]
    fn well_known_agent_contracts_accept_valid_outputs() {
        let cases = [
            (
                Some("architect"),
                "architecture",
                json!({"agent_result": {"summary": "atomic plan"}}),
            ),
            (
                Some("coder"),
                "codegen",
                json!({"agent_result": {"summary": "patched", "files_changed": ["src/lib.rs"]}}),
            ),
            (
                Some("qa"),
                "qa",
                json!({"agent_result": {"test_results": "cargo test passed"}}),
            ),
            (
                Some("security"),
                "security",
                json!({"agent_result": {"risk": "low"}}),
            ),
            (
                Some("critic"),
                "review",
                json!({"agent_result": {
                    "decision": "approve",
                    "reason": "contract is satisfied",
                    "target_task": "task-1",
                    "blocking_issues": [],
                    "remediation_proposal": {"assigned_agent": "none", "goal": "none"}
                }}),
            ),
        ];

        for (agent, kind, result) in cases {
            assert!(
                validate_agent_result(agent, kind, &result).is_ok(),
                "expected {agent:?} to pass"
            );
        }
    }
}
