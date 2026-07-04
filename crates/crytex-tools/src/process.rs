use async_trait::async_trait;
use crytex_core::services::{ExecutionRequest, SandboxNetwork, SandboxResources};
use serde_json::Value;
use std::path::PathBuf;
use std::time::Duration;
use tokio::process::Command;

use crate::policy::Capability;
use crate::sandbox::PathSandbox;
use crate::schema::{
    Tool, ToolContext, ToolError, ToolResult, ToolSchema, optional_str, require_str, result_ok,
};

/// Run an external command inside the project workspace.
///
/// The command is executed directly (no shell), with `cwd` restricted to the
/// project root. Requires `SHELL` capability.
pub struct RunCommand;

impl RunCommand {
    pub fn new() -> Self {
        Self
    }

    fn sanitize_args(args: &[String]) -> Result<(), ToolError> {
        let forbidden = [';', '|', '&', '$', '`', '\n'];
        for arg in args {
            if arg.contains(forbidden) {
                return Err(ToolError::Forbidden(format!(
                    "argument contains forbidden shell metacharacter: {}",
                    arg
                )));
            }
        }
        Ok(())
    }

    fn is_network_command(command: &str) -> bool {
        matches!(
            command.to_ascii_lowercase().as_str(),
            "curl" | "wget" | "nc" | "netcat" | "telnet" | "ssh" | "ftp" | "sftp"
        )
    }
}

impl Default for RunCommand {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for RunCommand {
    fn name(&self) -> &str {
        "run_command"
    }

    fn description(&self) -> &str {
        "Run an external command (argv, no shell) inside the project workspace."
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name().into(),
            description: self.description().into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string", "description": "executable name" },
                    "args": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "command arguments"
                    },
                    "cwd": { "type": "string", "description": "working directory relative to project root", "default": "." }
                }
            }),
            required: vec!["command".into()],
        }
    }

    fn required_capabilities(&self) -> Capability {
        Capability::READ.union(Capability::SHELL)
    }

    async fn execute(&self, ctx: &ToolContext, args: Value) -> ToolResult {
        self.schema().validate_required(&args)?;
        let command = require_str(&args, "command")?;
        let cmd_args: Vec<String> = args
            .get("args")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        let cwd = optional_str(&args, "cwd").unwrap_or_else(|| ".".to_string());

        Self::sanitize_args(&cmd_args)?;

        if Self::is_network_command(&command) && !ctx.permissions.contains(Capability::NETWORK) {
            return Err(ToolError::Forbidden(format!(
                "network capability is required to run '{}'",
                command
            )));
        }

        let path_sandbox = PathSandbox::new(&ctx.project_root, "");
        let host_cwd = path_sandbox.resolve(&cwd)?;
        let relative_cwd = host_cwd
            .strip_prefix(&ctx.project_root)
            .unwrap_or_else(|_| std::path::Path::new(""));
        let guest_cwd = PathBuf::from("/workspace").join(relative_cwd);

        let mut full_command = vec![command];
        full_command.extend(cmd_args);

        let read_only = !ctx.permissions.contains(Capability::WRITE);
        let network = if ctx.permissions.contains(Capability::NETWORK) {
            SandboxNetwork::Allow
        } else {
            SandboxNetwork::Deny
        };

        let request = ExecutionRequest::new(full_command)
            .cwd(guest_cwd)
            .mount(
                ctx.project_root.clone(),
                PathBuf::from("/workspace"),
                read_only,
            )
            .network(network)
            .resources(SandboxResources {
                timeout_seconds: ctx.timeout_seconds,
                ..Default::default()
            });

        let result = ctx.sandbox.execute(request).await?;

        if result.exit_code != 0 {
            return Err(ToolError::Process {
                exit_code: Some(result.exit_code as i32),
                stderr: result.stderr,
            });
        }

        result_ok(serde_json::json!({
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
        }))
    }
}

/// Run a command with a timeout.
pub async fn run_with_timeout(
    command: &str,
    args: &[String],
    cwd: &std::path::Path,
    timeout_seconds: u64,
) -> Result<std::process::Output, ToolError> {
    let output = tokio::time::timeout(
        Duration::from_secs(timeout_seconds),
        Command::new(command)
            .args(args)
            .current_dir(cwd)
            .kill_on_drop(true)
            .output(),
    )
    .await
    .map_err(|_| ToolError::Timeout(timeout_seconds))?
    .map_err(ToolError::Io)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::policy::Capability;

    fn shell_ctx() -> ToolContext {
        ToolContext {
            permissions: Capability::SHELL.union(Capability::READ),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn run_command_uses_sandbox_backend() {
        let tool = RunCommand::new();
        #[cfg(target_os = "windows")]
        let args = json!({
            "command": "cmd",
            "args": ["/c", "echo", "sandbox-hello"]
        });
        #[cfg(not(target_os = "windows"))]
        let args = json!({
            "command": "sh",
            "args": ["-c", "echo sandbox-hello"]
        });

        let result = tool.execute(&shell_ctx(), args).await.unwrap();

        assert!(
            result
                .get("stdout")
                .unwrap()
                .as_str()
                .unwrap()
                .contains("sandbox-hello")
        );
        assert_eq!(result.get("exit_code").unwrap().as_i64(), Some(0));
    }

    #[tokio::test]
    async fn tool_without_network_capability_cannot_run_curl() {
        let tool = RunCommand::new();
        let ctx = ToolContext {
            permissions: Capability::SHELL.union(Capability::READ),
            ..Default::default()
        };
        let args = json!({
            "command": "curl",
            "args": ["http://example.com"]
        });

        let err = tool.execute(&ctx, args).await.unwrap_err();

        assert!(
            matches!(err, ToolError::Forbidden(_)),
            "expected Forbidden, got {:?}",
            err
        );
    }
}
