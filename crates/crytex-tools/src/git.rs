use async_trait::async_trait;
use serde_json::Value;

use crate::policy::Capability;
use crate::schema::{Tool, ToolContext, ToolError, ToolResult, ToolSchema, require_str, result_ok};

fn open_repo(ctx: &ToolContext) -> Result<git2::Repository, ToolError> {
    git2::Repository::discover(&ctx.project_root).map_err(|e| ToolError::Git(e.to_string()))
}

/// Show working tree status.
pub struct GitStatus;

impl GitStatus {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GitStatus {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for GitStatus {
    fn name(&self) -> &str {
        "git_status"
    }

    fn description(&self) -> &str {
        "Show the git status of the project repository."
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name().into(),
            description: self.description().into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
            required: vec![],
        }
    }

    fn required_capabilities(&self) -> Capability {
        Capability::READ.union(Capability::GIT)
    }

    async fn execute(&self, ctx: &ToolContext, _args: Value) -> ToolResult {
        let repo = open_repo(ctx)?;
        let statuses = repo
            .statuses(None)
            .map_err(|e| ToolError::Git(e.to_string()))?;
        let mut entries = Vec::new();
        for status in statuses.iter() {
            let path = status.path().unwrap_or("?").to_string();
            let status_text = format!("{:?}", status.status());
            entries.push(serde_json::json!({ "path": path, "status": status_text }));
        }
        result_ok(serde_json::json!({ "entries": entries }))
    }
}

/// Show git diff for the working tree.
pub struct GitDiff;

impl GitDiff {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GitDiff {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for GitDiff {
    fn name(&self) -> &str {
        "git_diff"
    }

    fn description(&self) -> &str {
        "Show the git diff of the working tree against HEAD."
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name().into(),
            description: self.description().into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
            required: vec![],
        }
    }

    fn required_capabilities(&self) -> Capability {
        Capability::READ.union(Capability::GIT)
    }

    async fn execute(&self, ctx: &ToolContext, _args: Value) -> ToolResult {
        let repo = open_repo(ctx)?;
        let head = repo.head().map_err(|e| ToolError::Git(e.to_string()))?;
        let head_tree = head
            .peel_to_tree()
            .map_err(|e| ToolError::Git(e.to_string()))?;
        let diff = repo
            .diff_tree_to_workdir_with_index(Some(&head_tree), None)
            .map_err(|e| ToolError::Git(e.to_string()))?;
        let mut buf = Vec::new();
        diff.print(git2::DiffFormat::Patch, |_delta, _hunk, line| {
            buf.extend_from_slice(line.content());
            true
        })
        .map_err(|e| ToolError::Git(e.to_string()))?;
        let text = String::from_utf8_lossy(&buf).to_string();
        result_ok(serde_json::json!({ "diff": text }))
    }
}

/// Commit staged/unstaged changes.
pub struct GitCommit;

impl GitCommit {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GitCommit {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for GitCommit {
    fn name(&self) -> &str {
        "git_commit"
    }

    fn description(&self) -> &str {
        "Stage all changes and create a git commit with the given message."
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name().into(),
            description: self.description().into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "message": { "type": "string", "description": "commit message" }
                }
            }),
            required: vec!["message".into()],
        }
    }

    fn required_capabilities(&self) -> Capability {
        Capability::READ
            .union(Capability::WRITE)
            .union(Capability::GIT)
    }

    async fn execute(&self, ctx: &ToolContext, args: Value) -> ToolResult {
        self.schema().validate_required(&args)?;
        let message = require_str(&args, "message")?;
        let repo = open_repo(ctx)?;

        // Stage all changes.
        let mut index = repo.index().map_err(|e| ToolError::Git(e.to_string()))?;
        index
            .add_all(["*"].iter(), git2::IndexAddOption::DEFAULT, None)
            .map_err(|e| ToolError::Git(e.to_string()))?;
        index.write().map_err(|e| ToolError::Git(e.to_string()))?;

        let tree_id = index
            .write_tree()
            .map_err(|e| ToolError::Git(e.to_string()))?;
        let tree = repo
            .find_tree(tree_id)
            .map_err(|e| ToolError::Git(e.to_string()))?;

        let sig = repo
            .signature()
            .or_else(|_| git2::Signature::now("crytex", "crytex@local"))
            .map_err(|e| ToolError::Git(format!("invalid signature: {e}")))?;
        let parent = repo.head().map_err(|e| ToolError::Git(e.to_string()))?;
        let parent_commit = parent
            .peel_to_commit()
            .map_err(|e| ToolError::Git(e.to_string()))?;

        let commit_id = repo
            .commit(Some("HEAD"), &sig, &sig, &message, &tree, &[&parent_commit])
            .map_err(|e| ToolError::Git(e.to_string()))?;

        result_ok(serde_json::json!({ "commit": commit_id.to_string() }))
    }
}
