use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

use crytex_core::services::{Embedder, SparseEmbedder, VectorStore};

use crate::schema::{Tool, ToolContext, ToolError, ToolResult, ToolSchema};

/// Registry of tools keyed by name.
#[derive(Default, Clone)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.get(name)
    }

    pub fn list(&self) -> Vec<&Arc<dyn Tool>> {
        self.tools.values().collect()
    }

    pub fn schemas(&self) -> Vec<ToolSchema> {
        self.tools.values().map(|t| t.schema()).collect()
    }

    pub async fn invoke(&self, ctx: &ToolContext, name: &str, args: Value) -> ToolResult {
        let tool = self
            .tools
            .get(name)
            .ok_or_else(|| ToolError::NotFound(name.to_string()))?;
        tool.invoke(ctx, args).await
    }
}

/// A strongly-typed registry builder for convenience.
pub struct TypedToolRegistry {
    inner: ToolRegistry,
}

impl TypedToolRegistry {
    pub fn new() -> Self {
        Self {
            inner: ToolRegistry::new(),
        }
    }

    pub fn with_default_coding_tools(mut self) -> Self {
        use crate::fs::{FsList, FsRead, FsWrite};
        use crate::git::{GitCommit, GitDiff, GitStatus};
        use crate::process::RunCommand;
        use crate::search::SearchCode;

        self.inner.register(Arc::new(FsRead::new()));
        self.inner.register(Arc::new(FsWrite::new()));
        self.inner.register(Arc::new(FsList::new()));
        self.inner.register(Arc::new(GitStatus::new()));
        self.inner.register(Arc::new(GitDiff::new()));
        self.inner.register(Arc::new(GitCommit::new()));
        self.inner.register(Arc::new(RunCommand::new()));
        self.inner.register(Arc::new(SearchCode::new()));
        self
    }

    pub fn with_semantic_search(
        mut self,
        embedder: Arc<dyn Embedder>,
        vector_store: Arc<dyn VectorStore>,
    ) -> Self {
        use crate::search::SearchSemantic;
        self.inner
            .register(Arc::new(SearchSemantic::new(embedder, vector_store)));
        self
    }

    pub fn with_sparse_search(
        mut self,
        sparse_embedder: Arc<dyn SparseEmbedder>,
        vector_store: Arc<dyn VectorStore>,
    ) -> Self {
        use crate::search::SearchSparse;
        self.inner
            .register(Arc::new(SearchSparse::new(sparse_embedder, vector_store)));
        self
    }

    pub fn build(self) -> ToolRegistry {
        self.inner
    }
}

impl Default for TypedToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}
