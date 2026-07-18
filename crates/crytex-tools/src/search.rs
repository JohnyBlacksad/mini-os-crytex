use async_trait::async_trait;
use ignore::gitignore::Gitignore;
use regex::Regex;
use serde_json::Value;
use std::sync::Arc;
use walkdir::WalkDir;

use crytex_core::services::{Embedder, SearchOptions, SearchResult, SparseEmbedder, VectorStore};

use crate::policy::Capability;
use crate::schema::{
    Tool, ToolContext, ToolError, ToolResult, ToolSchema, optional_str, optional_usize,
    require_str, result_ok,
};

/// Search code by content or filename inside the project workspace.
pub struct SearchCode;

impl SearchCode {
    pub fn new() -> Self {
        Self
    }
}

impl Default for SearchCode {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for SearchCode {
    fn name(&self) -> &str {
        "search_code"
    }

    fn description(&self) -> &str {
        "Search for files or content inside the project workspace."
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name().into(),
            description: self.description().into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "substring or regex to search" },
                    "path": { "type": "string", "default": ".", "description": "directory to search" },
                    "kind": { "type": "string", "enum": ["content", "filename"], "default": "content" }
                }
            }),
            required: vec!["query".into()],
        }
    }

    fn required_capabilities(&self) -> Capability {
        Capability::READ
    }

    async fn execute(&self, ctx: &ToolContext, args: Value) -> ToolResult {
        self.schema().validate_required(&args)?;
        let query = require_str(&args, "query")?;
        let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
        let kind = args
            .get("kind")
            .and_then(|v| v.as_str())
            .unwrap_or("content");

        let regex = Regex::new(&query).map_err(|e| ToolError::InvalidArgs(e.to_string()))?;
        let search_root = ctx.project_root.join(path);

        let mut matches = Vec::new();
        let walker = WalkDir::new(&search_root)
            .follow_links(false)
            .into_iter()
            .filter_entry(|e| !is_hidden(e.file_name()));

        for entry in walker {
            let entry = entry.map_err(|e| {
                let msg = e.to_string();
                e.into_io_error()
                    .map(|io| ToolError::FileSystem {
                        path: search_root.to_string_lossy().to_string(),
                        source: io,
                    })
                    .unwrap_or_else(|| ToolError::Search(msg))
            })?;
            if !entry.file_type().is_file() {
                continue;
            }

            let relative = entry
                .path()
                .strip_prefix(&ctx.project_root)
                .unwrap_or(entry.path())
                .to_string_lossy()
                .to_string();

            if kind == "filename" {
                if regex.is_match(&entry.file_name().to_string_lossy()) {
                    matches.push(serde_json::json!({ "path": relative }));
                }
                continue;
            }

            // Respect .gitignore by skipping ignored files.
            if is_ignored(entry.path(), &ctx.project_root) {
                continue;
            }

            let content = tokio::fs::read_to_string(entry.path())
                .await
                .unwrap_or_default();
            if regex.is_match(&content) {
                matches.push(serde_json::json!({ "path": relative }));
            }
        }

        result_ok(serde_json::json!({ "matches": matches, "count": matches.len() }))
    }
}

/// Semantic search over indexed project chunks using an embedding model.
pub struct SearchSemantic {
    embedder: Arc<dyn Embedder>,
    vector_store: Arc<dyn VectorStore>,
}

impl SearchSemantic {
    pub fn new(embedder: Arc<dyn Embedder>, vector_store: Arc<dyn VectorStore>) -> Self {
        Self {
            embedder,
            vector_store,
        }
    }
}

#[async_trait]
impl Tool for SearchSemantic {
    fn name(&self) -> &str {
        "search_semantic"
    }

    fn description(&self) -> &str {
        "Search indexed code, documentation, and experience chunks by semantic similarity."
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name().into(),
            description: self.description().into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "natural-language query" },
                    "project_id": { "type": "string", "description": "optional project scope" },
                    "collection": { "type": "string", "description": "collection to search (code_chunks, doc_chunks, experience); defaults to all" },
                    "limit": { "type": "integer", "default": 5, "description": "maximum number of results" }
                }
            }),
            required: vec!["query".into()],
        }
    }

    fn required_capabilities(&self) -> Capability {
        Capability::READ
    }

    async fn execute(&self, _ctx: &ToolContext, args: Value) -> ToolResult {
        self.schema().validate_required(&args)?;
        let query = require_str(&args, "query")?;
        let project_id = optional_str(&args, "project_id");
        let collection = optional_str(&args, "collection");
        let limit = optional_usize(&args, "limit")?.unwrap_or(5);

        let vector = self
            .embedder
            .embed(&query)
            .await
            .map_err(|e| ToolError::Embedding(e.to_string()))?;

        let filter = project_id
            .as_ref()
            .map(|id| serde_json::json!({"project_id": {"match": {"value": id}}}));
        let options = SearchOptions {
            limit,
            filter,
            score_threshold: None,
        };

        let collections: Vec<&str> = match collection.as_deref() {
            Some(name) => vec![name],
            None => vec!["code_chunks", "doc_chunks", "experience"],
        };

        let mut combined = Vec::new();
        for name in collections {
            let mut results = self
                .vector_store
                .search(name, &vector, options.clone())
                .await
                .map_err(|e| ToolError::VectorStore(e.to_string()))?;
            combined.append(&mut results);
        }

        combined.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        combined.truncate(limit);

        let results: Vec<Value> = combined.into_iter().map(format_result).collect();
        result_ok(serde_json::json!({ "results": results, "count": results.len() }))
    }
}

/// Sparse (BM25/keyword) search over indexed project chunks using a sparse
/// embedder.
pub struct SearchSparse {
    sparse_embedder: Arc<dyn SparseEmbedder>,
    vector_store: Arc<dyn VectorStore>,
}

impl SearchSparse {
    pub fn new(
        sparse_embedder: Arc<dyn SparseEmbedder>,
        vector_store: Arc<dyn VectorStore>,
    ) -> Self {
        Self {
            sparse_embedder,
            vector_store,
        }
    }
}

#[async_trait]
impl Tool for SearchSparse {
    fn name(&self) -> &str {
        "search_sparse"
    }

    fn description(&self) -> &str {
        "Search indexed code, documentation, and experience chunks by lexical/BM25 similarity."
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name().into(),
            description: self.description().into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "natural-language or keyword query" },
                    "project_id": { "type": "string", "description": "optional project scope" },
                    "collection": { "type": "string", "description": "collection to search (code_chunks, doc_chunks, experience); defaults to all" },
                    "limit": { "type": "integer", "default": 5, "description": "maximum number of results" }
                }
            }),
            required: vec!["query".into()],
        }
    }

    fn required_capabilities(&self) -> Capability {
        Capability::READ
    }

    async fn execute(&self, _ctx: &ToolContext, args: Value) -> ToolResult {
        self.schema().validate_required(&args)?;
        let query = require_str(&args, "query")?;
        let project_id = optional_str(&args, "project_id");
        let collection = optional_str(&args, "collection");
        let limit = optional_usize(&args, "limit")?.unwrap_or(5);

        let sparse_vector = self
            .sparse_embedder
            .embed_query(&query)
            .await
            .map_err(|e| ToolError::Embedding(e.to_string()))?;

        let filter = project_id
            .as_ref()
            .map(|id| serde_json::json!({"project_id": {"match": {"value": id}}}));
        let options = SearchOptions {
            limit,
            filter,
            score_threshold: None,
        };

        let collections: Vec<&str> = match collection.as_deref() {
            Some(name) => vec![name],
            None => vec!["code_chunks", "doc_chunks", "experience"],
        };

        let mut combined = Vec::new();
        for name in collections {
            let mut results = self
                .vector_store
                .search_sparse(name, &sparse_vector, options.clone())
                .await
                .map_err(|e| ToolError::VectorStore(e.to_string()))?;
            combined.append(&mut results);
        }

        combined.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        combined.truncate(limit);

        let results: Vec<Value> = combined.into_iter().map(format_result).collect();
        result_ok(serde_json::json!({ "results": results, "count": results.len() }))
    }
}

fn format_result(result: SearchResult) -> Value {
    serde_json::json!({
        "id": result.id,
        "score": result.score,
        "source": result.payload.get("source").and_then(|v| v.as_str()).unwrap_or(""),
        "text": result.payload.get("text").and_then(|v| v.as_str()).unwrap_or(""),
    })
}

fn is_hidden(name: &std::ffi::OsStr) -> bool {
    name.to_string_lossy().starts_with('.')
}

fn is_ignored(path: &std::path::Path, project_root: &std::path::Path) -> bool {
    let mut builder = ignore::gitignore::GitignoreBuilder::new(project_root);
    let gitignore_path = project_root.join(".gitignore");
    if gitignore_path.exists() {
        let _ = builder.add(gitignore_path);
    }
    let gitignore = builder.build().unwrap_or_else(|_| Gitignore::empty());
    let matched = gitignore.matched(path, path.is_dir());
    matched.is_ignore()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crytex_core::services::{
        Embedder, EmbeddingError, MockSparseEmbedder, SearchOptions, SparseEmbedder, SparseVector,
        SparseVectorPoint, VectorPoint, VectorStoreError,
    };
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    /// Deterministic embedder that assigns axis-aligned vectors based on keywords.
    struct KeywordEmbedder;

    #[async_trait]
    impl Embedder for KeywordEmbedder {
        async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
            let mut vec = vec![0.0f32; 8];
            let lower = text.to_lowercase();
            if lower.contains("fetch") || lower.contains("http") {
                vec[0] = 1.0;
            }
            if lower.contains("main") || lower.contains("hello") {
                vec[1] = 1.0;
            }
            Ok(vec)
        }

        async fn dimension(&self) -> Result<usize, EmbeddingError> {
            Ok(8)
        }
    }

    #[derive(Default)]
    struct TestVectorStore {
        collections: Mutex<HashMap<String, Vec<VectorPoint>>>,
        sparse_collections: Mutex<HashMap<String, Vec<SparseVectorPoint>>>,
    }

    fn sparse_dot(a: &SparseVector, b: &SparseVector) -> f32 {
        let mut a: Vec<_> = a.indices.iter().zip(a.values.iter()).collect();
        let mut b: Vec<_> = b.indices.iter().zip(b.values.iter()).collect();
        a.sort_by_key(|(i, _)| *i);
        b.sort_by_key(|(i, _)| *i);
        let mut i = 0;
        let mut j = 0;
        let mut score = 0.0f32;
        while i < a.len() && j < b.len() {
            match a[i].0.cmp(b[j].0) {
                std::cmp::Ordering::Equal => {
                    score += *a[i].1 * *b[j].1;
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
            }
        }
        score
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 {
            0.0
        } else {
            dot / (na * nb)
        }
    }

    #[async_trait]
    impl VectorStore for TestVectorStore {
        async fn create_collection(
            &self,
            _collection: &str,
            _dim: usize,
        ) -> Result<(), VectorStoreError> {
            Ok(())
        }
        async fn delete_collection(&self, _collection: &str) -> Result<(), VectorStoreError> {
            Ok(())
        }
        async fn upsert(
            &self,
            _collection: &str,
            _points: Vec<VectorPoint>,
        ) -> Result<(), VectorStoreError> {
            Ok(())
        }
        async fn search(
            &self,
            collection: &str,
            vector: &[f32],
            options: SearchOptions,
        ) -> Result<Vec<SearchResult>, VectorStoreError> {
            let cols = self.collections.lock().unwrap();
            let points = cols.get(collection).cloned().unwrap_or_default();
            let project_filter = options
                .filter
                .as_ref()
                .and_then(|f| f.get("project_id"))
                .and_then(|f| f.get("match"))
                .and_then(|f| f.get("value"))
                .and_then(|v| v.as_str());
            let mut results: Vec<SearchResult> = points
                .iter()
                .filter(|p| {
                    project_filter.is_none_or(|expected| {
                        p.payload.get("project_id").and_then(|v| v.as_str()) == Some(expected)
                    })
                })
                .map(|p| SearchResult {
                    id: p.id.clone(),
                    score: cosine(vector, &p.vector),
                    payload: p.payload.clone(),
                })
                .collect();
            results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            results.truncate(options.limit);
            Ok(results)
        }

        async fn supports_sparse(&self) -> bool {
            true
        }

        async fn create_sparse_collection(
            &self,
            _collection: &str,
            _dim: usize,
        ) -> Result<(), VectorStoreError> {
            Ok(())
        }

        async fn upsert_with_sparse(
            &self,
            _collection: &str,
            _points: Vec<SparseVectorPoint>,
        ) -> Result<(), VectorStoreError> {
            Ok(())
        }

        async fn search_sparse(
            &self,
            collection: &str,
            vector: &SparseVector,
            options: SearchOptions,
        ) -> Result<Vec<SearchResult>, VectorStoreError> {
            let cols = self.sparse_collections.lock().unwrap();
            let points = cols.get(collection).cloned().unwrap_or_default();
            let project_filter = options
                .filter
                .as_ref()
                .and_then(|f| f.get("project_id"))
                .and_then(|f| f.get("match"))
                .and_then(|f| f.get("value"))
                .and_then(|v| v.as_str());
            let mut results: Vec<SearchResult> = points
                .iter()
                .filter(|p| {
                    project_filter.is_none_or(|expected| {
                        p.payload.get("project_id").and_then(|v| v.as_str()) == Some(expected)
                    })
                })
                .map(|p| SearchResult {
                    id: p.id.clone(),
                    score: sparse_dot(vector, &p.sparse_vector),
                    payload: p.payload.clone(),
                })
                .collect();
            results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            results.truncate(options.limit);
            Ok(results)
        }
    }

    #[tokio::test]
    async fn search_semantic_returns_relevant_code_chunks() {
        let vector_store = Arc::new(TestVectorStore::default());
        {
            let mut cols = vector_store.collections.lock().unwrap();
            cols.insert(
                "code_chunks".into(),
                vec![
                    VectorPoint {
                        id: "chunk-1".into(),
                        vector: vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        payload: serde_json::json!({
                            "project_id": "proj-1",
                            "source": "src/http.rs",
                            "text": "pub fn fetch(url: &str) -> Result<String, Error> { ... }",
                        }),
                    },
                    VectorPoint {
                        id: "chunk-2".into(),
                        vector: vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        payload: serde_json::json!({
                            "project_id": "proj-1",
                            "source": "src/main.rs",
                            "text": "fn main() { println!(\"hello\"); }",
                        }),
                    },
                ],
            );
        }
        let embedder: Arc<dyn Embedder> = Arc::new(KeywordEmbedder);
        let tool = SearchSemantic::new(embedder, vector_store);
        let ctx = ToolContext {
            project_root: std::path::PathBuf::from("."),
            ..Default::default()
        };

        let args = serde_json::json!({
            "query": "HTTP client fetch function",
            "project_id": "proj-1",
            "collection": "code_chunks",
            "limit": 1,
        });
        let result = tool.execute(&ctx, args).await.unwrap();
        let results = result.get("results").unwrap().as_array().unwrap();
        assert_eq!(results.len(), 1);
        assert!(
            results[0]
                .get("text")
                .unwrap()
                .as_str()
                .unwrap()
                .contains("fetch")
        );
    }

    #[tokio::test]
    async fn search_sparse_returns_relevant_code_chunks() {
        let sparse_embedder = MockSparseEmbedder;
        let vector_store = Arc::new(TestVectorStore::default());
        let sparse_fetch = sparse_embedder
            .embed_document("pub fn fetch url")
            .await
            .unwrap();
        let sparse_main = sparse_embedder
            .embed_document("fn main hello")
            .await
            .unwrap();
        {
            let mut cols = vector_store.sparse_collections.lock().unwrap();
            cols.insert(
                "code_chunks".into(),
                vec![
                    SparseVectorPoint {
                        id: "chunk-1".into(),
                        vector: vec![0.0; 8],
                        sparse_vector: sparse_fetch,
                        payload: serde_json::json!({
                            "project_id": "proj-1",
                            "source": "src/http.rs",
                            "text": "pub fn fetch(url: &str) -> Result<String, Error> { ... }",
                        }),
                    },
                    SparseVectorPoint {
                        id: "chunk-2".into(),
                        vector: vec![0.0; 8],
                        sparse_vector: sparse_main,
                        payload: serde_json::json!({
                            "project_id": "proj-1",
                            "source": "src/main.rs",
                            "text": "fn main() { println!(\"hello\"); }",
                        }),
                    },
                ],
            );
        }

        let sparse_embedder: Arc<dyn SparseEmbedder> = Arc::new(MockSparseEmbedder);
        let tool = SearchSparse::new(sparse_embedder, vector_store);
        let ctx = ToolContext {
            project_root: std::path::PathBuf::from("."),
            ..Default::default()
        };

        let args = serde_json::json!({
            "query": "fetch url",
            "project_id": "proj-1",
            "collection": "code_chunks",
            "limit": 1,
        });
        let result = tool.execute(&ctx, args).await.unwrap();
        let results = result.get("results").unwrap().as_array().unwrap();
        assert_eq!(results.len(), 1);
        assert!(
            results[0]
                .get("text")
                .unwrap()
                .as_str()
                .unwrap()
                .contains("fetch")
        );
    }
}
