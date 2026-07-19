//! Project indexer: walks a project, chunks code/docs, embeds and upserts to a vector store.

use std::path::Path;
use std::sync::Arc;

use crytex_doc::{
    Chunk, ChunkKind, chunk_code,
    chunking::chunk_code_with_graph,
    graph::{CodeGraph, builder::CodeGraphBuilder},
    parse_doc, parse_pdf_bytes, walk_project,
};

use crate::services::{
    Embedder, EmbeddingError, SearchOptions, SparseEmbedder, SparseVectorPoint, VectorPoint,
    VectorStore, VectorStoreError,
};

/// Errors that can occur during indexing.
#[derive(Debug, thiserror::Error)]
pub enum IndexerError {
    #[error("chunk error: {0}")]
    Chunk(String),
    #[error("embedding error: {0}")]
    Embedding(#[from] EmbeddingError),
    #[error("vector store error: {0}")]
    VectorStore(#[from] VectorStoreError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Indexes a project into `code_chunks` and `doc_chunks` collections.
#[derive(Clone)]
pub struct ProjectIndexer {
    embedder: Arc<dyn Embedder>,
    vector_store: Arc<dyn VectorStore>,
    sparse_embedder: Option<Arc<dyn SparseEmbedder>>,
}

impl ProjectIndexer {
    pub fn new(embedder: Arc<dyn Embedder>, vector_store: Arc<dyn VectorStore>) -> Self {
        Self {
            embedder,
            vector_store,
            sparse_embedder: None,
        }
    }

    /// Attach a sparse embedder (e.g. BM25) so that chunks are also indexed as
    /// sparse vectors when the store supports them.
    pub fn with_sparse_embedder(mut self, sparse_embedder: Arc<dyn SparseEmbedder>) -> Self {
        self.sparse_embedder = Some(sparse_embedder);
        self
    }

    /// Index `project_root` for `project_id`.
    pub async fn index(
        &self,
        project_id: &str,
        project_root: &Path,
    ) -> Result<IndexStats, IndexerError> {
        let mut stats = IndexStats::default();
        let files = walk_project(project_root).map_err(|e| IndexerError::Chunk(e.to_string()))?;
        let code_graph = CodeGraphBuilder::new()
            .index_project(project_root)
            .map_err(|e| IndexerError::Chunk(e.to_string()))?;

        for path in files {
            let relative = Path::new(&path)
                .strip_prefix(project_root)
                .unwrap_or(Path::new(&path))
                .to_string_lossy()
                .to_string()
                .replace('\\', "/");
            let Ok(bytes) = tokio::fs::read(&path).await else {
                continue;
            };

            let chunks = self.chunk_file(&path, &relative, &bytes, Some(&code_graph))?;
            if chunks.is_empty() {
                continue;
            }

            let collection = if chunks[0].kind == ChunkKind::Code {
                "code_chunks"
            } else {
                "doc_chunks"
            };

            self.upsert_chunks(project_id, &relative, collection, chunks)
                .await?;
            stats.files_indexed += 1;
            stats.chunks_indexed += 1; // approximate, updated below per batch
        }

        Ok(stats)
    }

    /// Index or re-index a single file for `project_id`.
    ///
    /// Any existing chunks for the same relative path are removed first so that
    /// stale chunks (e.g. renamed or deleted symbols) do not remain in the
    /// index.
    pub async fn index_file(
        &self,
        project_id: &str,
        project_root: &Path,
        relative_path: &str,
    ) -> Result<IndexStats, IndexerError> {
        let full_path = project_root.join(relative_path);
        let bytes = tokio::fs::read(&full_path).await?;
        let code_graph = CodeGraphBuilder::new()
            .index_project(project_root)
            .map_err(|e| IndexerError::Chunk(e.to_string()))?;
        let normalized_relative_path = relative_path.replace('\\', "/");
        let chunks = self.chunk_file(
            full_path.to_string_lossy().as_ref(),
            &normalized_relative_path,
            &bytes,
            Some(&code_graph),
        )?;
        if chunks.is_empty() {
            self.remove_file(project_id, relative_path).await?;
            return Ok(IndexStats::default());
        }

        let collection = if chunks[0].kind == ChunkKind::Code {
            "code_chunks"
        } else {
            "doc_chunks"
        };

        self.remove_file(project_id, &normalized_relative_path)
            .await?;
        self.upsert_chunks(project_id, &normalized_relative_path, collection, chunks)
            .await?;

        Ok(IndexStats {
            files_indexed: 1,
            chunks_indexed: 1,
        })
    }

    /// Remove all chunks belonging to `relative_path` from both code and doc
    /// collections.
    pub async fn remove_file(
        &self,
        project_id: &str,
        relative_path: &str,
    ) -> Result<(), IndexerError> {
        let filter = serde_json::json!({
            "project_id": {"match": {"value": project_id}},
            "relative_path": {"match": {"value": relative_path}}
        });
        for collection in ["code_chunks", "doc_chunks"] {
            match self
                .vector_store
                .delete_by_filter(collection, filter.clone())
                .await
            {
                Ok(()) | Err(VectorStoreError::Collection(_)) => {}
                Err(error) => return Err(error.into()),
            }
        }
        Ok(())
    }

    fn chunk_file(
        &self,
        full_path: &str,
        relative_path: &str,
        bytes: &[u8],
        code_graph: Option<&CodeGraph>,
    ) -> Result<Vec<Chunk>, IndexerError> {
        let ext = Path::new(full_path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        let code_exts = [
            "rs", "py", "js", "jsx", "ts", "tsx", "go", "java", "c", "cpp", "cc", "h", "hpp",
        ];
        let doc_exts = ["md", "html", "htm"];

        if code_exts.contains(&ext) {
            let source = std::str::from_utf8(bytes).map_err(|error| {
                IndexerError::Chunk(format!("invalid UTF-8 code file: {error}"))
            })?;
            match code_graph {
                Some(graph) => chunk_code_with_graph(relative_path, source, graph)
                    .map_err(|e| IndexerError::Chunk(e.to_string())),
                None => chunk_code(relative_path, source)
                    .map_err(|e| IndexerError::Chunk(e.to_string())),
            }
        } else if doc_exts.contains(&ext) {
            let source = std::str::from_utf8(bytes)
                .map_err(|error| IndexerError::Chunk(format!("invalid UTF-8 document: {error}")))?;
            Ok(parse_doc(relative_path, source))
        } else if ext == "pdf" {
            parse_pdf_bytes(relative_path, bytes).map_err(|e| IndexerError::Chunk(e.to_string()))
        } else {
            Ok(Vec::new())
        }
    }

    async fn upsert_chunks(
        &self,
        project_id: &str,
        relative_path: &str,
        collection: &str,
        chunks: Vec<Chunk>,
    ) -> Result<(), IndexerError> {
        if chunks.is_empty() {
            return Ok(());
        }

        let dim = self.embedder.dimension().await?;

        // Always embed dense vectors first; they are the fallback if the store
        // does not support sparse vectors.
        let mut dense_points = Vec::with_capacity(chunks.len());
        for chunk in &chunks {
            let vector = self.embedder.embed(&chunk.text).await?;
            dense_points.push((chunk, vector));
        }

        if let Some(sparse_embedder) = &self.sparse_embedder {
            match self
                .vector_store
                .create_sparse_collection(collection, dim)
                .await
            {
                Ok(()) => {
                    let mut sparse_points = Vec::with_capacity(dense_points.len());
                    for (chunk, vector) in &dense_points {
                        let sparse_vector = sparse_embedder.embed_document(&chunk.text).await?;
                        sparse_points.push(SparseVectorPoint {
                            id: chunk.id.clone(),
                            vector: vector.clone(),
                            sparse_vector,
                            payload: Self::chunk_payload(project_id, relative_path, chunk),
                        });
                    }

                    match self
                        .vector_store
                        .upsert_with_sparse(collection, sparse_points)
                        .await
                    {
                        Ok(()) => return Ok(()),
                        Err(VectorStoreError::Unsupported(_)) => {
                            // Fall through to dense-only upsert.
                        }
                        Err(e) => return Err(e.into()),
                    }
                }
                Err(VectorStoreError::Unsupported(_)) => {
                    // Store does not support sparse vectors; fall through.
                }
                Err(e) => return Err(e.into()),
            }
        }

        self.vector_store.create_collection(collection, dim).await?;
        let points = dense_points
            .into_iter()
            .map(|(chunk, vector)| VectorPoint {
                id: chunk.id.clone(),
                vector,
                payload: Self::chunk_payload(project_id, relative_path, chunk),
            })
            .collect();
        self.vector_store.upsert(collection, points).await?;
        Ok(())
    }

    fn chunk_payload(project_id: &str, relative_path: &str, chunk: &Chunk) -> serde_json::Value {
        serde_json::json!({
            "project_id": project_id,
            "source": chunk.source,
            "relative_path": relative_path,
            "kind": chunk.kind_str(),
            "language": chunk.language,
            "summary": chunk.summary,
            "text": chunk.text,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "symbol_id": chunk.symbol_id.as_deref(),
            "related_symbols": chunk
                .related_symbols
                .iter()
                .map(|id| id.as_str())
                .collect::<Vec<_>>(),
        })
    }
}

/// Search already-indexed project chunks by sparse (e.g. BM25) similarity.
pub async fn search_sparse_chunks(
    vector_store: &dyn VectorStore,
    collection: &str,
    sparse_embedder: &dyn SparseEmbedder,
    project_id: &str,
    query: &str,
    limit: usize,
) -> Result<Vec<crate::services::SearchResult>, IndexerError> {
    let sparse = sparse_embedder.embed_query(query).await?;
    Ok(vector_store
        .search_sparse(
            collection,
            &sparse,
            SearchOptions {
                limit,
                filter: Some(serde_json::json!({"project_id": {"match": {"value": project_id}}})),
                score_threshold: None,
            },
        )
        .await?)
}

/// Statistics returned by an indexing run.
#[derive(Debug, Clone, Default)]
pub struct IndexStats {
    pub files_indexed: usize,
    pub chunks_indexed: usize,
}

/// Search already-indexed project chunks by semantic similarity.
pub async fn search_chunks(
    vector_store: &dyn VectorStore,
    collection: &str,
    embedder: &dyn Embedder,
    project_id: &str,
    query: &str,
    limit: usize,
) -> Result<Vec<crate::services::SearchResult>, IndexerError> {
    let vector = embedder.embed(query).await?;
    Ok(vector_store
        .search(
            collection,
            &vector,
            SearchOptions {
                limit,
                filter: Some(serde_json::json!({"project_id": {"match": {"value": project_id}}})),
                score_threshold: None,
            },
        )
        .await?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::{
        MockEmbedder, MockSparseEmbedder, SearchResult, SparseVector, SparseVectorPoint,
        VectorPoint, VectorStoreError,
    };
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    #[derive(Debug, Default)]
    struct TestVectorStore {
        collections: Mutex<HashMap<String, Vec<VectorPoint>>>,
        sparse_collections: Mutex<HashMap<String, Vec<SparseVectorPoint>>>,
    }

    fn payload_matches_project(payload: &serde_json::Value, expected: &str) -> bool {
        payload
            .get("project_id")
            .and_then(|v| v.as_str())
            .map(|v| v == expected)
            .unwrap_or(false)
    }

    fn payload_matches_filter(payload: &serde_json::Value, filter: &serde_json::Value) -> bool {
        let Some(obj) = filter.as_object() else {
            return true;
        };
        for (key, clause) in obj {
            if let Some(match_clause) = clause.get("match") {
                let expected = match_clause.get("value");
                if payload.get(key) != expected {
                    return false;
                }
            }
        }
        true
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

    #[async_trait::async_trait]
    impl VectorStore for TestVectorStore {
        async fn create_collection(
            &self,
            collection: &str,
            _dim: usize,
        ) -> Result<(), VectorStoreError> {
            self.collections
                .lock()
                .unwrap()
                .entry(collection.into())
                .or_default();
            Ok(())
        }
        async fn delete_collection(&self, collection: &str) -> Result<(), VectorStoreError> {
            self.collections.lock().unwrap().remove(collection);
            Ok(())
        }
        async fn upsert(
            &self,
            collection: &str,
            points: Vec<VectorPoint>,
        ) -> Result<(), VectorStoreError> {
            self.collections
                .lock()
                .unwrap()
                .entry(collection.into())
                .or_default()
                .extend(points);
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
                    project_filter
                        .is_none_or(|expected| payload_matches_project(&p.payload, expected))
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
            collection: &str,
            _dim: usize,
        ) -> Result<(), VectorStoreError> {
            self.sparse_collections
                .lock()
                .unwrap()
                .entry(collection.into())
                .or_default();
            Ok(())
        }

        async fn upsert_with_sparse(
            &self,
            collection: &str,
            points: Vec<SparseVectorPoint>,
        ) -> Result<(), VectorStoreError> {
            self.sparse_collections
                .lock()
                .unwrap()
                .entry(collection.into())
                .or_default()
                .extend(points);
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
                    project_filter
                        .is_none_or(|expected| payload_matches_project(&p.payload, expected))
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

        async fn delete_by_filter(
            &self,
            collection: &str,
            filter: serde_json::Value,
        ) -> Result<(), VectorStoreError> {
            let dense_exists = self.collections.lock().unwrap().contains_key(collection);
            let sparse_exists = self
                .sparse_collections
                .lock()
                .unwrap()
                .contains_key(collection);
            if !dense_exists && !sparse_exists {
                return Err(VectorStoreError::Collection(format!(
                    "collection {collection} does not exist"
                )));
            }

            {
                let mut cols = self.collections.lock().unwrap();
                if let Some(points) = cols.get_mut(collection) {
                    points.retain(|p| !payload_matches_filter(&p.payload, &filter));
                }
            }
            {
                let mut cols = self.sparse_collections.lock().unwrap();
                if let Some(points) = cols.get_mut(collection) {
                    points.retain(|p| !payload_matches_filter(&p.payload, &filter));
                }
            }
            Ok(())
        }
    }

    #[tokio::test]
    async fn indexer_respects_gitignore() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir(root.join(".git")).unwrap();
        std::fs::create_dir(root.join("src")).unwrap();
        std::fs::write(root.join("src/lib.rs"), "fn kept() {}\n").unwrap();
        std::fs::write(root.join(".gitignore"), "ignored/\n").unwrap();
        std::fs::create_dir(root.join("ignored")).unwrap();
        std::fs::write(root.join("ignored/secret.rs"), "fn ignored() {}\n").unwrap();

        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new(8));
        let vector_store: Arc<dyn VectorStore> = Arc::new(TestVectorStore::default());
        let indexer = ProjectIndexer::new(embedder, vector_store.clone());
        let stats = indexer.index("proj-1", root).await.unwrap();

        assert_eq!(stats.files_indexed, 1, "gitignored files should be skipped");
        let results = vector_store
            .search(
                "code_chunks",
                &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                SearchOptions {
                    limit: 10,
                    filter: Some(serde_json::json!({"project_id": {"match": {"value": "proj-1"}}})),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let texts: Vec<String> = results
            .iter()
            .map(|r| r.payload.get("text").unwrap().as_str().unwrap().to_string())
            .collect();
        assert!(texts.iter().any(|t| t.contains("fn kept")));
        assert!(!texts.iter().any(|t| t.contains("fn ignored")));
    }

    #[tokio::test]
    async fn indexer_chunks_rust_function_by_symbol() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir(root.join(".git")).unwrap();
        std::fs::write(
            root.join("main.rs"),
            "fn add(a: i32, b: i32) -> i32 { a + b }\n",
        )
        .unwrap();

        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new(8));
        let vector_store: Arc<dyn VectorStore> = Arc::new(TestVectorStore::default());
        let indexer = ProjectIndexer::new(embedder, vector_store.clone());
        indexer.index("proj-1", root).await.unwrap();

        let results = vector_store
            .search(
                "code_chunks",
                &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                SearchOptions {
                    limit: 10,
                    filter: Some(serde_json::json!({"project_id": {"match": {"value": "proj-1"}}})),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(!results.is_empty());
        let text = results[0].payload.get("text").unwrap().as_str().unwrap();
        assert!(text.contains("fn add"));
    }

    #[tokio::test]
    async fn indexer_indexes_mixed_code_markdown_and_pdf_with_overlap_and_ast_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir(root.join(".git")).unwrap();
        std::fs::create_dir(root.join("src")).unwrap();
        std::fs::create_dir(root.join("docs")).unwrap();
        std::fs::write(
            root.join("src/lib.rs"),
            "pub fn fetch_context() -> &'static str { \"RAG_CONTEXT\" }\n",
        )
        .unwrap();
        std::fs::write(
            root.join("docs/guide.md"),
            format!(
                "{}\n{}",
                "RAG_CONTEXT markdown ".repeat(90),
                "overlap marker ".repeat(90)
            ),
        )
        .unwrap();
        std::fs::write(
            root.join("docs/spec.pdf"),
            minimal_pdf_with_text(
                "RAG_CONTEXT pdf requirements mention rerank and LoRA evolution.",
            ),
        )
        .unwrap();

        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new(8));
        let vector_store: Arc<dyn VectorStore> = Arc::new(TestVectorStore::default());
        let indexer = ProjectIndexer::new(embedder, vector_store.clone());
        indexer.index("proj-mixed", root).await.unwrap();

        let code = vector_store
            .search(
                "code_chunks",
                &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                SearchOptions {
                    limit: 10,
                    filter: Some(
                        serde_json::json!({"project_id": {"match": {"value": "proj-mixed"}}}),
                    ),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(code.iter().any(|result| {
            result.payload["relative_path"] == "src/lib.rs"
                && result.payload["symbol_id"].as_str().is_some()
                && result.payload["language"] == "rust"
        }));

        let docs = vector_store
            .search(
                "doc_chunks",
                &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                SearchOptions {
                    limit: 20,
                    filter: Some(
                        serde_json::json!({"project_id": {"match": {"value": "proj-mixed"}}}),
                    ),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(docs.iter().any(|result| {
            result.payload["relative_path"] == "docs/spec.pdf"
                && result.payload["language"] == "pdf"
        }));
        let guide_chunks: Vec<_> = docs
            .iter()
            .filter(|result| result.payload["relative_path"] == "docs/guide.md")
            .collect();
        assert!(guide_chunks.len() > 1);
        assert!(guide_chunks.windows(2).any(|pair| {
            pair[1].payload["text"]
                .as_str()
                .unwrap_or_default()
                .contains("overlap marker")
        }));
    }

    fn minimal_pdf_with_text(text: &str) -> Vec<u8> {
        let escaped = text
            .replace('\\', "\\\\")
            .replace('(', "\\(")
            .replace(')', "\\)");
        let objects = [
            "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n".to_string(),
            "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n".to_string(),
            "3 0 obj\n<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >>\nendobj\n".to_string(),
            "4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n".to_string(),
            format!(
                "5 0 obj\n<< /Length {} >>\nstream\nBT /F1 12 Tf 72 720 Td ({}) Tj ET\nendstream\nendobj\n",
                33 + escaped.len(),
                escaped
            ),
        ];
        let mut pdf = String::from("%PDF-1.4\n");
        let mut offsets = vec![0usize];
        for object in objects {
            offsets.push(pdf.len());
            pdf.push_str(&object);
        }
        let xref_offset = pdf.len();
        pdf.push_str("xref\n0 6\n0000000000 65535 f \n");
        for offset in offsets.iter().skip(1) {
            pdf.push_str(&format!("{offset:010} 00000 n \n"));
        }
        pdf.push_str(&format!(
            "trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
        ));
        pdf.into_bytes()
    }

    #[tokio::test]
    async fn indexer_persists_graph_symbol_metadata_for_code_chunks() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir(root.join(".git")).unwrap();
        std::fs::write(
            root.join("lib.rs"),
            "fn helper() -> i32 { 1 }\nfn caller() -> i32 { helper() }\n",
        )
        .unwrap();

        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new(8));
        let vector_store: Arc<dyn VectorStore> = Arc::new(TestVectorStore::default());
        let indexer = ProjectIndexer::new(embedder, vector_store.clone());
        indexer.index("proj-graph", root).await.unwrap();

        let results = vector_store
            .search(
                "code_chunks",
                &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                SearchOptions {
                    limit: 10,
                    filter: Some(
                        serde_json::json!({"project_id": {"match": {"value": "proj-graph"}}}),
                    ),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        let helper = results
            .iter()
            .find(|result| {
                result
                    .payload
                    .get("text")
                    .and_then(|value| value.as_str())
                    .is_some_and(|text| text.contains("fn helper"))
            })
            .expect("helper chunk should be indexed");

        assert!(
            helper
                .payload
                .get("symbol_id")
                .and_then(|value| value.as_str())
                .is_some(),
            "code chunk payload should include graph symbol_id: {}",
            helper.payload
        );
        assert!(
            helper
                .payload
                .get("related_symbols")
                .and_then(|value| value.as_array())
                .is_some_and(|symbols| !symbols.is_empty()),
            "code chunk payload should include related_symbols from the code graph: {}",
            helper.payload
        );
    }

    #[tokio::test]
    async fn semantic_search_finds_function_by_description() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir(root.join(".git")).unwrap();
        std::fs::write(
            root.join("main.rs"),
            "fn handle_error() { println!(\"ok\"); }\n",
        )
        .unwrap();

        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new(8));
        let vector_store: Arc<dyn VectorStore> = Arc::new(TestVectorStore::default());
        let indexer = ProjectIndexer::new(embedder.clone(), vector_store.clone());
        indexer.index("proj-1", root).await.unwrap();

        let results = search_chunks(
            vector_store.as_ref(),
            "code_chunks",
            embedder.as_ref(),
            "proj-1",
            "handle_error",
            5,
        )
        .await
        .unwrap();
        assert!(!results.is_empty());
        let text = results[0].payload.get("text").unwrap().as_str().unwrap();
        assert!(text.contains("handle_error"));
    }

    #[tokio::test]
    async fn indexer_indexes_chunks_with_sparse_vectors() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir(root.join(".git")).unwrap();
        std::fs::write(
            root.join("main.rs"),
            "fn fetch_url() -> String { \"ok\".to_string() }\n",
        )
        .unwrap();

        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new(8));
        let sparse_embedder: Arc<dyn SparseEmbedder> = Arc::new(MockSparseEmbedder);
        let vector_store: Arc<dyn VectorStore> = Arc::new(TestVectorStore::default());
        let indexer = ProjectIndexer::new(embedder, vector_store.clone())
            .with_sparse_embedder(sparse_embedder.clone());
        indexer.index("proj-1", root).await.unwrap();

        assert!(vector_store.supports_sparse().await);
        let results = search_sparse_chunks(
            vector_store.as_ref(),
            "code_chunks",
            sparse_embedder.as_ref(),
            "proj-1",
            "fetch_url",
            5,
        )
        .await
        .unwrap();
        assert!(!results.is_empty());
        let text = results[0].payload.get("text").unwrap().as_str().unwrap();
        assert!(text.contains("fetch_url"));
    }

    #[tokio::test]
    async fn index_file_indexes_single_file() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("lib.rs"), "fn indexed_function() -> i32 { 42 }\n").unwrap();

        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new(16));
        let vector_store: Arc<dyn VectorStore> = Arc::new(TestVectorStore::default());
        let indexer = ProjectIndexer::new(embedder.clone(), vector_store.clone());
        let stats = indexer.index_file("proj-1", root, "lib.rs").await.unwrap();

        assert_eq!(stats.files_indexed, 1);
        let results = search_chunks(
            vector_store.as_ref(),
            "code_chunks",
            embedder.as_ref(),
            "proj-1",
            "indexed_function",
            5,
        )
        .await
        .unwrap();
        assert!(!results.is_empty());
        let text = results[0].payload.get("text").unwrap().as_str().unwrap();
        assert!(text.contains("indexed_function"));
    }

    #[tokio::test]
    async fn index_file_ignores_missing_collection_during_stale_cleanup() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(
            root.join("main.rs"),
            "fn first_file_after_empty_index() {}\n",
        )
        .unwrap();

        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new(16));
        let vector_store: Arc<dyn VectorStore> = Arc::new(TestVectorStore::default());
        let indexer = ProjectIndexer::new(embedder.clone(), vector_store.clone());

        indexer
            .index_file("proj-1", root, "main.rs")
            .await
            .expect("first single-file index should tolerate absent collections");

        let results = search_chunks(
            vector_store.as_ref(),
            "code_chunks",
            embedder.as_ref(),
            "proj-1",
            "first_file_after_empty_index",
            5,
        )
        .await
        .unwrap();

        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn remove_file_deletes_indexed_chunks() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("main.rs"), "fn keep() {}\n").unwrap();

        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new(16));
        let vector_store: Arc<dyn VectorStore> = Arc::new(TestVectorStore::default());
        let indexer = ProjectIndexer::new(embedder.clone(), vector_store.clone());
        indexer.index_file("proj-1", root, "main.rs").await.unwrap();
        indexer.remove_file("proj-1", "main.rs").await.unwrap();

        let results = search_chunks(
            vector_store.as_ref(),
            "code_chunks",
            embedder.as_ref(),
            "proj-1",
            "keep",
            5,
        )
        .await
        .unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn index_file_replaces_stale_chunks() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("main.rs"), "fn old_func() {}\n").unwrap();

        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new(16));
        let test_store = Arc::new(TestVectorStore::default());
        let vector_store: Arc<dyn VectorStore> = test_store.clone();
        let indexer = ProjectIndexer::new(embedder.clone(), vector_store.clone());
        indexer.index_file("proj-1", root, "main.rs").await.unwrap();

        std::fs::write(root.join("main.rs"), "fn new_func() {}\n").unwrap();
        indexer.index_file("proj-1", root, "main.rs").await.unwrap();

        let cols = test_store.collections.lock().unwrap();
        let code = cols
            .get("code_chunks")
            .expect("code_chunks collection missing");
        assert_eq!(code.len(), 1, "only the new chunk should remain");
        let text = code[0].payload.get("text").unwrap().as_str().unwrap();
        assert!(text.contains("new_func"));
        assert!(!text.contains("old_func"));
    }
}
