//! Session memory bank for long-term project and task context.
//!
//! `MemoryBankService` stores structured facts (`MemoryEntry`) in SQLite and,
//! when an embedder and vector store are configured, indexes them for semantic
//! retrieval.  It is the backing store for mental models and contextual recall.

use std::sync::Arc;

use async_trait::async_trait;
use thiserror::Error;

use crate::models::MemoryEntry;
use crate::persistence::{MemoryEntryRepository, PersistenceError};
use crate::services::embedder::{Embedder, EmbeddingError};
use crate::services::vector_store::{SearchOptions, VectorPoint, VectorStore, VectorStoreError};

const MEMORY_COLLECTION: &str = "memory_bank";

/// Errors returned by [`MemoryBankService`].
#[derive(Debug, Error)]
pub enum MemoryBankError {
    #[error("persistence error: {0}")]
    Persistence(#[from] PersistenceError),
    #[error("embedding error: {0}")]
    Embedding(#[from] EmbeddingError),
    #[error("vector store error: {0}")]
    VectorStore(#[from] VectorStoreError),
    #[error("serialization error: {0}")]
    Serialization(String),
}

/// Stores and retrieves memory entries for projects and sessions.
#[async_trait]
pub trait MemoryBankService: Send + Sync {
    /// Store a memory entry.
    async fn remember(&self, entry: &MemoryEntry) -> Result<(), MemoryBankError>;

    /// List recent memory entries, optionally filtered by project and kind.
    async fn recall(
        &self,
        project_id: Option<&str>,
        kind: Option<&str>,
        limit: usize,
    ) -> Result<Vec<MemoryEntry>, MemoryBankError>;

    /// Search memory semantically for entries relevant to `query`.
    async fn recall_semantic(
        &self,
        project_id: Option<&str>,
        query: &str,
        limit: usize,
    ) -> Result<Vec<MemoryEntry>, MemoryBankError>;

    /// Return a simple concatenated summary of all entries in a session.
    async fn summarize_session(&self, session_id: &str) -> Result<Option<String>, MemoryBankError>;

    /// Build a lightweight mental model for a project from its stored memory entries.
    async fn mental_model_for_project(
        &self,
        project_id: &str,
    ) -> Result<serde_json::Value, MemoryBankError>;
}

/// Default implementation backed by a [`MemoryEntryRepository`] and an optional
/// vector index.
pub struct MemoryBankServiceImpl {
    repository: Arc<dyn MemoryEntryRepository>,
    embedder: Option<Arc<dyn Embedder>>,
    vector_store: Option<Arc<dyn VectorStore>>,
}

impl MemoryBankServiceImpl {
    /// Create a memory bank that only supports structured recall.
    pub fn new(repository: Arc<dyn MemoryEntryRepository>) -> Self {
        Self {
            repository,
            embedder: None,
            vector_store: None,
        }
    }

    /// Enable semantic search by pairing an embedder with a vector store.
    pub fn with_semantic_index(
        mut self,
        embedder: Arc<dyn Embedder>,
        vector_store: Arc<dyn VectorStore>,
    ) -> Self {
        self.embedder = Some(embedder);
        self.vector_store = Some(vector_store);
        self
    }

    async fn ensure_collection(&self, dim: usize) -> Result<(), VectorStoreError> {
        if let Some(store) = &self.vector_store {
            store.create_collection(MEMORY_COLLECTION, dim).await?;
        }
        Ok(())
    }

    async fn index_entry(&self, entry: &MemoryEntry) -> Result<(), MemoryBankError> {
        let (embedder, store) = match (&self.embedder, &self.vector_store) {
            (Some(e), Some(s)) => (e, s),
            _ => return Ok(()),
        };

        let dim = embedder.dimension().await?;
        let vector = embedder.embed(&entry.text).await?;
        self.ensure_collection(dim).await?;

        let payload = serde_json::json!({
            "id": entry.id,
            "project_id": entry.project_id,
            "session_id": entry.session_id,
            "kind": entry.kind,
            "text": entry.text,
        });
        store
            .upsert(
                MEMORY_COLLECTION,
                vec![VectorPoint {
                    id: entry.id.clone(),
                    vector,
                    payload,
                }],
            )
            .await?;
        Ok(())
    }

    fn project_filter(project_id: Option<&str>) -> Option<serde_json::Value> {
        project_id.map(|pid| {
            serde_json::json!({
                "must": [
                    {"key": "project_id", "match": {"value": pid}}
                ]
            })
        })
    }
}

#[async_trait]
impl MemoryBankService for MemoryBankServiceImpl {
    async fn remember(&self, entry: &MemoryEntry) -> Result<(), MemoryBankError> {
        self.repository.insert_memory_entry(entry).await?;
        self.index_entry(entry).await?;
        Ok(())
    }

    async fn recall(
        &self,
        project_id: Option<&str>,
        kind: Option<&str>,
        limit: usize,
    ) -> Result<Vec<MemoryEntry>, MemoryBankError> {
        Ok(self
            .repository
            .list_memory_entries(project_id, kind, limit)
            .await?)
    }

    async fn recall_semantic(
        &self,
        project_id: Option<&str>,
        query: &str,
        limit: usize,
    ) -> Result<Vec<MemoryEntry>, MemoryBankError> {
        let (embedder, store) = match (&self.embedder, &self.vector_store) {
            (Some(e), Some(s)) => (e, s),
            _ => return self.recall(project_id, None, limit).await,
        };

        let dim = embedder.dimension().await?;
        let vector = embedder.embed(query).await?;
        self.ensure_collection(dim).await?;

        let results = store
            .search(
                MEMORY_COLLECTION,
                &vector,
                SearchOptions {
                    limit,
                    filter: Self::project_filter(project_id),
                    ..SearchOptions::default()
                },
            )
            .await?;

        let ids: Vec<String> = results.iter().map(|r| r.id.clone()).collect();
        let all = self
            .repository
            .list_memory_entries(None, None, usize::MAX)
            .await?;
        let by_id: std::collections::HashMap<_, _> =
            all.into_iter().map(|e| (e.id.clone(), e)).collect();

        let mut entries = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(entry) = by_id.get(&id).cloned() {
                entries.push(entry);
            }
        }
        Ok(entries)
    }

    async fn summarize_session(&self, session_id: &str) -> Result<Option<String>, MemoryBankError> {
        let entries = self
            .repository
            .list_memory_entries_by_session(session_id)
            .await?;
        if entries.is_empty() {
            return Ok(None);
        }
        let lines: Vec<String> = entries
            .into_iter()
            .map(|e| format!("[{}] {}", e.kind, e.text))
            .collect();
        Ok(Some(lines.join("\n")))
    }

    async fn mental_model_for_project(
        &self,
        project_id: &str,
    ) -> Result<serde_json::Value, MemoryBankError> {
        let entries = self
            .repository
            .list_memory_entries(Some(project_id), None, 10_000)
            .await?;
        let mut kinds = serde_json::Map::new();
        for entry in entries {
            let key = entry.kind.clone();
            let array = kinds
                .entry(key)
                .or_insert_with(|| serde_json::json!([]))
                .as_array_mut()
                .ok_or_else(|| {
                    MemoryBankError::Serialization(
                        "expected JSON array for memory entry kind".to_string(),
                    )
                })?;
            array.push(serde_json::json!({
                "id": entry.id,
                "text": entry.text,
                "session_id": entry.session_id,
                "created_at": entry.created_at,
                "metadata": entry.metadata,
            }));
        }
        Ok(serde_json::json!({
            "project_id": project_id,
            "kinds": kinds,
        }))
    }
}

/// Convenience builder for [`MemoryEntry`].
#[derive(Debug, Default)]
pub struct MemoryEntryBuilder {
    id: Option<String>,
    project_id: Option<String>,
    session_id: Option<String>,
    kind: Option<String>,
    text: Option<String>,
    metadata: Option<serde_json::Value>,
    created_at: Option<i64>,
}

impl MemoryEntryBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn project_id(mut self, project_id: impl Into<String>) -> Self {
        self.project_id = Some(project_id.into());
        self
    }

    pub fn session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    pub fn kind(mut self, kind: impl Into<String>) -> Self {
        self.kind = Some(kind.into());
        self
    }

    pub fn text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }

    pub fn metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    pub fn created_at(mut self, created_at: i64) -> Self {
        self.created_at = Some(created_at);
        self
    }

    pub fn build(self) -> MemoryEntry {
        MemoryEntry {
            id: self.id.unwrap_or_else(|| ulid::Ulid::new().to_string()),
            project_id: self.project_id,
            session_id: self.session_id,
            kind: self.kind.unwrap_or_else(|| "note".to_string()),
            text: self.text.unwrap_or_default(),
            metadata: self.metadata.unwrap_or_default(),
            created_at: self
                .created_at
                .unwrap_or_else(|| chrono::Utc::now().timestamp_millis()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::MemoryEntryRepository;
    use crate::services::embedder::MockEmbedder;
    use crate::services::vector_store::{SearchOptions, VectorPoint, VectorStore};
    use std::collections::HashMap;
    use std::sync::Mutex;

    #[derive(Default)]
    struct MemoryMemoryEntryRepository {
        entries: Mutex<HashMap<String, MemoryEntry>>,
    }

    #[async_trait]
    impl MemoryEntryRepository for MemoryMemoryEntryRepository {
        async fn insert_memory_entry(&self, entry: &MemoryEntry) -> Result<(), PersistenceError> {
            self.entries
                .lock()
                .unwrap()
                .insert(entry.id.clone(), entry.clone());
            Ok(())
        }

        async fn list_memory_entries(
            &self,
            project_id: Option<&str>,
            kind: Option<&str>,
            _limit: usize,
        ) -> Result<Vec<MemoryEntry>, PersistenceError> {
            let entries = self.entries.lock().unwrap();
            let mut out: Vec<_> = entries
                .values()
                .filter(|e| project_id.is_none_or(|pid| e.project_id.as_deref() == Some(pid)))
                .filter(|e| kind.is_none_or(|k| e.kind == k))
                .cloned()
                .collect();
            out.sort_by_key(|b| std::cmp::Reverse(b.created_at));
            Ok(out)
        }

        async fn list_memory_entries_by_session(
            &self,
            session_id: &str,
        ) -> Result<Vec<MemoryEntry>, PersistenceError> {
            let entries = self.entries.lock().unwrap();
            let mut out: Vec<_> = entries
                .values()
                .filter(|e| e.session_id.as_deref() == Some(session_id))
                .cloned()
                .collect();
            out.sort_by_key(|a| a.created_at);
            Ok(out)
        }
    }

    #[derive(Default)]
    struct MemoryVectorStore {
        collections: Mutex<HashMap<String, Vec<VectorPoint>>>,
    }

    #[async_trait]
    impl VectorStore for MemoryVectorStore {
        async fn create_collection(
            &self,
            _collection: &str,
            _dim: usize,
        ) -> Result<(), VectorStoreError> {
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
            let mut collections = self.collections.lock().unwrap();
            let vec = collections.entry(collection.to_string()).or_default();
            for point in points {
                if let Some(existing) = vec.iter_mut().find(|p| p.id == point.id) {
                    *existing = point;
                } else {
                    vec.push(point);
                }
            }
            Ok(())
        }

        async fn search(
            &self,
            collection: &str,
            vector: &[f32],
            options: SearchOptions,
        ) -> Result<Vec<crate::services::vector_store::SearchResult>, VectorStoreError> {
            let collections = self.collections.lock().unwrap();
            let points = collections.get(collection).cloned().unwrap_or_default();
            let mut scored: Vec<_> = points
                .iter()
                .map(|p| {
                    let score = cosine_similarity(vector, &p.vector);
                    (p.clone(), score)
                })
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let results: Vec<_> = scored
                .into_iter()
                .take(options.limit)
                .map(|(p, score)| crate::services::vector_store::SearchResult {
                    id: p.id,
                    score,
                    payload: p.payload,
                })
                .collect();
            Ok(results)
        }
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let mut dot = 0.0f32;
        let mut na = 0.0f32;
        let mut nb = 0.0f32;
        for i in 0..len {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        dot / (na.sqrt() * nb.sqrt() + 1e-10)
    }

    #[tokio::test]
    async fn remember_and_recall_structured_entries() {
        let repo = Arc::new(MemoryMemoryEntryRepository::default());
        let service = MemoryBankServiceImpl::new(repo);

        let entry = MemoryEntryBuilder::new()
            .id("m1")
            .project_id("p1")
            .kind("goal")
            .text("Use tokio channels")
            .build();
        service.remember(&entry).await.unwrap();

        let recalled = service.recall(Some("p1"), Some("goal"), 10).await.unwrap();
        assert_eq!(recalled.len(), 1);
        assert_eq!(recalled[0].text, "Use tokio channels");
    }

    #[tokio::test]
    async fn semantic_recall_finds_closest_memory() {
        let repo = Arc::new(MemoryMemoryEntryRepository::default());
        let store = Arc::new(MemoryVectorStore::default());
        let embedder = Arc::new(MockEmbedder::new(8));
        let service = MemoryBankServiceImpl::new(repo).with_semantic_index(embedder, store);

        let e1 = MemoryEntryBuilder::new().id("m1").text("aaaaaaaa").build();
        let e2 = MemoryEntryBuilder::new().id("m2").text("zzzzzzzz").build();
        service.remember(&e1).await.unwrap();
        service.remember(&e2).await.unwrap();

        let results = service.recall_semantic(None, "aaaa", 1).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "m1");
    }

    #[tokio::test]
    async fn mental_model_groups_entries_by_kind() {
        let repo = Arc::new(MemoryMemoryEntryRepository::default());
        let service = MemoryBankServiceImpl::new(repo);

        service
            .remember(
                &MemoryEntryBuilder::new()
                    .project_id("p1")
                    .kind("goal")
                    .text("Use async traits")
                    .build(),
            )
            .await
            .unwrap();
        service
            .remember(
                &MemoryEntryBuilder::new()
                    .project_id("p1")
                    .kind("decision")
                    .text("Pick SQLite")
                    .build(),
            )
            .await
            .unwrap();

        let model = service.mental_model_for_project("p1").await.unwrap();
        assert_eq!(model["project_id"], "p1");
        assert_eq!(model["kinds"]["goal"].as_array().unwrap().len(), 1);
        assert_eq!(model["kinds"]["decision"].as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn summarize_session_joins_entries() {
        let repo = Arc::new(MemoryMemoryEntryRepository::default());
        let service = MemoryBankServiceImpl::new(repo);

        let e1 = MemoryEntryBuilder::new()
            .session_id("s1")
            .kind("note")
            .text("first")
            .created_at(1)
            .build();
        let e2 = MemoryEntryBuilder::new()
            .session_id("s1")
            .kind("decision")
            .text("second")
            .created_at(2)
            .build();
        service.remember(&e1).await.unwrap();
        service.remember(&e2).await.unwrap();

        let summary = service.summarize_session("s1").await.unwrap();
        assert!(summary.is_some());
        let text = summary.unwrap();
        assert!(text.contains("[note] first"));
        assert!(text.contains("[decision] second"));
    }
}
