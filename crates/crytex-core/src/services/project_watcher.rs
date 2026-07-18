//! Incremental project indexing via file-system notifications.
//!
//! [`ProjectWatcher`] watches a project directory, debounces change events, and
//! applies per-file incremental updates through [`ProjectIndexer`]. After a
//! batch of changes is processed it publishes [`Event::ProjectContextUpdated`]
//! so that agents can refresh their context.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use notify_debouncer_mini::{DebounceEventResult, new_debouncer};
use thiserror::Error;
use tracing::{info, warn};
use ulid::Ulid;

use crate::bus::Event;
use crate::indexer::{IndexerError, ProjectIndexer};
use crate::services::EventService;

/// Errors that can occur while watching a project.
#[derive(Debug, Error)]
pub enum WatcherError {
    #[error("notify error: {0}")]
    Notify(#[from] notify::Error),
    #[error("watcher task failed: {0}")]
    Join(String),
    #[error("event channel closed")]
    ChannelClosed,
    #[error("indexer error: {0}")]
    Indexer(#[from] IndexerError),
}

/// Watches a project directory and incrementally updates its vector index.
pub struct ProjectWatcher {
    indexer: ProjectIndexer,
    event_service: Arc<dyn EventService>,
    debounce_ms: u64,
}

impl ProjectWatcher {
    /// Create a watcher for `indexer` that publishes lifecycle events through
    /// `event_service`.
    pub fn new(indexer: ProjectIndexer, event_service: Arc<dyn EventService>) -> Self {
        Self {
            indexer,
            event_service,
            debounce_ms: 500,
        }
    }

    /// Override the default 500 ms debounce window.
    pub fn with_debounce(mut self, debounce_ms: u64) -> Self {
        self.debounce_ms = debounce_ms;
        self
    }

    /// Start watching `root_path` for changes belonging to `project_id`.
    ///
    /// The watcher runs until `shutdown` fires or the event channel is closed.
    /// On shutdown the background file-system watcher is stopped and the
    /// method returns.
    pub async fn watch(
        self,
        project_id: String,
        root_path: PathBuf,
        shutdown: tokio::sync::oneshot::Receiver<()>,
    ) -> Result<(), WatcherError> {
        self.watch_inner(project_id, root_path, shutdown, None)
            .await
    }

    /// Start watching and signal once the filesystem backend is subscribed.
    pub async fn watch_with_ready(
        self,
        project_id: String,
        root_path: PathBuf,
        shutdown: tokio::sync::oneshot::Receiver<()>,
        ready: tokio::sync::oneshot::Sender<Result<(), String>>,
    ) -> Result<(), WatcherError> {
        self.watch_inner(project_id, root_path, shutdown, Some(ready))
            .await
    }

    async fn watch_inner(
        self,
        project_id: String,
        root_path: PathBuf,
        mut shutdown: tokio::sync::oneshot::Receiver<()>,
        ready: Option<tokio::sync::oneshot::Sender<Result<(), String>>>,
    ) -> Result<(), WatcherError> {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Vec<PathBuf>>();
        let debounce_ms = self.debounce_ms;
        let root = root_path.clone();
        let (blocking_shutdown_tx, blocking_shutdown_rx) = std::sync::mpsc::channel::<()>();

        let watcher_task = tokio::task::spawn_blocking(move || -> Result<(), WatcherError> {
            let (debounce_tx, debounce_rx) = std::sync::mpsc::channel::<DebounceEventResult>();
            let mut ready = ready;
            let mut debouncer = match new_debouncer(Duration::from_millis(debounce_ms), debounce_tx)
            {
                Ok(debouncer) => debouncer,
                Err(error) => {
                    if let Some(ready) = ready.take() {
                        let _ = ready.send(Err(error.to_string()));
                    }
                    return Err(error.into());
                }
            };
            if let Err(error) = debouncer
                .watcher()
                .watch(&root, notify::RecursiveMode::Recursive)
            {
                if let Some(ready) = ready.take() {
                    let _ = ready.send(Err(error.to_string()));
                }
                return Err(error.into());
            }
            if let Some(ready) = ready.take() {
                let _ = ready.send(Ok(()));
            }

            loop {
                match debounce_rx.recv_timeout(Duration::from_millis(100)) {
                    Ok(result) => match result {
                        Ok(events) => {
                            let paths: Vec<PathBuf> = events.into_iter().map(|e| e.path).collect();
                            let unique: Vec<PathBuf> = paths
                                .into_iter()
                                .collect::<HashSet<_>>()
                                .into_iter()
                                .collect();
                            if !unique.is_empty() && tx.send(unique).is_err() {
                                break;
                            }
                        }
                        Err(err) => {
                            warn!(%err, "file watcher error");
                        }
                    },
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                        if blocking_shutdown_rx.try_recv().is_ok() {
                            break;
                        }
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                }
            }
            Ok(())
        });

        loop {
            tokio::select! {
                Some(batch) = rx.recv() => {
                    for path in batch {
                        let Some(relative) = path
                            .strip_prefix(&root_path)
                            .ok()
                            .map(|p| p.to_string_lossy().to_string())
                        else {
                            continue;
                        };

                        if path.is_dir() {
                            continue;
                        }

                        let result = if path.exists() {
                            self.indexer
                                .index_file(&project_id, &root_path, &relative)
                                .await
                                .map(|_| ())
                        } else {
                            self.indexer.remove_file(&project_id, &relative).await
                        };

                        if let Err(err) = result {
                            warn!(%relative, %err, "incremental index update failed");
                        }
                    }

                    info!(%project_id, "project context incrementally updated");
                    self.event_service.publish(Event::ProjectContextUpdated {
                        project_id: project_id.clone(),
                        snapshot_id: Ulid::new().to_string(),
                    });
                }
                _ = &mut shutdown => {
                    let _ = blocking_shutdown_tx.send(());
                    break;
                }
            }
        }

        watcher_task
            .await
            .map_err(|err| WatcherError::Join(err.to_string()))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::EventBus;
    use crate::services::embedder::Embedder;
    use crate::services::{
        EventServiceImpl, MockEmbedder, SearchOptions, SearchResult, VectorPoint, VectorStore,
        VectorStoreError,
    };
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::time::Duration;

    #[derive(Debug, Default)]
    struct TestStore {
        collections: Mutex<HashMap<String, Vec<VectorPoint>>>,
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 {
            0.0
        } else {
            dot / (na * nb)
        }
    }

    fn payload_matches(payload: &serde_json::Value, filter: &serde_json::Value) -> bool {
        let Some(obj) = filter.as_object() else {
            return true;
        };
        for (key, clause) in obj {
            if let Some(match_clause) = clause.get("match")
                && payload.get(key) != match_clause.get("value")
            {
                return false;
            }
        }
        true
    }

    #[async_trait::async_trait]
    impl VectorStore for TestStore {
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
            let mut results: Vec<SearchResult> = points
                .iter()
                .filter(|p| {
                    options
                        .filter
                        .as_ref()
                        .is_none_or(|f| payload_matches(&p.payload, f))
                })
                .map(|p| SearchResult {
                    id: p.id.clone(),
                    score: cosine_similarity(vector, &p.vector),
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
            let mut cols = self.collections.lock().unwrap();
            if let Some(points) = cols.get_mut(collection) {
                points.retain(|p| !payload_matches(&p.payload, &filter));
            }
            Ok(())
        }
    }

    #[tokio::test]
    async fn watcher_indexes_changed_file_after_debounce() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();

        let embedder = Arc::new(MockEmbedder::new(16));
        let store: Arc<dyn VectorStore> = Arc::new(TestStore::default());
        let indexer = ProjectIndexer::new(embedder.clone(), store.clone());

        let event_bus = Arc::new(EventBus::new());
        let event_service: Arc<dyn EventService> = Arc::new(EventServiceImpl::new(event_bus));
        let watcher = ProjectWatcher::new(indexer, event_service).with_debounce(50);

        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        let handle = tokio::spawn(watcher.watch("proj-1".into(), root.clone(), shutdown_rx));
        tokio::time::sleep(Duration::from_millis(100)).await;

        tokio::fs::write(root.join("lib.rs"), "fn alpha() {}\n")
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(300)).await;

        let results = store
            .search(
                "code_chunks",
                &embedder.embed("alpha").await.unwrap(),
                SearchOptions {
                    limit: 10,
                    filter: Some(serde_json::json!({
                        "project_id": {"match": {"value": "proj-1"}}
                    })),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(!results.is_empty(), "watcher should index the changed file");

        shutdown_tx.send(()).unwrap();
        handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn watcher_removes_deleted_file_from_index() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        tokio::fs::write(root.join("main.rs"), "fn beta() {}\n")
            .await
            .unwrap();

        let embedder = Arc::new(MockEmbedder::new(16));
        let store: Arc<dyn VectorStore> = Arc::new(TestStore::default());
        let indexer = ProjectIndexer::new(embedder.clone(), store.clone());
        indexer
            .index_file("proj-1", &root, "main.rs")
            .await
            .unwrap();

        let event_service: Arc<dyn EventService> =
            Arc::new(EventServiceImpl::new(Arc::new(EventBus::new())));
        let watcher = ProjectWatcher::new(indexer, event_service).with_debounce(50);
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        let handle = tokio::spawn(watcher.watch("proj-1".into(), root.clone(), shutdown_rx));
        tokio::time::sleep(Duration::from_millis(100)).await;

        tokio::fs::remove_file(root.join("main.rs")).await.unwrap();
        tokio::time::sleep(Duration::from_millis(300)).await;

        let results = store
            .search(
                "code_chunks",
                &embedder.embed("beta").await.unwrap(),
                SearchOptions {
                    limit: 10,
                    filter: Some(serde_json::json!({
                        "project_id": {"match": {"value": "proj-1"}}
                    })),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(
            results.is_empty(),
            "watcher should remove deleted file chunks"
        );

        shutdown_tx.send(()).unwrap();
        handle.await.unwrap().unwrap();
    }

    struct FailingVectorStore;

    #[async_trait::async_trait]
    impl VectorStore for FailingVectorStore {
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
            Err(VectorStoreError::Upsert("simulated".into()))
        }
        async fn search(
            &self,
            _collection: &str,
            _vector: &[f32],
            _options: SearchOptions,
        ) -> Result<Vec<SearchResult>, VectorStoreError> {
            Ok(vec![])
        }
    }

    #[tokio::test]
    async fn watcher_survives_indexing_failure() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();

        let embedder = Arc::new(MockEmbedder::new(16));
        let store: Arc<dyn VectorStore> = Arc::new(FailingVectorStore);
        let indexer = ProjectIndexer::new(embedder, store);
        let event_service: Arc<dyn EventService> =
            Arc::new(EventServiceImpl::new(Arc::new(EventBus::new())));
        let watcher = ProjectWatcher::new(indexer, event_service).with_debounce(50);
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        let handle = tokio::spawn(watcher.watch("proj-1".into(), root.clone(), shutdown_rx));
        tokio::time::sleep(Duration::from_millis(100)).await;

        tokio::fs::write(root.join("bad.rs"), "fn x() {}\n")
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(300)).await;

        shutdown_tx.send(()).unwrap();
        handle.await.unwrap().unwrap();
    }
}
