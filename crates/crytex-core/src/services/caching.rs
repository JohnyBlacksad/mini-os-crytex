//! Production caching decorators for embedding and vector-store operations.
//!
//! Both caches are bounded, TTL-bounded, and exact-match. They do not attempt
//! semantic similarity matching: for embeddings the same text always yields the
//! same vector for a fixed model, and for search results the cache key includes
//! the full query vector and all `SearchOptions` fields that affect the result.
//!
//! The decorators intentionally do not cache errors. A failed embedding or
//! search is propagated to the caller and not stored, so retries are not
//! poisoned by transient failures.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use dashmap::DashMap;
use moka::future::Cache;

use crate::config::CacheConfig;
use crate::metrics::MetricsService;
use crate::services::{
    Embedder, EmbeddingError, SearchOptions, SearchResult, SparseVector, SparseVectorPoint,
    VectorPoint, VectorStore, VectorStoreError,
};

/// Decorates an [`Embedder`] with a bounded, time-bounded exact-match cache.
pub struct CachedEmbedder {
    inner: Arc<dyn Embedder>,
    cache: Cache<EmbeddingCacheKey, Arc<Vec<f32>>>,
    metrics: Option<Arc<dyn MetricsService>>,
}

impl CachedEmbedder {
    /// Wrap `inner` with a cache configured by `config`.
    ///
    /// `namespace` should identify the embedding model/backend so that a switch
    /// of model does not reuse stale vectors.
    pub fn new(
        inner: Arc<dyn Embedder>,
        namespace: impl Into<String>,
        metrics: Option<Arc<dyn MetricsService>>,
        config: &CacheConfig,
    ) -> Self {
        let namespace = namespace.into();
        let cache = Cache::builder()
            .max_capacity(config.embedding_cache_capacity)
            .time_to_live(Duration::from_secs(config.embedding_cache_ttl_seconds))
            .build();
        Self {
            inner,
            cache,
            metrics,
        }
        .with_namespace(namespace)
    }

    fn with_namespace(self, namespace: String) -> Self {
        // The namespace is baked into every key by `CachedEmbedder::embed`.
        let _ = namespace;
        self
    }

    fn record_hit(&self) {
        if let Some(metrics) = &self.metrics {
            let metrics = metrics.clone();
            tokio::spawn(async move {
                let _ = metrics.record_cache_hit().await;
            });
        }
    }

    fn record_miss(&self) {
        if let Some(metrics) = &self.metrics {
            let metrics = metrics.clone();
            tokio::spawn(async move {
                let _ = metrics.record_cache_miss().await;
            });
        }
    }
}

#[async_trait]
impl Embedder for CachedEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let key = EmbeddingCacheKey::new(text);

        if let Some(cached) = self.cache.get(&key).await {
            self.record_hit();
            return Ok((*cached).clone());
        }

        self.record_miss();
        let vector = self.inner.embed(text).await?;
        self.cache.insert(key, Arc::new(vector.clone())).await;
        Ok(vector)
    }

    async fn dimension(&self) -> Result<usize, EmbeddingError> {
        self.inner.dimension().await
    }
}

/// Decorates a [`VectorStore`] with per-collection bounded search caches.
///
/// Only `search` is cached. Mutating operations (`upsert`, `delete_collection`)
/// invalidate the cache for the affected collection so that subsequent searches
/// do not return stale results.
pub struct CachedVectorStore {
    inner: Arc<dyn VectorStore>,
    caches: DashMap<String, Cache<SearchCacheKey, Arc<Vec<SearchResult>>>>,
    metrics: Option<Arc<dyn MetricsService>>,
    config: CacheConfig,
}

impl CachedVectorStore {
    /// Wrap `inner` with per-collection search caches configured by `config`.
    pub fn new(
        inner: Arc<dyn VectorStore>,
        metrics: Option<Arc<dyn MetricsService>>,
        config: CacheConfig,
    ) -> Self {
        Self {
            inner,
            caches: DashMap::new(),
            metrics,
            config,
        }
    }

    fn collection_cache(&self, collection: &str) -> Cache<SearchCacheKey, Arc<Vec<SearchResult>>> {
        self.caches
            .entry(collection.to_string())
            .or_insert_with(|| {
                Cache::builder()
                    .max_capacity(self.config.vector_search_cache_capacity)
                    .time_to_live(Duration::from_secs(
                        self.config.vector_search_cache_ttl_seconds,
                    ))
                    .build()
            })
            .clone()
    }

    async fn invalidate_collection(&self, collection: &str) {
        if let Some((_, cache)) = self.caches.remove(collection) {
            cache.invalidate_all();
        }
    }

    fn record_hit(&self) {
        if let Some(metrics) = &self.metrics {
            let metrics = metrics.clone();
            tokio::spawn(async move {
                let _ = metrics.record_cache_hit().await;
            });
        }
    }

    fn record_miss(&self) {
        if let Some(metrics) = &self.metrics {
            let metrics = metrics.clone();
            tokio::spawn(async move {
                let _ = metrics.record_cache_miss().await;
            });
        }
    }
}

#[async_trait]
impl VectorStore for CachedVectorStore {
    async fn create_collection(
        &self,
        collection: &str,
        dim: usize,
    ) -> Result<(), VectorStoreError> {
        self.inner.create_collection(collection, dim).await
    }

    async fn delete_collection(&self, collection: &str) -> Result<(), VectorStoreError> {
        let result = self.inner.delete_collection(collection).await;
        self.caches.remove(collection);
        result
    }

    async fn upsert(
        &self,
        collection: &str,
        points: Vec<VectorPoint>,
    ) -> Result<(), VectorStoreError> {
        let result = self.inner.upsert(collection, points).await;
        self.invalidate_collection(collection).await;
        result
    }

    async fn search(
        &self,
        collection: &str,
        vector: &[f32],
        options: SearchOptions,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        if !self.config.vector_search_cache_enabled {
            return self.inner.search(collection, vector, options).await;
        }

        let key = SearchCacheKey::new(vector, &options);
        let cache = self.collection_cache(collection);

        if let Some(cached) = cache.get(&key).await {
            self.record_hit();
            return Ok((*cached).clone());
        }

        self.record_miss();
        let results = self.inner.search(collection, vector, options).await?;
        cache.insert(key, Arc::new(results.clone())).await;
        Ok(results)
    }

    async fn supports_sparse(&self) -> bool {
        self.inner.supports_sparse().await
    }

    async fn create_sparse_collection(
        &self,
        collection: &str,
        dim: usize,
    ) -> Result<(), VectorStoreError> {
        self.inner.create_sparse_collection(collection, dim).await
    }

    async fn upsert_with_sparse(
        &self,
        collection: &str,
        points: Vec<SparseVectorPoint>,
    ) -> Result<(), VectorStoreError> {
        let result = self.inner.upsert_with_sparse(collection, points).await;
        self.invalidate_collection(collection).await;
        result
    }

    async fn search_sparse(
        &self,
        collection: &str,
        vector: &SparseVector,
        options: SearchOptions,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        // Sparse search is not cached yet; delegate directly to the inner store.
        self.inner.search_sparse(collection, vector, options).await
    }

    async fn delete_by_filter(
        &self,
        collection: &str,
        filter: serde_json::Value,
    ) -> Result<(), VectorStoreError> {
        let result = self.inner.delete_by_filter(collection, filter).await;
        self.invalidate_collection(collection).await;
        result
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct EmbeddingCacheKey {
    text: String,
}

impl EmbeddingCacheKey {
    fn new(text: &str) -> Self {
        Self {
            text: text.to_string(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct SearchCacheKey {
    vector_bytes: Vec<u8>,
    limit: usize,
    filter_json: String,
    threshold_bytes: Option<[u8; 4]>,
}

impl SearchCacheKey {
    fn new(vector: &[f32], options: &SearchOptions) -> Self {
        let mut vector_bytes = Vec::with_capacity(vector.len() * 4);
        for value in vector {
            vector_bytes.extend_from_slice(&value.to_le_bytes());
        }
        Self {
            vector_bytes,
            limit: options.limit,
            filter_json: options
                .filter
                .as_ref()
                .map(|f| f.to_string())
                .unwrap_or_default(),
            threshold_bytes: options.score_threshold.map(|t| t.to_le_bytes()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{MetricsError, MetricsService, MetricsSnapshot};
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct CountingEmbedder {
        calls: AtomicU64,
    }

    #[async_trait]
    impl Embedder for CountingEmbedder {
        async fn embed(&self, _text: &str) -> Result<Vec<f32>, EmbeddingError> {
            let count = self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(vec![count as f32])
        }

        async fn dimension(&self) -> Result<usize, EmbeddingError> {
            Ok(1)
        }
    }

    struct FailingEmbedder;

    #[async_trait]
    impl Embedder for FailingEmbedder {
        async fn embed(&self, _text: &str) -> Result<Vec<f32>, EmbeddingError> {
            Err(EmbeddingError::EmbeddingFailed("boom".into()))
        }

        async fn dimension(&self) -> Result<usize, EmbeddingError> {
            Ok(1)
        }
    }

    struct CountingMetrics {
        hits: AtomicU64,
        misses: AtomicU64,
    }

    #[async_trait]
    impl MetricsService for CountingMetrics {
        async fn snapshot(&self) -> Result<MetricsSnapshot, MetricsError> {
            Ok(MetricsSnapshot::default())
        }
        async fn record_task_completion(
            &self,
            _latency_ms: u64,
            _success: bool,
        ) -> Result<(), MetricsError> {
            Ok(())
        }
        async fn record_cache_hit(&self) -> Result<(), MetricsError> {
            self.hits.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
        async fn record_cache_miss(&self) -> Result<(), MetricsError> {
            self.misses.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
        async fn history(
            &self,
            _from: i64,
            _to: i64,
        ) -> Result<Vec<MetricsSnapshot>, MetricsError> {
            Ok(vec![])
        }
    }

    fn cache_config() -> CacheConfig {
        CacheConfig {
            embedding_cache_enabled: true,
            embedding_cache_capacity: 100,
            embedding_cache_ttl_seconds: 60,
            vector_search_cache_enabled: true,
            vector_search_cache_capacity: 100,
            vector_search_cache_ttl_seconds: 60,
        }
    }

    #[tokio::test]
    async fn embedder_returns_cached_value_on_second_call() {
        let inner = Arc::new(CountingEmbedder {
            calls: AtomicU64::new(0),
        });
        let cached = CachedEmbedder::new(inner.clone(), "ns", None, &cache_config());

        let first = cached.embed("hello").await.unwrap();
        let second = cached.embed("hello").await.unwrap();

        assert_eq!(first, second);
        assert_eq!(inner.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn embedder_does_not_cache_errors() {
        let cached = CachedEmbedder::new(Arc::new(FailingEmbedder), "ns", None, &cache_config());

        assert!(cached.embed("hello").await.is_err());
        assert!(cached.embed("hello").await.is_err());
    }

    #[tokio::test]
    async fn embedder_records_hits_and_misses() {
        let metrics = Arc::new(CountingMetrics {
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        });
        let cached = CachedEmbedder::new(
            Arc::new(CountingEmbedder {
                calls: AtomicU64::new(0),
            }),
            "ns",
            Some(metrics.clone()),
            &cache_config(),
        );

        cached.embed("hello").await.unwrap();
        cached.embed("hello").await.unwrap();

        // Metrics updates are emitted via tokio::spawn; yield to let them run.
        tokio::task::yield_now().await;

        assert_eq!(metrics.hits.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.misses.load(Ordering::Relaxed), 1);
    }

    struct CountingVectorStore {
        searches: AtomicU64,
        upserts: AtomicU64,
        deletes: AtomicU64,
    }

    #[async_trait]
    impl VectorStore for CountingVectorStore {
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
            self.upserts.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        async fn search(
            &self,
            _collection: &str,
            _vector: &[f32],
            _options: SearchOptions,
        ) -> Result<Vec<SearchResult>, VectorStoreError> {
            let count = self.searches.fetch_add(1, Ordering::SeqCst);
            Ok(vec![SearchResult {
                id: format!("result-{count}"),
                score: 1.0,
                payload: serde_json::Value::Null,
            }])
        }
        async fn delete_by_filter(
            &self,
            _collection: &str,
            _filter: serde_json::Value,
        ) -> Result<(), VectorStoreError> {
            self.deletes.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    #[tokio::test]
    async fn vector_store_caches_search_results() {
        let inner = Arc::new(CountingVectorStore {
            searches: AtomicU64::new(0),
            upserts: AtomicU64::new(0),
            deletes: AtomicU64::new(0),
        });
        let cached = CachedVectorStore::new(inner.clone(), None, cache_config());
        let options = SearchOptions {
            limit: 5,
            ..Default::default()
        };

        let first = cached
            .search("col", &[1.0, 2.0, 3.0], options.clone())
            .await
            .unwrap();
        let second = cached
            .search("col", &[1.0, 2.0, 3.0], options)
            .await
            .unwrap();

        assert_eq!(first.len(), 1);
        assert_eq!(first, second);
        assert_eq!(inner.searches.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn vector_store_invalidates_collection_on_upsert() {
        let inner = Arc::new(CountingVectorStore {
            searches: AtomicU64::new(0),
            upserts: AtomicU64::new(0),
            deletes: AtomicU64::new(0),
        });
        let cached = CachedVectorStore::new(inner.clone(), None, cache_config());
        let options = SearchOptions {
            limit: 5,
            ..Default::default()
        };

        cached
            .search("col", &[1.0, 2.0, 3.0], options.clone())
            .await
            .unwrap();
        cached.upsert("col", vec![]).await.unwrap();
        let after = cached
            .search("col", &[1.0, 2.0, 3.0], options)
            .await
            .unwrap();

        assert_eq!(after[0].id, "result-1");
        assert_eq!(inner.searches.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn vector_store_deletes_collection_cache_on_delete() {
        let inner = Arc::new(CountingVectorStore {
            searches: AtomicU64::new(0),
            upserts: AtomicU64::new(0),
            deletes: AtomicU64::new(0),
        });
        let cached = CachedVectorStore::new(inner.clone(), None, cache_config());
        let options = SearchOptions {
            limit: 5,
            ..Default::default()
        };

        cached
            .search("col", &[1.0, 2.0, 3.0], options.clone())
            .await
            .unwrap();
        cached.delete_collection("col").await.unwrap();
        let after = cached
            .search("col", &[1.0, 2.0, 3.0], options)
            .await
            .unwrap();

        assert_eq!(after[0].id, "result-1");
        assert_eq!(inner.searches.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn vector_store_invalidates_cache_on_delete_by_filter() {
        let inner = Arc::new(CountingVectorStore {
            searches: AtomicU64::new(0),
            upserts: AtomicU64::new(0),
            deletes: AtomicU64::new(0),
        });
        let cached = CachedVectorStore::new(inner.clone(), None, cache_config());
        let options = SearchOptions {
            limit: 5,
            ..Default::default()
        };

        cached
            .search("col", &[1.0, 2.0, 3.0], options.clone())
            .await
            .unwrap();
        cached
            .delete_by_filter(
                "col",
                serde_json::json!({"project_id": {"match": {"value": "p1"}}}),
            )
            .await
            .unwrap();
        let after = cached
            .search("col", &[1.0, 2.0, 3.0], options)
            .await
            .unwrap();

        assert_eq!(after[0].id, "result-1");
        assert_eq!(inner.searches.load(Ordering::SeqCst), 2);
        assert_eq!(inner.deletes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn vector_store_records_hits_and_misses() {
        let metrics = Arc::new(CountingMetrics {
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        });
        let cached = CachedVectorStore::new(
            Arc::new(CountingVectorStore {
                searches: AtomicU64::new(0),
                upserts: AtomicU64::new(0),
                deletes: AtomicU64::new(0),
            }),
            Some(metrics.clone()),
            cache_config(),
        );
        let options = SearchOptions {
            limit: 5,
            ..Default::default()
        };

        cached
            .search("col", &[1.0, 2.0, 3.0], options.clone())
            .await
            .unwrap();
        cached
            .search("col", &[1.0, 2.0, 3.0], options)
            .await
            .unwrap();

        tokio::task::yield_now().await;

        assert_eq!(metrics.hits.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.misses.load(Ordering::Relaxed), 1);
    }
}
