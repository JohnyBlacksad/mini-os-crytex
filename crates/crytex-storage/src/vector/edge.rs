//! Embedded vector-store implementation backed by Qdrant Edge.
//!
//! Each logical collection is stored as a separate on-disk shard under the
//! configured base directory. Qdrant Edge runs in-process and synchronously,
//! so all blocking shard operations are executed via
//! [`tokio::task::spawn_blocking`].

use std::collections::HashMap;
use std::convert::TryFrom;
use std::path::PathBuf;
use std::sync::Arc;

use crytex_core::services::vector_store::{
    SearchOptions, SearchResult, SparseVector, SparseVectorPoint, VectorPoint, VectorStore,
    VectorStoreError,
};
use qdrant_edge::external::ordered_float::OrderedFloat;
use qdrant_edge::{
    Condition, Distance, EdgeConfigBuilder, EdgeShard, EdgeSparseVectorParamsBuilder,
    EdgeVectorParamsBuilder, FieldCondition, Filter, JsonPath, Match, MatchValue, Modifier,
    NamedQuery, PointId, PointInsertOperations, PointOperations, PointStruct, QueryEnum,
    QueryRequest, ScoringQuery, ValueVariants, Vector, VectorInternal, Vectors,
    WithPayloadInterface, WithVector,
};
use serde_json::Value;
use tokio::sync::RwLock;
use uuid::Uuid;

const SEGMENTS_DIR: &str = "segments";
const ORIGINAL_ID_KEY: &str = "__crytex_id";

/// In-process, on-disk vector store powered by Qdrant Edge.
#[derive(Debug)]
pub struct EdgeVectorStore {
    base_path: PathBuf,
    shards: RwLock<HashMap<String, Arc<EdgeShard>>>,
}

impl EdgeVectorStore {
    /// Open (or create) an embedded vector store at `base_path`.
    pub fn new(base_path: impl Into<PathBuf>) -> Result<Self, VectorStoreError> {
        let base_path = base_path.into();
        std::fs::create_dir_all(&base_path).map_err(|e| {
            VectorStoreError::Collection(format!(
                "failed to create vector store directory {}: {e}",
                base_path.display()
            ))
        })?;
        Ok(Self {
            base_path,
            shards: RwLock::new(HashMap::new()),
        })
    }

    fn collection_path(&self, collection: &str) -> PathBuf {
        self.base_path.join(sanitize_collection_name(collection))
    }

    async fn get_shard(&self, collection: &str) -> Result<Arc<EdgeShard>, VectorStoreError> {
        let shards = self.shards.read().await;
        shards.get(collection).cloned().ok_or_else(|| {
            VectorStoreError::Collection(format!("collection {collection} does not exist"))
        })
    }
}

#[async_trait::async_trait]
impl VectorStore for EdgeVectorStore {
    async fn create_collection(
        &self,
        collection: &str,
        dim: usize,
    ) -> Result<(), VectorStoreError> {
        if self.shards.read().await.contains_key(collection) {
            return Ok(());
        }

        let path = self.collection_path(collection);
        std::fs::create_dir_all(&path).map_err(|e| {
            VectorStoreError::Collection(format!(
                "failed to create collection directory {}: {e}",
                path.display()
            ))
        })?;

        let config = EdgeConfigBuilder::new()
            .vector(
                qdrant_edge::DEFAULT_VECTOR_NAME,
                EdgeVectorParamsBuilder::new(dim, Distance::Cosine).build(),
            )
            .build();

        let shard = if path.join(SEGMENTS_DIR).exists() {
            EdgeShard::load(&path, Some(config))
        } else {
            EdgeShard::new(&path, config)
        }
        .map_err(|e| VectorStoreError::Collection(e.to_string()))?;

        let mut shards = self.shards.write().await;
        shards.insert(collection.to_string(), Arc::new(shard));
        Ok(())
    }

    async fn supports_sparse(&self) -> bool {
        true
    }

    async fn create_sparse_collection(
        &self,
        collection: &str,
        dim: usize,
    ) -> Result<(), VectorStoreError> {
        if self.shards.read().await.contains_key(collection) {
            return Ok(());
        }

        // Remove any existing dense-only shard so the new config can declare the
        // sparse vector. Callers that need to preserve data should reindex into a
        // fresh collection.
        let _ = self.delete_collection(collection).await;

        let path = self.collection_path(collection);
        std::fs::create_dir_all(&path).map_err(|e| {
            VectorStoreError::Collection(format!(
                "failed to create collection directory {}: {e}",
                path.display()
            ))
        })?;

        let config = EdgeConfigBuilder::new()
            .vector(
                qdrant_edge::DEFAULT_VECTOR_NAME,
                EdgeVectorParamsBuilder::new(dim, Distance::Cosine).build(),
            )
            .sparse_vector(
                "bm25",
                EdgeSparseVectorParamsBuilder::new()
                    .modifier(Modifier::Idf)
                    .build(),
            )
            .build();

        let shard = EdgeShard::new(&path, config)
            .map_err(|e| VectorStoreError::Collection(e.to_string()))?;

        let mut shards = self.shards.write().await;
        shards.insert(collection.to_string(), Arc::new(shard));
        Ok(())
    }

    async fn delete_collection(&self, collection: &str) -> Result<(), VectorStoreError> {
        let mut shards = self.shards.write().await;
        shards.remove(collection);
        drop(shards);

        let path = self.collection_path(collection);
        if path.exists() {
            std::fs::remove_dir_all(&path).map_err(|e| {
                VectorStoreError::Collection(format!(
                    "failed to remove collection directory {}: {e}",
                    path.display()
                ))
            })?;
        }
        Ok(())
    }

    async fn upsert(
        &self,
        collection: &str,
        points: Vec<VectorPoint>,
    ) -> Result<(), VectorStoreError> {
        let shard = self.get_shard(collection).await?;

        tokio::task::spawn_blocking(move || {
            let edge_points: Vec<PointStruct> = points
                .into_iter()
                .map(|p| -> Result<PointStruct, VectorStoreError> {
                    let payload = payload_with_original_id(p.payload, &p.id)?;
                    Ok(PointStruct::new(
                        point_id_from_str(&p.id)?,
                        qdrant_edge::Vectors::from(p.vector),
                        payload,
                    ))
                })
                .collect::<Result<Vec<_>, _>>()?;

            let persisted: Vec<qdrant_edge::PointStructPersisted> =
                edge_points.into_iter().map(|p| p.0).collect();

            let operation = qdrant_edge::UpdateOperation::PointOperation(
                PointOperations::UpsertPoints(PointInsertOperations::from(persisted)),
            );

            shard
                .update(operation)
                .map_err(|e| VectorStoreError::Upsert(e.to_string()))
        })
        .await
        .map_err(|e| VectorStoreError::Upsert(format!("upsert task panicked: {e}")))?
    }

    async fn upsert_with_sparse(
        &self,
        collection: &str,
        points: Vec<SparseVectorPoint>,
    ) -> Result<(), VectorStoreError> {
        let shard = self.get_shard(collection).await?;

        tokio::task::spawn_blocking(move || {
            let edge_points: Vec<PointStruct> = points
                .into_iter()
                .map(|p| -> Result<PointStruct, VectorStoreError> {
                    let payload = payload_with_original_id(p.payload, &p.id)?;
                    let sparse =
                        Vector::new_sparse(p.sparse_vector.indices, p.sparse_vector.values)
                            .map_err(|e| VectorStoreError::Upsert(e.to_string()))?;
                    let vectors = Vectors::new_named([
                        (
                            qdrant_edge::DEFAULT_VECTOR_NAME,
                            Vector::new_dense(p.vector),
                        ),
                        ("bm25", sparse),
                    ]);
                    Ok(PointStruct::new(
                        point_id_from_str(&p.id)?,
                        vectors,
                        payload,
                    ))
                })
                .collect::<Result<Vec<_>, _>>()?;

            let persisted: Vec<qdrant_edge::PointStructPersisted> =
                edge_points.into_iter().map(|p| p.0).collect();

            let operation = qdrant_edge::UpdateOperation::PointOperation(
                PointOperations::UpsertPoints(PointInsertOperations::from(persisted)),
            );

            shard
                .update(operation)
                .map_err(|e| VectorStoreError::Upsert(e.to_string()))?;
            shard
                .optimize()
                .map_err(|e| VectorStoreError::Upsert(e.to_string()))?;
            Ok(())
        })
        .await
        .map_err(|e| VectorStoreError::Upsert(format!("upsert task panicked: {e}")))?
    }

    async fn search(
        &self,
        collection: &str,
        vector: &[f32],
        options: SearchOptions,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let shard = self.get_shard(collection).await?;
        let vector = vector.to_vec();
        let filter = build_filter(options.filter.as_ref())?;

        tokio::task::spawn_blocking(move || {
            let request = QueryRequest {
                prefetches: vec![],
                query: Some(ScoringQuery::Vector(QueryEnum::from(vector))),
                filter,
                score_threshold: options.score_threshold.map(OrderedFloat),
                limit: options.limit,
                offset: 0,
                params: None,
                with_vector: WithVector::Bool(false),
                with_payload: WithPayloadInterface::Bool(true),
            };

            let scored = shard
                .query(request)
                .map_err(|e| VectorStoreError::Search(e.to_string()))?;

            Ok(scored
                .into_iter()
                .map(|p| {
                    let mut payload = p
                        .payload
                        .map(|payload| Value::Object(payload.0.into_iter().collect()))
                        .unwrap_or_else(|| Value::Object(serde_json::Map::new()));
                    let id = payload
                        .as_object_mut()
                        .and_then(|map| map.remove(ORIGINAL_ID_KEY))
                        .and_then(|v| v.as_str().map(String::from))
                        .unwrap_or_else(|| p.id.to_string());
                    SearchResult {
                        id,
                        score: p.score,
                        payload,
                    }
                })
                .collect())
        })
        .await
        .map_err(|e| VectorStoreError::Search(format!("search task panicked: {e}")))?
    }

    async fn search_sparse(
        &self,
        collection: &str,
        vector: &SparseVector,
        options: SearchOptions,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let shard = self.get_shard(collection).await?;
        let sparse = qdrant_edge::SparseVector {
            indices: vector.indices.clone(),
            values: vector.values.clone(),
        };
        let filter = build_filter(options.filter.as_ref())?;

        tokio::task::spawn_blocking(move || {
            let request = QueryRequest {
                prefetches: vec![],
                query: Some(ScoringQuery::Vector(QueryEnum::Nearest(NamedQuery::new(
                    VectorInternal::Sparse(sparse),
                    "bm25",
                )))),
                filter,
                score_threshold: options.score_threshold.map(OrderedFloat),
                limit: options.limit,
                offset: 0,
                params: None,
                with_vector: WithVector::Bool(false),
                with_payload: WithPayloadInterface::Bool(true),
            };

            let scored = shard
                .query(request)
                .map_err(|e| VectorStoreError::Search(e.to_string()))?;

            Ok(scored
                .into_iter()
                .map(|p| {
                    let mut payload = p
                        .payload
                        .map(|payload| Value::Object(payload.0.into_iter().collect()))
                        .unwrap_or_else(|| Value::Object(serde_json::Map::new()));
                    let id = payload
                        .as_object_mut()
                        .and_then(|map| map.remove(ORIGINAL_ID_KEY))
                        .and_then(|v| v.as_str().map(String::from))
                        .unwrap_or_else(|| p.id.to_string());
                    SearchResult {
                        id,
                        score: p.score,
                        payload,
                    }
                })
                .collect())
        })
        .await
        .map_err(|e| VectorStoreError::Search(format!("search task panicked: {e}")))?
    }

    async fn delete_by_filter(
        &self,
        collection: &str,
        filter: Value,
    ) -> Result<(), VectorStoreError> {
        let shard = self.get_shard(collection).await?;
        let filter = build_filter(Some(&filter))?.ok_or_else(|| {
            VectorStoreError::Search("delete_by_filter requires a non-empty filter".into())
        })?;

        tokio::task::spawn_blocking(move || {
            let operation = qdrant_edge::UpdateOperation::PointOperation(
                PointOperations::DeletePointsByFilter(filter),
            );
            shard
                .update(operation)
                .map_err(|e| VectorStoreError::Upsert(e.to_string()))
        })
        .await
        .map_err(|e| VectorStoreError::Upsert(format!("delete task panicked: {e}")))?
    }
}

fn payload_with_original_id(payload: Value, original_id: &str) -> Result<Value, VectorStoreError> {
    let Value::Object(mut map) = payload else {
        return Err(VectorStoreError::Upsert(format!(
            "payload must be a JSON object, got {payload}"
        )));
    };
    map.insert(
        ORIGINAL_ID_KEY.to_string(),
        Value::String(original_id.to_string()),
    );
    Ok(Value::Object(map))
}

fn point_id_from_str(id: &str) -> Result<PointId, VectorStoreError> {
    if let Ok(num) = id.parse::<u64>() {
        return Ok(PointId::NumId(num));
    }
    let uuid = Uuid::new_v5(&Uuid::NAMESPACE_OID, id.as_bytes());
    Ok(PointId::Uuid(uuid))
}

fn build_filter(filter: Option<&Value>) -> Result<Option<Filter>, VectorStoreError> {
    let Some(obj) = filter.and_then(Value::as_object) else {
        return Ok(None);
    };

    let mut conditions = Vec::new();
    for (key, clause) in obj {
        let match_clause = clause.get("match").ok_or_else(|| {
            VectorStoreError::Search(format!("unsupported filter clause for {key}"))
        })?;
        let value = match_clause.get("value").ok_or_else(|| {
            VectorStoreError::Search(format!("filter match for {key} missing value"))
        })?;

        let variant = match value {
            Value::String(s) => ValueVariants::String(s.clone()),
            Value::Bool(b) => ValueVariants::Bool(*b),
            Value::Number(n) => {
                let i = n.as_i64().ok_or_else(|| {
                    VectorStoreError::Search(format!("filter value {value} is not an integer"))
                })?;
                ValueVariants::Integer(i)
            }
            _ => {
                return Err(VectorStoreError::Search(format!(
                    "unsupported filter value type: {value}"
                )));
            }
        };

        let json_path = JsonPath::try_from(key.as_str())
            .map_err(|()| VectorStoreError::Search(format!("invalid filter key {key}")))?;
        conditions.push(Condition::Field(FieldCondition::new_match(
            json_path,
            Match::Value(MatchValue { value: variant }),
        )));
    }

    Ok(Some(Filter {
        must: Some(conditions),
        should: None,
        must_not: None,
        min_should: None,
    }))
}

fn sanitize_collection_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crytex_core::services::{Embedder, SparseEmbedder};
    use serde_json::json;

    fn point(id: &str, vec: Vec<f32>, payload: Value) -> VectorPoint {
        VectorPoint {
            id: id.into(),
            vector: vec,
            payload,
        }
    }

    #[tokio::test]
    async fn edge_vector_store_upsert_and_search() {
        let dir = tempfile::tempdir().unwrap();
        let store = EdgeVectorStore::new(dir.path()).unwrap();

        store.create_collection("code_chunks", 3).await.unwrap();
        store
            .upsert(
                "code_chunks",
                vec![
                    point("1", vec![1.0, 0.0, 0.0], json!({"text": "fn a()"})),
                    point("2", vec![0.0, 1.0, 0.0], json!({"text": "fn b()"})),
                ],
            )
            .await
            .unwrap();

        let results = store
            .search(
                "code_chunks",
                &[1.0, 0.0, 0.0],
                SearchOptions {
                    limit: 1,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "1");
        assert!(results[0].score > 0.99);
    }

    #[tokio::test]
    async fn edge_vector_store_filters_by_project_id() {
        let dir = tempfile::tempdir().unwrap();
        let store = EdgeVectorStore::new(dir.path()).unwrap();

        store.create_collection("code_chunks", 3).await.unwrap();
        store
            .upsert(
                "code_chunks",
                vec![
                    point(
                        "p1",
                        vec![1.0, 0.0, 0.0],
                        json!({"project_id": "p1", "text": "fn a()"}),
                    ),
                    point(
                        "p2",
                        vec![1.0, 0.0, 0.0],
                        json!({"project_id": "p2", "text": "fn b()"}),
                    ),
                ],
            )
            .await
            .unwrap();

        let results = store
            .search(
                "code_chunks",
                &[1.0, 0.0, 0.0],
                SearchOptions {
                    limit: 10,
                    filter: Some(json!({"project_id": { "match": { "value": "p1" } }})),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "p1");
    }

    #[tokio::test]
    async fn edge_vector_store_creates_sparse_collection() {
        let dir = tempfile::tempdir().unwrap();
        let store = EdgeVectorStore::new(dir.path()).unwrap();

        assert!(store.supports_sparse().await);
        store
            .create_sparse_collection("code_chunks", 3)
            .await
            .unwrap();

        // A sparse-only upsert should succeed after sparse collection creation.
        store
            .upsert_with_sparse(
                "code_chunks",
                vec![SparseVectorPoint {
                    id: "s1".into(),
                    vector: vec![1.0, 0.0, 0.0],
                    sparse_vector: SparseVector {
                        indices: vec![1],
                        values: vec![1.0],
                    },
                    payload: serde_json::json!({"text": "hello"}),
                }],
            )
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn edge_vector_store_upserts_and_searches_sparse() {
        use qdrant_edge::bm25_embed::{EdgeBm25, EdgeBm25Config};

        let dir = tempfile::tempdir().unwrap();
        let store = EdgeVectorStore::new(dir.path()).unwrap();
        store
            .create_sparse_collection("code_chunks", 3)
            .await
            .unwrap();

        let bm25 = EdgeBm25::new(EdgeBm25Config::default()).unwrap();

        let docs = [
            ("chunk-1", "the quick brown fox jumps over the lazy dog"),
            ("chunk-2", "a lazy dog sleeps all day"),
        ];
        let mut points = Vec::new();
        for (id, text) in docs {
            let sparse = bm25.embed_document(text);
            points.push(SparseVectorPoint {
                id: id.into(),
                vector: vec![0.0; 3],
                sparse_vector: SparseVector {
                    indices: sparse.indices,
                    values: sparse.values,
                },
                payload: serde_json::json!({"text": text}),
            });
        }
        store
            .upsert_with_sparse("code_chunks", points)
            .await
            .unwrap();

        let query = bm25.embed_query("fox");
        let results = store
            .search_sparse(
                "code_chunks",
                &SparseVector {
                    indices: query.indices,
                    values: query.values,
                },
                SearchOptions {
                    limit: 5,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert!(!results.is_empty(), "expected sparse hits for 'fox'");
        assert_eq!(results[0].id, "chunk-1");
        let text = results[0]
            .payload
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap();
        assert!(text.contains("fox"));
    }

    #[tokio::test]
    async fn edge_vector_store_filters_sparse_by_project_id() {
        use qdrant_edge::bm25_embed::{EdgeBm25, EdgeBm25Config};

        let dir = tempfile::tempdir().unwrap();
        let store = EdgeVectorStore::new(dir.path()).unwrap();
        store
            .create_sparse_collection("code_chunks", 3)
            .await
            .unwrap();

        let bm25 = EdgeBm25::new(EdgeBm25Config::default()).unwrap();
        let docs = [
            ("p1", "project one fetch url"),
            ("p2", "project two post body"),
        ];
        let points: Vec<SparseVectorPoint> = docs
            .iter()
            .map(|(id, text)| {
                let sparse = bm25.embed_document(text);
                SparseVectorPoint {
                    id: (*id).into(),
                    vector: vec![0.0; 3],
                    sparse_vector: SparseVector {
                        indices: sparse.indices,
                        values: sparse.values,
                    },
                    payload: serde_json::json!({"project_id": id, "text": text}),
                }
            })
            .collect();
        store
            .upsert_with_sparse("code_chunks", points)
            .await
            .unwrap();

        let query = bm25.embed_query("fetch");
        let results = store
            .search_sparse(
                "code_chunks",
                &SparseVector {
                    indices: query.indices,
                    values: query.values,
                },
                SearchOptions {
                    limit: 5,
                    filter: Some(serde_json::json!({"project_id": {"match": {"value": "p1"}}})),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "p1");
    }

    #[tokio::test]
    async fn edge_vector_store_delete_collection_removes_data() {
        let dir = tempfile::tempdir().unwrap();
        let store = EdgeVectorStore::new(dir.path()).unwrap();

        store.create_collection("temp", 2).await.unwrap();
        store
            .upsert(
                "temp",
                vec![point(
                    "1",
                    vec![1.0, 0.0],
                    Value::Object(serde_json::Map::new()),
                )],
            )
            .await
            .unwrap();

        store.delete_collection("temp").await.unwrap();

        let err = store
            .upsert(
                "temp",
                vec![point(
                    "1",
                    vec![1.0, 0.0],
                    Value::Object(serde_json::Map::new()),
                )],
            )
            .await
            .unwrap_err();
        assert!(matches!(err, VectorStoreError::Collection(_)));
    }

    #[tokio::test]
    async fn edge_vector_store_delete_by_filter_removes_matching_points() {
        let dir = tempfile::tempdir().unwrap();
        let store = EdgeVectorStore::new(dir.path()).unwrap();

        store.create_collection("code_chunks", 2).await.unwrap();
        store
            .upsert(
                "code_chunks",
                vec![
                    point(
                        "a",
                        vec![1.0, 0.0],
                        json!({"project_id": "p1", "relative_path": "a.rs"}),
                    ),
                    point(
                        "b",
                        vec![0.0, 1.0],
                        json!({"project_id": "p1", "relative_path": "b.rs"}),
                    ),
                    point(
                        "c",
                        vec![1.0, 1.0],
                        json!({"project_id": "p2", "relative_path": "a.rs"}),
                    ),
                ],
            )
            .await
            .unwrap();

        store
            .delete_by_filter(
                "code_chunks",
                json!({
                    "project_id": {"match": {"value": "p1"}},
                    "relative_path": {"match": {"value": "a.rs"}}
                }),
            )
            .await
            .unwrap();

        let results = store
            .search(
                "code_chunks",
                &[1.0, 0.0],
                SearchOptions {
                    limit: 10,
                    filter: Some(json!({"project_id": {"match": {"value": "p1"}}})),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let ids: Vec<_> = results.iter().map(|r| r.id.as_str()).collect();
        assert!(!ids.contains(&"a"), "filtered point should be deleted");
        assert!(ids.contains(&"b"));

        let p2 = store
            .search(
                "code_chunks",
                &[1.0, 0.0],
                SearchOptions {
                    limit: 10,
                    filter: Some(json!({"project_id": {"match": {"value": "p2"}}})),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(p2.iter().any(|r| r.id == "c"));
    }

    #[tokio::test]
    async fn edge_hybrid_search_fuses_dense_and_sparse() {
        use crate::sparse_embedder::EdgeBm25SparseEmbedder;
        use crytex_core::services::MockEmbedder;
        use crytex_core::services::hybrid::{HybridRetriever, ReciprocalRankFusion};

        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(EdgeVectorStore::new(dir.path()).unwrap());
        store
            .create_sparse_collection("code_chunks", 3)
            .await
            .unwrap();

        let sparse_embedder =
            Arc::new(EdgeBm25SparseEmbedder::with_language(Some("english".into())).unwrap());
        let embedder = Arc::new(MockEmbedder::new(3));
        let query = "bar";
        let query_vector = embedder.embed(query).await.unwrap();

        // chunk-dense is the best dense match; chunk-sparse is the best BM25 match.
        let points = vec![
            SparseVectorPoint {
                id: "chunk-dense".into(),
                vector: query_vector.clone(),
                sparse_vector: sparse_embedder
                    .embed_document("semantic similarity text")
                    .await
                    .unwrap(),
                payload: json!({"project_id": "proj-1", "text": "semantic similarity text"}),
            },
            SparseVectorPoint {
                id: "chunk-sparse".into(),
                vector: vec![0.0, 1.0, 0.0],
                sparse_vector: sparse_embedder
                    .embed_document("contains the bar keyword")
                    .await
                    .unwrap(),
                payload: json!({"project_id": "proj-1", "text": "contains the bar keyword"}),
            },
        ];
        store
            .upsert_with_sparse("code_chunks", points)
            .await
            .unwrap();

        let fusion = Arc::new(ReciprocalRankFusion::default());
        let retriever = HybridRetriever::new(embedder, store, Some(sparse_embedder), fusion);

        let results = retriever
            .search(query, "proj-1", &["code_chunks"], 5, 5)
            .await
            .unwrap();

        let ids: Vec<_> = results.iter().map(|r| r.id.as_str()).collect();
        assert!(ids.contains(&"chunk-dense"), "dense match missing: {ids:?}");
        assert!(
            ids.contains(&"chunk-sparse"),
            "sparse match missing: {ids:?}"
        );
    }
}
