//! In-memory vector-store fallback for tests and offline use.

use std::collections::HashMap;
use std::sync::Mutex;

use crytex_core::services::vector_store::{
    SearchOptions, SearchResult, VectorPoint, VectorStore, VectorStoreError,
};

#[derive(Debug, Default)]
struct Collection {
    dim: usize,
    points: HashMap<String, VectorPoint>,
}

/// Thread-safe in-memory vector store.
#[derive(Debug, Default)]
pub struct MemoryVectorStore {
    collections: Mutex<HashMap<String, Collection>>,
}

impl MemoryVectorStore {
    pub fn new() -> Self {
        Self::default()
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Check whether a point payload matches a simple filter object.
///
/// Supported filter shape (mirrors a tiny subset of Qdrant's DSL):
/// `{ "key": { "match": { "value": "..." } } }`
fn payload_matches(payload: &serde_json::Value, filter: &serde_json::Value) -> bool {
    let Some(obj) = filter.as_object() else {
        return true;
    };
    for (key, clause) in obj {
        let payload_value = payload.get(key);
        if let Some(match_clause) = clause.get("match") {
            let expected = match_clause.get("value");
            if payload_value != expected {
                return false;
            }
        }
    }
    true
}

#[async_trait::async_trait]
impl VectorStore for MemoryVectorStore {
    async fn create_collection(
        &self,
        collection: &str,
        dim: usize,
    ) -> Result<(), VectorStoreError> {
        let mut collections = self.collections.lock().unwrap();
        collections.insert(
            collection.to_string(),
            Collection {
                dim,
                points: HashMap::new(),
            },
        );
        Ok(())
    }

    async fn delete_collection(&self, collection: &str) -> Result<(), VectorStoreError> {
        let mut collections = self.collections.lock().unwrap();
        collections.remove(collection);
        Ok(())
    }

    async fn upsert(
        &self,
        collection: &str,
        points: Vec<VectorPoint>,
    ) -> Result<(), VectorStoreError> {
        let mut collections = self.collections.lock().unwrap();
        let col = collections.get_mut(collection).ok_or_else(|| {
            VectorStoreError::Collection(format!("collection {} does not exist", collection))
        })?;
        for point in points {
            if point.vector.len() != col.dim {
                return Err(VectorStoreError::DimensionMismatch {
                    expected: col.dim,
                    actual: point.vector.len(),
                });
            }
            col.points.insert(point.id.clone(), point);
        }
        Ok(())
    }

    async fn search(
        &self,
        collection: &str,
        vector: &[f32],
        options: SearchOptions,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let collections = self.collections.lock().unwrap();
        let col = collections.get(collection).ok_or_else(|| {
            VectorStoreError::Collection(format!("collection {} does not exist", collection))
        })?;
        if vector.len() != col.dim {
            return Err(VectorStoreError::DimensionMismatch {
                expected: col.dim,
                actual: vector.len(),
            });
        }

        let limit = options.limit.max(1);
        let mut results: Vec<SearchResult> = col
            .points
            .values()
            .filter(|p| {
                options
                    .filter
                    .as_ref()
                    .map(|f| payload_matches(&p.payload, f))
                    .unwrap_or(true)
            })
            .map(|p| SearchResult {
                id: p.id.clone(),
                score: cosine_similarity(vector, &p.vector),
                payload: p.payload.clone(),
            })
            .filter(|r| {
                options
                    .score_threshold
                    .map(|t| r.score >= t)
                    .unwrap_or(true)
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(limit);
        Ok(results)
    }

    async fn delete_by_filter(
        &self,
        collection: &str,
        filter: serde_json::Value,
    ) -> Result<(), VectorStoreError> {
        let mut collections = self.collections.lock().unwrap();
        let Some(col) = collections.get_mut(collection) else {
            return Ok(());
        };
        col.points.retain(|_, point| !payload_matches(&point.payload, &filter));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn point(id: &str, vec: Vec<f32>, payload: serde_json::Value) -> VectorPoint {
        VectorPoint {
            id: id.into(),
            vector: vec,
            payload,
        }
    }

    #[tokio::test]
    async fn memory_vector_store_upsert_and_search() {
        let store = MemoryVectorStore::new();
        store.create_collection("code_chunks", 3).await.unwrap();
        store
            .upsert(
                "code_chunks",
                vec![
                    point("a", vec![1.0, 0.0, 0.0], json!({"text": "fn a()"})),
                    point("b", vec![0.0, 1.0, 0.0], json!({"text": "fn b()"})),
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
        assert_eq!(results[0].id, "a");
        assert!(results[0].score > 0.99);
    }

    #[tokio::test]
    async fn vector_store_filters_by_project_id() {
        let store = MemoryVectorStore::new();
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
    async fn delete_by_filter_removes_matching_points() {
        let store = MemoryVectorStore::new();
        store.create_collection("code_chunks", 3).await.unwrap();
        store
            .upsert(
                "code_chunks",
                vec![
                    point(
                        "a",
                        vec![1.0, 0.0, 0.0],
                        json!({"project_id": "p1", "relative_path": "a.rs"}),
                    ),
                    point(
                        "b",
                        vec![0.0, 1.0, 0.0],
                        json!({"project_id": "p1", "relative_path": "b.rs"}),
                    ),
                    point(
                        "c",
                        vec![1.0, 1.0, 0.0],
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
                &[1.0, 0.0, 0.0],
                SearchOptions {
                    limit: 10,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let ids: Vec<_> = results.iter().map(|r| r.id.as_str()).collect();
        assert!(!ids.contains(&"a"), "filtered point should be deleted");
        assert!(ids.contains(&"b"));
        assert!(ids.contains(&"c"));
    }
}
