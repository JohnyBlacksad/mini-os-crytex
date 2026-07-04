//! Production Qdrant vector-store implementation.
//!
//! This module is compiled unconditionally; if no Qdrant server is available,
//! callers can fall back to [`MemoryVectorStore`](super::memory::MemoryVectorStore).

use crytex_core::services::vector_store::{
    SearchOptions, SearchResult, VectorPoint, VectorStore, VectorStoreError,
};
use qdrant_client::qdrant::{
    Condition, CreateCollectionBuilder, DeletePointsBuilder, Distance, Filter, PointStruct,
    QueryPointsBuilder, UpsertPointsBuilder, VectorParamsBuilder, r#match::MatchValue,
    point_id::PointIdOptions, points_selector::PointsSelectorOneOf,
};
use qdrant_client::{Payload, Qdrant};

#[derive(Clone)]
pub struct QdrantVectorStore {
    client: Qdrant,
}

impl std::fmt::Debug for QdrantVectorStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QdrantVectorStore").finish_non_exhaustive()
    }
}

impl QdrantVectorStore {
    /// Connect to a Qdrant server at the given gRPC URL (e.g. `http://localhost:6334`).
    pub fn new(url: &str) -> Result<Self, VectorStoreError> {
        let client = Qdrant::from_url(url)
            .build()
            .map_err(|e| VectorStoreError::Collection(e.to_string()))?;
        Ok(Self { client })
    }

    /// Wrap an existing `Qdrant` client.
    pub fn from_client(client: Qdrant) -> Self {
        Self { client }
    }
}

fn payload_to_qdrant(value: serde_json::Value) -> Result<Payload, VectorStoreError> {
    Payload::try_from(value).map_err(|e| VectorStoreError::Upsert(e.to_string()))
}

fn json_to_match_value(value: &serde_json::Value) -> Result<MatchValue, VectorStoreError> {
    match value {
        serde_json::Value::String(s) => Ok(MatchValue::Keyword(s.clone())),
        serde_json::Value::Bool(b) => Ok(MatchValue::Boolean(*b)),
        serde_json::Value::Number(n) => {
            let i = n.as_i64().ok_or_else(|| {
                VectorStoreError::Search(format!("filter value {value} is not an integer"))
            })?;
            Ok(MatchValue::Integer(i))
        }
        _ => Err(VectorStoreError::Search(format!(
            "unsupported filter value type: {value}"
        ))),
    }
}

fn build_filter(filter: &serde_json::Value) -> Result<Filter, VectorStoreError> {
    let Some(obj) = filter.as_object() else {
        return Ok(Filter::default());
    };
    let mut conditions = Vec::new();
    for (key, clause) in obj {
        if let Some(match_clause) = clause.get("match") {
            if let Some(value) = match_clause.get("value") {
                conditions.push(Condition::matches(key, json_to_match_value(value)?));
            } else {
                return Err(VectorStoreError::Search(format!(
                    "filter match for {key} missing value"
                )));
            }
        } else {
            return Err(VectorStoreError::Search(format!(
                "unsupported filter clause for {key}"
            )));
        }
    }
    Ok(Filter::all(conditions))
}

fn point_id_to_string(id: qdrant_client::qdrant::PointId) -> String {
    match id.point_id_options {
        Some(PointIdOptions::Num(n)) => n.to_string(),
        Some(PointIdOptions::Uuid(s)) => s,
        None => String::new(),
    }
}

#[async_trait::async_trait]
impl VectorStore for QdrantVectorStore {
    async fn create_collection(
        &self,
        collection: &str,
        dim: usize,
    ) -> Result<(), VectorStoreError> {
        let exists = self
            .client
            .collection_exists(collection)
            .await
            .map_err(|e| VectorStoreError::Collection(e.to_string()))?;
        if exists {
            return Ok(());
        }
        self.client
            .create_collection(
                CreateCollectionBuilder::new(collection)
                    .vectors_config(VectorParamsBuilder::new(dim as u64, Distance::Cosine)),
            )
            .await
            .map_err(|e| VectorStoreError::Collection(e.to_string()))?;
        Ok(())
    }

    async fn delete_collection(&self, collection: &str) -> Result<(), VectorStoreError> {
        self.client
            .delete_collection(collection)
            .await
            .map_err(|e| VectorStoreError::Collection(e.to_string()))?;
        Ok(())
    }

    async fn upsert(
        &self,
        collection: &str,
        points: Vec<VectorPoint>,
    ) -> Result<(), VectorStoreError> {
        let qdrant_points: Vec<PointStruct> = points
            .into_iter()
            .map(|p| {
                let payload = payload_to_qdrant(p.payload)?;
                Ok::<_, VectorStoreError>(PointStruct::new(p.id, p.vector, payload))
            })
            .collect::<Result<Vec<_>, _>>()?;

        self.client
            .upsert_points(UpsertPointsBuilder::new(collection, qdrant_points))
            .await
            .map_err(|e| VectorStoreError::Upsert(e.to_string()))?;
        Ok(())
    }

    async fn search(
        &self,
        collection: &str,
        vector: &[f32],
        options: SearchOptions,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let mut builder = QueryPointsBuilder::new(collection)
            .query(vector.to_vec())
            .limit(options.limit as u64)
            .with_payload(true);

        if let Some(filter) = options.filter {
            builder = builder.filter(build_filter(&filter)?);
        }

        let response = self
            .client
            .query(builder)
            .await
            .map_err(|e| VectorStoreError::Search(e.to_string()))?;

        let threshold = options.score_threshold.unwrap_or(f32::NEG_INFINITY);
        Ok(response
            .result
            .into_iter()
            .filter(|p| p.score >= threshold)
            .map(|p| SearchResult {
                id: p.id.map(point_id_to_string).unwrap_or_default(),
                score: p.score,
                payload: serde_json::Value::from(Payload::from(p.payload)),
            })
            .collect())
    }

    async fn delete_by_filter(
        &self,
        collection: &str,
        filter: serde_json::Value,
    ) -> Result<(), VectorStoreError> {
        let filter = build_filter(&filter)?;
        self.client
            .delete_points(
                DeletePointsBuilder::new(collection).points(PointsSelectorOneOf::Filter(filter)),
            )
            .await
            .map_err(|e| VectorStoreError::Upsert(e.to_string()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore = "requires a running Qdrant instance"]
    async fn qdrant_client_creates_collection_and_searches() {
        let store = QdrantVectorStore::new("http://localhost:6334").unwrap();
        let _ = store.delete_collection("test_code_chunks").await;
        store
            .create_collection("test_code_chunks", 3)
            .await
            .unwrap();
        store
            .upsert(
                "test_code_chunks",
                vec![VectorPoint {
                    id: "a".into(),
                    vector: vec![1.0, 0.0, 0.0],
                    payload: serde_json::json!({"text": "fn a()"}),
                }],
            )
            .await
            .unwrap();

        let results = store
            .search(
                "test_code_chunks",
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
}
