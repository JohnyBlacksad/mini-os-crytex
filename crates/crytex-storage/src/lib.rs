pub mod files;
pub mod graph;
pub mod persistence;
pub mod sparse_embedder;
pub mod vector;

use std::sync::Arc;

use crytex_core::models::Experience;
use crytex_core::services::{Embedder, VectorPoint, VectorStore};
use graph::GraphStore;
use serde::de::Error as SerdeErrorTrait;

/// Top-level storage facade.
///
/// Keeps the SQLite graph store and an optional vector store for embeddings.
/// When both an embedder and a vector store are configured, experience records
/// that carry `text` are also upserted to the `experience` collection.
#[derive(Clone)]
pub struct Storage {
    pub graph: GraphStore,
    embedder: Option<Arc<dyn Embedder>>,
    vector_store: Option<Arc<dyn VectorStore>>,
}

impl Storage {
    pub async fn new(db_path: &str) -> Result<Self, crate::graph::Error> {
        let graph = GraphStore::new(db_path).await?;
        Ok(Self {
            graph,
            embedder: None,
            vector_store: None,
        })
    }

    /// Attach an embedder and vector store for experience indexing.
    pub fn with_experience_vector_store(
        mut self,
        embedder: Arc<dyn Embedder>,
        vector_store: Arc<dyn VectorStore>,
    ) -> Self {
        self.embedder = Some(embedder);
        self.vector_store = Some(vector_store);
        self
    }

    pub async fn insert_experience(&self, exp: &Experience) -> Result<(), crate::graph::Error> {
        self.graph.insert_experience(exp).await?;

        if let (Some(embedder), Some(vector_store)) = (&self.embedder, &self.vector_store)
            && let Some(text) = exp.text.as_deref()
        {
            let dim = embedder
                .dimension()
                .await
                .map_err(|e| crate::graph::Error::Serde(serde_json::Error::custom(e)))?;
            let vector = embedder
                .embed(text)
                .await
                .map_err(|e| crate::graph::Error::Serde(serde_json::Error::custom(e)))?;
            vector_store
                .create_collection("experience", dim)
                .await
                .map_err(|e| crate::graph::Error::Serde(serde_json::Error::custom(e)))?;
            vector_store
                .upsert(
                    "experience",
                    vec![VectorPoint {
                        id: exp.id.clone(),
                        vector,
                        payload: serde_json::json!({
                            "task_id": exp.task_id,
                            "project_id": exp.project_id,
                            "prompt_version_id": exp.prompt_version_id,
                            "text": text,
                            "reward": exp.reward,
                        }),
                    }],
                )
                .await
                .map_err(|e| crate::graph::Error::Serde(serde_json::Error::custom(e)))?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crytex_core::models::{Project, Task, TaskStatus};
    use crytex_core::persistence::ExperienceRepository;
    use crytex_core::services::MockEmbedder;
    use vector::memory::MemoryVectorStore;

    fn sample_project() -> Project {
        Project {
            id: "proj-1".into(),
            name: "Test".into(),
            root_path: "/tmp".into(),
            created_at: 1,
            updated_at: 1,
            metadata: serde_json::Value::Null,
        }
    }

    fn sample_task(id: &str) -> Task {
        Task {
            id: id.into(),
            project_id: "proj-1".into(),
            parent_id: None,
            title: "task".into(),
            description: None,
            kind: "codegen".into(),
            status: TaskStatus::Completed,
            assigned_agent: None,
            priority: 0,
            created_at: 1,
            started_at: None,
            finished_at: None,
            payload: serde_json::Value::Null,
            result: None,
            iteration_count: 0,
            priority_score: 0.0,
            critic_score: None,
            human_score: None,
            prompt_version_id: None,
            lora_adapter_id: None,
            trace_id: "trace-1".into(),
        }
    }

    fn sample_experience(id: &str, task_id: &str, text: Option<&str>) -> Experience {
        Experience {
            id: id.into(),
            task_id: task_id.into(),
            project_id: Some("proj-1".into()),
            prompt_version_id: Some("pv1".into()),
            text: text.map(|s| s.into()),
            critic_score: Some(4.0),
            human_score: Some(5.0),
            reward: 4.4,
            comment: Some("good".into()),
            created_at: 1234567890,
        }
    }

    async fn seed_project_and_task(storage: &Storage) {
        storage
            .graph
            .insert_project(&sample_project())
            .await
            .unwrap();
        storage.graph.insert_task(&sample_task("t1")).await.unwrap();
    }

    #[tokio::test]
    async fn experience_survives_restart() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("exp.db");
        {
            let storage = Storage::new(p.to_str().unwrap()).await.unwrap();
            seed_project_and_task(&storage).await;
            let exp = sample_experience("e1", "t1", None);
            storage.insert_experience(&exp).await.unwrap();
        }

        // Re-open the same database file.
        let storage = Storage::new(p.to_str().unwrap()).await.unwrap();
        let loaded = storage.list_experiences_by_task("t1").await.unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].id, "e1");
        assert_eq!(loaded[0].reward, 4.4);
    }

    #[tokio::test]
    async fn experience_upserts_vector_embedding() {
        let storage = Storage::new(":memory:").await.unwrap();
        seed_project_and_task(&storage).await;
        let vector_store: Arc<dyn VectorStore> = Arc::new(MemoryVectorStore::new());
        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new(8));
        let storage = storage.with_experience_vector_store(embedder.clone(), vector_store.clone());

        let exp = sample_experience("e1", "t1", Some("successful refactoring"));
        storage.insert_experience(&exp).await.unwrap();

        let query = embedder.embed("successful refactoring").await.unwrap();
        let results = vector_store
            .search(
                "experience",
                &query,
                crytex_core::services::SearchOptions {
                    limit: 10,
                    filter: Some(serde_json::json!({"project_id": {"match": {"value": "proj-1"}}})),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "e1");
        assert!(results[0].score > 0.99);
    }
}
