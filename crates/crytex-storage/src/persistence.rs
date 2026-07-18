use async_trait::async_trait;
use crytex_core::metrics::{MetricsError, MetricsRepository, MetricsSnapshot};
use crytex_core::models::{
    AgentLog, Artifact, BenchmarkResult, BenchmarkRun, BenchmarkRunSummary, Experience,
    LoraAdapter, MemoryEntry, Project, ProjectSnapshot, PromptVersion, Task, TaskDependency,
    TaskStatus, TrainingExample, TrainingJob,
};
use crytex_core::persistence::{
    ArtifactRepository, BenchmarkResultRepository, ExperienceRepository, LogRepository,
    LoraAdapterRepository, MemoryEntryRepository, PersistenceError, ProjectRepository,
    ProjectSnapshotRepository, PromptVersionRepository, TaskRepository, TrainingExampleRepository,
    TrainingJobRepository,
};

use crate::Storage;

fn map_error(e: crate::graph::Error) -> PersistenceError {
    PersistenceError::Database(e.to_string())
}

fn map_metrics_error(e: crate::graph::Error) -> MetricsError {
    MetricsError::Persistence(e.to_string())
}

#[async_trait]
impl ProjectRepository for Storage {
    async fn insert_project(&self, project: &Project) -> Result<(), PersistenceError> {
        self.graph.insert_project(project).await.map_err(map_error)
    }

    async fn get_project(&self, id: &str) -> Result<Option<Project>, PersistenceError> {
        self.graph.get_project(id).await.map_err(map_error)
    }

    async fn list_projects(&self) -> Result<Vec<Project>, PersistenceError> {
        self.graph.list_projects().await.map_err(map_error)
    }
}

#[async_trait]
impl TaskRepository for Storage {
    async fn insert_task(&self, task: &Task) -> Result<(), PersistenceError> {
        self.graph.insert_task(task).await.map_err(map_error)
    }

    async fn update_task(&self, task: &Task) -> Result<(), PersistenceError> {
        self.graph.insert_task(task).await.map_err(map_error)
    }

    async fn update_task_status(
        &self,
        id: &str,
        status: TaskStatus,
        result: Option<serde_json::Value>,
    ) -> Result<(), PersistenceError> {
        self.graph
            .update_task_status(id, status, result)
            .await
            .map_err(map_error)
    }

    async fn get_task(&self, id: &str) -> Result<Option<Task>, PersistenceError> {
        self.graph.get_task(id).await.map_err(map_error)
    }

    async fn list_tasks_by_project(&self, project_id: &str) -> Result<Vec<Task>, PersistenceError> {
        self.graph
            .list_tasks_by_project(project_id)
            .await
            .map_err(map_error)
    }

    async fn list_all_tasks(&self) -> Result<Vec<Task>, PersistenceError> {
        let projects = self.graph.list_projects().await.map_err(map_error)?;
        let mut tasks = Vec::new();
        for project in projects {
            let mut project_tasks = self
                .graph
                .list_tasks_by_project(&project.id)
                .await
                .map_err(map_error)?;
            tasks.append(&mut project_tasks);
        }
        Ok(tasks)
    }

    async fn list_ready_tasks(&self) -> Result<Vec<Task>, PersistenceError> {
        self.graph.list_ready_tasks().await.map_err(map_error)
    }

    async fn add_dependency(&self, dep: &TaskDependency) -> Result<(), PersistenceError> {
        self.graph.add_dependency(dep).await.map_err(map_error)
    }

    async fn list_dependencies(&self) -> Result<Vec<TaskDependency>, PersistenceError> {
        self.graph.list_dependencies().await.map_err(map_error)
    }
}

#[async_trait]
impl ArtifactRepository for Storage {
    async fn insert_artifact(&self, artifact: &Artifact) -> Result<(), PersistenceError> {
        self.graph
            .insert_artifact(artifact)
            .await
            .map_err(map_error)
    }

    async fn list_artifacts_by_task(
        &self,
        task_id: &str,
    ) -> Result<Vec<Artifact>, PersistenceError> {
        self.graph
            .list_artifacts_by_task(task_id)
            .await
            .map_err(map_error)
    }
}

#[async_trait]
impl LogRepository for Storage {
    async fn insert_agent_log(&self, log: &AgentLog) -> Result<(), PersistenceError> {
        self.graph.insert_agent_log(log).await.map_err(map_error)
    }

    async fn list_logs_by_task(&self, task_id: &str) -> Result<Vec<AgentLog>, PersistenceError> {
        self.graph
            .list_logs_by_task(task_id)
            .await
            .map_err(map_error)
    }

    async fn list_logs_by_project(
        &self,
        project_id: &str,
    ) -> Result<Vec<AgentLog>, PersistenceError> {
        self.graph
            .list_logs_by_project(project_id)
            .await
            .map_err(map_error)
    }
}

#[async_trait]
impl ExperienceRepository for Storage {
    async fn insert_experience(&self, exp: &Experience) -> Result<(), PersistenceError> {
        self.insert_experience(exp)
            .await
            .map_err(|e| PersistenceError::Database(e.to_string()))
    }

    async fn list_experiences_by_task(
        &self,
        task_id: &str,
    ) -> Result<Vec<Experience>, PersistenceError> {
        self.graph
            .list_experiences_by_task(task_id)
            .await
            .map_err(map_error)
    }

    async fn list_experiences_by_prompt_version(
        &self,
        prompt_version_id: &str,
    ) -> Result<Vec<Experience>, PersistenceError> {
        self.graph
            .list_experiences_by_prompt_version(prompt_version_id)
            .await
            .map_err(map_error)
    }
}

#[async_trait]
impl TrainingExampleRepository for Storage {
    async fn insert_training_example(
        &self,
        example: &TrainingExample,
    ) -> Result<(), PersistenceError> {
        self.graph
            .insert_training_example(example)
            .await
            .map_err(map_error)
    }

    async fn list_training_examples_by_kind(
        &self,
        task_kind: &str,
    ) -> Result<Vec<TrainingExample>, PersistenceError> {
        self.graph
            .list_training_examples_by_kind(task_kind)
            .await
            .map_err(map_error)
    }

    async fn count_training_examples_by_kind(
        &self,
        task_kind: &str,
    ) -> Result<usize, PersistenceError> {
        self.graph
            .count_training_examples_by_kind(task_kind)
            .await
            .map_err(map_error)
    }

    async fn list_training_examples_by_project(
        &self,
        project_id: &str,
    ) -> Result<Vec<TrainingExample>, PersistenceError> {
        self.graph
            .list_training_examples_by_project(project_id)
            .await
            .map_err(map_error)
    }

    async fn list_training_examples_by_role(
        &self,
        agent_role: &str,
    ) -> Result<Vec<TrainingExample>, PersistenceError> {
        self.graph
            .list_training_examples_by_role(agent_role)
            .await
            .map_err(map_error)
    }

    async fn count_training_examples_by_role(
        &self,
        agent_role: &str,
    ) -> Result<usize, PersistenceError> {
        self.graph
            .count_training_examples_by_role(agent_role)
            .await
            .map_err(map_error)
    }
}

#[async_trait]
impl LoraAdapterRepository for Storage {
    async fn insert_lora_adapter(&self, adapter: &LoraAdapter) -> Result<(), PersistenceError> {
        self.graph
            .insert_lora_adapter(adapter)
            .await
            .map_err(map_error)
    }

    async fn get_lora_adapter(&self, id: &str) -> Result<Option<LoraAdapter>, PersistenceError> {
        self.graph.get_lora_adapter(id).await.map_err(map_error)
    }

    async fn list_lora_adapters_by_kind(
        &self,
        task_kind: &str,
    ) -> Result<Vec<LoraAdapter>, PersistenceError> {
        self.graph
            .list_lora_adapters_by_kind(task_kind)
            .await
            .map_err(map_error)
    }

    async fn list_lora_adapters_by_project(
        &self,
        project_id: &str,
    ) -> Result<Vec<LoraAdapter>, PersistenceError> {
        self.graph
            .list_lora_adapters_by_project(project_id)
            .await
            .map_err(map_error)
    }

    async fn list_lora_adapters_by_role(
        &self,
        agent_role: &str,
    ) -> Result<Vec<LoraAdapter>, PersistenceError> {
        self.graph
            .list_lora_adapters_by_role(agent_role)
            .await
            .map_err(map_error)
    }

    async fn set_lora_adapter_active(
        &self,
        id: &str,
        active: bool,
    ) -> Result<(), PersistenceError> {
        self.graph
            .set_lora_adapter_active(id, active)
            .await
            .map_err(map_error)
    }
}

#[async_trait]
impl PromptVersionRepository for Storage {
    async fn insert_prompt_version(&self, version: &PromptVersion) -> Result<(), PersistenceError> {
        self.graph
            .insert_prompt_version(version)
            .await
            .map_err(map_error)
    }

    async fn update_prompt_version(&self, version: &PromptVersion) -> Result<(), PersistenceError> {
        self.graph
            .insert_prompt_version(version)
            .await
            .map_err(map_error)
    }

    async fn get_prompt_version(
        &self,
        id: &str,
    ) -> Result<Option<PromptVersion>, PersistenceError> {
        self.graph.get_prompt_version(id).await.map_err(map_error)
    }

    async fn list_prompt_versions_by_agent(
        &self,
        agent: &str,
    ) -> Result<Vec<PromptVersion>, PersistenceError> {
        self.graph
            .list_prompt_versions_by_agent(agent)
            .await
            .map_err(map_error)
    }

    async fn get_active_prompt_version(
        &self,
        agent: &str,
    ) -> Result<Option<PromptVersion>, PersistenceError> {
        self.graph
            .get_active_prompt_version(agent)
            .await
            .map_err(map_error)
    }

    async fn set_active_prompt_version(
        &self,
        id: &str,
        agent: &str,
    ) -> Result<(), PersistenceError> {
        self.graph
            .set_active_prompt_version(id, agent)
            .await
            .map_err(map_error)
    }
}

#[async_trait]
impl ProjectSnapshotRepository for Storage {
    async fn insert_project_snapshot(
        &self,
        snapshot: &ProjectSnapshot,
    ) -> Result<(), PersistenceError> {
        self.graph
            .insert_project_snapshot(snapshot)
            .await
            .map_err(map_error)
    }

    async fn get_project_snapshot(
        &self,
        id: &str,
    ) -> Result<Option<ProjectSnapshot>, PersistenceError> {
        self.graph.get_project_snapshot(id).await.map_err(map_error)
    }

    async fn list_project_snapshots(
        &self,
        project_id: &str,
    ) -> Result<Vec<ProjectSnapshot>, PersistenceError> {
        self.graph
            .list_project_snapshots(project_id)
            .await
            .map_err(map_error)
    }
}

#[async_trait]
impl TrainingJobRepository for Storage {
    async fn insert_training_job(&self, job: &TrainingJob) -> Result<(), PersistenceError> {
        self.graph.insert_training_job(job).await.map_err(map_error)
    }

    async fn update_training_job(&self, job: &TrainingJob) -> Result<(), PersistenceError> {
        self.graph.insert_training_job(job).await.map_err(map_error)
    }

    async fn get_training_job(&self, id: &str) -> Result<Option<TrainingJob>, PersistenceError> {
        self.graph.get_training_job(id).await.map_err(map_error)
    }

    async fn list_training_jobs_by_kind(
        &self,
        task_kind: &str,
    ) -> Result<Vec<TrainingJob>, PersistenceError> {
        self.graph
            .list_training_jobs_by_kind(task_kind)
            .await
            .map_err(map_error)
    }
}

#[async_trait]
impl MemoryEntryRepository for Storage {
    async fn insert_memory_entry(&self, entry: &MemoryEntry) -> Result<(), PersistenceError> {
        self.graph
            .insert_memory_entry(entry)
            .await
            .map_err(map_error)
    }

    async fn list_memory_entries(
        &self,
        project_id: Option<&str>,
        kind: Option<&str>,
        limit: usize,
    ) -> Result<Vec<MemoryEntry>, PersistenceError> {
        self.graph
            .list_memory_entries(project_id, kind, limit)
            .await
            .map_err(map_error)
    }

    async fn list_memory_entries_by_session(
        &self,
        session_id: &str,
    ) -> Result<Vec<MemoryEntry>, PersistenceError> {
        self.graph
            .list_memory_entries_by_session(session_id)
            .await
            .map_err(map_error)
    }
}

#[async_trait]
impl MetricsRepository for Storage {
    async fn insert_metric(&self, snapshot: &MetricsSnapshot) -> Result<(), MetricsError> {
        self.graph
            .insert_metric(snapshot)
            .await
            .map_err(map_metrics_error)
    }

    async fn list_metrics(&self, from: i64, to: i64) -> Result<Vec<MetricsSnapshot>, MetricsError> {
        self.graph
            .list_metrics(from, to)
            .await
            .map_err(map_metrics_error)
    }
}

#[async_trait]
impl BenchmarkResultRepository for Storage {
    async fn insert_run(&self, run: &BenchmarkRun) -> Result<(), PersistenceError> {
        self.graph
            .insert_benchmark_run(run)
            .await
            .map_err(map_error)
    }

    async fn get_run(&self, id: &str) -> Result<Option<BenchmarkRun>, PersistenceError> {
        self.graph.get_benchmark_run(id).await.map_err(map_error)
    }

    async fn list_runs(&self, limit: usize) -> Result<Vec<BenchmarkRunSummary>, PersistenceError> {
        self.graph
            .list_benchmark_runs(limit)
            .await
            .map_err(map_error)
    }

    async fn insert_result(
        &self,
        _run_id: &str,
        result: &BenchmarkResult,
    ) -> Result<(), PersistenceError> {
        self.graph
            .insert_benchmark_result(result)
            .await
            .map_err(map_error)
    }

    async fn list_results(&self, run_id: &str) -> Result<Vec<BenchmarkResult>, PersistenceError> {
        self.graph
            .list_benchmark_results(run_id)
            .await
            .map_err(map_error)
    }
}
