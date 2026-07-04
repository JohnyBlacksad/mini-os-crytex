use crate::models::{
    AgentLog, Artifact, BenchmarkResult, BenchmarkRun, BenchmarkRunSummary, Experience,
    LoraAdapter, MemoryEntry, Project, ProjectSnapshot, PromptVersion, Task, TaskDependency,
    TaskStatus, TrainingExample, TrainingJob,
};
use async_trait::async_trait;
use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PersistenceError {
    #[error("database error: {0}")]
    Database(String),
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("invalid data: {0}")]
    InvalidData(String),
}

#[async_trait]
pub trait ProjectRepository: Send + Sync {
    async fn insert_project(&self, project: &Project) -> Result<(), PersistenceError>;
    async fn get_project(&self, id: &str) -> Result<Option<Project>, PersistenceError>;
    async fn list_projects(&self) -> Result<Vec<Project>, PersistenceError>;
}

#[async_trait]
pub trait TaskRepository: Send + Sync {
    async fn insert_task(&self, task: &Task) -> Result<(), PersistenceError>;
    async fn update_task(&self, task: &Task) -> Result<(), PersistenceError>;
    async fn update_task_status(
        &self,
        id: &str,
        status: TaskStatus,
        result: Option<Value>,
    ) -> Result<(), PersistenceError>;
    async fn get_task(&self, id: &str) -> Result<Option<Task>, PersistenceError>;
    async fn list_tasks_by_project(&self, project_id: &str) -> Result<Vec<Task>, PersistenceError>;
    async fn list_all_tasks(&self) -> Result<Vec<Task>, PersistenceError>;
    async fn list_ready_tasks(&self) -> Result<Vec<Task>, PersistenceError>;
    async fn add_dependency(&self, dep: &TaskDependency) -> Result<(), PersistenceError>;
}

#[async_trait]
pub trait ArtifactRepository: Send + Sync {
    async fn insert_artifact(&self, artifact: &Artifact) -> Result<(), PersistenceError>;
    async fn list_artifacts_by_task(
        &self,
        task_id: &str,
    ) -> Result<Vec<Artifact>, PersistenceError>;
}

#[async_trait]
pub trait LogRepository: Send + Sync {
    async fn insert_agent_log(&self, log: &AgentLog) -> Result<(), PersistenceError>;
    async fn list_logs_by_task(&self, task_id: &str) -> Result<Vec<AgentLog>, PersistenceError>;
    async fn list_logs_by_project(
        &self,
        project_id: &str,
    ) -> Result<Vec<AgentLog>, PersistenceError>;
}

#[async_trait]
pub trait ExperienceRepository: Send + Sync {
    async fn insert_experience(&self, exp: &Experience) -> Result<(), PersistenceError>;
    async fn list_experiences_by_task(
        &self,
        task_id: &str,
    ) -> Result<Vec<Experience>, PersistenceError>;
    async fn list_experiences_by_prompt_version(
        &self,
        prompt_version_id: &str,
    ) -> Result<Vec<Experience>, PersistenceError>;
}

#[async_trait]
pub trait TrainingExampleRepository: Send + Sync {
    async fn insert_training_example(
        &self,
        example: &TrainingExample,
    ) -> Result<(), PersistenceError>;
    async fn list_training_examples_by_kind(
        &self,
        task_kind: &str,
    ) -> Result<Vec<TrainingExample>, PersistenceError>;
    async fn count_training_examples_by_kind(
        &self,
        task_kind: &str,
    ) -> Result<usize, PersistenceError>;
    async fn list_training_examples_by_project(
        &self,
        project_id: &str,
    ) -> Result<Vec<TrainingExample>, PersistenceError>;
    async fn list_training_examples_by_role(
        &self,
        agent_role: &str,
    ) -> Result<Vec<TrainingExample>, PersistenceError>;
    async fn count_training_examples_by_role(
        &self,
        agent_role: &str,
    ) -> Result<usize, PersistenceError>;
}

#[async_trait]
pub trait LoraAdapterRepository: Send + Sync {
    async fn insert_lora_adapter(&self, adapter: &LoraAdapter) -> Result<(), PersistenceError>;
    async fn get_lora_adapter(&self, id: &str) -> Result<Option<LoraAdapter>, PersistenceError>;
    async fn list_lora_adapters_by_kind(
        &self,
        task_kind: &str,
    ) -> Result<Vec<LoraAdapter>, PersistenceError>;
    async fn list_lora_adapters_by_project(
        &self,
        project_id: &str,
    ) -> Result<Vec<LoraAdapter>, PersistenceError>;
    async fn list_lora_adapters_by_role(
        &self,
        agent_role: &str,
    ) -> Result<Vec<LoraAdapter>, PersistenceError>;
    async fn set_lora_adapter_active(&self, id: &str, active: bool)
    -> Result<(), PersistenceError>;
}

#[async_trait]
pub trait PromptVersionRepository: Send + Sync {
    async fn insert_prompt_version(&self, version: &PromptVersion) -> Result<(), PersistenceError>;
    async fn update_prompt_version(&self, version: &PromptVersion) -> Result<(), PersistenceError>;
    async fn get_prompt_version(&self, id: &str)
    -> Result<Option<PromptVersion>, PersistenceError>;
    async fn list_prompt_versions_by_agent(
        &self,
        agent: &str,
    ) -> Result<Vec<PromptVersion>, PersistenceError>;
    async fn get_active_prompt_version(
        &self,
        agent: &str,
    ) -> Result<Option<PromptVersion>, PersistenceError>;
    async fn set_active_prompt_version(
        &self,
        id: &str,
        agent: &str,
    ) -> Result<(), PersistenceError>;
}

#[async_trait]
pub trait ProjectSnapshotRepository: Send + Sync {
    async fn insert_project_snapshot(
        &self,
        snapshot: &ProjectSnapshot,
    ) -> Result<(), PersistenceError>;
    async fn get_project_snapshot(
        &self,
        id: &str,
    ) -> Result<Option<ProjectSnapshot>, PersistenceError>;
    async fn list_project_snapshots(
        &self,
        project_id: &str,
    ) -> Result<Vec<ProjectSnapshot>, PersistenceError>;
}

#[async_trait]
pub trait MemoryEntryRepository: Send + Sync {
    async fn insert_memory_entry(&self, entry: &MemoryEntry) -> Result<(), PersistenceError>;
    async fn list_memory_entries(
        &self,
        project_id: Option<&str>,
        kind: Option<&str>,
        limit: usize,
    ) -> Result<Vec<MemoryEntry>, PersistenceError>;
    async fn list_memory_entries_by_session(
        &self,
        session_id: &str,
    ) -> Result<Vec<MemoryEntry>, PersistenceError>;
}

#[async_trait]
pub trait TrainingJobRepository: Send + Sync {
    async fn insert_training_job(&self, job: &TrainingJob) -> Result<(), PersistenceError>;
    async fn update_training_job(&self, job: &TrainingJob) -> Result<(), PersistenceError>;
    async fn get_training_job(&self, id: &str) -> Result<Option<TrainingJob>, PersistenceError>;
    async fn list_training_jobs_by_kind(
        &self,
        task_kind: &str,
    ) -> Result<Vec<TrainingJob>, PersistenceError>;
}

#[async_trait]
pub trait BenchmarkResultRepository: Send + Sync {
    async fn insert_run(&self, run: &BenchmarkRun) -> Result<(), PersistenceError>;
    async fn get_run(&self, id: &str) -> Result<Option<BenchmarkRun>, PersistenceError>;
    async fn list_runs(&self, limit: usize) -> Result<Vec<BenchmarkRunSummary>, PersistenceError>;
    async fn insert_result(
        &self,
        run_id: &str,
        result: &BenchmarkResult,
    ) -> Result<(), PersistenceError>;
    async fn list_results(&self, run_id: &str) -> Result<Vec<BenchmarkResult>, PersistenceError>;
}

pub trait Persistence:
    ProjectRepository
    + TaskRepository
    + ArtifactRepository
    + LogRepository
    + ExperienceRepository
    + TrainingExampleRepository
    + LoraAdapterRepository
    + PromptVersionRepository
    + ProjectSnapshotRepository
    + MemoryEntryRepository
    + TrainingJobRepository
    + BenchmarkResultRepository
{
}

impl<T> Persistence for T where
    T: ProjectRepository
        + TaskRepository
        + ArtifactRepository
        + LogRepository
        + ExperienceRepository
        + TrainingExampleRepository
        + LoraAdapterRepository
        + PromptVersionRepository
        + ProjectSnapshotRepository
        + MemoryEntryRepository
        + TrainingJobRepository
        + BenchmarkResultRepository
{
}

/// In-memory implementation of [`TaskRepository`] for tests.
#[derive(Default)]
pub struct MemoryTaskRepository {
    tasks: std::sync::Mutex<std::collections::HashMap<String, Task>>,
    deps: std::sync::Mutex<Vec<TaskDependency>>,
    experiences: std::sync::Mutex<std::collections::HashMap<String, Vec<Experience>>>,
    prompt_versions: std::sync::Mutex<std::collections::HashMap<String, PromptVersion>>,
    training_jobs: std::sync::Mutex<std::collections::HashMap<String, TrainingJob>>,
    benchmark_runs: std::sync::Mutex<std::collections::HashMap<String, BenchmarkRun>>,
    benchmark_results: std::sync::Mutex<std::collections::HashMap<String, Vec<BenchmarkResult>>>,
}

impl MemoryTaskRepository {
    pub fn new() -> Self {
        Self::default()
    }

    fn lock_guard<T>(
        mutex: &std::sync::Mutex<T>,
    ) -> Result<std::sync::MutexGuard<'_, T>, PersistenceError> {
        mutex
            .lock()
            .map_err(|_| PersistenceError::Database("mutex poisoned".into()))
    }
}

#[async_trait]
impl TaskRepository for MemoryTaskRepository {
    async fn insert_task(&self, task: &Task) -> Result<(), PersistenceError> {
        Self::lock_guard(&self.tasks)?.insert(task.id.clone(), task.clone());
        Ok(())
    }

    async fn update_task(&self, task: &Task) -> Result<(), PersistenceError> {
        Self::lock_guard(&self.tasks)?.insert(task.id.clone(), task.clone());
        Ok(())
    }

    async fn update_task_status(
        &self,
        id: &str,
        status: TaskStatus,
        result: Option<Value>,
    ) -> Result<(), PersistenceError> {
        let mut tasks = Self::lock_guard(&self.tasks)?;
        let task = tasks
            .get_mut(id)
            .ok_or_else(|| PersistenceError::Database(id.into()))?;
        task.status = status;
        task.result = result;
        Ok(())
    }

    async fn get_task(&self, id: &str) -> Result<Option<Task>, PersistenceError> {
        Ok(Self::lock_guard(&self.tasks)?.get(id).cloned())
    }

    async fn list_tasks_by_project(&self, project_id: &str) -> Result<Vec<Task>, PersistenceError> {
        Ok(Self::lock_guard(&self.tasks)?
            .values()
            .filter(|t| t.project_id == project_id)
            .cloned()
            .collect())
    }

    async fn list_all_tasks(&self) -> Result<Vec<Task>, PersistenceError> {
        Ok(Self::lock_guard(&self.tasks)?.values().cloned().collect())
    }

    async fn list_ready_tasks(&self) -> Result<Vec<Task>, PersistenceError> {
        Ok(vec![])
    }

    async fn add_dependency(&self, dep: &TaskDependency) -> Result<(), PersistenceError> {
        Self::lock_guard(&self.deps)?.push(dep.clone());
        Ok(())
    }
}

#[async_trait]
impl ExperienceRepository for MemoryTaskRepository {
    async fn insert_experience(&self, exp: &Experience) -> Result<(), PersistenceError> {
        Self::lock_guard(&self.experiences)?
            .entry(exp.task_id.clone())
            .or_default()
            .push(exp.clone());
        Ok(())
    }

    async fn list_experiences_by_task(
        &self,
        task_id: &str,
    ) -> Result<Vec<Experience>, PersistenceError> {
        Ok(Self::lock_guard(&self.experiences)?
            .get(task_id)
            .cloned()
            .unwrap_or_default())
    }

    async fn list_experiences_by_prompt_version(
        &self,
        prompt_version_id: &str,
    ) -> Result<Vec<Experience>, PersistenceError> {
        Ok(Self::lock_guard(&self.experiences)?
            .values()
            .flat_map(|v| v.iter())
            .filter(|e| e.prompt_version_id.as_deref() == Some(prompt_version_id))
            .cloned()
            .collect())
    }
}

#[async_trait]
impl PromptVersionRepository for MemoryTaskRepository {
    async fn insert_prompt_version(&self, version: &PromptVersion) -> Result<(), PersistenceError> {
        Self::lock_guard(&self.prompt_versions)?.insert(version.id.clone(), version.clone());
        Ok(())
    }

    async fn update_prompt_version(&self, version: &PromptVersion) -> Result<(), PersistenceError> {
        Self::lock_guard(&self.prompt_versions)?.insert(version.id.clone(), version.clone());
        Ok(())
    }

    async fn get_prompt_version(
        &self,
        id: &str,
    ) -> Result<Option<PromptVersion>, PersistenceError> {
        Ok(Self::lock_guard(&self.prompt_versions)?.get(id).cloned())
    }

    async fn list_prompt_versions_by_agent(
        &self,
        agent: &str,
    ) -> Result<Vec<PromptVersion>, PersistenceError> {
        Ok(Self::lock_guard(&self.prompt_versions)?
            .values()
            .filter(|v| v.agent == agent)
            .cloned()
            .collect())
    }

    async fn get_active_prompt_version(
        &self,
        agent: &str,
    ) -> Result<Option<PromptVersion>, PersistenceError> {
        Ok(Self::lock_guard(&self.prompt_versions)?
            .values()
            .find(|v| v.agent == agent && v.active)
            .cloned())
    }

    async fn set_active_prompt_version(
        &self,
        id: &str,
        agent: &str,
    ) -> Result<(), PersistenceError> {
        let mut versions = Self::lock_guard(&self.prompt_versions)?;
        for v in versions.values_mut() {
            if v.agent == agent {
                v.active = false;
            }
        }
        if let Some(v) = versions.get_mut(id) {
            v.active = true;
        }
        Ok(())
    }
}

#[async_trait]
impl TrainingJobRepository for MemoryTaskRepository {
    async fn insert_training_job(&self, job: &TrainingJob) -> Result<(), PersistenceError> {
        Self::lock_guard(&self.training_jobs)?.insert(job.id.clone(), job.clone());
        Ok(())
    }

    async fn update_training_job(&self, job: &TrainingJob) -> Result<(), PersistenceError> {
        Self::lock_guard(&self.training_jobs)?.insert(job.id.clone(), job.clone());
        Ok(())
    }

    async fn get_training_job(&self, id: &str) -> Result<Option<TrainingJob>, PersistenceError> {
        Ok(Self::lock_guard(&self.training_jobs)?.get(id).cloned())
    }

    async fn list_training_jobs_by_kind(
        &self,
        task_kind: &str,
    ) -> Result<Vec<TrainingJob>, PersistenceError> {
        Ok(Self::lock_guard(&self.training_jobs)?
            .values()
            .filter(|j| j.task_kind == task_kind)
            .cloned()
            .collect())
    }
}

#[async_trait]
impl BenchmarkResultRepository for MemoryTaskRepository {
    async fn insert_run(&self, run: &BenchmarkRun) -> Result<(), PersistenceError> {
        Self::lock_guard(&self.benchmark_runs)?.insert(run.summary.id.clone(), run.clone());
        Ok(())
    }

    async fn get_run(&self, id: &str) -> Result<Option<BenchmarkRun>, PersistenceError> {
        Ok(Self::lock_guard(&self.benchmark_runs)?.get(id).cloned())
    }

    async fn list_runs(&self, limit: usize) -> Result<Vec<BenchmarkRunSummary>, PersistenceError> {
        let runs = Self::lock_guard(&self.benchmark_runs)?;
        let mut summaries: Vec<_> = runs.values().map(|r| r.summary.clone()).collect();
        summaries.sort_by(|a, b| b.id.cmp(&a.id));
        summaries.truncate(limit);
        Ok(summaries)
    }

    async fn insert_result(
        &self,
        run_id: &str,
        result: &BenchmarkResult,
    ) -> Result<(), PersistenceError> {
        Self::lock_guard(&self.benchmark_results)?
            .entry(run_id.into())
            .or_default()
            .push(result.clone());
        Ok(())
    }

    async fn list_results(&self, run_id: &str) -> Result<Vec<BenchmarkResult>, PersistenceError> {
        Ok(Self::lock_guard(&self.benchmark_results)?
            .get(run_id)
            .cloned()
            .unwrap_or_default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Task, TaskStatus};

    fn sample_task() -> Task {
        Task {
            id: "t1".into(),
            project_id: "p1".into(),
            parent_id: None,
            title: "task".into(),
            description: None,
            kind: "codegen".into(),
            status: TaskStatus::Pending,
            assigned_agent: None,
            priority: 0,
            payload: serde_json::Value::Null,
            result: None,
            created_at: 0,
            started_at: None,
            finished_at: None,
            iteration_count: 0,
            priority_score: 0.0,
            critic_score: None,
            human_score: None,
            prompt_version_id: None,
            lora_adapter_id: None,
            trace_id: "trace-1".into(),
        }
    }

    #[tokio::test]
    async fn update_task_persists_critic_score_and_iteration_count() {
        let repo = MemoryTaskRepository::new();
        let mut task = sample_task();
        repo.insert_task(&task).await.unwrap();

        task.critic_score = Some(4.2);
        task.iteration_count = 3;
        repo.update_task(&task).await.unwrap();

        let loaded = repo.get_task("t1").await.unwrap().expect("task exists");
        assert_eq!(loaded.critic_score, Some(4.2));
        assert_eq!(loaded.iteration_count, 3);
    }
}
