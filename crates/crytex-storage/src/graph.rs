use chrono::Utc;
use crytex_core::metrics::MetricsSnapshot;
use crytex_core::models::{
    AgentLog, Artifact, BenchmarkResult, BenchmarkRun, BenchmarkRunSummary, BenchmarkVariant,
    Experience, LoraAdapter, MemoryEntry, Project, ProjectSnapshot, PromptVersion, Task,
    TaskDependency, TaskStatus, TrainingExample, TrainingJob, TrainingJobStatus,
};
use rusqlite::Connection;
use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::Mutex;
use ulid::Ulid;

const CURRENT_SCHEMA_VERSION: u32 = 14;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("serde error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("unknown task status: {0}")]
    InvalidTaskStatus(String),
}

#[derive(Clone)]
pub struct GraphStore {
    conn: Arc<Mutex<Connection>>,
}

fn map_project_row(row: &rusqlite::Row<'_>) -> Result<Project, Error> {
    Ok(Project {
        id: row.get(0)?,
        name: row.get(1)?,
        root_path: row.get(2)?,
        created_at: row.get(3)?,
        updated_at: row.get(4)?,
        metadata: serde_json::from_str(&row.get::<_, String>(5)?)?,
    })
}

fn map_lora_adapter_row(row: &rusqlite::Row<'_>) -> Result<LoraAdapter, Error> {
    let task_kind: Option<String> = row.get(5)?;
    let agent_role: Option<String> = row.get(9)?;
    Ok(LoraAdapter {
        id: row.get(0)?,
        project_id: row.get(1)?,
        name: row.get(2)?,
        file_path: row.get(3)?,
        base_model: row.get(4)?,
        task_kind: task_kind.filter(|s| !s.is_empty()),
        agent_role: agent_role.filter(|s| !s.is_empty()),
        metrics: serde_json::from_str(&row.get::<_, String>(6)?)?,
        created_at: row.get(7)?,
        active: row.get::<_, i32>(8)? != 0,
    })
}

fn map_task_row(row: &rusqlite::Row<'_>) -> Result<Task, Error> {
    let status_str: String = row.get(6)?;
    Ok(Task {
        id: row.get(0)?,
        project_id: row.get(1)?,
        parent_id: as_optional_string(row.get(2)?),
        title: row.get(3)?,
        description: as_optional_string(row.get(4)?),
        kind: row.get(5)?,
        status: TaskStatus::from_str(&status_str).map_err(|_| {
            rusqlite::Error::InvalidColumnType(
                6,
                "TaskStatus".to_string(),
                rusqlite::types::Type::Text,
            )
        })?,
        assigned_agent: as_optional_string(row.get(7)?),
        priority: row.get(8)?,
        created_at: row.get(9)?,
        started_at: row.get(10)?,
        finished_at: row.get(11)?,
        payload: serde_json::from_str(&row.get::<_, String>(12)?)?,
        result: row
            .get::<_, Option<String>>(13)?
            .filter(|s| !s.is_empty())
            .map(|s| serde_json::from_str(&s))
            .transpose()?,
        iteration_count: row.get::<_, Option<i64>>(14)?.unwrap_or(0) as u32,
        priority_score: row.get::<_, Option<f64>>(15)?.unwrap_or(0.0),
        critic_score: row.get(16)?,
        human_score: row.get(17)?,
        prompt_version_id: as_optional_string(row.get(18)?),
        lora_adapter_id: as_optional_string(row.get(19)?),
        trace_id: row.get::<_, Option<String>>(20)?.unwrap_or_default(),
    })
}

fn as_optional_string(value: Option<String>) -> Option<String> {
    value.filter(|s| !s.is_empty())
}

impl GraphStore {
    pub async fn new(db_path: &str) -> Result<Self, Error> {
        let conn = Connection::open(db_path)?;
        conn.execute("PRAGMA foreign_keys = ON", [])?;
        let store = Self {
            conn: Arc::new(Mutex::new(conn)),
        };
        store.init_schema().await?;
        Ok(store)
    }

    async fn migrate_trace_id(conn: &Connection) -> Result<(), Error> {
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM pragma_table_info('tasks') WHERE name = 'trace_id'",
            [],
            |row| row.get(0),
        )?;
        if count == 0 {
            conn.execute("ALTER TABLE tasks ADD COLUMN trace_id TEXT DEFAULT ''", [])?;
        }
        Ok(())
    }

    async fn migrate_task_recovery_columns(conn: &Connection) -> Result<(), Error> {
        for (column, ddl) in [
            (
                "iteration_count",
                "ALTER TABLE tasks ADD COLUMN iteration_count INTEGER DEFAULT 0",
            ),
            (
                "priority_score",
                "ALTER TABLE tasks ADD COLUMN priority_score REAL DEFAULT 0.0",
            ),
            (
                "critic_score",
                "ALTER TABLE tasks ADD COLUMN critic_score REAL",
            ),
            (
                "human_score",
                "ALTER TABLE tasks ADD COLUMN human_score REAL",
            ),
            (
                "prompt_version_id",
                "ALTER TABLE tasks ADD COLUMN prompt_version_id TEXT",
            ),
            (
                "lora_adapter_id",
                "ALTER TABLE tasks ADD COLUMN lora_adapter_id TEXT",
            ),
        ] {
            let count: i64 = conn.query_row(
                "SELECT COUNT(*) FROM pragma_table_info('tasks') WHERE name = ?1",
                [column],
                |row| row.get(0),
            )?;
            if count == 0 {
                conn.execute(ddl, [])?;
            }
        }
        Ok(())
    }

    async fn migrate_lora_task_kind(conn: &Connection) -> Result<(), Error> {
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM pragma_table_info('lora_adapters') WHERE name = 'task_kind'",
            [],
            |row| row.get(0),
        )?;
        if count == 0 {
            conn.execute("ALTER TABLE lora_adapters ADD COLUMN task_kind TEXT", [])?;
        }
        Ok(())
    }

    async fn migrate_lora_agent_role(conn: &Connection) -> Result<(), Error> {
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM pragma_table_info('lora_adapters') WHERE name = 'agent_role'",
            [],
            |row| row.get(0),
        )?;
        if count == 0 {
            conn.execute("ALTER TABLE lora_adapters ADD COLUMN agent_role TEXT", [])?;
        }
        Ok(())
    }

    async fn migrate_training_example_agent_role(conn: &Connection) -> Result<(), Error> {
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM pragma_table_info('training_examples') WHERE name = 'agent_role'",
            [],
            |row| row.get(0),
        )?;
        if count == 0 {
            conn.execute(
                "ALTER TABLE training_examples ADD COLUMN agent_role TEXT",
                [],
            )?;
        }
        Ok(())
    }

    async fn migrate_training_example_dataset_fields(conn: &Connection) -> Result<(), Error> {
        for (column, ddl) in [
            (
                "model_id",
                "ALTER TABLE training_examples ADD COLUMN model_id TEXT",
            ),
            (
                "rag_evidence_ids",
                "ALTER TABLE training_examples ADD COLUMN rag_evidence_ids TEXT DEFAULT '[]'",
            ),
            (
                "accepted_output",
                "ALTER TABLE training_examples ADD COLUMN accepted_output TEXT",
            ),
            (
                "rejected_output",
                "ALTER TABLE training_examples ADD COLUMN rejected_output TEXT",
            ),
            (
                "critic_feedback",
                "ALTER TABLE training_examples ADD COLUMN critic_feedback TEXT",
            ),
            (
                "failure_type",
                "ALTER TABLE training_examples ADD COLUMN failure_type TEXT",
            ),
        ] {
            let count: i64 = conn.query_row(
                "SELECT COUNT(*) FROM pragma_table_info('training_examples') WHERE name = ?1",
                [column],
                |row| row.get(0),
            )?;
            if count == 0 {
                conn.execute(ddl, [])?;
            }
        }
        Ok(())
    }

    async fn migrate_prompt_version_metrics(conn: &Connection) -> Result<(), Error> {
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM pragma_table_info('prompt_versions') WHERE name = 'metrics'",
            [],
            |row| row.get(0),
        )?;
        if count == 0 {
            conn.execute(
                "ALTER TABLE prompt_versions ADD COLUMN metrics TEXT DEFAULT '{}'",
                [],
            )?;
        }
        Ok(())
    }

    async fn init_schema(&self) -> Result<(), Error> {
        let conn = self.conn.lock().await;
        conn.execute_batch(
            r#"
CREATE TABLE IF NOT EXISTS projects (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    root_path   TEXT NOT NULL,
    created_at  INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL,
    metadata    TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS tasks (
    id          TEXT PRIMARY KEY,
    project_id  TEXT NOT NULL REFERENCES projects(id),
    parent_id   TEXT,
    title       TEXT NOT NULL,
    description TEXT,
    kind        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    assigned_agent TEXT,
    priority    INTEGER DEFAULT 0,
    created_at  INTEGER NOT NULL,
    started_at  INTEGER,
    finished_at INTEGER,
    payload     TEXT DEFAULT '{}',
    result      TEXT,
    iteration_count INTEGER DEFAULT 0,
    priority_score REAL DEFAULT 0.0,
    critic_score REAL,
    human_score REAL,
    prompt_version_id TEXT,
    lora_adapter_id TEXT,
    trace_id TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS task_dependencies (
    task_id       TEXT NOT NULL REFERENCES tasks(id),
    depends_on    TEXT NOT NULL REFERENCES tasks(id),
    dep_type      TEXT NOT NULL DEFAULT 'sequential',
    PRIMARY KEY (task_id, depends_on)
);

CREATE TABLE IF NOT EXISTS artifacts (
    id          TEXT PRIMARY KEY,
    task_id     TEXT NOT NULL REFERENCES tasks(id),
    file_path   TEXT NOT NULL,
    file_type   TEXT NOT NULL,
    commit_sha  TEXT,
    created_at  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS lora_adapters (
    id          TEXT PRIMARY KEY,
    project_id  TEXT REFERENCES projects(id),
    name        TEXT NOT NULL,
    file_path   TEXT NOT NULL,
    base_model  TEXT NOT NULL,
    task_kind   TEXT,
    metrics     TEXT DEFAULT '{}',
    created_at  INTEGER NOT NULL,
    active      INTEGER DEFAULT 0,
    agent_role  TEXT
);

CREATE TABLE IF NOT EXISTS prompt_versions (
    id          TEXT PRIMARY KEY,
    agent       TEXT NOT NULL,
    project_id  TEXT REFERENCES projects(id),
    system_prompt TEXT NOT NULL,
    fitness     REAL,
    parent_id   TEXT REFERENCES prompt_versions(id),
    metrics     TEXT DEFAULT '{}',
    created_at  INTEGER NOT NULL,
    active      INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS project_snapshots (
    id          TEXT PRIMARY KEY,
    project_id  TEXT NOT NULL REFERENCES projects(id),
    name        TEXT NOT NULL,
    state_json  TEXT NOT NULL,
    created_at  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_log (
    id          TEXT PRIMARY KEY,
    project_id  TEXT REFERENCES projects(id),
    task_id     TEXT REFERENCES tasks(id),
    agent       TEXT NOT NULL,
    action      TEXT NOT NULL,
    message     TEXT,
    level       TEXT NOT NULL DEFAULT 'info',
    timestamp   INTEGER NOT NULL,
    metadata    TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS metrics (
    id          TEXT PRIMARY KEY,
    timestamp   INTEGER NOT NULL,
    snapshot    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiences (
    id                TEXT PRIMARY KEY,
    task_id           TEXT NOT NULL REFERENCES tasks(id),
    project_id        TEXT,
    prompt_version_id TEXT,
    text              TEXT,
    critic_score      REAL,
    human_score       REAL,
    reward            REAL NOT NULL,
    comment           TEXT,
    created_at        INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_experiences_task ON experiences(task_id);
CREATE INDEX IF NOT EXISTS idx_experiences_prompt_version ON experiences(prompt_version_id);
CREATE INDEX IF NOT EXISTS idx_experiences_project ON experiences(project_id);

CREATE TABLE IF NOT EXISTS training_examples (
    id                TEXT PRIMARY KEY,
    task_id           TEXT NOT NULL REFERENCES tasks(id),
    project_id        TEXT REFERENCES projects(id),
    prompt_version_id TEXT,
    task_kind         TEXT NOT NULL,
    input_text        TEXT NOT NULL,
    output_text       TEXT NOT NULL,
    reward            REAL NOT NULL,
    created_at        INTEGER NOT NULL,
    agent_role        TEXT,
    model_id          TEXT,
    rag_evidence_ids  TEXT DEFAULT '[]',
    accepted_output   TEXT,
    rejected_output   TEXT,
    critic_feedback   TEXT,
    failure_type      TEXT
);
CREATE INDEX IF NOT EXISTS idx_training_examples_kind ON training_examples(task_kind);
CREATE INDEX IF NOT EXISTS idx_training_examples_project ON training_examples(project_id);
CREATE INDEX IF NOT EXISTS idx_training_examples_role ON training_examples(agent_role);

CREATE TABLE IF NOT EXISTS training_jobs (
    id          TEXT PRIMARY KEY,
    task_kind   TEXT NOT NULL,
    status      TEXT NOT NULL,
    started_at  INTEGER NOT NULL,
    finished_at INTEGER,
    adapter_id  TEXT,
    metrics     TEXT DEFAULT '{}',
    error_message TEXT
);
CREATE INDEX IF NOT EXISTS idx_training_jobs_kind ON training_jobs(task_kind);
CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON training_jobs(status);

CREATE TABLE IF NOT EXISTS memory_entries (
    id          TEXT PRIMARY KEY,
    project_id  TEXT,
    session_id  TEXT,
    kind        TEXT NOT NULL,
    text        TEXT NOT NULL,
    metadata    TEXT DEFAULT '{}',
    created_at  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_entries_project ON memory_entries(project_id);
CREATE INDEX IF NOT EXISTS idx_memory_entries_session ON memory_entries(session_id);
CREATE INDEX IF NOT EXISTS idx_memory_entries_kind ON memory_entries(kind);

CREATE TABLE IF NOT EXISTS benchmark_runs (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    project_id      TEXT REFERENCES projects(id),
    golden_set_path TEXT NOT NULL,
    variant_name    TEXT NOT NULL,
    agent_role      TEXT,
    lora_adapter_id TEXT,
    prompt_version_id TEXT,
    backend_id      TEXT,
    scorer_kind     TEXT NOT NULL,
    started_at      INTEGER NOT NULL,
    finished_at     INTEGER,
    pass_count      INTEGER NOT NULL,
    fail_count      INTEGER NOT NULL,
    total_cases     INTEGER NOT NULL,
    pass_rate       REAL NOT NULL,
    mean_latency_ms REAL,
    total_tokens    INTEGER,
    metadata        TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_benchmark_runs_project ON benchmark_runs(project_id);
CREATE INDEX IF NOT EXISTS idx_benchmark_runs_name ON benchmark_runs(name);

CREATE TABLE IF NOT EXISTS benchmark_results (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES benchmark_runs(id) ON DELETE CASCADE,
    case_id         TEXT NOT NULL,
    case_input      TEXT NOT NULL,
    expected        TEXT,
    actual          TEXT,
    passed          INTEGER NOT NULL,
    score_value     REAL NOT NULL,
    latency_ms      INTEGER,
    token_usage     TEXT,
    explanation     TEXT,
    metadata        TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_benchmark_results_run ON benchmark_results(run_id);
CREATE INDEX IF NOT EXISTS idx_benchmark_results_case ON benchmark_results(case_id);

CREATE TABLE IF NOT EXISTS ab_test_reports (
    id              TEXT PRIMARY KEY,
    baseline_run_id TEXT NOT NULL REFERENCES benchmark_runs(id),
    challenger_run_id TEXT NOT NULL REFERENCES benchmark_runs(id),
    created_at      INTEGER NOT NULL,
    significance_level REAL NOT NULL,
    baseline_pass_rate REAL NOT NULL,
    challenger_pass_rate REAL NOT NULL,
    delta_pass_rate REAL NOT NULL,
    mc_nemar_p_value REAL NOT NULL,
    winner          TEXT NOT NULL,
    metadata        TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_ab_test_baseline ON ab_test_reports(baseline_run_id);
CREATE INDEX IF NOT EXISTS idx_ab_test_challenger ON ab_test_reports(challenger_run_id);
"#,
        )?;
        Self::migrate_trace_id(&conn).await?;
        Self::migrate_task_recovery_columns(&conn).await?;
        Self::migrate_lora_task_kind(&conn).await?;
        Self::migrate_lora_agent_role(&conn).await?;
        Self::migrate_training_example_agent_role(&conn).await?;
        Self::migrate_training_example_dataset_fields(&conn).await?;
        Self::migrate_prompt_version_metrics(&conn).await?;
        conn.pragma_update(None, "user_version", CURRENT_SCHEMA_VERSION)?;
        Ok(())
    }

    pub async fn schema_version(&self) -> Result<u32, Error> {
        let conn = self.conn.lock().await;
        let version = conn.pragma_query_value(None, "user_version", |row| row.get(0))?;
        Ok(version)
    }

    pub async fn insert_project(&self, project: &Project) -> Result<(), Error> {
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO projects (id, name, root_path, created_at, updated_at, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                root_path = excluded.root_path,
                updated_at = excluded.updated_at,
                metadata = excluded.metadata",
            [
                &project.id,
                &project.name,
                &project.root_path,
                &project.created_at.to_string(),
                &project.updated_at.to_string(),
                &project.metadata.to_string(),
            ],
        )?;
        Ok(())
    }

    pub async fn list_projects(&self) -> Result<Vec<Project>, Error> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, name, root_path, created_at, updated_at, metadata FROM projects",
        )?;
        let mut rows = stmt.query([])?;
        let mut projects = Vec::new();
        while let Some(row) = rows.next()? {
            projects.push(map_project_row(row)?);
        }
        Ok(projects)
    }

    pub async fn get_project(&self, id: &str) -> Result<Option<Project>, Error> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, name, root_path, created_at, updated_at, metadata FROM projects WHERE id = ?1",
        )?;
        let mut rows = stmt.query([id])?;
        match rows.next()? {
            Some(row) => Ok(Some(map_project_row(row)?)),
            None => Ok(None),
        }
    }

    pub async fn insert_task(&self, task: &Task) -> Result<(), Error> {
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO tasks (
                id, project_id, parent_id, title, description, kind, status,
                assigned_agent, priority, created_at, started_at, finished_at,
                payload, result, iteration_count, priority_score, critic_score,
                human_score, prompt_version_id, lora_adapter_id, trace_id
             )
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21)
             ON CONFLICT(id) DO UPDATE SET
                project_id = excluded.project_id,
                parent_id = excluded.parent_id,
                title = excluded.title,
                description = excluded.description,
                kind = excluded.kind,
                status = excluded.status,
                assigned_agent = excluded.assigned_agent,
                priority = excluded.priority,
                started_at = excluded.started_at,
                finished_at = excluded.finished_at,
                payload = excluded.payload,
                result = excluded.result,
                iteration_count = excluded.iteration_count,
                priority_score = excluded.priority_score,
                critic_score = excluded.critic_score,
                human_score = excluded.human_score,
                prompt_version_id = excluded.prompt_version_id,
                lora_adapter_id = excluded.lora_adapter_id,
                trace_id = excluded.trace_id",
            [
                &task.id as &dyn rusqlite::ToSql,
                &task.project_id as &dyn rusqlite::ToSql,
                &task.parent_id.as_deref() as &dyn rusqlite::ToSql,
                &task.title as &dyn rusqlite::ToSql,
                &task.description.as_deref() as &dyn rusqlite::ToSql,
                &task.kind as &dyn rusqlite::ToSql,
                &task.status.as_str() as &dyn rusqlite::ToSql,
                &task.assigned_agent.as_deref() as &dyn rusqlite::ToSql,
                &task.priority as &dyn rusqlite::ToSql,
                &task.created_at as &dyn rusqlite::ToSql,
                &task.started_at as &dyn rusqlite::ToSql,
                &task.finished_at as &dyn rusqlite::ToSql,
                &task.payload.to_string() as &dyn rusqlite::ToSql,
                &task.result.as_ref().map(|v| v.to_string()) as &dyn rusqlite::ToSql,
                &task.iteration_count as &dyn rusqlite::ToSql,
                &task.priority_score as &dyn rusqlite::ToSql,
                &task.critic_score as &dyn rusqlite::ToSql,
                &task.human_score as &dyn rusqlite::ToSql,
                &task.prompt_version_id.as_deref() as &dyn rusqlite::ToSql,
                &task.lora_adapter_id.as_deref() as &dyn rusqlite::ToSql,
                &task.trace_id as &dyn rusqlite::ToSql,
            ],
        )?;
        Ok(())
    }

    pub async fn update_task_status(
        &self,
        id: &str,
        status: TaskStatus,
        result: Option<serde_json::Value>,
    ) -> Result<(), Error> {
        let now = Utc::now().timestamp();
        let conn = self.conn.lock().await;
        match status {
            TaskStatus::InProgress => {
                conn.execute(
                    "UPDATE tasks SET status = ?1, started_at = ?2 WHERE id = ?3",
                    [status.as_str(), &now.to_string(), id],
                )?;
            }
            TaskStatus::Completed
            | TaskStatus::Done
            | TaskStatus::Failed
            | TaskStatus::Cancelled => {
                conn.execute(
                    "UPDATE tasks SET status = ?1, finished_at = ?2, result = ?3 WHERE id = ?4",
                    [
                        status.as_str(),
                        &now.to_string(),
                        &result.map(|v| v.to_string()).unwrap_or_default(),
                        id,
                    ],
                )?;
            }
            TaskStatus::Review => {
                conn.execute(
                    "UPDATE tasks SET status = ?1, started_at = NULL, finished_at = NULL, result = ?2 WHERE id = ?3",
                    [
                        status.as_str(),
                        &result.map(|v| v.to_string()).unwrap_or_default(),
                        id,
                    ],
                )?;
            }
            TaskStatus::Backlog
            | TaskStatus::Ready
            | TaskStatus::Pending
            | TaskStatus::Remediation
            | TaskStatus::Blocked => {
                conn.execute(
                    "UPDATE tasks SET status = ?1, started_at = NULL, finished_at = NULL, result = NULL WHERE id = ?2",
                    [status.as_str(), id],
                )?;
            }
        }
        Ok(())
    }

    pub async fn get_task(&self, id: &str) -> Result<Option<Task>, Error> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, project_id, parent_id, title, description, kind, status, assigned_agent, priority, created_at, started_at, finished_at, payload, result, iteration_count, priority_score, critic_score, human_score, prompt_version_id, lora_adapter_id, trace_id FROM tasks WHERE id = ?1",
        )?;
        let mut rows = stmt.query([id])?;
        match rows.next()? {
            Some(row) => Ok(Some(map_task_row(row)?)),
            None => Ok(None),
        }
    }

    pub async fn list_tasks_by_project(&self, project_id: &str) -> Result<Vec<Task>, Error> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, project_id, parent_id, title, description, kind, status, assigned_agent, priority, created_at, started_at, finished_at, payload, result, iteration_count, priority_score, critic_score, human_score, prompt_version_id, lora_adapter_id, trace_id FROM tasks WHERE project_id = ?1",
        )?;
        let mut rows = stmt.query([project_id])?;
        let mut tasks = Vec::new();
        while let Some(row) = rows.next()? {
            tasks.push(map_task_row(row)?);
        }
        Ok(tasks)
    }

    pub async fn list_ready_tasks(&self) -> Result<Vec<Task>, Error> {
        let pending = {
            let conn = self.conn.lock().await;
            let mut stmt = conn.prepare(
                "SELECT id, project_id, parent_id, title, description, kind, status, assigned_agent, priority, created_at, started_at, finished_at, payload, result, iteration_count, priority_score, critic_score, human_score, prompt_version_id, lora_adapter_id, trace_id FROM tasks WHERE status IN ('pending', 'ready')",
            )?;
            let mut rows = stmt.query([])?;
            let mut tasks = Vec::new();
            while let Some(row) = rows.next()? {
                tasks.push(map_task_row(row)?);
            }
            tasks
        };

        let mut ready = Vec::new();
        for task in pending {
            let deps_finished = self.are_dependencies_finished(&task.id).await?;
            if deps_finished {
                ready.push(task);
            }
        }
        Ok(ready)
    }

    async fn are_dependencies_finished(&self, task_id: &str) -> Result<bool, Error> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT COUNT(*) FROM task_dependencies td
             JOIN tasks t ON td.depends_on = t.id
             WHERE td.task_id = ?1 AND t.status != 'completed'",
        )?;
        let count: i64 = stmt.query_row([task_id], |row| row.get(0))?;
        Ok(count == 0)
    }

    pub async fn add_dependency(&self, dep: &TaskDependency) -> Result<(), Error> {
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO task_dependencies (task_id, depends_on, dep_type)
             VALUES (?1, ?2, ?3)
             ON CONFLICT(task_id, depends_on) DO NOTHING",
            [&dep.task_id, &dep.depends_on, &dep.dep_type],
        )?;
        Ok(())
    }

    pub async fn list_dependencies(&self) -> Result<Vec<TaskDependency>, Error> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT task_id, depends_on, dep_type
             FROM task_dependencies",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(TaskDependency {
                task_id: row.get(0)?,
                depends_on: row.get(1)?,
                dep_type: row.get(2)?,
            })
        })?;
        let mut deps = Vec::new();
        for dep in rows {
            deps.push(dep?);
        }
        Ok(deps)
    }

    pub async fn insert_artifact(&self, artifact: &Artifact) -> Result<(), Error> {
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO artifacts (id, task_id, file_path, file_type, commit_sha, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(id) DO UPDATE SET
                file_path = excluded.file_path,
                file_type = excluded.file_type,
                commit_sha = excluded.commit_sha",
            [
                &artifact.id,
                &artifact.task_id,
                &artifact.file_path,
                &artifact.file_type,
                &artifact
                    .commit_sha
                    .as_ref()
                    .map(|s| s.as_str().to_string())
                    .unwrap_or_default(),
                &artifact.created_at.to_string(),
            ],
        )?;
        Ok(())
    }

    pub async fn insert_lora_adapter(&self, lora: &LoraAdapter) -> Result<(), Error> {
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO lora_adapters (id, project_id, name, file_path, base_model, task_kind, metrics, created_at, active, agent_role)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
             ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                file_path = excluded.file_path,
                task_kind = excluded.task_kind,
                active = excluded.active,
                agent_role = excluded.agent_role",
            rusqlite::params![
                &lora.id,
                lora.project_id.as_deref(),
                &lora.name,
                &lora.file_path,
                &lora.base_model,
                lora.task_kind.as_deref(),
                lora.metrics.to_string(),
                lora.created_at,
                if lora.active { 1 } else { 0 },
                lora.agent_role.as_deref(),
            ],
        )?;
        Ok(())
    }

    pub async fn get_lora_adapter(&self, id: &str) -> Result<Option<LoraAdapter>, Error> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, project_id, name, file_path, base_model, task_kind, metrics, created_at, active, agent_role
             FROM lora_adapters WHERE id = ?1",
        )?;
        let mut rows = stmt.query([id])?;
        if let Some(row) = rows.next()? {
            Ok(Some(map_lora_adapter_row(row)?))
        } else {
            Ok(None)
        }
    }

    pub async fn list_lora_adapters_by_kind(
        &self,
        task_kind: &str,
    ) -> Result<Vec<LoraAdapter>, Error> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, project_id, name, file_path, base_model, task_kind, metrics, created_at, active, agent_role
             FROM lora_adapters WHERE task_kind = ?1 ORDER BY created_at DESC",
        )?;
        let mut rows = stmt.query([task_kind])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(map_lora_adapter_row(row)?);
        }
        Ok(out)
    }

    pub async fn list_lora_adapters_by_project(
        &self,
        project_id: &str,
    ) -> Result<Vec<LoraAdapter>, Error> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, project_id, name, file_path, base_model, task_kind, metrics, created_at, active, agent_role
             FROM lora_adapters WHERE project_id = ?1 ORDER BY created_at DESC",
        )?;
        let mut rows = stmt.query([project_id])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(map_lora_adapter_row(row)?);
        }
        Ok(out)
    }

    pub async fn list_lora_adapters_by_role(
        &self,
        agent_role: &str,
    ) -> Result<Vec<LoraAdapter>, Error> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, project_id, name, file_path, base_model, task_kind, metrics, created_at, active, agent_role
             FROM lora_adapters WHERE agent_role = ?1 ORDER BY created_at DESC",
        )?;
        let mut rows = stmt.query([agent_role])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(map_lora_adapter_row(row)?);
        }
        Ok(out)
    }

    pub async fn set_lora_adapter_active(&self, id: &str, active: bool) -> Result<(), Error> {
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE lora_adapters SET active = ?1 WHERE id = ?2",
            [if active { "1" } else { "0" }, id],
        )?;
        Ok(())
    }

    pub async fn insert_agent_log(&self, log: &AgentLog) -> Result<(), Error> {
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO agent_log (id, project_id, task_id, agent, action, message, level, timestamp, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            [
                &log.id as &dyn rusqlite::ToSql,
                &log.project_id.as_deref(),
                &log.task_id.as_deref(),
                &log.agent,
                &log.action,
                &log.message.as_deref(),
                &log.level,
                &log.timestamp,
                &log.metadata.to_string(),
            ],
        )?;
        Ok(())
    }

    pub async fn list_logs_by_task(&self, task_id: &str) -> Result<Vec<AgentLog>, Error> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, project_id, task_id, agent, action, message, level, timestamp, metadata
             FROM agent_log WHERE task_id = ?1 ORDER BY timestamp",
        )?;
        let rows = stmt.query_map([task_id], |row| {
            Ok(AgentLog {
                id: row.get(0)?,
                project_id: row.get(1)?,
                task_id: row.get(2)?,
                agent: row.get(3)?,
                action: row.get(4)?,
                message: row.get(5)?,
                level: row.get(6)?,
                timestamp: row.get(7)?,
                metadata: serde_json::from_str(&row.get::<_, String>(8)?).unwrap_or_default(),
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Error::Sqlite)
    }

    pub async fn list_logs_by_project(&self, project_id: &str) -> Result<Vec<AgentLog>, Error> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, project_id, task_id, agent, action, message, level, timestamp, metadata
             FROM agent_log WHERE project_id = ?1 ORDER BY timestamp",
        )?;
        let rows = stmt.query_map([project_id], |row| {
            Ok(AgentLog {
                id: row.get(0)?,
                project_id: row.get(1)?,
                task_id: row.get(2)?,
                agent: row.get(3)?,
                action: row.get(4)?,
                message: row.get(5)?,
                level: row.get(6)?,
                timestamp: row.get(7)?,
                metadata: serde_json::from_str(&row.get::<_, String>(8)?).unwrap_or_default(),
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Error::Sqlite)
    }

    pub async fn insert_metric(&self, snapshot: &MetricsSnapshot) -> Result<(), Error> {
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO metrics (id, timestamp, snapshot) VALUES (?1, ?2, ?3)",
            [
                &Ulid::new().to_string() as &dyn rusqlite::ToSql,
                &snapshot.timestamp as &dyn rusqlite::ToSql,
                &serde_json::to_string(snapshot)? as &dyn rusqlite::ToSql,
            ],
        )?;
        Ok(())
    }

    pub async fn list_metrics(&self, from: i64, to: i64) -> Result<Vec<MetricsSnapshot>, Error> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT snapshot FROM metrics WHERE timestamp >= ?1 AND timestamp <= ?2 ORDER BY timestamp",
        )?;
        let rows = stmt.query_map([from, to], |row| {
            let json: String = row.get(0)?;
            Ok(serde_json::from_str(&json).unwrap_or_default())
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Error::Sqlite)
    }

    pub async fn insert_project_snapshot(&self, snapshot: &ProjectSnapshot) -> Result<(), Error> {
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO project_snapshots (id, project_id, name, state_json, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                state_json = excluded.state_json",
            [
                &snapshot.id as &dyn rusqlite::ToSql,
                &snapshot.project_id as &dyn rusqlite::ToSql,
                &snapshot.name as &dyn rusqlite::ToSql,
                &snapshot.state_json.to_string() as &dyn rusqlite::ToSql,
                &snapshot.created_at as &dyn rusqlite::ToSql,
            ],
        )?;
        Ok(())
    }

    pub async fn get_project_snapshot(&self, id: &str) -> Result<Option<ProjectSnapshot>, Error> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, project_id, name, state_json, created_at FROM project_snapshots WHERE id = ?1",
        )?;
        let mut rows = stmt.query([id])?;
        match rows.next()? {
            Some(row) => Ok(Some(ProjectSnapshot {
                id: row.get(0)?,
                project_id: row.get(1)?,
                name: row.get(2)?,
                state_json: serde_json::from_str(&row.get::<_, String>(3)?)?,
                created_at: row.get(4)?,
            })),
            None => Ok(None),
        }
    }

    pub async fn list_project_snapshots(
        &self,
        project_id: &str,
    ) -> Result<Vec<ProjectSnapshot>, Error> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, project_id, name, state_json, created_at
             FROM project_snapshots WHERE project_id = ?1 ORDER BY created_at DESC",
        )?;
        let mut rows = stmt.query([project_id])?;
        let mut snapshots = Vec::new();
        while let Some(row) = rows.next()? {
            snapshots.push(ProjectSnapshot {
                id: row.get(0)?,
                project_id: row.get(1)?,
                name: row.get(2)?,
                state_json: serde_json::from_str(&row.get::<_, String>(3)?)?,
                created_at: row.get(4)?,
            });
        }
        Ok(snapshots)
    }

    pub async fn list_artifacts_by_task(&self, task_id: &str) -> Result<Vec<Artifact>, Error> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, task_id, file_path, file_type, commit_sha, created_at
             FROM artifacts WHERE task_id = ?1 ORDER BY created_at",
        )?;
        let rows = stmt.query_map([task_id], |row| {
            Ok(Artifact {
                id: row.get(0)?,
                task_id: row.get(1)?,
                file_path: row.get(2)?,
                file_type: row.get(3)?,
                commit_sha: row.get(4)?,
                created_at: row.get(5)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Error::Sqlite)
    }

    pub async fn insert_prompt_version(&self, version: &PromptVersion) -> Result<(), Error> {
        let conn = self.conn.lock().await;
        let metrics = serde_json::to_string(&version.metrics)?;
        conn.execute(
            "INSERT INTO prompt_versions (id, agent, project_id, system_prompt, fitness, parent_id, metrics, created_at, active)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
             ON CONFLICT(id) DO UPDATE SET
                agent = excluded.agent,
                project_id = excluded.project_id,
                system_prompt = excluded.system_prompt,
                fitness = excluded.fitness,
                parent_id = excluded.parent_id,
                metrics = excluded.metrics,
                active = excluded.active",
            [
                &version.id as &dyn rusqlite::ToSql,
                &version.agent,
                &version.project_id.as_deref(),
                &version.system_prompt,
                &version.fitness as &dyn rusqlite::ToSql,
                &version.parent_id.as_deref(),
                &metrics,
                &version.created_at,
                &(if version.active { 1i32 } else { 0i32 }),
            ],
        )?;
        Ok(())
    }

    pub async fn get_prompt_version(&self, id: &str) -> Result<Option<PromptVersion>, Error> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, agent, project_id, system_prompt, fitness, parent_id, metrics, created_at, active
             FROM prompt_versions WHERE id = ?1",
        )?;
        let mut rows = stmt.query([id])?;
        match rows.next()? {
            Some(row) => Ok(Some(map_prompt_version_row(row)?)),
            None => Ok(None),
        }
    }

    pub async fn list_prompt_versions_by_agent(
        &self,
        agent: &str,
    ) -> Result<Vec<PromptVersion>, Error> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, agent, project_id, system_prompt, fitness, parent_id, metrics, created_at, active
             FROM prompt_versions WHERE agent = ?1 ORDER BY created_at",
        )?;
        let rows = stmt.query_map([agent], map_prompt_version_row)?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Error::Sqlite)
    }

    pub async fn get_active_prompt_version(
        &self,
        agent: &str,
    ) -> Result<Option<PromptVersion>, Error> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, agent, project_id, system_prompt, fitness, parent_id, metrics, created_at, active
             FROM prompt_versions WHERE agent = ?1 AND active = 1 LIMIT 1",
        )?;
        let mut rows = stmt.query([agent])?;
        match rows.next()? {
            Some(row) => Ok(Some(map_prompt_version_row(row)?)),
            None => Ok(None),
        }
    }

    pub async fn set_active_prompt_version(&self, id: &str, agent: &str) -> Result<(), Error> {
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE prompt_versions SET active = 0 WHERE agent = ?1",
            [agent],
        )?;
        conn.execute(
            "UPDATE prompt_versions SET active = 1 WHERE id = ?1 AND agent = ?2",
            [id, agent],
        )?;
        Ok(())
    }

    pub async fn insert_experience(&self, exp: &Experience) -> Result<(), Error> {
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO experiences (id, task_id, project_id, prompt_version_id, text, critic_score, human_score, reward, comment, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
             ON CONFLICT(id) DO UPDATE SET
                task_id = excluded.task_id,
                project_id = excluded.project_id,
                prompt_version_id = excluded.prompt_version_id,
                text = excluded.text,
                critic_score = excluded.critic_score,
                human_score = excluded.human_score,
                reward = excluded.reward,
                comment = excluded.comment,
                created_at = excluded.created_at",
            rusqlite::params![
                &exp.id,
                &exp.task_id,
                exp.project_id.as_deref().unwrap_or(""),
                exp.prompt_version_id.as_deref().unwrap_or(""),
                exp.text.as_deref().unwrap_or(""),
                exp.critic_score,
                exp.human_score,
                exp.reward,
                exp.comment.as_deref().unwrap_or(""),
                exp.created_at,
            ],
        )?;
        Ok(())
    }

    pub async fn list_experiences_by_task(&self, task_id: &str) -> Result<Vec<Experience>, Error> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, task_id, project_id, prompt_version_id, text, critic_score, human_score, reward, comment, created_at
             FROM experiences WHERE task_id = ?1",
        )?;
        let mut rows = stmt.query([task_id])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(map_experience_row(row)?);
        }
        Ok(out)
    }

    pub async fn list_experiences_by_prompt_version(
        &self,
        prompt_version_id: &str,
    ) -> Result<Vec<Experience>, Error> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, task_id, project_id, prompt_version_id, text, critic_score, human_score, reward, comment, created_at
             FROM experiences WHERE prompt_version_id = ?1",
        )?;
        let mut rows = stmt.query([prompt_version_id])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(map_experience_row(row)?);
        }
        Ok(out)
    }

    pub async fn insert_training_example(&self, example: &TrainingExample) -> Result<(), Error> {
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO training_examples (id, task_id, project_id, prompt_version_id, task_kind, input_text, output_text, reward, created_at, agent_role, model_id, rag_evidence_ids, accepted_output, rejected_output, critic_feedback, failure_type)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)
             ON CONFLICT(id) DO UPDATE SET
                task_id = excluded.task_id,
                project_id = excluded.project_id,
                prompt_version_id = excluded.prompt_version_id,
                task_kind = excluded.task_kind,
                input_text = excluded.input_text,
                output_text = excluded.output_text,
                reward = excluded.reward,
                created_at = excluded.created_at,
                agent_role = excluded.agent_role,
                model_id = excluded.model_id,
                rag_evidence_ids = excluded.rag_evidence_ids,
                accepted_output = excluded.accepted_output,
                rejected_output = excluded.rejected_output,
                critic_feedback = excluded.critic_feedback,
                failure_type = excluded.failure_type",
            rusqlite::params![
                &example.id,
                &example.task_id,
                example.project_id.as_deref().unwrap_or(""),
                example.prompt_version_id.as_deref().unwrap_or(""),
                &example.task_kind,
                &example.input_text,
                &example.output_text,
                example.reward,
                example.created_at,
                example.agent_role.as_deref().unwrap_or(""),
                example.model_id.as_deref().unwrap_or(""),
                serde_json::to_string(&example.rag_evidence_ids)?,
                example.accepted_output.as_deref().unwrap_or(""),
                example.rejected_output.as_deref().unwrap_or(""),
                example.critic_feedback.as_deref().unwrap_or(""),
                example.failure_type.as_deref().unwrap_or(""),
            ],
        )?;
        Ok(())
    }

    pub async fn list_training_examples_by_kind(
        &self,
        task_kind: &str,
    ) -> Result<Vec<TrainingExample>, Error> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, task_id, project_id, prompt_version_id, task_kind, input_text, output_text, reward, created_at, agent_role, model_id, rag_evidence_ids, accepted_output, rejected_output, critic_feedback, failure_type
             FROM training_examples WHERE task_kind = ?1 ORDER BY created_at",
        )?;
        let mut rows = stmt.query([task_kind])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(map_training_example_row(row)?);
        }
        Ok(out)
    }

    pub async fn count_training_examples_by_kind(&self, task_kind: &str) -> Result<usize, Error> {
        let conn = self.conn.lock().await;
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM training_examples WHERE task_kind = ?1",
            [task_kind],
            |row| row.get(0),
        )?;
        Ok(count as usize)
    }

    pub async fn list_training_examples_by_project(
        &self,
        project_id: &str,
    ) -> Result<Vec<TrainingExample>, Error> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, task_id, project_id, prompt_version_id, task_kind, input_text, output_text, reward, created_at, agent_role, model_id, rag_evidence_ids, accepted_output, rejected_output, critic_feedback, failure_type
             FROM training_examples WHERE project_id = ?1 ORDER BY created_at",
        )?;
        let mut rows = stmt.query([project_id])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(map_training_example_row(row)?);
        }
        Ok(out)
    }

    pub async fn list_training_examples_by_role(
        &self,
        agent_role: &str,
    ) -> Result<Vec<TrainingExample>, Error> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, task_id, project_id, prompt_version_id, task_kind, input_text, output_text, reward, created_at, agent_role, model_id, rag_evidence_ids, accepted_output, rejected_output, critic_feedback, failure_type
             FROM training_examples WHERE agent_role = ?1 ORDER BY created_at",
        )?;
        let mut rows = stmt.query([agent_role])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(map_training_example_row(row)?);
        }
        Ok(out)
    }

    pub async fn count_training_examples_by_role(&self, agent_role: &str) -> Result<usize, Error> {
        let conn = self.conn.lock().await;
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM training_examples WHERE agent_role = ?1",
            [agent_role],
            |row| row.get(0),
        )?;
        Ok(count as usize)
    }

    pub async fn insert_training_job(&self, job: &TrainingJob) -> Result<(), Error> {
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO training_jobs (id, task_kind, status, started_at, finished_at, adapter_id, metrics, error_message)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
             ON CONFLICT(id) DO UPDATE SET
                task_kind = excluded.task_kind,
                status = excluded.status,
                started_at = excluded.started_at,
                finished_at = excluded.finished_at,
                adapter_id = excluded.adapter_id,
                metrics = excluded.metrics,
                error_message = excluded.error_message",
            rusqlite::params![
                &job.id,
                &job.task_kind,
                job.status.as_str(),
                job.started_at,
                job.finished_at,
                job.adapter_id.as_deref(),
                job.metrics.to_string(),
                job.error_message.as_deref(),
            ],
        )?;
        Ok(())
    }

    pub async fn get_training_job(&self, id: &str) -> Result<Option<TrainingJob>, Error> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, task_kind, status, started_at, finished_at, adapter_id, metrics, error_message
             FROM training_jobs WHERE id = ?1",
        )?;
        let mut rows = stmt.query([id])?;
        if let Some(row) = rows.next()? {
            Ok(Some(map_training_job_row(row)?))
        } else {
            Ok(None)
        }
    }

    pub async fn list_training_jobs_by_kind(
        &self,
        task_kind: &str,
    ) -> Result<Vec<TrainingJob>, Error> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, task_kind, status, started_at, finished_at, adapter_id, metrics, error_message
             FROM training_jobs WHERE task_kind = ?1 ORDER BY started_at DESC",
        )?;
        let mut rows = stmt.query([task_kind])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(map_training_job_row(row)?);
        }
        Ok(out)
    }

    pub async fn insert_memory_entry(&self, entry: &MemoryEntry) -> Result<(), Error> {
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO memory_entries (id, project_id, session_id, kind, text, metadata, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(id) DO UPDATE SET
                project_id = excluded.project_id,
                session_id = excluded.session_id,
                kind = excluded.kind,
                text = excluded.text,
                metadata = excluded.metadata,
                created_at = excluded.created_at",
            rusqlite::params![
                &entry.id,
                entry.project_id.as_deref().unwrap_or(""),
                entry.session_id.as_deref().unwrap_or(""),
                &entry.kind,
                &entry.text,
                entry.metadata.to_string(),
                entry.created_at,
            ],
        )?;
        Ok(())
    }

    pub async fn list_memory_entries(
        &self,
        project_id: Option<&str>,
        kind: Option<&str>,
        limit: usize,
    ) -> Result<Vec<MemoryEntry>, Error> {
        let conn = self.conn.lock().await;
        let mut sql = String::from(
            "SELECT id, project_id, session_id, kind, text, metadata, created_at FROM memory_entries WHERE 1=1",
        );
        if project_id.is_some() {
            sql.push_str(" AND project_id = ?1");
        }
        if kind.is_some() {
            sql.push_str(" AND kind = ?2");
        }
        sql.push_str(" ORDER BY created_at DESC LIMIT ?3");
        let mut stmt = conn.prepare(&sql)?;
        let project_param = project_id.unwrap_or("");
        let kind_param = kind.unwrap_or("");
        let mut rows = stmt.query(rusqlite::params![project_param, kind_param, limit as i64])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(map_memory_entry_row(row)?);
        }
        Ok(out)
    }

    pub async fn list_memory_entries_by_session(
        &self,
        session_id: &str,
    ) -> Result<Vec<MemoryEntry>, Error> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, project_id, session_id, kind, text, metadata, created_at
             FROM memory_entries WHERE session_id = ?1 ORDER BY created_at",
        )?;
        let mut rows = stmt.query([session_id])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(map_memory_entry_row(row)?);
        }
        Ok(out)
    }
}

fn map_experience_row(row: &rusqlite::Row<'_>) -> Result<Experience, Error> {
    Ok(Experience {
        id: row.get(0)?,
        task_id: row.get(1)?,
        project_id: as_optional_string(row.get(2)?),
        prompt_version_id: as_optional_string(row.get(3)?),
        text: as_optional_string(row.get(4)?),
        critic_score: row.get(5)?,
        human_score: row.get(6)?,
        reward: row.get(7)?,
        comment: as_optional_string(row.get(8)?),
        created_at: row.get(9)?,
    })
}

fn map_training_example_row(row: &rusqlite::Row<'_>) -> Result<TrainingExample, Error> {
    let agent_role: Option<String> = row.get(9)?;
    let rag_evidence_ids = row
        .get::<_, Option<String>>(11)?
        .filter(|value| !value.is_empty())
        .map(|value| serde_json::from_str(&value))
        .transpose()?
        .unwrap_or_default();
    Ok(TrainingExample {
        id: row.get(0)?,
        task_id: row.get(1)?,
        project_id: as_optional_string(row.get(2)?),
        prompt_version_id: as_optional_string(row.get(3)?),
        task_kind: row.get(4)?,
        agent_role: agent_role.filter(|s| !s.is_empty()),
        model_id: as_optional_string(row.get(10)?),
        rag_evidence_ids,
        input_text: row.get(5)?,
        output_text: row.get(6)?,
        accepted_output: as_optional_string(row.get(12)?),
        rejected_output: as_optional_string(row.get(13)?),
        critic_feedback: as_optional_string(row.get(14)?),
        failure_type: as_optional_string(row.get(15)?),
        reward: row.get(7)?,
        created_at: row.get(8)?,
    })
}

fn map_training_job_row(row: &rusqlite::Row<'_>) -> Result<TrainingJob, Error> {
    let status_str: String = row.get(2)?;
    Ok(TrainingJob {
        id: row.get(0)?,
        task_kind: row.get(1)?,
        status: TrainingJobStatus::from_str(&status_str).map_err(|_| {
            rusqlite::Error::InvalidColumnType(
                2,
                "TrainingJobStatus".to_string(),
                rusqlite::types::Type::Text,
            )
        })?,
        started_at: row.get(3)?,
        finished_at: row.get(4)?,
        adapter_id: as_optional_string(row.get(5)?),
        metrics: serde_json::from_str(&row.get::<_, String>(6)?)?,
        error_message: as_optional_string(row.get(7)?),
    })
}

fn map_memory_entry_row(row: &rusqlite::Row<'_>) -> Result<MemoryEntry, Error> {
    Ok(MemoryEntry {
        id: row.get(0)?,
        project_id: as_optional_string(row.get(1)?),
        session_id: as_optional_string(row.get(2)?),
        kind: row.get(3)?,
        text: row.get(4)?,
        metadata: serde_json::from_str(&row.get::<_, String>(5)?)?,
        created_at: row.get(6)?,
    })
}

fn map_prompt_version_row(row: &rusqlite::Row<'_>) -> Result<PromptVersion, rusqlite::Error> {
    Ok(PromptVersion {
        id: row.get(0)?,
        agent: row.get(1)?,
        project_id: as_optional_string(row.get(2)?),
        system_prompt: row.get(3)?,
        fitness: row.get(4)?,
        parent_id: as_optional_string(row.get(5)?),
        metrics: serde_json::from_str(&row.get::<_, String>(6)?).map_err(|error| {
            rusqlite::Error::FromSqlConversionFailure(
                6,
                rusqlite::types::Type::Text,
                Box::new(error),
            )
        })?,
        created_at: row.get(7)?,
        active: row.get::<_, i32>(8)? != 0,
    })
}

fn map_benchmark_run_row(row: &rusqlite::Row<'_>) -> Result<BenchmarkRun, Error> {
    let started_at = row.get::<_, i64>(10)?;
    let finished_at = row.get::<_, Option<i64>>(11)?;
    Ok(BenchmarkRun {
        summary: BenchmarkRunSummary {
            id: row.get(0)?,
            name: row.get(1)?,
            golden_set_path: std::path::PathBuf::from(row.get::<_, String>(3)?),
            variant_name: row.get(4)?,
            pass_count: row.get::<_, i64>(12)? as usize,
            fail_count: row.get::<_, i64>(13)? as usize,
            total_cases: row.get::<_, i64>(14)? as usize,
            pass_rate: row.get::<_, f64>(15)?,
            mean_latency_ms: row.get::<_, Option<f64>>(16)?.unwrap_or(0.0),
            total_tokens: row.get::<_, Option<i64>>(17)?.unwrap_or(0) as usize,
        },
        project_id: as_optional_string(row.get(2)?),
        variant: BenchmarkVariant {
            name: row.get(4)?,
            agent_role: as_optional_string(row.get(5)?),
            lora_adapter_id: as_optional_string(row.get(6)?),
            prompt_version_id: as_optional_string(row.get(7)?),
            backend_id: as_optional_string(row.get(8)?),
        },
        scorer_kind: row.get(9)?,
        started_at: chrono::DateTime::from_timestamp(started_at, 0).unwrap_or_default(),
        finished_at: finished_at.and_then(|t| chrono::DateTime::from_timestamp(t, 0)),
        results: Vec::new(),
        metadata: serde_json::from_str(&row.get::<_, String>(18)?)?,
    })
}

fn map_benchmark_result_row(row: &rusqlite::Row<'_>) -> Result<BenchmarkResult, Error> {
    Ok(BenchmarkResult {
        id: row.get(0)?,
        run_id: row.get(1)?,
        case_id: row.get(2)?,
        case_input: serde_json::from_str(&row.get::<_, String>(3)?)?,
        expected: row
            .get::<_, Option<String>>(4)?
            .map(|s| serde_json::from_str(&s))
            .transpose()?,
        actual: serde_json::from_str(&row.get::<_, String>(5)?)?,
        passed: row.get::<_, i32>(6)? != 0,
        score_value: row.get::<_, f64>(7)?,
        latency_ms: row.get::<_, Option<i64>>(8)?.unwrap_or(0) as u64,
        token_usage: row
            .get::<_, Option<String>>(9)?
            .map(|s| serde_json::from_str(&s))
            .transpose()?,
        explanation: as_optional_string(row.get(10)?),
        metadata: serde_json::from_str(&row.get::<_, String>(11)?)?,
    })
}

impl GraphStore {
    pub async fn insert_benchmark_run(&self, run: &BenchmarkRun) -> Result<(), Error> {
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO benchmark_runs (
                id, name, project_id, golden_set_path, variant_name, agent_role,
                lora_adapter_id, prompt_version_id, backend_id, scorer_kind,
                started_at, finished_at, pass_count, fail_count, total_cases,
                pass_rate, mean_latency_ms, total_tokens, metadata
             )
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19)
             ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                project_id = excluded.project_id,
                golden_set_path = excluded.golden_set_path,
                variant_name = excluded.variant_name,
                agent_role = excluded.agent_role,
                lora_adapter_id = excluded.lora_adapter_id,
                prompt_version_id = excluded.prompt_version_id,
                backend_id = excluded.backend_id,
                scorer_kind = excluded.scorer_kind,
                started_at = excluded.started_at,
                finished_at = excluded.finished_at,
                pass_count = excluded.pass_count,
                fail_count = excluded.fail_count,
                total_cases = excluded.total_cases,
                pass_rate = excluded.pass_rate,
                mean_latency_ms = excluded.mean_latency_ms,
                total_tokens = excluded.total_tokens,
                metadata = excluded.metadata",
            [
                &run.summary.id as &dyn rusqlite::ToSql,
                &run.summary.name as &dyn rusqlite::ToSql,
                &run.project_id.as_deref() as &dyn rusqlite::ToSql,
                &run.summary.golden_set_path.to_string_lossy().to_string() as &dyn rusqlite::ToSql,
                &run.summary.variant_name as &dyn rusqlite::ToSql,
                &run.variant.agent_role.as_deref() as &dyn rusqlite::ToSql,
                &run.variant.lora_adapter_id.as_deref() as &dyn rusqlite::ToSql,
                &run.variant.prompt_version_id.as_deref() as &dyn rusqlite::ToSql,
                &run.variant.backend_id.as_deref() as &dyn rusqlite::ToSql,
                &run.scorer_kind as &dyn rusqlite::ToSql,
                &run.started_at.timestamp() as &dyn rusqlite::ToSql,
                &run.finished_at.map(|t| t.timestamp()) as &dyn rusqlite::ToSql,
                &(run.summary.pass_count as i64) as &dyn rusqlite::ToSql,
                &(run.summary.fail_count as i64) as &dyn rusqlite::ToSql,
                &(run.summary.total_cases as i64) as &dyn rusqlite::ToSql,
                &run.summary.pass_rate as &dyn rusqlite::ToSql,
                &run.summary.mean_latency_ms as &dyn rusqlite::ToSql,
                &(run.summary.total_tokens as i64) as &dyn rusqlite::ToSql,
                &run.metadata.to_string() as &dyn rusqlite::ToSql,
            ],
        )?;
        Ok(())
    }

    pub async fn get_benchmark_run(&self, id: &str) -> Result<Option<BenchmarkRun>, Error> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, name, project_id, golden_set_path, variant_name, agent_role,
                    lora_adapter_id, prompt_version_id, backend_id, scorer_kind,
                    started_at, finished_at, pass_count, fail_count, total_cases,
                    pass_rate, mean_latency_ms, total_tokens, metadata
             FROM benchmark_runs WHERE id = ?1",
        )?;
        let mut rows = stmt.query([id])?;
        match rows.next()? {
            Some(row) => Ok(Some(map_benchmark_run_row(row)?)),
            None => Ok(None),
        }
    }

    pub async fn list_benchmark_runs(
        &self,
        limit: usize,
    ) -> Result<Vec<BenchmarkRunSummary>, Error> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, name, project_id, golden_set_path, variant_name, agent_role,
                    lora_adapter_id, prompt_version_id, backend_id, scorer_kind,
                    started_at, finished_at, pass_count, fail_count, total_cases,
                    pass_rate, mean_latency_ms, total_tokens, metadata
             FROM benchmark_runs
             ORDER BY started_at DESC
             LIMIT ?1",
        )?;
        let mut rows = stmt.query([limit as i64])?;
        let mut runs = Vec::new();
        while let Some(row) = rows.next()? {
            runs.push(map_benchmark_run_row(row)?.summary);
        }
        Ok(runs)
    }

    pub async fn insert_benchmark_result(&self, result: &BenchmarkResult) -> Result<(), Error> {
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO benchmark_results (
                id, run_id, case_id, case_input, expected, actual, passed,
                score_value, latency_ms, token_usage, explanation, metadata
             )
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
             ON CONFLICT(id) DO UPDATE SET
                run_id = excluded.run_id,
                case_id = excluded.case_id,
                case_input = excluded.case_input,
                expected = excluded.expected,
                actual = excluded.actual,
                passed = excluded.passed,
                score_value = excluded.score_value,
                latency_ms = excluded.latency_ms,
                token_usage = excluded.token_usage,
                explanation = excluded.explanation,
                metadata = excluded.metadata",
            [
                &result.id as &dyn rusqlite::ToSql,
                &result.run_id as &dyn rusqlite::ToSql,
                &result.case_id as &dyn rusqlite::ToSql,
                &result.case_input.to_string() as &dyn rusqlite::ToSql,
                &result.expected.as_ref().map(|v| v.to_string()) as &dyn rusqlite::ToSql,
                &result.actual.to_string() as &dyn rusqlite::ToSql,
                &(if result.passed { 1 } else { 0 }) as &dyn rusqlite::ToSql,
                &result.score_value as &dyn rusqlite::ToSql,
                &(result.latency_ms as i64) as &dyn rusqlite::ToSql,
                &result
                    .token_usage
                    .as_ref()
                    .map(|u| serde_json::to_string(u).unwrap_or_default())
                    as &dyn rusqlite::ToSql,
                &result.explanation.as_deref() as &dyn rusqlite::ToSql,
                &result.metadata.to_string() as &dyn rusqlite::ToSql,
            ],
        )?;
        Ok(())
    }

    pub async fn list_benchmark_results(
        &self,
        run_id: &str,
    ) -> Result<Vec<BenchmarkResult>, Error> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, run_id, case_id, case_input, expected, actual, passed,
                    score_value, latency_ms, token_usage, explanation, metadata
             FROM benchmark_results
             WHERE run_id = ?1",
        )?;
        let mut rows = stmt.query([run_id])?;
        let mut results = Vec::new();
        while let Some(row) = rows.next()? {
            results.push(map_benchmark_result_row(row)?);
        }
        Ok(results)
    }
}

#[cfg(test)]
impl GraphStore {
    async fn raw_execute(
        &self,
        sql: &str,
        params: &[&dyn rusqlite::ToSql],
    ) -> Result<usize, Error> {
        let conn = self.conn.lock().await;
        Ok(conn.execute(sql, params)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_task(id: &str, project_id: &str, status: TaskStatus) -> Task {
        Task {
            id: id.to_string(),
            project_id: project_id.to_string(),
            parent_id: None,
            title: "Test".to_string(),
            description: None,
            kind: "codegen".to_string(),
            status,
            assigned_agent: None,
            priority: 0,
            created_at: Utc::now().timestamp(),
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

    #[tokio::test]
    async fn test_project_crud() {
        let store = GraphStore::new(":memory:").await.unwrap();
        let project = Project {
            id: "proj-1".to_string(),
            name: "Test Project".to_string(),
            root_path: "/tmp/test".to_string(),
            created_at: Utc::now().timestamp(),
            updated_at: Utc::now().timestamp(),
            metadata: serde_json::json!({"key": "value"}),
        };
        store.insert_project(&project).await.unwrap();
        let fetched = store.get_project("proj-1").await.unwrap();
        assert!(fetched.is_some());
        assert_eq!(fetched.unwrap().name, "Test Project");
    }

    #[tokio::test]
    async fn test_task_crud() {
        let store = GraphStore::new(":memory:").await.unwrap();
        let project = Project {
            id: "proj-1".to_string(),
            name: "Test Project".to_string(),
            root_path: "/tmp/test".to_string(),
            created_at: Utc::now().timestamp(),
            updated_at: Utc::now().timestamp(),
            metadata: serde_json::Value::Null,
        };
        store.insert_project(&project).await.unwrap();

        let task = sample_task("task-1", "proj-1", TaskStatus::Pending);
        store.insert_task(&task).await.unwrap();
        let fetched = store.get_task("task-1").await.unwrap();
        assert!(fetched.is_some());
        assert_eq!(fetched.unwrap().title, "Test");
    }

    #[tokio::test]
    async fn project_level_agent_log_preserves_null_task_id() {
        let store = GraphStore::new(":memory:").await.unwrap();
        let project = Project {
            id: "proj-1".to_string(),
            name: "Test Project".to_string(),
            root_path: "/tmp/test".to_string(),
            created_at: Utc::now().timestamp(),
            updated_at: Utc::now().timestamp(),
            metadata: serde_json::Value::Null,
        };
        store.insert_project(&project).await.unwrap();

        store
            .insert_agent_log(&AgentLog {
                id: "log-1".to_string(),
                project_id: Some("proj-1".to_string()),
                task_id: None,
                agent: "runner".to_string(),
                action: "run_started".to_string(),
                message: None,
                level: "info".to_string(),
                timestamp: Utc::now().timestamp_millis(),
                metadata: serde_json::json!({ "trace_id": "trace-run" }),
            })
            .await
            .unwrap();

        let logs = store.list_logs_by_project("proj-1").await.unwrap();
        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0].action, "run_started");
        assert_eq!(logs[0].task_id, None);
    }

    #[tokio::test]
    async fn test_task_status_update() {
        let store = GraphStore::new(":memory:").await.unwrap();
        let project = Project {
            id: "proj-1".to_string(),
            name: "Test".to_string(),
            root_path: "/tmp".to_string(),
            created_at: Utc::now().timestamp(),
            updated_at: Utc::now().timestamp(),
            metadata: serde_json::Value::Null,
        };
        store.insert_project(&project).await.unwrap();

        let task = sample_task("task-1", "proj-1", TaskStatus::Pending);
        store.insert_task(&task).await.unwrap();

        store
            .update_task_status("task-1", TaskStatus::InProgress, None)
            .await
            .unwrap();
        let fetched = store.get_task("task-1").await.unwrap().unwrap();
        assert_eq!(fetched.status, TaskStatus::InProgress);
        assert!(fetched.started_at.is_some());
    }

    #[tokio::test]
    async fn old_database_fixture_migrates_to_current_schema_version() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("old-v2.db");
        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch(
                r#"
                PRAGMA user_version = 2;
                CREATE TABLE projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    root_path TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    metadata TEXT DEFAULT '{}'
                );
                CREATE TABLE tasks (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id),
                    parent_id TEXT,
                    title TEXT NOT NULL,
                    description TEXT,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    assigned_agent TEXT,
                    priority INTEGER DEFAULT 0,
                    created_at INTEGER NOT NULL,
                    started_at INTEGER,
                    finished_at INTEGER,
                    payload TEXT DEFAULT '{}',
                    result TEXT
                );
                INSERT INTO projects (id, name, root_path, created_at, updated_at, metadata)
                VALUES ('proj-old', 'Old', '/tmp/old', 1, 1, '{}');
                INSERT INTO tasks (id, project_id, title, kind, status, created_at, payload)
                VALUES ('task-old', 'proj-old', 'Old task', 'codegen', 'in_progress', 1, '{}');
                "#,
            )
            .unwrap();
        }

        let store = GraphStore::new(db_path.to_str().unwrap()).await.unwrap();

        assert_eq!(store.schema_version().await.unwrap(), 14);
        let task = store.get_task("task-old").await.unwrap().unwrap();
        assert_eq!(task.status, TaskStatus::InProgress);
        assert_eq!(task.trace_id, "");
    }

    #[tokio::test]
    async fn get_project_with_invalid_metadata_returns_serde_error() {
        let store = GraphStore::new(":memory:").await.unwrap();
        store
            .raw_execute(
                "INSERT INTO projects (id, name, root_path, created_at, updated_at, metadata)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                &[
                    &"proj-1" as &dyn rusqlite::ToSql,
                    &"Test",
                    &"/tmp",
                    &0i64,
                    &0i64,
                    &"not valid json",
                ],
            )
            .await
            .unwrap();

        let result = store.get_project("proj-1").await;
        assert!(matches!(result, Err(Error::Serde(_))));
    }

    #[tokio::test]
    async fn prompt_version_crud_and_active_switching() {
        let store = GraphStore::new(":memory:").await.unwrap();
        let v1 = PromptVersion {
            id: "pv1".to_string(),
            agent: "coder".to_string(),
            project_id: None,
            system_prompt: "base".to_string(),
            fitness: None,
            parent_id: None,
            metrics: serde_json::json!({ "seed": true }),
            created_at: 1,
            active: true,
        };
        store.insert_prompt_version(&v1).await.unwrap();

        let v2 = PromptVersion {
            id: "pv2".to_string(),
            agent: "coder".to_string(),
            project_id: None,
            system_prompt: "mutant".to_string(),
            fitness: Some(4.5),
            parent_id: Some("pv1".to_string()),
            metrics: serde_json::json!({ "prompt_benchmark_gate": { "accepted": true } }),
            created_at: 2,
            active: false,
        };
        store.insert_prompt_version(&v2).await.unwrap();

        let fetched = store.get_prompt_version("pv2").await.unwrap().unwrap();
        assert_eq!(fetched.system_prompt, "mutant");
        assert_eq!(fetched.parent_id, Some("pv1".to_string()));
        assert_eq!(
            fetched.metrics["prompt_benchmark_gate"]["accepted"],
            serde_json::Value::Bool(true)
        );

        let coder_versions = store.list_prompt_versions_by_agent("coder").await.unwrap();
        assert_eq!(coder_versions.len(), 2);

        store
            .set_active_prompt_version("pv2", "coder")
            .await
            .unwrap();
        let active = store
            .get_active_prompt_version("coder")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(active.id, "pv2");

        let v1_after = store.get_prompt_version("pv1").await.unwrap().unwrap();
        assert!(!v1_after.active);
    }

    #[tokio::test]
    async fn get_task_with_invalid_payload_returns_serde_error() {
        let store = GraphStore::new(":memory:").await.unwrap();
        store
            .raw_execute(
                "INSERT INTO projects (id, name, root_path, created_at, updated_at, metadata)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                &[
                    &"proj-1" as &dyn rusqlite::ToSql,
                    &"Test",
                    &"/tmp",
                    &0i64,
                    &0i64,
                    &"{}",
                ],
            )
            .await
            .unwrap();
        store
            .raw_execute(
                "INSERT INTO tasks (id, project_id, parent_id, title, description, kind, status, assigned_agent, priority, created_at, started_at, finished_at, payload, result)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
                &[
                    &"task-1" as &dyn rusqlite::ToSql,
                    &"proj-1",
                    &"",
                    &"Test",
                    &"",
                    &"codegen",
                    &"pending",
                    &"",
                    &0i64,
                    &0i64,
                    &Option::<i64>::None,
                    &Option::<i64>::None,
                    &"not valid json",
                    &Option::<String>::None,
                ],
            )
            .await
            .unwrap();

        let result = store.get_task("task-1").await;
        assert!(matches!(result, Err(Error::Serde(_))));
    }

    #[tokio::test]
    async fn training_example_round_trips_through_repository() {
        let store = GraphStore::new(":memory:").await.unwrap();
        let project = Project {
            id: "proj-1".to_string(),
            name: "Test".to_string(),
            root_path: "/tmp".to_string(),
            created_at: Utc::now().timestamp(),
            updated_at: Utc::now().timestamp(),
            metadata: serde_json::Value::Null,
        };
        store.insert_project(&project).await.unwrap();
        let task = sample_task("task-1", "proj-1", TaskStatus::Completed);
        store.insert_task(&task).await.unwrap();
        let example = TrainingExample {
            id: "te-1".into(),
            task_id: "task-1".into(),
            project_id: Some("proj-1".into()),
            prompt_version_id: Some("pv-1".into()),
            task_kind: "codegen".into(),
            agent_role: Some("coder".into()),
            model_id: Some("qwen3.5:9b".into()),
            rag_evidence_ids: vec!["chunk-a".into(), "chunk-b".into()],
            input_text: "system prompt\n\nImplement X".into(),
            output_text: "fn x() {}".into(),
            accepted_output: Some("fn x() {}".into()),
            rejected_output: Some("fn bad() {}".into()),
            critic_feedback: Some("missing tests".into()),
            failure_type: Some("missing-tests".into()),
            reward: 4.5,
            created_at: Utc::now().timestamp(),
        };
        store.insert_training_example(&example).await.unwrap();

        let by_kind = store
            .list_training_examples_by_kind("codegen")
            .await
            .unwrap();
        assert_eq!(by_kind.len(), 1);
        assert_eq!(by_kind[0], example);

        let by_project = store
            .list_training_examples_by_project("proj-1")
            .await
            .unwrap();
        assert_eq!(by_project.len(), 1);

        let count = store
            .count_training_examples_by_kind("codegen")
            .await
            .unwrap();
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn lora_adapter_round_trips_through_repository() {
        let store = GraphStore::new(":memory:").await.unwrap();
        let project = Project {
            id: "proj-1".to_string(),
            name: "Test".to_string(),
            root_path: "/tmp".to_string(),
            created_at: Utc::now().timestamp(),
            updated_at: Utc::now().timestamp(),
            metadata: serde_json::Value::Null,
        };
        store.insert_project(&project).await.unwrap();
        let adapter = LoraAdapter {
            id: "lora-1".into(),
            project_id: Some("proj-1".into()),
            name: "codegen-v1".into(),
            file_path: "/tmp/adapters/codegen-v1.safetensors".into(),
            base_model: "mistral-7b".into(),
            task_kind: Some("codegen".into()),
            agent_role: Some("coder".into()),
            metrics: serde_json::json!({"loss": 0.1}),
            created_at: Utc::now().timestamp(),
            active: true,
        };
        store.insert_lora_adapter(&adapter).await.unwrap();

        let fetched = store.get_lora_adapter("lora-1").await.unwrap().unwrap();
        assert_eq!(fetched.id, "lora-1");
        assert_eq!(fetched.task_kind, Some("codegen".into()));
        assert!(fetched.active);

        let by_kind = store.list_lora_adapters_by_kind("codegen").await.unwrap();
        assert_eq!(by_kind.len(), 1);

        store
            .set_lora_adapter_active("lora-1", false)
            .await
            .unwrap();
        let updated = store.get_lora_adapter("lora-1").await.unwrap().unwrap();
        assert!(!updated.active);
    }
}
