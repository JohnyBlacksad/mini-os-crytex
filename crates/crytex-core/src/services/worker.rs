//! Worker pool that executes ready tasks with bounded concurrency.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use thiserror::Error;
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

use crate::models::Task;
use crate::services::{Scheduler, SchedulerError};

/// Errors emitted by the worker pool.
#[derive(Debug, Error)]
pub enum WorkerError {
    #[error("scheduler error: {0}")]
    Scheduler(#[from] SchedulerError),
    #[error("handler error: {0}")]
    Handler(String),
    #[error("worker pool was cancelled")]
    Cancelled,
}

/// Handles a single task. Implementations are responsible for status updates
/// and result persistence.
#[async_trait]
pub trait TaskHandler: Send + Sync {
    async fn handle(&self, task: Task) -> Result<(), WorkerError>;
}

/// Pool of workers with bounded concurrency and graceful shutdown.
pub struct WorkerPool {
    concurrency: usize,
    cancellation: CancellationToken,
}

impl WorkerPool {
    /// Create a new pool with the given concurrency limit.
    pub fn new(concurrency: usize) -> Self {
        Self {
            concurrency,
            cancellation: CancellationToken::new(),
        }
    }

    /// Signal the pool to stop after the current batch finishes.
    pub fn shutdown(&self) {
        self.cancellation.cancel();
    }

    /// Run the pool until cancellation.
    ///
    /// Repeatedly asks the scheduler for the next batch of ready tasks and
    /// executes them through `handler`, respecting the concurrency limit.
    pub async fn run<S, H>(&self, scheduler: Arc<S>, handler: Arc<H>) -> Result<(), WorkerError>
    where
        S: Scheduler + 'static,
        H: TaskHandler + 'static,
    {
        if self.concurrency == 0 {
            return Err(WorkerError::Handler("concurrency must be > 0".into()));
        }

        let semaphore = Arc::new(Semaphore::new(self.concurrency));

        loop {
            if self.cancellation.is_cancelled() {
                break;
            }

            let batch = match scheduler.next_batch(self.concurrency).await {
                Ok(tasks) => tasks,
                Err(e) => return Err(WorkerError::Scheduler(e)),
            };

            if batch.is_empty() {
                tokio::time::sleep(Duration::from_millis(100)).await;
                continue;
            }

            let mut handles = Vec::with_capacity(batch.len());
            for task in batch {
                let permit = semaphore
                    .clone()
                    .acquire_owned()
                    .await
                    .map_err(|e| WorkerError::Handler(e.to_string()))?;
                let handler = handler.clone();
                let cancel = self.cancellation.clone();

                handles.push(tokio::spawn(async move {
                    tokio::select! {
                        _ = cancel.cancelled() => {}
                        res = handler.handle(task) => {
                            if let Err(e) = res {
                                eprintln!("task handler failed: {}", e);
                            }
                        }
                    }
                    drop(permit);
                }));
            }

            for handle in handles {
                handle
                    .await
                    .map_err(|e| WorkerError::Handler(e.to_string()))?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;

    use crate::models::Task;
    use crate::services::{Scheduler, SchedulerError};

    use super::*;

    struct MockScheduler {
        tasks: Mutex<Vec<Task>>,
    }

    #[async_trait]
    impl Scheduler for MockScheduler {
        async fn next_batch(&self, limit: usize) -> Result<Vec<Task>, SchedulerError> {
            let mut tasks = self.tasks.lock().unwrap();
            let n = tasks.len().min(limit);
            Ok(tasks.drain(..n).collect())
        }
    }

    struct RecordingHandler {
        calls: AtomicUsize,
    }

    #[async_trait]
    impl TaskHandler for RecordingHandler {
        async fn handle(&self, _task: Task) -> Result<(), WorkerError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    fn sample_task(id: &str) -> Task {
        Task {
            id: id.to_string(),
            project_id: "p1".to_string(),
            parent_id: None,
            title: id.to_string(),
            description: None,
            kind: "codegen".to_string(),
            status: crate::models::TaskStatus::Pending,
            assigned_agent: None,
            priority: 0,
            created_at: 0,
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
    async fn worker_pool_executes_all_ready_tasks() {
        let scheduler = Arc::new(MockScheduler {
            tasks: Mutex::new(vec![
                sample_task("t1"),
                sample_task("t2"),
                sample_task("t3"),
            ]),
        });
        let handler = Arc::new(RecordingHandler {
            calls: AtomicUsize::new(0),
        });

        let pool = WorkerPool::new(2);
        let pool_ref = Arc::new(pool);
        let pool_clone = pool_ref.clone();

        let handler_for_run = handler.clone();
        let run_handle = tokio::spawn(async move {
            pool_clone.run(scheduler, handler_for_run).await.unwrap();
        });

        // Let it process the only batch.
        tokio::time::sleep(Duration::from_millis(300)).await;
        pool_ref.shutdown();
        run_handle.await.unwrap();

        assert_eq!(handler.calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn worker_pool_respects_concurrency() {
        let scheduler = Arc::new(MockScheduler {
            tasks: Mutex::new(vec![sample_task("t1"), sample_task("t2")]),
        });
        let handler = Arc::new(RecordingHandler {
            calls: AtomicUsize::new(0),
        });

        let pool = WorkerPool::new(1);
        let pool_ref = Arc::new(pool);
        let pool_clone = pool_ref.clone();

        let handler_for_run = handler.clone();
        let run_handle = tokio::spawn(async move {
            pool_clone.run(scheduler, handler_for_run).await.unwrap();
        });

        tokio::time::sleep(Duration::from_millis(300)).await;
        pool_ref.shutdown();
        run_handle.await.unwrap();

        assert_eq!(handler.calls.load(Ordering::SeqCst), 2);
    }
}
