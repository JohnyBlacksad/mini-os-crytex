//! System and business metrics collection.
//!
//! Provides snapshots of CPU/RAM/disk/network usage and GPU state via
//! `sysinfo` and `nvml-wrapper`. Business metrics (tasks completed/failed,
//! latency) are tracked in-memory and persisted to SQLite history.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use sysinfo::{Disks, Networks, System};
use thiserror::Error;

/// Errors that can occur while collecting metrics.
#[derive(Debug, Error)]
pub enum MetricsError {
    #[error("persistence error: {0}")]
    Persistence(String),
    #[error("system information unavailable: {0}")]
    Unavailable(String),
}

/// GPU-specific metrics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GpuMetrics {
    pub name: String,
    pub usage_percent: u32,
    pub memory_used_mb: u64,
    pub memory_total_mb: u64,
    pub temperature_c: Option<u32>,
    pub power_draw_w: Option<u32>,
    pub fan_speed_percent: Option<u32>,
}

/// A point-in-time snapshot of system and business metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct MetricsSnapshot {
    pub timestamp: i64,
    pub cpu_usage_percent: f32,
    pub memory_used_mb: u64,
    pub memory_total_mb: u64,
    pub swap_used_mb: u64,
    pub swap_total_mb: u64,
    pub disk_used_gb: u64,
    pub disk_total_gb: u64,
    pub network_rx_mb: u64,
    pub network_tx_mb: u64,
    pub gpus: Vec<GpuMetrics>,
    pub tasks_completed: u64,
    pub tasks_failed: u64,
    pub average_latency_ms: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

/// Repository for metric history.
#[async_trait]
pub trait MetricsRepository: Send + Sync {
    async fn insert_metric(&self, snapshot: &MetricsSnapshot) -> Result<(), MetricsError>;
    async fn list_metrics(&self, from: i64, to: i64) -> Result<Vec<MetricsSnapshot>, MetricsError>;
}

/// Collects metrics and tracks business counters.
#[async_trait]
pub trait MetricsService: Send + Sync {
    async fn snapshot(&self) -> Result<MetricsSnapshot, MetricsError>;
    async fn record_task_completion(
        &self,
        latency_ms: u64,
        success: bool,
    ) -> Result<(), MetricsError>;
    async fn record_cache_hit(&self) -> Result<(), MetricsError>;
    async fn record_cache_miss(&self) -> Result<(), MetricsError>;
    async fn history(&self, from: i64, to: i64) -> Result<Vec<MetricsSnapshot>, MetricsError>;
}

/// Default implementation using `sysinfo` and optional NVIDIA GPU data.
pub struct MetricsServiceImpl<R> {
    system: Arc<std::sync::Mutex<System>>,
    networks: Arc<std::sync::Mutex<Networks>>,
    repo: Arc<R>,
    tasks_completed: AtomicU64,
    tasks_failed: AtomicU64,
    total_latency_ms: AtomicU64,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
}

impl<R> MetricsServiceImpl<R>
where
    R: MetricsRepository + 'static,
{
    pub fn new(repo: Arc<R>) -> Self {
        Self {
            system: Arc::new(std::sync::Mutex::new(System::new_all())),
            networks: Arc::new(std::sync::Mutex::new(Networks::new_with_refreshed_list())),
            repo,
            tasks_completed: AtomicU64::new(0),
            tasks_failed: AtomicU64::new(0),
            total_latency_ms: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
        }
    }

    fn collect_system(&self) -> Result<SystemMetrics, MetricsError> {
        let mut system = self
            .system
            .lock()
            .map_err(|e| MetricsError::Unavailable(format!("cannot lock system monitor: {e}")))?;
        system.refresh_all();

        let mut networks = self
            .networks
            .lock()
            .map_err(|e| MetricsError::Unavailable(format!("cannot lock network monitor: {e}")))?;
        networks.refresh(true);

        let memory_used = system.used_memory();
        let memory_total = system.total_memory();
        let swap_used = system.used_swap();
        let swap_total = system.total_swap();

        let cpu_usage = system.cpus().iter().map(|c| c.cpu_usage()).sum::<f32>()
            / system.cpus().len().max(1) as f32;

        let disks = Disks::new_with_refreshed_list();
        let (disk_used, disk_total) = disks
            .iter()
            .map(|d| {
                let total = d.total_space();
                let available = d.available_space();
                (total.saturating_sub(available), total)
            })
            .fold((0, 0), |(u, t), (du, dt)| (u + du, t + dt));

        let (network_rx, network_tx) = networks
            .values()
            .map(|data| (data.total_received(), data.total_transmitted()))
            .fold((0, 0), |(rx, tx), (drx, dtx)| (rx + drx, tx + dtx));

        Ok(SystemMetrics {
            cpu_usage_percent: cpu_usage,
            memory_used_mb: bytes_to_mb(memory_used),
            memory_total_mb: bytes_to_mb(memory_total),
            swap_used_mb: bytes_to_mb(swap_used),
            swap_total_mb: bytes_to_mb(swap_total),
            disk_used_gb: bytes_to_gb(disk_used),
            disk_total_gb: bytes_to_gb(disk_total),
            network_rx_mb: bytes_to_mb(network_rx),
            network_tx_mb: bytes_to_mb(network_tx),
        })
    }

    fn collect_gpus() -> Vec<GpuMetrics> {
        let Ok(nvml) = nvml_wrapper::Nvml::init() else {
            return Vec::new();
        };
        let count = nvml.device_count().unwrap_or(0);
        (0..count)
            .filter_map(|i| {
                let device = nvml.device_by_index(i).ok()?;
                let name = device.name().unwrap_or_default();
                let usage = device.utilization_rates().ok()?;
                let memory = device.memory_info().ok()?;
                let temperature = device
                    .temperature(nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu)
                    .ok();
                let power = device.power_usage().ok().map(|mw| mw / 1000);
                let fan = device.fan_speed(0).ok();

                Some(GpuMetrics {
                    name,
                    usage_percent: usage.gpu,
                    memory_used_mb: bytes_to_mb(memory.used),
                    memory_total_mb: bytes_to_mb(memory.total),
                    temperature_c: temperature,
                    power_draw_w: power,
                    fan_speed_percent: fan,
                })
            })
            .collect()
    }
}

#[async_trait]
impl<R> MetricsService for MetricsServiceImpl<R>
where
    R: MetricsRepository + 'static,
{
    async fn snapshot(&self) -> Result<MetricsSnapshot, MetricsError> {
        let system = self.collect_system()?;
        let gpus = Self::collect_gpus();

        let completed = self.tasks_completed.load(Ordering::Relaxed);
        let total = self.total_latency_ms.load(Ordering::Relaxed);
        let avg_latency = total.checked_div(completed).unwrap_or(0);

        let snapshot = MetricsSnapshot {
            timestamp: chrono::Utc::now().timestamp_millis(),
            cpu_usage_percent: system.cpu_usage_percent,
            memory_used_mb: system.memory_used_mb,
            memory_total_mb: system.memory_total_mb,
            swap_used_mb: system.swap_used_mb,
            swap_total_mb: system.swap_total_mb,
            disk_used_gb: system.disk_used_gb,
            disk_total_gb: system.disk_total_gb,
            network_rx_mb: system.network_rx_mb,
            network_tx_mb: system.network_tx_mb,
            gpus,
            tasks_completed: completed,
            tasks_failed: self.tasks_failed.load(Ordering::Relaxed),
            average_latency_ms: avg_latency,
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
        };

        self.repo.insert_metric(&snapshot).await?;
        Ok(snapshot)
    }

    async fn record_task_completion(
        &self,
        latency_ms: u64,
        success: bool,
    ) -> Result<(), MetricsError> {
        if success {
            self.tasks_completed.fetch_add(1, Ordering::Relaxed);
            self.total_latency_ms
                .fetch_add(latency_ms, Ordering::Relaxed);
        } else {
            self.tasks_failed.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    async fn record_cache_hit(&self) -> Result<(), MetricsError> {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    async fn record_cache_miss(&self) -> Result<(), MetricsError> {
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    async fn history(&self, from: i64, to: i64) -> Result<Vec<MetricsSnapshot>, MetricsError> {
        self.repo.list_metrics(from, to).await
    }
}

struct SystemMetrics {
    cpu_usage_percent: f32,
    memory_used_mb: u64,
    memory_total_mb: u64,
    swap_used_mb: u64,
    swap_total_mb: u64,
    disk_used_gb: u64,
    disk_total_gb: u64,
    network_rx_mb: u64,
    network_tx_mb: u64,
}

fn bytes_to_mb(bytes: u64) -> u64 {
    bytes / 1024 / 1024
}

fn bytes_to_gb(bytes: u64) -> u64 {
    bytes / 1024 / 1024 / 1024
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::Mutex;

    #[derive(Default)]
    struct MockRepo {
        metrics: Mutex<Vec<MetricsSnapshot>>,
    }

    #[async_trait]
    impl MetricsRepository for MockRepo {
        async fn insert_metric(&self, snapshot: &MetricsSnapshot) -> Result<(), MetricsError> {
            self.metrics.lock().unwrap().push(snapshot.clone());
            Ok(())
        }

        async fn list_metrics(
            &self,
            from: i64,
            to: i64,
        ) -> Result<Vec<MetricsSnapshot>, MetricsError> {
            Ok(self
                .metrics
                .lock()
                .unwrap()
                .iter()
                .filter(|m| m.timestamp >= from && m.timestamp <= to)
                .cloned()
                .collect())
        }
    }

    fn service() -> MetricsServiceImpl<MockRepo> {
        MetricsServiceImpl::new(Arc::new(MockRepo::default()))
    }

    #[tokio::test]
    async fn snapshot_returns_non_empty_system_values() {
        let svc = service();
        let snap = svc.snapshot().await.unwrap();
        assert!(snap.memory_total_mb > 0);
        assert!(snap.timestamp > 0);
    }

    #[tokio::test]
    async fn snapshot_is_persisted_to_repository() {
        let svc = service();
        svc.snapshot().await.unwrap();
        let history = svc.history(0, i64::MAX).await.unwrap();
        assert_eq!(history.len(), 1);
    }

    #[tokio::test]
    async fn record_task_completion_updates_counters() {
        let svc = service();
        svc.record_task_completion(100, true).await.unwrap();
        svc.record_task_completion(200, true).await.unwrap();
        svc.record_task_completion(50, false).await.unwrap();

        let snap = svc.snapshot().await.unwrap();
        assert_eq!(snap.tasks_completed, 2);
        assert_eq!(snap.tasks_failed, 1);
        assert_eq!(snap.average_latency_ms, 150);
    }

    #[tokio::test]
    async fn record_cache_counters_update_snapshot() {
        let svc = service();
        svc.record_cache_hit().await.unwrap();
        svc.record_cache_hit().await.unwrap();
        svc.record_cache_miss().await.unwrap();

        let snap = svc.snapshot().await.unwrap();
        assert_eq!(snap.cache_hits, 2);
        assert_eq!(snap.cache_misses, 1);
    }
}
