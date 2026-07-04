//! Alerting service for metrics thresholds.
//!
//! Evaluates a [`MetricsSnapshot`] against user-defined [`AlertThresholds`] and
//! emits [`Event::Alert`] messages on the event bus when a threshold is crossed.

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::bus::Event;
use crate::metrics::{GpuMetrics, MetricsSnapshot};
use crate::services::event_service::EventService;

/// Severity of an alert.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AlertSeverity {
    /// Degraded condition; human attention may be required soon.
    Warning,
    /// Severe condition; immediate action is recommended.
    Critical,
}

/// A single triggered alert.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Alert {
    pub severity: AlertSeverity,
    pub message: String,
    pub metric: String,
    pub value: String,
    pub threshold: String,
}

/// Threshold configuration. Any `None` threshold is not evaluated.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct AlertThresholds {
    /// CPU usage percentage that triggers a warning.
    pub cpu_percent_warning: Option<f32>,
    /// CPU usage percentage that triggers a critical alert.
    pub cpu_percent_critical: Option<f32>,
    /// RAM usage percentage that triggers a warning.
    pub ram_percent_warning: Option<f32>,
    /// RAM usage percentage that triggers a critical alert.
    pub ram_percent_critical: Option<f32>,
    /// GPU temperature (°C) that triggers a warning.
    pub gpu_temp_warning_c: Option<u32>,
    /// GPU temperature (°C) that triggers a critical alert.
    pub gpu_temp_critical_c: Option<u32>,
    /// Minimum acceptable free disk percentage.
    pub disk_free_percent_warning: Option<u8>,
    /// Minimum success rate (0.0..=1.0) before a warning is raised.
    pub success_rate_warning: Option<f64>,
}

/// Evaluates metrics and publishes alerts.
#[async_trait]
pub trait AlertService: Send + Sync {
    /// Evaluate `snapshot` against configured thresholds.
    ///
    /// Returns the list of alerts that were emitted. Implementations are
    /// expected to publish each alert on the event bus.
    async fn check(&self, snapshot: &MetricsSnapshot) -> Vec<Alert>;
}

/// Default [`AlertService`] implementation.
pub struct AlertServiceImpl<E: EventService> {
    thresholds: AlertThresholds,
    events: Arc<E>,
}

impl<E: EventService> AlertServiceImpl<E> {
    pub fn new(thresholds: AlertThresholds, events: Arc<E>) -> Self {
        Self { thresholds, events }
    }

    fn emit_alert(&self, alert: Alert, alerts: &mut Vec<Alert>) {
        self.events.publish(Event::Alert {
            severity: alert.severity,
            message: alert.message.clone(),
            metric: alert.metric.clone(),
            value: alert.value.clone(),
            threshold: alert.threshold.clone(),
        });
        alerts.push(alert);
    }

    fn gpu_temp_alert(&self, gpu: &GpuMetrics) -> Option<Alert> {
        let temp = gpu.temperature_c.unwrap_or(0);
        if let Some(critical) = self.thresholds.gpu_temp_critical_c
            && temp >= critical
        {
            return Some(Alert {
                severity: AlertSeverity::Critical,
                message: format!(
                    "GPU {} temperature {}°C exceeds critical threshold {}°C",
                    gpu.name, temp, critical
                ),
                metric: format!("gpu.{}.temperature_c", gpu.name),
                value: gpu
                    .temperature_c
                    .map_or_else(|| "unknown".into(), |v| v.to_string()),
                threshold: critical.to_string(),
            });
        }
        if let Some(warning) = self.thresholds.gpu_temp_warning_c
            && temp >= warning
        {
            return Some(Alert {
                severity: AlertSeverity::Warning,
                message: format!(
                    "GPU {} temperature {}°C exceeds warning threshold {}°C",
                    gpu.name, temp, warning
                ),
                metric: format!("gpu.{}.temperature_c", gpu.name),
                value: gpu
                    .temperature_c
                    .map_or_else(|| "unknown".into(), |v| v.to_string()),
                threshold: warning.to_string(),
            });
        }
        None
    }

    fn cpu_alert(&self, snapshot: &MetricsSnapshot) -> Option<Alert> {
        let usage = snapshot.cpu_usage_percent;
        if let Some(critical) = self.thresholds.cpu_percent_critical
            && usage >= critical
        {
            return Some(Alert {
                severity: AlertSeverity::Critical,
                message: format!(
                    "CPU usage {:.1}% exceeds critical threshold {:.1}%",
                    usage, critical
                ),
                metric: "cpu.usage_percent".into(),
                value: usage.to_string(),
                threshold: critical.to_string(),
            });
        }
        if let Some(warning) = self.thresholds.cpu_percent_warning
            && usage >= warning
        {
            return Some(Alert {
                severity: AlertSeverity::Warning,
                message: format!(
                    "CPU usage {:.1}% exceeds warning threshold {:.1}%",
                    usage, warning
                ),
                metric: "cpu.usage_percent".into(),
                value: usage.to_string(),
                threshold: warning.to_string(),
            });
        }
        None
    }

    fn ram_alert(&self, snapshot: &MetricsSnapshot) -> Option<Alert> {
        let usage = Self::ram_percent(snapshot);
        if let Some(critical) = self.thresholds.ram_percent_critical
            && usage >= critical
        {
            return Some(Alert {
                severity: AlertSeverity::Critical,
                message: format!(
                    "RAM usage {:.1}% exceeds critical threshold {:.1}%",
                    usage, critical
                ),
                metric: "memory.usage_percent".into(),
                value: usage.to_string(),
                threshold: critical.to_string(),
            });
        }
        if let Some(warning) = self.thresholds.ram_percent_warning
            && usage >= warning
        {
            return Some(Alert {
                severity: AlertSeverity::Warning,
                message: format!(
                    "RAM usage {:.1}% exceeds warning threshold {:.1}%",
                    usage, warning
                ),
                metric: "memory.usage_percent".into(),
                value: usage.to_string(),
                threshold: warning.to_string(),
            });
        }
        None
    }

    fn disk_alert(&self, snapshot: &MetricsSnapshot) -> Option<Alert> {
        let min_free = self.thresholds.disk_free_percent_warning?;
        let free = Self::disk_free_percent(snapshot);
        if free >= min_free as f32 {
            return None;
        }
        Some(Alert {
            severity: AlertSeverity::Warning,
            message: format!("Free disk space {:.1}% below threshold {}%", free, min_free),
            metric: "disk.free_percent".into(),
            value: free.to_string(),
            threshold: min_free.to_string(),
        })
    }

    fn success_rate_alert(&self, snapshot: &MetricsSnapshot) -> Option<Alert> {
        let min_rate = self.thresholds.success_rate_warning?;
        let rate = Self::success_rate(snapshot);
        if rate >= min_rate {
            return None;
        }
        Some(Alert {
            severity: AlertSeverity::Warning,
            message: format!(
                "Task success rate {:.1}% below threshold {:.1}%",
                rate * 100.0,
                min_rate * 100.0
            ),
            metric: "business.success_rate".into(),
            value: rate.to_string(),
            threshold: min_rate.to_string(),
        })
    }

    fn ram_percent(snapshot: &MetricsSnapshot) -> f32 {
        if snapshot.memory_total_mb == 0 {
            return 0.0;
        }
        100.0 * snapshot.memory_used_mb as f32 / snapshot.memory_total_mb as f32
    }

    fn disk_free_percent(snapshot: &MetricsSnapshot) -> f32 {
        if snapshot.disk_total_gb == 0 {
            return 0.0;
        }
        let free_gb = snapshot.disk_total_gb.saturating_sub(snapshot.disk_used_gb) as f32;
        100.0 * free_gb / snapshot.disk_total_gb as f32
    }

    fn success_rate(snapshot: &MetricsSnapshot) -> f64 {
        let total = snapshot
            .tasks_completed
            .saturating_add(snapshot.tasks_failed);
        if total == 0 {
            return 1.0;
        }
        snapshot.tasks_completed as f64 / total as f64
    }
}

#[async_trait]
impl<E: EventService> AlertService for AlertServiceImpl<E> {
    async fn check(&self, snapshot: &MetricsSnapshot) -> Vec<Alert> {
        let mut alerts = Vec::new();

        for gpu in &snapshot.gpus {
            if let Some(alert) = self.gpu_temp_alert(gpu) {
                self.emit_alert(alert, &mut alerts);
            }
        }

        if let Some(alert) = self.cpu_alert(snapshot) {
            self.emit_alert(alert, &mut alerts);
        }

        if let Some(alert) = self.ram_alert(snapshot) {
            self.emit_alert(alert, &mut alerts);
        }

        if let Some(alert) = self.disk_alert(snapshot) {
            self.emit_alert(alert, &mut alerts);
        }

        if let Some(alert) = self.success_rate_alert(snapshot) {
            self.emit_alert(alert, &mut alerts);
        }

        alerts
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    struct MockEvents {
        published: Mutex<Vec<Event>>,
    }

    #[async_trait]
    impl EventService for MockEvents {
        fn publish(&self, event: Event) {
            self.published.lock().unwrap().push(event);
        }

        fn subscribe(&self) -> tokio::sync::broadcast::Receiver<Event> {
            let (tx, _rx) = tokio::sync::broadcast::channel(1);
            tx.subscribe()
        }

        async fn start_handler(&self, _handler: Arc<dyn crate::services::EventHandler>) {}
    }

    fn snapshot() -> MetricsSnapshot {
        MetricsSnapshot {
            timestamp: 0,
            cpu_usage_percent: 50.0,
            memory_used_mb: 4_096,
            memory_total_mb: 16_384,
            swap_used_mb: 0,
            swap_total_mb: 0,
            disk_used_gb: 500,
            disk_total_gb: 1_000,
            network_rx_mb: 0,
            network_tx_mb: 0,
            gpus: vec![],
            tasks_completed: 8,
            tasks_failed: 2,
            average_latency_ms: 100,
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    #[tokio::test]
    async fn no_alerts_when_thresholds_not_exceeded() {
        let events = Arc::new(MockEvents {
            published: Mutex::new(Vec::new()),
        });
        let svc = AlertServiceImpl::new(AlertThresholds::default(), events.clone());
        let alerts = svc.check(&snapshot()).await;
        assert!(alerts.is_empty());
        assert!(events.published.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn fires_warning_when_cpu_usage_exceeds() {
        let events = Arc::new(MockEvents {
            published: Mutex::new(Vec::new()),
        });
        let thresholds = AlertThresholds {
            cpu_percent_warning: Some(40.0),
            ..Default::default()
        };
        let svc = AlertServiceImpl::new(thresholds, events.clone());
        let mut snap = snapshot();
        snap.cpu_usage_percent = 55.0;

        let alerts = svc.check(&snap).await;

        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].severity, AlertSeverity::Warning);
        assert!(events.published.lock().unwrap().iter().any(|e| matches!(
            e,
            Event::Alert {
                severity: AlertSeverity::Warning,
                ..
            }
        )));
    }

    #[tokio::test]
    async fn fires_critical_when_gpu_temp_exceeds() {
        let events = Arc::new(MockEvents {
            published: Mutex::new(Vec::new()),
        });
        let thresholds = AlertThresholds {
            gpu_temp_critical_c: Some(85),
            ..Default::default()
        };
        let svc = AlertServiceImpl::new(thresholds, events.clone());
        let mut snap = snapshot();
        snap.gpus = vec![crate::metrics::GpuMetrics {
            name: "RTX 4090".into(),
            usage_percent: 90,
            memory_used_mb: 20_000,
            memory_total_mb: 24_000,
            temperature_c: Some(90),
            power_draw_w: Some(350),
            fan_speed_percent: Some(80),
        }];

        let alerts = svc.check(&snap).await;

        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].severity, AlertSeverity::Critical);
        assert!(alerts[0].metric.contains("temperature_c"));
    }

    #[tokio::test]
    async fn fires_warning_when_free_disk_space_below_threshold() {
        let events = Arc::new(MockEvents {
            published: Mutex::new(Vec::new()),
        });
        let thresholds = AlertThresholds {
            disk_free_percent_warning: Some(50),
            ..Default::default()
        };
        let svc = AlertServiceImpl::new(thresholds, events.clone());
        let mut snap = snapshot();
        snap.disk_used_gb = 800;
        snap.disk_total_gb = 1_000;

        let alerts = svc.check(&snap).await;

        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].metric, "disk.free_percent");
    }

    #[tokio::test]
    async fn fires_warning_when_success_rate_below_threshold() {
        let events = Arc::new(MockEvents {
            published: Mutex::new(Vec::new()),
        });
        let thresholds = AlertThresholds {
            success_rate_warning: Some(0.9),
            ..Default::default()
        };
        let svc = AlertServiceImpl::new(thresholds, events.clone());
        let mut snap = snapshot();
        snap.tasks_completed = 5;
        snap.tasks_failed = 5;

        let alerts = svc.check(&snap).await;

        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].metric, "business.success_rate");
        assert!(alerts[0].message.contains("50.0%"));
    }

    #[tokio::test]
    async fn warning_is_suppressed_by_critical_for_same_metric() {
        let events = Arc::new(MockEvents {
            published: Mutex::new(Vec::new()),
        });
        let thresholds = AlertThresholds {
            cpu_percent_warning: Some(30.0),
            cpu_percent_critical: Some(60.0),
            ..Default::default()
        };
        let svc = AlertServiceImpl::new(thresholds, events.clone());
        let mut snap = snapshot();
        snap.cpu_usage_percent = 75.0;

        let alerts = svc.check(&snap).await;

        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].severity, AlertSeverity::Critical);
    }
}
