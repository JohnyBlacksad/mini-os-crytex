//! Hardware detection for local inference.
//!
//! Provides a small, mockable abstraction over GPU/driver detection so the
//! kernel can default to GPU and fall back to CPU without hard-coding
//! platform-specific checks.

/// Detected compute device for local inference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceKind {
    /// No usable GPU; run on CPU.
    Cpu,
    /// NVIDIA GPU with detected VRAM (MiB).
    Cuda {
        name: String,
        vram_mb: usize,
        driver_version: String,
    },
    /// Apple Silicon / Metal GPU.
    Metal { name: String },
}

impl DeviceKind {
    pub fn is_gpu(&self) -> bool {
        !matches!(self, DeviceKind::Cpu)
    }

    pub fn vram_mb(&self) -> Option<usize> {
        match self {
            DeviceKind::Cuda { vram_mb, .. } => Some(*vram_mb),
            _ => None,
        }
    }
}

/// Recommends device and `gpu_layers` for a local backend.
#[derive(Debug, Clone)]
pub struct HardwareRecommendation {
    pub device: DeviceKind,
    /// `None` means "GPU with automatic layer placement".
    pub gpu_layers: Option<usize>,
    pub reason: String,
}

/// Detects available hardware.
pub trait HardwareDetector: Send + Sync {
    fn detect(&self) -> DeviceKind;
}

/// Platform-aware detector that shells out to `nvidia-smi` on Windows/Linux
/// and checks Metal on macOS.
#[derive(Debug, Clone, Default)]
pub struct SystemHardwareDetector;

impl SystemHardwareDetector {
    pub fn new() -> Self {
        Self
    }
}

impl HardwareDetector for SystemHardwareDetector {
    fn detect(&self) -> DeviceKind {
        #[cfg(target_os = "macos")]
        {
            if let Some(metal) = detect_metal() {
                return metal;
            }
        }

        match detect_nvidia_gpu() {
            Some(device) => device,
            None => DeviceKind::Cpu,
        }
    }
}

/// Parse `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits`.
fn detect_nvidia_gpu() -> Option<DeviceKind> {
    let output = run_nvidia_smi()?;
    parse_nvidia_smi_csv(&output)
}

fn run_nvidia_smi() -> Option<String> {
    let candidates = [
        "nvidia-smi",
        #[cfg(target_os = "windows")]
        "C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe",
    ];

    for cmd in candidates {
        if let Ok(out) = std::process::Command::new(cmd)
            .args([
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ])
            .output()
            && out.status.success()
        {
            return String::from_utf8(out.stdout).ok();
        }
    }
    None
}

fn parse_nvidia_smi_csv(output: &str) -> Option<DeviceKind> {
    let line = output.lines().next()?.trim();
    if line.is_empty() {
        return None;
    }
    let mut parts = line.split(',').map(|s| s.trim());
    let name = parts.next()?.to_string();
    let vram_str = parts.next()?;
    let driver = parts.next().unwrap_or("unknown").to_string();

    // memory.total can be reported as "24564 MiB" or just "24564".
    let vram_mb: usize = vram_str
        .split_whitespace()
        .next()
        .and_then(|s| s.parse().ok())?;

    Some(DeviceKind::Cuda {
        name,
        vram_mb,
        driver_version: driver,
    })
}

#[cfg(target_os = "macos")]
fn detect_metal() -> Option<DeviceKind> {
    use std::process::Command;
    let out = Command::new("system_profiler")
        .args(["SPDisplaysDataType", "-json"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8(out.stdout).ok()?;
    let json: serde_json::Value = serde_json::from_str(&text).ok()?;
    let entries = json.get("SPDisplaysDataType")?.as_array()?;
    for entry in entries {
        if entry.get("spdisplays_metal_status")?.as_str() == Some("supported") {
            let name = entry
                .get("sppci_model")
                .or_else(|| entry.get("_name"))?
                .as_str()?
                .to_string();
            return Some(DeviceKind::Metal { name });
        }
    }
    None
}

/// Build a recommendation for the local `mistral.rs` backend.
pub fn recommend_local_device(
    detector: &dyn HardwareDetector,
    user_override: Option<usize>,
) -> HardwareRecommendation {
    let device = detector.detect();

    if let Some(layers) = user_override {
        return HardwareRecommendation {
            device: device.clone(),
            gpu_layers: Some(layers),
            reason: format!("user override: gpu_layers={}", layers),
        };
    }

    match &device {
        DeviceKind::Cpu => HardwareRecommendation {
            device,
            gpu_layers: Some(0),
            reason: "no usable GPU detected; using CPU".to_string(),
        },
        DeviceKind::Cuda { name, vram_mb, .. } => HardwareRecommendation {
            device: device.clone(),
            gpu_layers: None,
            reason: format!("CUDA GPU detected: {} ({} MiB)", name, vram_mb),
        },
        DeviceKind::Metal { name } => HardwareRecommendation {
            device: device.clone(),
            gpu_layers: None,
            reason: format!("Metal GPU detected: {}", name),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixedDetector(DeviceKind);

    impl HardwareDetector for FixedDetector {
        fn detect(&self) -> DeviceKind {
            self.0.clone()
        }
    }

    #[test]
    fn recommendation_defaults_to_gpu() {
        let detector = FixedDetector(DeviceKind::Cuda {
            name: "RTX 4090".into(),
            vram_mb: 24_000,
            driver_version: "531".into(),
        });
        let rec = recommend_local_device(&detector, None);
        assert!(rec.device.is_gpu());
        assert_eq!(rec.gpu_layers, None);
    }

    #[test]
    fn recommendation_falls_back_to_cpu() {
        let detector = FixedDetector(DeviceKind::Cpu);
        let rec = recommend_local_device(&detector, None);
        assert!(!rec.device.is_gpu());
        assert_eq!(rec.gpu_layers, Some(0));
    }

    #[test]
    fn user_override_respected() {
        let detector = FixedDetector(DeviceKind::Cuda {
            name: "RTX 4090".into(),
            vram_mb: 24_000,
            driver_version: "531".into(),
        });
        let rec = recommend_local_device(&detector, Some(0));
        assert_eq!(rec.gpu_layers, Some(0));
    }

    #[test]
    fn parse_nvidia_smi_csv_works() {
        let output = "NVIDIA GeForce RTX 4090, 24564 MiB, 531.41\n";
        let device = parse_nvidia_smi_csv(output).unwrap();
        assert_eq!(
            device,
            DeviceKind::Cuda {
                name: "NVIDIA GeForce RTX 4090".into(),
                vram_mb: 24564,
                driver_version: "531.41".into(),
            }
        );
    }
}
