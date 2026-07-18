//! Hardware detection for local inference.
//!
//! Provides a small, mockable abstraction over GPU/driver detection so the
//! kernel can default to GPU and fall back to CPU without hard-coding
//! platform-specific checks.

use serde::{Deserialize, Serialize};
use std::path::Path;

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

/// CUDA build/runtime readiness reported before a GPU inference attempt.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CudaToolchainStatus {
    pub gpu_detected: bool,
    pub nvcc_available: bool,
    pub msvc_cl_available: bool,
    pub msvc_cl_path: Option<String>,
    pub nvcc_ccbin: Option<String>,
    pub recommended_nvcc_ccbin: Option<String>,
    pub ready: bool,
    pub diagnostics: Vec<String>,
}

/// Pure inputs for CUDA preflight evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaToolchainProbe {
    pub device: DeviceKind,
    pub nvcc_available: bool,
    pub msvc_cl_available: bool,
    pub msvc_cl_path: Option<String>,
    pub nvcc_ccbin: Option<String>,
    pub nvcc_ccbin_exists: bool,
    pub is_windows: bool,
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

/// Detect whether the local CUDA stack is ready enough to build/run CUDA backends.
pub fn detect_cuda_toolchain_status(detector: &dyn HardwareDetector) -> CudaToolchainStatus {
    let nvcc_ccbin = std::env::var("NVCC_CCBIN")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let nvcc_ccbin_exists = nvcc_ccbin
        .as_deref()
        .map(Path::new)
        .is_some_and(Path::exists);

    build_cuda_toolchain_status(CudaToolchainProbe {
        device: detector.detect(),
        nvcc_available: command_can_start("nvcc", &["--version"]),
        msvc_cl_available: command_can_start("cl.exe", &["/help"]),
        msvc_cl_path: find_msvc_cl_path(),
        nvcc_ccbin,
        nvcc_ccbin_exists,
        is_windows: cfg!(target_os = "windows"),
    })
}

/// Evaluate CUDA readiness from pre-collected facts.
pub fn build_cuda_toolchain_status(probe: CudaToolchainProbe) -> CudaToolchainStatus {
    let gpu_detected = matches!(probe.device, DeviceKind::Cuda { .. });
    let recommended_nvcc_ccbin = probe
        .nvcc_ccbin
        .clone()
        .filter(|_| probe.nvcc_ccbin_exists)
        .or_else(|| probe.msvc_cl_path.clone());
    let windows_compiler_ready =
        !probe.is_windows || probe.msvc_cl_available || recommended_nvcc_ccbin.is_some();
    let ready = gpu_detected && probe.nvcc_available && windows_compiler_ready;
    let mut diagnostics = Vec::new();

    if !gpu_detected {
        diagnostics.push("CUDA GPU was not detected; GPU inference will fall back or fail.".into());
    }
    if !probe.nvcc_available {
        diagnostics.push("nvcc is not available in PATH; CUDA-enabled crates cannot build.".into());
    }
    if probe.is_windows && !probe.msvc_cl_available && recommended_nvcc_ccbin.is_none() {
        diagnostics.push(
            "cl.exe is not available in PATH and NVCC_CCBIN does not point to an existing compiler; run from a VS Developer PowerShell or set NVCC_CCBIN.".into(),
        );
    } else if probe.is_windows && !probe.msvc_cl_available && probe.nvcc_ccbin_exists {
        diagnostics.push("cl.exe is not in PATH; using NVCC_CCBIN for CUDA builds.".into());
    } else if probe.is_windows && !probe.msvc_cl_available && probe.msvc_cl_path.is_some() {
        diagnostics
            .push("cl.exe is not in PATH; use recommended_nvcc_ccbin for CUDA builds.".into());
    }

    CudaToolchainStatus {
        gpu_detected,
        nvcc_available: probe.nvcc_available,
        msvc_cl_available: probe.msvc_cl_available,
        msvc_cl_path: probe.msvc_cl_path,
        nvcc_ccbin: probe.nvcc_ccbin,
        recommended_nvcc_ccbin,
        ready,
        diagnostics,
    }
}

fn command_can_start(program: &str, args: &[&str]) -> bool {
    std::process::Command::new(program)
        .args(args)
        .output()
        .is_ok()
}

#[cfg(target_os = "windows")]
fn find_msvc_cl_path() -> Option<String> {
    find_msvc_cl_path_with_vswhere(
        "C:\\Program Files (x86)\\Microsoft Visual Studio\\Installer\\vswhere.exe",
    )
    .or_else(find_msvc_cl_path_in_default_roots)
}

#[cfg(not(target_os = "windows"))]
fn find_msvc_cl_path() -> Option<String> {
    None
}

#[cfg(target_os = "windows")]
fn find_msvc_cl_path_with_vswhere(vswhere: &str) -> Option<String> {
    let out = std::process::Command::new(vswhere)
        .args([
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-find",
            "VC\\Tools\\MSVC\\**\\bin\\Hostx64\\x64\\cl.exe",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8(out.stdout)
        .ok()?
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && Path::new(line).exists())
        .map(ToOwned::to_owned)
}

#[cfg(target_os = "windows")]
fn find_msvc_cl_path_in_default_roots() -> Option<String> {
    [
        "C:\\Program Files\\Microsoft Visual Studio",
        "C:\\Program Files (x86)\\Microsoft Visual Studio",
    ]
    .into_iter()
    .filter_map(|root| std::fs::read_dir(root).ok())
    .flat_map(|entries| entries.filter_map(Result::ok))
    .map(|entry| entry.path())
    .filter(|path| path.is_dir())
    .find_map(find_host_x64_cl_under)
}

#[cfg(target_os = "windows")]
fn find_host_x64_cl_under(root: std::path::PathBuf) -> Option<String> {
    let tools = root.join("VC").join("Tools").join("MSVC");
    let versions = std::fs::read_dir(tools).ok()?;
    versions
        .filter_map(Result::ok)
        .map(|entry| {
            entry
                .path()
                .join("bin")
                .join("Hostx64")
                .join("x64")
                .join("cl.exe")
        })
        .find(|path| path.exists())
        .map(|path| path.display().to_string())
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

    #[test]
    fn cuda_preflight_is_ready_when_gpu_nvcc_and_windows_compiler_are_available() {
        let status = build_cuda_toolchain_status(CudaToolchainProbe {
            device: DeviceKind::Cuda {
                name: "RTX 5080".into(),
                vram_mb: 16_303,
                driver_version: "596.36".into(),
            },
            nvcc_available: true,
            msvc_cl_available: true,
            msvc_cl_path: None,
            nvcc_ccbin: None,
            nvcc_ccbin_exists: false,
            is_windows: true,
        });

        assert!(status.gpu_detected);
        assert!(status.nvcc_available);
        assert!(status.msvc_cl_available);
        assert!(status.ready);
        assert!(status.diagnostics.is_empty());
    }

    #[test]
    fn cuda_preflight_reports_missing_cl_exe_when_nvcc_cannot_build_on_windows() {
        let status = build_cuda_toolchain_status(CudaToolchainProbe {
            device: DeviceKind::Cuda {
                name: "RTX 5080".into(),
                vram_mb: 16_303,
                driver_version: "596.36".into(),
            },
            nvcc_available: true,
            msvc_cl_available: false,
            msvc_cl_path: None,
            nvcc_ccbin: None,
            nvcc_ccbin_exists: false,
            is_windows: true,
        });

        assert!(status.gpu_detected);
        assert!(status.nvcc_available);
        assert!(!status.msvc_cl_available);
        assert!(!status.ready);
        assert!(
            status
                .diagnostics
                .iter()
                .any(|line| line.contains("cl.exe"))
        );
    }

    #[test]
    fn cuda_preflight_accepts_existing_nvcc_ccbin_when_cl_exe_is_not_on_path() {
        let status = build_cuda_toolchain_status(CudaToolchainProbe {
            device: DeviceKind::Cuda {
                name: "RTX 5080".into(),
                vram_mb: 16_303,
                driver_version: "596.36".into(),
            },
            nvcc_available: true,
            msvc_cl_available: false,
            msvc_cl_path: None,
            nvcc_ccbin: Some("C:\\BuildTools\\VC\\Tools\\MSVC\\cl.exe".into()),
            nvcc_ccbin_exists: true,
            is_windows: true,
        });

        assert!(status.ready);
        assert_eq!(
            status.nvcc_ccbin.as_deref(),
            Some("C:\\BuildTools\\VC\\Tools\\MSVC\\cl.exe")
        );
        assert_eq!(
            status.recommended_nvcc_ccbin.as_deref(),
            Some("C:\\BuildTools\\VC\\Tools\\MSVC\\cl.exe")
        );
    }

    #[test]
    fn cuda_preflight_recommends_discovered_cl_path_when_cl_exe_is_not_on_path() {
        let status = build_cuda_toolchain_status(CudaToolchainProbe {
            device: DeviceKind::Cuda {
                name: "RTX 5080".into(),
                vram_mb: 16_303,
                driver_version: "596.36".into(),
            },
            nvcc_available: true,
            msvc_cl_available: false,
            msvc_cl_path: Some(
                "C:\\Program Files\\Microsoft Visual Studio\\18\\Community\\VC\\Tools\\MSVC\\14.51.36231\\bin\\Hostx64\\x64\\cl.exe".into(),
            ),
            nvcc_ccbin: None,
            nvcc_ccbin_exists: false,
            is_windows: true,
        });

        assert!(status.ready);
        assert!(status.nvcc_ccbin.is_none());
        assert_eq!(status.msvc_cl_path, status.recommended_nvcc_ccbin);
        assert!(
            status
                .diagnostics
                .iter()
                .any(|line| line.contains("recommended_nvcc_ccbin"))
        );
    }

    #[test]
    fn cuda_preflight_is_not_ready_without_a_cuda_gpu() {
        let status = build_cuda_toolchain_status(CudaToolchainProbe {
            device: DeviceKind::Cpu,
            nvcc_available: true,
            msvc_cl_available: true,
            msvc_cl_path: None,
            nvcc_ccbin: None,
            nvcc_ccbin_exists: false,
            is_windows: true,
        });

        assert!(!status.gpu_detected);
        assert!(!status.ready);
        assert!(status.diagnostics.iter().any(|line| line.contains("GPU")));
    }
}
