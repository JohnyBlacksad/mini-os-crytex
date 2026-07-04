//! Wasmtime-backed WASM sandbox implementing capability-based isolation.

use std::path::PathBuf;

use anyhow::Context;
use wasmtime::{Config, Engine, Linker, Module, Store, StoreLimits, StoreLimitsBuilder};
use wasmtime_wasi::pipe::{ClosedInputStream, MemoryOutputPipe};
use wasmtime_wasi::preview1::{self, WasiP1Ctx};
use wasmtime_wasi::{DirPerms, FilePerms, I32Exit, WasiCtxBuilder};

/// A preopened directory exposed to the guest under a virtual path.
#[derive(Clone, Debug)]
pub struct PreopenDir {
    pub host_path: PathBuf,
    pub guest_path: String,
    pub read: bool,
    pub write: bool,
}

/// Configuration for a single WASM sandbox execution.
#[derive(Clone, Debug)]
pub struct WasiConfig {
    /// Maximum linear memory size in bytes (per memory).
    pub max_memory_bytes: usize,
    /// Maximum WebAssembly stack size in bytes.
    pub max_stack_bytes: usize,
    /// Initial fuel budget. Execution traps when exhausted.
    pub fuel: u64,
    /// Directories the guest is allowed to access.
    pub preopened_dirs: Vec<PreopenDir>,
    /// Maximum bytes captured from stdout and stderr.
    pub capture_capacity: usize,
    /// Arguments passed to the guest.
    pub args: Vec<String>,
    /// Environment variables passed to the guest.
    pub env: Vec<(String, String)>,
}

impl Default for WasiConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: 64 * 1024 * 1024,
            max_stack_bytes: 512 * 1024,
            fuel: 10_000_000,
            preopened_dirs: Vec::new(),
            capture_capacity: 64 * 1024,
            args: Vec::new(),
            env: Vec::new(),
        }
    }
}

/// Result of executing a WASM module inside the sandbox.
#[derive(Clone, Debug)]
pub struct WasiRunResult {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub fuel_consumed: u64,
}

/// Errors emitted by the WASM sandbox.
#[derive(thiserror::Error, Debug)]
pub enum WasiError {
    #[error("wasmtime engine error: {0}")]
    Engine(#[from] anyhow::Error),
    #[error("wasm trap: {0}")]
    Trap(String),
    #[error("guest exceeded memory limit")]
    MemoryLimit,
    #[error("guest exceeded fuel budget")]
    FuelExhausted,
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

impl WasiError {
    fn classify_trap(err: &wasmtime::Error) -> Self {
        if err.is::<I32Exit>() {
            // Not a real trap; handled by the caller.
            WasiError::Engine(anyhow::Error::msg(err.to_string()))
        } else if err.to_string().contains("out of fuel") {
            WasiError::FuelExhausted
        } else if err.to_string().contains("memory") {
            WasiError::MemoryLimit
        } else {
            WasiError::Trap(err.to_string())
        }
    }
}

/// Wasmtime-based implementation of a WASM sandbox.
pub struct WasiBackend;

impl WasiBackend {
    pub fn new() -> Self {
        Self
    }

    /// Execute the module's `_start` function under the configured policy.
    pub async fn run(
        &self,
        wasm_bytes: &[u8],
        config: &WasiConfig,
    ) -> Result<WasiRunResult, WasiError> {
        let mut engine_config = Config::new();
        engine_config
            .async_support(true)
            .consume_fuel(true)
            .max_wasm_stack(config.max_stack_bytes);

        let engine = Engine::new(&engine_config)?;
        let module = Module::new(&engine, wasm_bytes)?;

        let mut linker: Linker<SandboxState> = Linker::new(&engine);
        preview1::add_to_linker_async(&mut linker, |state| &mut state.wasi)?;

        let stdout_pipe = MemoryOutputPipe::new(config.capture_capacity);
        let stderr_pipe = MemoryOutputPipe::new(config.capture_capacity);

        let mut builder = WasiCtxBuilder::new();
        builder
            .stdin(ClosedInputStream)
            .stdout(stdout_pipe.clone())
            .stderr(stderr_pipe.clone())
            .args(&config.args)
            .envs(&config.env);

        for preopen in &config.preopened_dirs {
            let dir_perms = match (preopen.read, preopen.write) {
                (true, true) => DirPerms::all(),
                (true, false) => DirPerms::READ,
                (false, true) => DirPerms::MUTATE,
                (false, false) => DirPerms::empty(),
            };
            let file_perms = match (preopen.read, preopen.write) {
                (true, true) => FilePerms::all(),
                (true, false) => FilePerms::READ,
                (false, true) => FilePerms::WRITE,
                (false, false) => FilePerms::empty(),
            };
            builder
                .preopened_dir(
                    &preopen.host_path,
                    &preopen.guest_path,
                    dir_perms,
                    file_perms,
                )
                .with_context(|| format!("failed to preopen {}", preopen.host_path.display()))?;
        }

        let wasi = builder.build_p1();
        let limits = StoreLimitsBuilder::new()
            .memory_size(config.max_memory_bytes)
            .instances(1)
            .memories(1)
            .build();

        let state = SandboxState { wasi, limits };
        let mut store = Store::new(&engine, state);
        store.limiter(|state| &mut state.limits);
        store.set_fuel(config.fuel)?;

        let initial_fuel = store.get_fuel()?;

        let instance = linker
            .instantiate_async(&mut store, &module)
            .await
            .map_err(|e| WasiError::classify_trap(&e))?;

        let func = instance
            .get_typed_func::<(), ()>(&mut store, "_start")
            .map_err(WasiError::Engine)?;

        let exit_code = match func.call_async(&mut store, ()).await {
            Ok(()) => 0,
            Err(err) => {
                if let Some(exit) = err.downcast_ref::<I32Exit>() {
                    exit.0
                } else {
                    return Err(WasiError::classify_trap(&err));
                }
            }
        };

        let final_fuel = store.get_fuel()?;
        let fuel_consumed = initial_fuel.saturating_sub(final_fuel);

        drop(store);

        let stdout = String::from_utf8_lossy(&stdout_pipe.contents()).to_string();
        let stderr = String::from_utf8_lossy(&stderr_pipe.contents()).to_string();

        Ok(WasiRunResult {
            exit_code,
            stdout,
            stderr,
            fuel_consumed,
        })
    }
}

impl Default for WasiBackend {
    fn default() -> Self {
        Self::new()
    }
}

struct SandboxState {
    wasi: WasiP1Ctx,
    limits: StoreLimits,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal hand-encoded WASM module exporting `_start` as a no-op function.
    fn empty_start_module() -> Vec<u8> {
        vec![
            0x00, 0x61, 0x73, 0x6d, // magic
            0x01, 0x00, 0x00, 0x00, // version
            // type section
            0x01, 0x04, 0x01, 0x60, 0x00, 0x00, // func section
            0x03, 0x02, 0x01, 0x00, // export section: "_start"
            0x07, 0x0a, 0x01, 0x06, 0x5f, 0x73, 0x74, 0x61, 0x72, 0x74, 0x00, 0x00,
            // code section
            0x0a, 0x04, 0x01, 0x02, 0x00, 0x0b,
        ]
    }

    #[tokio::test]
    async fn should_run_noop_module_with_exit_code_zero() {
        let sandbox = WasiBackend::new();
        let config = WasiConfig::default();

        let result = sandbox.run(&empty_start_module(), &config).await.unwrap();

        assert_eq!(result.exit_code, 0);
        assert!(result.stdout.is_empty());
        assert!(result.stderr.is_empty());
        assert!(result.fuel_consumed > 0);
    }

    #[tokio::test]
    async fn should_return_error_when_start_export_missing() {
        // Minimal module with no exports.
        let wasm = vec![
            0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x04, 0x01, 0x60, 0x00, 0x00,
            0x03, 0x02, 0x01, 0x00, 0x0a, 0x04, 0x01, 0x02, 0x00, 0x0b,
        ];

        let sandbox = WasiBackend::new();
        let err = sandbox
            .run(&wasm, &WasiConfig::default())
            .await
            .unwrap_err();

        assert!(
            matches!(err, WasiError::Engine(_)),
            "expected missing _start to be an engine error, got {err:?}"
        );
    }

    #[test]
    fn default_config_uses_deny_first_policy() {
        let config = WasiConfig::default();

        assert!(config.preopened_dirs.is_empty());
        assert!(config.env.is_empty());
        assert!(config.args.is_empty());
        assert!(config.fuel > 0);
        assert!(config.max_memory_bytes > 0);
    }
}
