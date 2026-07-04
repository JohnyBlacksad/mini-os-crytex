# crytex-sandbox

Secure, pluggable sandbox backends for running agent tasks.

## Backends

- **`DockerBackend`** (preferred): runs each command in an ephemeral Docker container.
  - Hardened: `cap_drop ALL`, `no-new-privileges`, read-only rootfs, non-root user.
  - Default-deny network; egress allowed only when `SandboxNetwork::Allow` is set.
  - Resource limits: memory, CPU shares, wall-clock timeout.
  - Bind-mounts project/cache directories into `/workspace`.
- **`HostBackend`** (fallback): runs commands directly on the host with `tokio::process::Command`.
  - No real isolation; only for local development when Docker is unavailable.
  - Translates guest working directories back to host paths using mounts.
- **`WasmtimeSandbox`**: WASM/WASI Preview 1 sandbox for isolated plugins and scripts.
  - Fuel, memory, and stack limits; preopened directories; stdout/stderr capture.

## Orchestration

`SandboxOrchestrator::auto().await` tries Docker first and falls back to the host.

## Images

Base images live in `images/`:

- `crytex/sandbox-rust`
- `crytex/sandbox-node`
- `crytex/sandbox-python`

Build them with:

```bash
docker build -t crytex/sandbox-rust:latest -f images/rust/Dockerfile images/rust/
docker build -t crytex/sandbox-node:latest -f images/node/Dockerfile images/node/
docker build -t crytex/sandbox-python:latest -f images/python/Dockerfile images/python/
```

## Testing

```bash
cargo test -p crytex-sandbox
```

Docker-backed tests require a running Docker daemon and are marked `#[ignore]` or skipped when Docker is unavailable.
