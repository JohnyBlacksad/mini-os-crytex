# Backend Acceptance Harness

`crytex backend-acceptance --full --json` is the canonical backend proof
command. It runs the complete backend CLI path and emits one JSON artifact.

## Command

```powershell
cargo run -p crytex-kernel -- backend-acceptance --full --json --deterministic
```

Optional report file:

```powershell
cargo run -p crytex-kernel -- backend-acceptance --full --json --deterministic --report-path reports\backend-acceptance.json
```

Real runtime modes:

```powershell
cargo run -p crytex-kernel -- backend-acceptance --full --json --runtime ollama --live-model qwen3.5:9b --live-url http://localhost:11434
cargo run -p crytex-kernel --features mistral -- backend-acceptance --full --json --runtime mistral --live-model A:\models\model.gguf
```

## Acceptance Chain

The harness proves this ordered chain:

```text
doctor -> project open -> index -> RAG rerank -> goal -> plan -> kanban
-> run -> critic -> remediation -> reward -> evolution evidence -> diag export
```

The command exits:

- `0` when every required gate passes;
- `1` for command/config/runtime errors;
- `2` when the harness ran but a backend acceptance gate failed.

## JSON Artifact

The top-level artifact contains:

- `proof_type = "backend_acceptance"`;
- `profile = "full"`;
- `runtime_mode = deterministic | ollama | mistral`;
- `doctor_status`;
- ordered `stages`;
- nested `kernel_proof`;
- final `passed` boolean.

The nested proof includes project id, trace id, indexed file/chunk counts,
diagnostics artifact path, benchmark ids, prompt evolution evidence, LoRA
adapter evidence, and individual proof gates.

## Test Profiles

Workspace profiles are defined in `Cargo.toml`:

```powershell
cargo test --profile fast -p crytex-core -p crytex-kernel
cargo test --profile integration -p crytex-core -p crytex-kernel
cargo test --profile runtime -p crytex-tauri --test e2e_ollama_start_run
cargo test --profile network -p crytex-core real_hf -- --ignored
cargo test --profile cuda -p crytex-kernel --features cuda
cargo test --profile full --workspace
```

Runtime/network/CUDA profiles are explicit by design: they may download models,
call local services, or require GPU toolchains.

## Windows Runtime Lock

`crates/crytex-tauri/tests/e2e_ollama_start_run.rs` serializes Ollama E2E tests
with a process-local Tokio mutex. This prevents Windows file/model locks from
making runtime tests flaky while still keeping the fast deterministic suite
parallel.
