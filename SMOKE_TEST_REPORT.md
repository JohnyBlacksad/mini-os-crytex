# Crytex Smoke Test Report

Date: 2026-07-04

## Environment

- Workspace: `A:\Projects\mini-os-crytex`
- Cargo target dir used: `B:\crytex-target-audit`
- Reason: drive `A:` has almost no free space, so default Cargo target on `A:` is not usable for full checks.

## Passed Checks

### Real Ollama Model E2E

Date: 2026-07-05

Commands:

```powershell
$env:CARGO_TARGET_DIR='B:\crytex-target-audit'
$env:CRYTEX_E2E_OLLAMA_MODEL='qwen3.5:9b'
cargo test -p crytex-inference-ollama --test e2e_ollama -- --ignored --nocapture
cargo test -p crytex-inference-ollama --all-targets
cargo clippy -p crytex-inference-ollama --all-targets -- -D warnings
```

Result:

- Added a real ignored E2E test: `crates/crytex-inference-ollama/tests/e2e_ollama.rs`.
- Added a runnable smoke wrapper: `scripts/e2e-ollama-smoke.ps1`.
- The E2E test checks local Ollama, pulls the configured model if missing, sends a real generation task through `crytex-inference-ollama`, and asserts non-empty content plus token usage.
- Real local inference passed with already installed model `qwen3.5:9b`.
- Observed output:

```text
CRYTEX_E2E_RESULT model=qwen3.5:9b finish_reason=stop usage=TokenUsage { prompt_tokens: 54, completion_tokens: 19, total_tokens: 73 } content=I am the local model successfully executing your Crytex E2E smoke test as requested.
```

- Fixed Ollama thinking-model handling by sending `think: false` for normal chat generation. Without this, `qwen3.5:9b` returned an empty `message.content` and placed text in `message.thinking`.

Blocked:

- Downloading default small model `smollm2:135m` failed because Ollama stores blobs at `A:\ai_model\ollama`, and drive `A:` does not have enough free space:

```text
Ollama pull failed for smollm2:135m: There is not enough space on the disk.
```

Next required fix:

- Move Ollama model storage to a disk with space, for example by configuring `OLLAMA_MODELS` to a `B:` path and restarting Ollama, then rerun:

```powershell
.\scripts\e2e-ollama-smoke.ps1 -Model smollm2:135m
```

### Tauri UI Goal-First Shell

Date: 2026-07-05

Commands:

```powershell
npm run build
$env:CARGO_TARGET_DIR='B:\crytex-target-audit'
npm run tauri dev
```

Result:

- Reworked the Tauri UI shell around the product model: Workspace, Goals, Generated Tasks, Runs, Index/RAG, Models, Evolution, Observe.
- The primary UI action is now `New goal`; manual task creation is labeled `Debug task`.
- Wired existing TypeScript API calls into the UI for `submit_goal`, `approve_plan`, `reject_plan`, and `start_run`.
- Added an IDE placeholder area as a first-class Workspace region, with explicit next IPC surface for file tree/editor/LSP.
- Added plan approval controls in Goals and Inspector.
- Added Runs view that clearly labels the current executor as `tauri_stub_run`.
- Added visible first-class placeholder screens for Index/RAG, Models, and Evolution instead of hiding those systems.
- Fixed the bottom Observe drawer layout so it remains visible inside a 720px-tall viewport.
- Fixed stale background refresh errors leaking into modal forms.
- Verified frontend build passes.
- Verified Tauri desktop dev starts and the `Crytex` window can call Rust IPC: active project loaded, Observe showed `list_projects`, `kanban_state`, `list_tasks`, and `get_project_state` with `0 errors`.

Limitations:

- Automated Windows input into the Tauri webview was blocked by the Computer Use safety guard after external input was detected, so the full click-through `New goal -> approve -> start run` path still needs a short manual test in the open Tauri window.
- Index/RAG, Models, Evolution, and IDE screens still use explicit pending placeholders where IPC is not implemented yet.

### Goal Plan Approval Gate

Date: 2026-07-05

Commands:

```powershell
$env:CARGO_TARGET_DIR='B:\crytex-target-audit'
cargo test -p crytex-tauri sqlite_state -- --nocapture
cargo test -p crytex-tauri --all-targets
cargo test -p crytex-core --lib --tests
cargo clippy -p crytex-core -p crytex-tauri --all-targets -- -D warnings
npm run build
```

Result:

- `submit_goal` now creates a root goal in `Review` after planning.
- Generated plan tasks are held in `Backlog` and are not ready for execution before human approval.
- `approve_plan` releases generated tasks into `Pending` and marks the root goal approved/completed.
- `reject_plan` cancels generated tasks, stores human feedback on the root goal, and returns the goal to `Pending` for revision.
- Tauri IPC and TypeScript API expose `approve_plan` and `reject_plan`.
- Verified: `crytex-tauri` 12 tests passed, `crytex-core` 217 unit tests + 5 integration tests passed, frontend build passed, clippy passed for `crytex-core` and `crytex-tauri`.

### Stub Run Execution Gate

Date: 2026-07-05

Commands:

```powershell
$env:CARGO_TARGET_DIR='B:\crytex-target-audit'
cargo test -p crytex-tauri sqlite_state_starts_stub_run_after_plan_approval -- --nocapture
cargo test -p crytex-tauri --all-targets
cargo test -p crytex-core --lib --tests
cargo clippy -p crytex-core -p crytex-tauri --all-targets -- -D warnings
npm run build
```

Result:

- Added `start_run` as a Tauri IPC command and TypeScript API method.
- `start_run` attaches a deterministic stub result to the next ready project task and moves it to `Review`.
- Stub execution no longer marks tasks as `Completed`, because no real agent/model/tool work happened.
- Dependency chains remain blocked until reviewed work is explicitly approved by a real completion path.
- Verified path: create project -> submit goal -> generated plan -> approve plan -> start run -> first ready task moves to review with a stub result.
- Stub task results are marked with `source = "tauri_stub_run"` so the UI can distinguish smoke output from real agent execution.
- TypeScript response contract now uses `review_tasks` instead of `completed_tasks`.
- Startup repair converts old incorrectly completed `tauri_stub_run` tasks back to `Review` without touching real completed tasks.
- Verified: `crytex-tauri` 14 tests passed, frontend build passed, clippy passed for `crytex-tauri`.

### Core

Command:

```powershell
$env:CARGO_TARGET_DIR='B:\crytex-target-audit'
cargo test -p crytex-core --lib --tests
```

Result:

- 216 unit tests passed.
- 5 integration tests passed.
- Covered: task graph, orchestrator, watcher/indexer, RAG search helpers, prompt evolution, LoRA evolution, LoRA router, model manager, audit logs, feedback loop, metrics, worker/scheduler, context assembly.

### Tauri Backend

Command:

```powershell
$env:CARGO_TARGET_DIR='B:\crytex-target-audit'
cargo test -p crytex-tauri --all-targets
```

Result:

- 11 tests passed.
- Covered: project commands, task commands, `submit_goal`, SQLite-backed app state, exported project state.

### Frontend Build

Command:

```powershell
npm run build
```

Result:

- TypeScript compile passed.
- Vite production build passed.

### Storage, IDE, Docs, Tools

Command:

```powershell
$env:CARGO_TARGET_DIR='B:\crytex-target-audit'
cargo test -p crytex-storage -p crytex-ide -p crytex-doc -p crytex-tools --all-targets
```

Result:

- `crytex-storage`: 21 passed, 1 ignored.
- `crytex-ide`: 6 passed.
- `crytex-doc`: 17 passed.
- `crytex-tools`: 21 passed.
- Covered: SQLite repositories, Qdrant Edge dense/sparse vector search, IDE/LSP bridge, AST/chunking, file/process/search tools and sandbox policy.

### Agents, Bench, Compression

Command:

```powershell
$env:CARGO_TARGET_DIR='B:\crytex-target-audit'
cargo test -p crytex-agents -p crytex-bench -p crytex-compress --all-targets
```

Result:

- `crytex-agents`: 45 passed.
- `crytex-bench`: 14 passed.
- `crytex-compress`: 74 passed.
- Covered: agent prompt contracts, tool loop, coder/architect/critic/qa/researcher/security agents, benchmark harness, A/B scoring, compression/CCR.

### Inference Backends

Commands:

```powershell
$env:CARGO_TARGET_DIR='B:\crytex-target-audit'
cargo test -p crytex-inference -p crytex-inference-openai -p crytex-inference-anthropic -p crytex-inference-ollama --all-targets
cargo test -p crytex-inference-onnx --all-targets
cargo test -p crytex-inference-candle -p crytex-inference-mistral --all-targets
```

Result:

- `crytex-inference`: 5 passed.
- `crytex-inference-openai`: 6 passed.
- `crytex-inference-anthropic`: 3 passed.
- `crytex-inference-ollama`: 1 passed.
- `crytex-inference-onnx`: 7 passed, 3 ignored.
- `crytex-inference-candle`: 6 passed.
- `crytex-inference-mistral`: 8 passed.

### Sandbox

Command:

```powershell
$env:CARGO_TARGET_DIR='B:\crytex-target-audit'
cargo test -p crytex-sandbox --all-targets
```

Result:

- 7 passed, 3 ignored.
- Docker tests are ignored because they require Docker/images.

### Workspace Clippy

Command:

```powershell
$env:CARGO_TARGET_DIR='B:\crytex-target-audit'
cargo clippy --workspace --all-targets -- -D warnings
```

Result:

- Passed for the whole workspace.

## What This Proves

- The Rust workspace compiles across all crates.
- The main backend modules are test-green.
- Tauri backend IPC layer compiles and has tested goal, plan approval/rejection, and run paths.
- Frontend TypeScript and Vite build are valid.
- Qdrant Edge in-process vector store works in tests.
- IDE/LSP bridge compiles and has basic behavior covered.
- Agents, benchmark, feedback, prompt evolution, LoRA evolution, and compression have unit/integration coverage.
- Tauri `start_run` can execute a ready task through a real local Ollama model and move the result to `Review` without falsely marking it `Completed`.
- Tauri state now has a real goal-first E2E proof: `create_project -> submit_goal -> generated_tasks -> approve_plan -> start_run -> Ollama qwen3.5:9b -> Review -> approve_task_review -> start_run -> Ollama qwen3.5:9b -> Review`.

## What Is Not Yet Proven

- Real manual Tauri desktop app flow was not exercised in a running window.
- Real Hugging Face model download from UI was not tested.
- Real ONNX embedding download/cache path was not tested; model-dependent tests are ignored.
- Real Docker sandbox execution was not tested; Docker tests are ignored.
- Full multi-agent execution with tools, file edits, benchmarks, and evolution loop was not tested end-to-end.
- Watcher/indexer auto-start from actual project-open UI flow is not wired yet.
- Real Hugging Face -> optimize -> serve -> run path is not wired into Tauri UI yet.

## Current Assessment

The backend is not a toy stub: the major subsystems compile and their isolated tests pass. As of 2026-07-05, two real local model execution paths are proven through Tauri state:

- Direct ready task: `create_project -> submit_task -> start_run -> Ollama qwen3.5:9b -> Review`.
- Goal-first path: `create_project -> submit_goal -> generated_tasks -> approve_plan -> start_run -> Ollama qwen3.5:9b -> Review -> approve_task_review -> start_run -> Ollama qwen3.5:9b -> Review`.

The weakest product gap is still the complete user-facing runtime flow: goal-first orchestration, project auto-indexing, model management, IDE, Observe, and evolution are not yet integrated as one desktop workflow. The next critical checks should be:

1. Wire model picker/download/optimization into Tauri instead of using `CRYTEX_TAURI_OLLAMA_MODEL`.
2. Wire watcher/indexer startup into project open/create.
3. Add Observe events for prompts, retrieval, model calls, file/tool events, retries, and evolution decisions.
4. Wire task review approval/rejection into the UI so the generated chain can continue by hand.
5. Rework the UI center around Agent Console + IDE, with Kanban as secondary inspection.

## 2026-07-05 Update: Tauri Real Model Run

Implemented and verified:

- Added `TaskExecutor` abstraction for `start_run`.
- Added `StubTaskExecutor` to preserve honest smoke behavior when no model is configured.
- Added `InferenceTaskExecutor` backed by `crytex-inference::InferenceManager`.
- Tauri runtime now uses Ollama when `CRYTEX_TAURI_OLLAMA_MODEL` is set; otherwise it falls back to stub.
- Added ignored E2E: `crates/crytex-tauri/tests/e2e_ollama_start_run.rs`.

Real E2E command:

```powershell
$env:CARGO_TARGET_DIR='B:\crytex-target-audit'
$env:CRYTEX_E2E_OLLAMA_MODEL='qwen3.5:9b'
$env:CRYTEX_E2E_OLLAMA_URL='http://127.0.0.1:11434'
cargo test -p crytex-tauri --test e2e_ollama_start_run -- --ignored --nocapture
```

Result before review-continuation support:

```text
CRYTEX_TAURI_E2E_RESULT model=qwen3.5:9b task_id=01KWQPX5TDCSHJRGWW9SM67WKS content=Crytex has successfully verified that the Tauri `start_run` command triggered a response from the local model, confirming the integration is active and operational within this session.
test start_run_executes_ready_task_with_real_ollama_model ... ok
```

Verification:

```powershell
$env:CARGO_TARGET_DIR='B:\crytex-target-audit'
cargo test -p crytex-tauri --all-targets
cargo clippy -p crytex-tauri --all-targets -- -D warnings
npm run build # from crates/crytex-tauri
```

## 2026-07-05 Update: Goal-First Ollama E2E

Added a second ignored E2E in `crates/crytex-tauri/tests/e2e_ollama_start_run.rs`:

```powershell
$env:CARGO_TARGET_DIR='B:\crytex-target-audit'
$env:CRYTEX_E2E_OLLAMA_MODEL='qwen3.5:9b'
$env:CRYTEX_E2E_OLLAMA_URL='http://127.0.0.1:11434'
cargo test -p crytex-tauri --test e2e_ollama_start_run goal_plan_approval_starts_first_generated_task_with_real_ollama_model -- --ignored --nocapture
```

Result:

```text
CRYTEX_TAURI_GOAL_E2E_RESULT model=qwen3.5:9b goal_id=01KWQSBTZNHFVE7NESK98VWTMW task_id=01KWQSBV0Z9XZVRFGJEVATPNZW content=...
test goal_plan_approval_starts_first_generated_task_with_real_ollama_model ... ok
```

This proves the core backend happy path up to human review. It intentionally stops after the first generated task because dependencies should remain blocked until the reviewed result is approved.

## 2026-07-05 Update: Task Review Continuation

Implemented and verified:

- Added `approve_task_review` and `reject_task_review` Tauri commands.
- `approve_task_review` records `human_score = 1.0`, marks the reviewed task `Completed`, and returns newly ready dependent tasks.
- `reject_task_review` records `human_score = 0.0`, retries the reviewed task with human feedback, and returns ready tasks.
- Added TypeScript API/types for task review decisions.
- Wired task review approval/rejection into the Tauri UI Inspector and made Run review tasks selectable.
- Extended the real Ollama goal-first E2E so it approves the `architect` review and then runs the unblocked `coder` task through the local model.

Real E2E result:

```text
CRYTEX_TAURI_GOAL_E2E_RESULT model=qwen3.5:9b goal_id=01KWQV21B4BACJ8JMYXEXCW8KQ architect_task_id=01KWQV21BK4NN9FZMAQKQ304KD coder_task_id=01KWQV21BTMJG7AJDX65Q9Y7J1 coder_content=...
test goal_plan_approval_starts_first_generated_task_with_real_ollama_model ... ok
```

Verification:

```powershell
$env:CARGO_TARGET_DIR='B:\crytex-target-audit'
cargo test -p crytex-tauri --all-targets
cargo clippy -p crytex-tauri --all-targets -- -D warnings
npm run build # from crates/crytex-tauri
```

UI verification:

```powershell
npm run build # from crates/crytex-tauri
npm run dev -- --host 127.0.0.1 --port 5177
```

Browser smoke via Edge confirmed the React shell renders. In plain web mode the command log records expected Tauri runtime errors because IPC is only available in the desktop app.
