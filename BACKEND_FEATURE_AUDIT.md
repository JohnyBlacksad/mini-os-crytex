# Crytex Backend Feature Audit

Date: 2026-07-16

Scope: backend only. UI is intentionally out of scope until the core can be proven from CLI/tests.

Verdict scale:

- `PROVEN`: the product claim is covered by direct executable evidence.
- `PARTIAL`: meaningful implementation exists, but the full architecture/product claim is not proven.
- `NOT PROVEN`: mostly interface, mock, placeholder, or missing integration.

Latest verification run:

```powershell
$env:CARGO_TARGET_DIR='B:\crytex-target-audit'
cargo test -p crytex-doc -p crytex-core -p crytex-bench -p crytex-compress -p crytex-storage -p crytex-inference-mistral
cargo clean
```

Result: all selected backend tests passed; one external Qdrant test is ignored; cleanup removed 13.6 GiB.

Later targeted verification also passed:

```powershell
$env:CARGO_TARGET_DIR='B:\crytex-target-audit'
cargo test -p crytex-bench lora_gate::tests
cargo test -p crytex-kernel factory::tests::create_lora_evolution_service_wires_benchmark_gate_before_promotion
cargo check -p crytex-kernel
cargo clean
```

Additional cleanup removed 18.2 GiB.

Latest LoRA observability verification:

```powershell
$env:CARGO_TARGET_DIR='B:\crytex-target-audit'
cargo test -p crytex-core lora_evolution::tests
cargo test -p crytex-tauri build_run_diagnostics_
```

Result: 19 LoRA evolution tests passed and 3 Tauri diagnostics tests passed at the time; the current targeted LoRA suite is 20 tests. `LoraEvolutionServiceImpl` now emits structured promotion/rejection `RunObserved` events with training job id, adapter id, task kind, metrics, triggering task trace id, and `benchmark_gate` metadata when a gate is present.

Latest Tauri product-path trigger verification:

```powershell
$env:CARGO_TARGET_DIR='B:\crytex-target-audit'
cargo test -p crytex-tauri sqlite_state_
```

Result: 20 Tauri app-state tests passed at the time; the current targeted app-state suite is 22 tests. Human approval of a normal work task now triggers LoRA golden-example collection and threshold-based training. Human rejection records a LoRA counter-example without breaking retry semantics. Critic/review gate tasks are excluded from the LoRA dataset so "critic passed to human review" does not become training data.

Latest approval-triggered LoRA diagnostics verification:

```powershell
$env:CARGO_TARGET_DIR='B:\crytex-target-audit'
cargo test -p crytex-tauri sqlite_state_
cargo test -p crytex-tauri build_run_diagnostics_
cargo test -p crytex-core lora_evolution::tests
cargo test -p crytex-storage lora
```

Result: 24 Tauri app-state tests, 3 Tauri diagnostics tests, 20 core LoRA evolution tests, 4 bench LoRA-gate tests, and 1 storage LoRA persistence test passed across the latest targeted checks. `sqlite_state_approval_triggered_lora_service_decision_reaches_diagnostics` proves the SQLite-backed desktop path now reaches `LoraEvolutionServiceImpl`, creates a training job and adapter through the deterministic mock trainer, emits `RunObserved`, persists it through `RunObservedAuditBridge`, and exports `lora_evolution[]` without manual event injection. `sqlite_state_approval_triggered_lora_benchmark_gate_reaches_diagnostics` proves the same desktop path can inject a `LoraBenchmarkGate`, call it before promotion, persist winner/p-value/pass-rate metadata, and export it through diagnostics. `sqlite_state_approval_triggered_real_bench_gate_uses_held_out_corpus` proves the desktop path can use the real `BenchLoraBenchmarkGate`, `DefaultBenchmarkHarness`, `ExactMatchScorer`, AB comparison, and a held-out JSONL corpus before promotion.

Latest LoRA request-routing verification:

```powershell
$env:CARGO_TARGET_DIR='B:\crytex-target-audit'
cargo test -p crytex-tauri sqlite_state_uses_promoted_lora_for_next_agent_request
cargo test -p crytex-inference-mistral generate_rejects_unregistered_request_lora_adapter_before_model_load
cargo test -p crytex-inference-mistral load_plan
cargo test -p crytex-inference-mistral
cargo test -p crytex-core unsupported_inference_error_maps_to_unsupported_service_error
cargo clean
```

Result: 1 targeted Tauri app-state test passed. `sqlite_state_uses_promoted_lora_for_next_agent_request` proves that after 50 approved `codegen`/`coder` tasks promote `codegen-v1`, the next real Tauri agent execution path selects the promoted adapter, persists it on the task, and sends it into `InferenceRequest.lora_adapter_id`.

Result: 1 targeted Mistral backend test passed. `generate_rejects_unregistered_request_lora_adapter_before_model_load` proves a request-level LoRA adapter id must be registered before generation; the backend now fails before model loading instead of silently forwarding an unknown adapter name into mistral.rs.

Result: 2 targeted Mistral load-plan tests passed, then the full `crytex-inference-mistral` suite passed with 11 tests. `plain_model_load_plan_uses_registered_lora_paths` proves registered LoRA adapter paths feed the plain-model mistral.rs `LoraModelBuilder`. `gguf_model_load_plan_rejects_registered_lora_until_supported` proves GGUF plus locally registered LoRA now returns typed unsupported evidence instead of pretending hot-swap works. `unsupported_inference_error_maps_to_unsupported_service_error` proves this unsupported evidence survives the core inference service boundary.

Latest unsupported-backend and artifact-boundary LoRA verification:

```powershell
$env:CARGO_TARGET_DIR='B:\crytex-target-audit'
cargo test -p crytex-inference-openai -p crytex-inference-anthropic -p crytex-inference-ollama -p crytex-inference-onnx lora_is_unsupported
cargo test -p crytex-core lora_evolution::tests
cargo test -p crytex-inference-mistral training::tests
cargo test -p crytex-inference-mistral
cargo test -p crytex-tauri sqlite_state_approval_triggered_lora_service_decision_reaches_diagnostics
cargo clean
```

Result: OpenAI, Anthropic, Ollama, and ONNX now return typed `UnsupportedOperation` for LoRA register/swap instead of pretending success or surfacing a generic load failure. The core LoRA evolution service now rejects a single-file fake adapter before adapter persistence or inference registration, requires a PEFT-like adapter directory with `adapter_config.json` and `adapter_model.safetensors`, enforces non-empty adapter weights plus max-size checks, and cleans up either file or directory artifacts on rollback. The deterministic Mistral mock trainer now emits that directory layout, Mistral `register_lora` independently rejects single-file adapters and malformed `adapter_config.json` before registry insertion, Mistral GGUF file/directory backends no longer advertise `"lora"` capability until GGUF LoRA is implemented, and the SQLite-backed Tauri LoRA diagnostics path still passes with the stricter validator.

Latest inference capability-report verification:

```powershell
$env:CARGO_TARGET_DIR='B:\crytex-target-audit'
cargo test -p crytex-inference
cargo test -p crytex-core backend_capability_reports_are_typed
cargo clean
```

Result: `BackendInfo` now builds a typed `BackendCapabilityReport` with `generate`, `chat`, `embed`, `rerank`, `lora`, and `hot_swap` booleans. `BackendRegistry` can list typed reports, and the core `InferenceService` trait exposes a default `backend_capability_reports()` method, so UI/diagnostics can consume capability truth without duplicating string parsing. LoRA does not imply true hot-swap: `hot_swap` is reported only from explicit `hot_swap`/`lora_hot_swap` capability strings.

Latest Tauri capability-report export verification:

```powershell
$env:CARGO_TARGET_DIR='B:\crytex-target-audit'
cargo test -p crytex-tauri sqlite_state_
cargo test -p crytex-tauri build_run_diagnostics_collects_trace_tasks_events_rag_and_reward_evidence
cd crates/crytex-tauri
npm test -- --run
npm run build
cargo clean
```

Result: `RuntimeStatus` now includes `backend_capabilities`, and `RunDiagnosticsReport.runtime` carries those typed capability reports to Observe/UI consumers. SQLite app-state tests prove stub runtime reports no capabilities, Ollama runtime reports generate/chat without LoRA/hot-swap, and downloaded GGUF Mistral runtime reports generate/chat without LoRA/hot-swap. The diagnostics builder test proves the report preserves those flags, while Vitest and `tsc && vite build` prove the frontend contract accepts the new field.

Latest managed-model product-path verification:

```powershell
$env:CARGO_TARGET_DIR='B:\crytex-target-audit'
cargo test -p crytex-tauri sqlite_state_adds_downloads_lists_and_activates_managed_model
cargo test -p crytex-tauri managed_model
cargo clean
```

Result: 1 targeted app-state test passed, then 7 managed-model tests passed. `sqlite_state_adds_downloads_lists_and_activates_managed_model` proves the desktop-facing backend can add a model, download it through the `ModelManager` abstraction, list it as `Downloaded`, activate it as the Mistral runtime, and expose runtime capability flags without manually writing the registry. This is still a mocked download proof; the real `HfHubDownloader` network path remains unproven.

Latest real Hugging Face downloader smoke:

```powershell
$env:CARGO_TARGET_DIR='B:\crytex-target-audit'
cargo test -p crytex-core real_hf_download_persists_registry_and_reloads_as_downloaded
cargo test -p crytex-core real_hf_download_persists_registry_and_reloads_as_downloaded -- --ignored --nocapture
cargo clean
```

Result: the first command proves the network smoke compiles and remains ignored by default; the second command executed the real network path successfully. It downloaded `sshleifer/tiny-gpt2/config.json` through `HfHubDownloader`, copied the file into Crytex managed cache, persisted `registry.toml`, and reloaded the model as `Downloaded`. This proves HF file download, not real LLM load/generation from the downloaded artifact.

Latest real Hugging Face GGUF downloader smoke:

```powershell
$env:CARGO_TARGET_DIR='B:\crytex-target-audit'
cargo test -p crytex-core real_hf_tiny_gguf_download_reloads_as_mistral_runtime_candidate
cargo test -p crytex-core real_hf_tiny_gguf_download_reloads_as_mistral_runtime_candidate -- --ignored --nocapture
cargo clean
```

Result: the first command proves the 83 MB network smoke compiles and remains ignored by default; the second command executed the real path successfully in 10.83s. It downloaded `tensorblock/tiny-random-minicpm-GGUF/tiny-random-minicpm-Q2_K.gguf`, copied it into Crytex managed cache, persisted/reloaded `registry.toml`, and verified it remains a Mistral runtime candidate with `Q2_K` metadata. This still does not prove actual mistral.rs generation from that file.

Latest real Mistral GGUF generation attempt:

```powershell
$env:CARGO_TARGET_DIR='B:\crytex-target-audit'
cargo test -p crytex-inference-mistral real_hf_tiny_gguf_downloaded_model_generates_with_mistralrs
cargo test -p crytex-inference-mistral real_hf_tiny_gguf_downloaded_model_generates_with_mistralrs -- --ignored --nocapture
$env:CRYTEX_RUN_SLOW_MISTRAL_SMOKE='1'
cargo test -p crytex-inference-mistral real_hf_tiny_gguf_downloaded_model_generates_with_mistralrs -- --ignored --nocapture
```

Result: the smoke now compiles and is safe by default; even explicit `--ignored` skips unless `CRYTEX_RUN_SLOW_MISTRAL_SMOKE=1` is set. The smoke is hardware-aware instead of CPU-only: `auto` detects CUDA/Metal, `CRYTEX_MISTRAL_SMOKE_DEVICE=cpu|gpu|auto` selects the mode, and `CRYTEX_MISTRAL_SMOKE_GPU_LAYERS` can override layer count. Unit tests prove auto uses CUDA without forcing `Some(0)`. `tensorblock/tiny-random-minicpm-GGUF` and `tensorblock/tiny-random-Llama-3-GGUF` both failed during mistral.rs/Candle model load with `unknown dtype for tensor 20`; `tensorblock/tiny-random-llama-GGUF` loaded further but failed because it lacks a chat template; `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` did not return one token within a 180 second CPU timeout. On this machine, NVIDIA RTX 5080 is detected. The initial CUDA build failed because `nvcc` could not find MSVC `cl.exe`; the next attempt found MSVC but failed on CUDA 13.2 CCCL requiring `/Zc:preprocessor`; later probes identified the optimized chunked GDN prefill kernel inside `mistralrs-core src/cuda/gdn.cu` as the hanging `cicc` path. The chunked GDN kernel now builds by preventing full unroll of the heavy `BK` dot-product/register loops with `#pragma unroll 1`; real TinyLlama GGUF generation on RTX 5080 passed without `-SkipGdnCuda`.

Latest CUDA toolchain preflight verification:

```powershell
$env:CARGO_TARGET_DIR='B:\crytex-target-audit'
cargo test -p crytex-core hardware
cargo test -p crytex-tauri runtime_status_includes_cuda_toolchain_preflight
cargo test -p crytex-inference-mistral smoke_runtime_
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
nvcc --version
cl.exe /help
cargo clean
```

Result: core now has a pure CUDA preflight evaluator and system detector that reports `gpu_detected`, `nvcc_available`, `msvc_cl_available`, `msvc_cl_path`, `NVCC_CCBIN`, `recommended_nvcc_ccbin`, `ready`, and human-readable diagnostics. Tauri `RuntimeStatus` exports this as `cuda_toolchain`, so UI/diagnostics can explain why CUDA is unavailable before a deep mistral.rs/Candle failure. The local machine reports `NVIDIA GeForce RTX 5080, 16303 MiB, driver 596.36` and `nvcc 13.2`; `cl.exe` is not in the current PowerShell PATH, but Visual Studio Build Tools `cl.exe` is discoverable. `scripts/mistral-cuda-build-probe.ps1` and `scripts/mistral-cuda-smoke.ps1` use that path as `NVCC_CCBIN`, set `CL=/MD /Zc:preprocessor /DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING`, set `CUDA_NVCC_FLAGS` with the same MSVC runtime/preprocessor flags, write stdout/stderr logs into `.crytex-smoke-logs/`, enforce a hard timeout, dump live build processes on timeout, and clean the target directory.

Latest CUDA build probe harness verification:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\mistral-cuda-build-probe.ps1 -TimeoutSeconds 5
```

Result: the probe timed out intentionally with exit `124`, wrote stdout/stderr log files, captured the live `cargo`/`rustc` process list, stopped build processes, and cleaned 166.6 MiB from `B:\crytex-target-audit`. This verifies the harness behavior, not a successful CUDA build. The short timeout kills ordinary Rust dependency compilation, so the resulting `windows-sys`/`syn`/`regex-syntax` failures in that log are expected artifacts of the forced timeout.

Latest real CUDA build probe:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\mistral-cuda-build-probe.ps1 -TimeoutSeconds 900 -IdleTimeoutSeconds 180
```

Result: the probe reached `candle-core`, `candle-nn`, and `mistralrs-vision`, then hit an idle timeout after 361 seconds. The last live build process was a single `nvcc` child with no CPU activity; the final log line before idling was `Compiling mistralrs-vision v0.8.1`. The probe cleaned 4.7 GiB from `B:\crytex-target-audit`. The next probe version records the full command line for the idle `nvcc` process so the exact CUDA source/kernel can be identified instead of only the owning crate/stage.

Latest CUDA kernel identification probe:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\mistral-cuda-build-probe.ps1 -TimeoutSeconds 900 -IdleTimeoutSeconds 120
powershell -ExecutionPolicy Bypass -File scripts\mistral-cuda-build-probe.ps1 -TimeoutSeconds 900 -IdleTimeoutSeconds 120 -ComputeCap 90
```

Result: both probes timed out cleanly and removed 4.7 GiB from `B:\crytex-target-audit`. The command-line dump identified the same blocker in both cases: `mistralrs-core` invokes `nvcc` for `src\cuda\gdn.cu`, producing `gdn-*.o`, and the process idles with no CPU/log output. The native RTX 5080 probe used `compute_120a/sm_120a`; the diagnostic fallback used `CUDA_COMPUTE_CAP=90` and compiled `compute_90a/sm_90a`. Because both hang on the same source file, this is not only a Blackwell `sm_120a` target issue; it is the `mistralrs-core` GDN CUDA kernel build path on this Windows/NVCC 13.2 setup.

Latest CUDA GDN kernel fix proof:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\mistral-gdn-kernel-probe.ps1 -TimeoutSeconds 120
powershell -ExecutionPolicy Bypass -File scripts\mistral-gdn-kernel-probe.ps1 -TimeoutSeconds 180
$env:CARGO_TARGET_DIR='B:\crytex-target-audit'
cargo test --manifest-path patches\mistralrs-core\Cargo.toml --features cuda cuda_chunked_gdn_matches_cpu_reference -- --nocapture
cargo clean
powershell -ExecutionPolicy Bypass -File scripts\mistral-cuda-build-probe.ps1 -TimeoutSeconds 1200 -IdleTimeoutSeconds 240
powershell -ExecutionPolicy Bypass -File scripts\mistral-cuda-smoke.ps1 -TimeoutSeconds 1200 -IdleTimeoutSeconds 240
```

Result: `scripts/mistral-gdn-kernel-probe.ps1` reproduced the old full `gdn.cu` timeout, proved disabling `chunked_gated_delta_rule_recurrence` made the file compile in 5 seconds, then proved the actual chunked kernel compiles after the heavy `BK` loops were marked `#pragma unroll 1`. A direct CUDA kernel correctness test then exposed and fixed a second real bug: recurrence kernels returned early for partial V tiles before `__syncthreads()`, so cooperative shared-memory loads were incomplete when `v_dim < BV` or on a tail tile. The fix keeps all threads participating in shared loads/barriers and gates only per-V computation/writes with `has_v`. `cargo test --manifest-path patches\mistralrs-core\Cargo.toml --features cuda cuda_chunked_gdn_matches_cpu_reference -- --nocapture` now passes against a CPU reference and cleanup removed 4.3 GiB. The default full `gdn.cu` probe passes in 62 seconds on `sm_120a`. The build/smoke path still builds all 6 CUDA kernels without `-SkipGdnCuda`; the TinyLlama GPU smoke forced `NVIDIA GeForce RTX 5080`, generated successfully, printed `test result: ok. 1 passed`, and cleaned 8.8 GiB. This proves default CUDA build/generation and direct chunked GDN kernel correctness.

Latest CUDA GDN-family MoE e2e fix proof:

```powershell
cargo test --manifest-path patches\mistralrs-quant\Cargo.toml --features cuda should_gather_unquantized_tensor_weights_without_requiring_qtensor -- --nocapture
powershell -ExecutionPolicy Bypass -File scripts\mistral-cuda-smoke.ps1 -Gdn -TimeoutSeconds 1500 -IdleTimeoutSeconds 300
```

Result: the first command was written RED around the real failure and initially failed with `indexed_moe_forward is only supported for quantized tensors (QTensor)`. The fix adds a local `mistralrs-quant` patch and makes CUDA `GgufMatMul::gather_forward` use the fused indexed MoE kernel only for `QTensor`; unquantized `Tensor/TensorF16` weights now use ordinary indexed tensor matmul instead of crashing. The same unit test then passed and cleaned 1.8 GiB. The `-Gdn` smoke built patched `mistralrs-quant` plus patched `mistralrs-core`, forced GPU runtime on `NVIDIA GeForce RTX 5080`, ran `real_hf_tiny_qwen3_next_gdn_generates_with_mistralrs`, printed `test result: ok. 1 passed`, finished in 301 seconds, and cleaned 9.0 GiB. This proves the tiny Qwen3-Next MoE/GDN e2e generation path on the local GPU; it does not yet prove automatic production-grade model optimization or broad GDN-family performance.

Latest product-contract follow-up: `cargo test -p crytex-tauri app_state::tests` passed with 28 tests. `RuntimeStatus` exports `compatibility_notes`; these now report that CUDA generation, direct GDN kernel correctness, and tiny Qwen3-Next GDN smoke are proven, while broader model-family compatibility still belongs in model/runtime diagnostics.

Latest model-family compatibility planning proof:

```powershell
cargo test -p crytex-core model_compatibility -- --nocapture
cargo test -p crytex-tauri managed_gguf_runtime_status_reports_cuda_gdn_compatibility -- --nocapture
```

Result: core now has `ModelCompatibilityPlanner`, a pre-load planner that classifies a managed model by format (`GGUF`, `HuggingFace`) and feature family (`dense`, `MoE`, `GDN`), then emits a typed `ExecutionStrategy` with actions, warnings, and blockers. The tests prove `tiny-random/qwen3-next-moe` is treated as MoE+GDN and planned as `CudaWithFallback` when runtime features are available; missing GDN CUDA support becomes a typed `Unsupported` blocker before model load; dense GGUF models are planned as `CudaFused`. Tauri `RuntimeStatus` now carries `model_compatibility: Option<ModelCompatibilityPlan>` for managed Mistral models, so the UI can show strategy/features/blockers instead of relying on model-specific trial-and-error. The core planner test passed 3 tests and cleaned 3.3 GiB; the Tauri runtime-status contract test passed and cleaned 24.0 GiB.

Defects found and fixed in that proof:

- Tauri approval scores are normalized (`1.0` approved, `0.0` rejected), while default LoRA thresholds expected a 0-5 scale. Desktop LoRA wiring now uses normalized `min_human_score` and validation reward thresholds.
- SQLite LoRA adapter persistence wrote absent `project_id` as an empty string, violating the `projects(id)` foreign key. Nullable LoRA fields now persist as `NULL`.
- Tauri agent execution did not attach the selected promoted LoRA before inference. A `LoraSelectingTaskExecutor` now selects a role/kind adapter through `LoraEvolutionService`, persists it to the task, and delegates the updated task to the real executor.
- Mistral request-level LoRA selection accepted unknown adapter ids. The backend now validates that the selected adapter was registered before generation starts.
- Mistral registered LoRA paths were not part of model loading. Plain-model loading now uses `LoraModelBuilder` when adapters are registered; GGUF registered LoRA is explicitly unsupported until a correct GGUF adapter ordering path is implemented.
- Ollama previously accepted LoRA registration/swap as `Ok(())`, which could make the system believe an adapter was active when it was ignored. Unsupported LoRA backends now fail explicitly.
- LoRA evolution previously accepted a lone `.safetensors` file from the deterministic trainer. The service now requires adapter config plus adapter weights before any promotion or backend registration.
- Mistral `register_lora` previously accepted arbitrary paths directly into its registry. It now validates the adapter directory, JSON config, declared `peft_type=LORA`, and non-empty weights before mutating backend state.
- Mistral `available_backends` previously advertised `"lora"` for GGUF models even though GGUF plus locally registered LoRA returns typed unsupported. GGUF file and GGUF-directory backends now hide the LoRA capability until that path is implemented.
- Backend capability consumers previously had to parse free-form strings directly. The inference layer now exposes typed capability reports and keeps `lora` separate from true `hot_swap`.
- Runtime status and run diagnostics previously did not expose those typed reports to the desktop/UI boundary. They now carry `backend_capabilities` as part of the serialized contract.
- Runtime status previously could say a CUDA GPU exists without explaining why CUDA execution still cannot build/run. It now carries `cuda_toolchain` readiness and diagnostics, including missing `nvcc`, missing Windows `cl.exe`, discovered MSVC `cl.exe`, `NVCC_CCBIN`, and recommended `NVCC_CCBIN` status.
- The desktop managed-model activation proof previously bypassed download by hand-writing the registry. A new app-state product-path test now goes through add/download/list/activate commands and verifies runtime status/capabilities.
- The HF downloader previously had only mocked coverage. A new ignored network smoke now proves the real `hf-hub` path downloads a tiny file, copies it to managed cache, persists the registry, and reloads as downloaded.
- The HF downloader now also has an ignored real GGUF smoke: it downloads a tiny `.gguf`, preserves Mistral backend metadata, and reloads as a downloaded runtime candidate.
- Real mistral.rs generation from a downloaded GGUF is now proven on the current RTX 5080 through the normal CUDA path without `-SkipGdnCuda`. CPU TinyLlama generation still timed out in earlier attempts and tiny-random GGUF still exposes unsupported dtype coverage, but the chunked GDN kernel compile hang was fixed by limiting unroll in heavy dot-product/register loops, direct chunked GDN CUDA output/state now matches a CPU reference, and the tiny Qwen3-Next MoE/GDN candidate now passes a real GPU generation smoke after the `mistralrs-quant` CUDA GGUF MoE fallback fix.

## Executive Verdict

Crytex has a real backend foundation: task graph, orchestration, agent execution contracts, audit events, project indexing, vector storage, context assembly, prompt evolution records, LoRA records/router, compression, benchmark harness, sandbox/tooling, and several inference backends.

It is not yet product-ready. The biggest missing proofs are:

1. Strict task atomicity from orchestrator plans.
2. Graph-aware AST metadata in RAG prompts/diagnostics, beyond persisted index payload.
3. Reranker diagnostics and Tauri/manual path proof.
4. Real downloaded model generation through the managed-model product path; direct RTX 5080 TinyLlama CUDA generation is proven, but the add/download/activate path still needs to drive the same real generation smoke.
5. Metadata-backed compatibility planning from real `config.json` and GGUF metadata, then performance regression coverage for the fixed chunked GDN prefill kernel and broader GDN-family compatibility/performance probes beyond the tiny Qwen3-Next smoke.
6. Prompt evolution benchmark promote/reject loop.
7. LoRA real trainer/hot-swap proof after the deterministic promote/reject loop, request-routing proof, and diagnostics export.
8. Real LoRA training/hot-swap behavior.
9. Full CLI happy path that proves the backend without Tauri.
10. Diagnostics covering model/RAG/rerank/compression/prompt/LoRA/benchmark decisions.

## Architecture Claims vs Reality

### 1. Task Orchestration

Status: `PARTIAL`

Claim: an AI teamlead/orchestrator decomposes user goals into atomic tasks and assigns them to agents in a chain.

Current evidence:

- `crates/crytex-core/src/services/orchestrator.rs`
- `codegen_decomposes_into_four_tasks_by_default` currently asserts the default chain, despite the name being stale.
- `codegen_tasks_have_serial_dependencies`
- `codegen_uses_planning_agent_when_configured`
- `codegen_falls_back_to_default_when_planning_agent_returns_no_subtasks`
- `crates/crytex-core/tests/codegen_pipeline.rs`
- Tauri e2e smoke verifies a generated architect/coder/qa/security/critic chain.

Reality:

- The orchestrator can create serial subtasks and can use an LLM planning agent.
- The default decomposition is a role-stage pipeline, not necessarily atomic work units.
- LLM-generated subtasks are accepted from JSON, but there is no strict validator that enforces one action, bounded scope, acceptance criteria, artifact schema, and dependency correctness.

Required proof:

- Add an `AtomicTaskPlanValidator`.
- Test that broad subtasks are rejected or split.
- Test that every generated subtask has one responsibility, acceptance criteria, expected artifact, agent role, and dependency metadata.
- Add a CLI/e2e goal where the architect returns 6-10 small tasks, not one large task.

### 2. Agent Chain and Artifact Handoff

Status: `PARTIAL`

Claim: agents execute in sequence, passing artifacts to the next agent; critic gates the final result; human approval/rejection closes the loop.

Current evidence:

- `crates/crytex-tauri/src/commands.rs` attaches upstream artifacts before execution.
- `attach_upstream_artifacts`
- `start_run_with_orchestrator`
- Tauri tests assert critic receives upstream artifacts.
- Real Ollama smoke path exists in `crates/crytex-tauri/tests/e2e_ollama_start_run.rs`.

Reality:

- Chain execution and upstream artifact passing exist.
- Critic review/human review gates exist.
- Core now owns shared artifact contract validation for `architect`, `coder`, `qa`, `security`, and `critic`.
- `AgentWorkflowNodeExecutor` rejects malformed agent outputs before writing them into workflow state.
- Tauri command execution uses the same core content validator for handoff envelopes, retries malformed outputs once, then fails with contract diagnostics if still invalid.
- Diagnostics export typed artifact lineage and artifact handoff rejection reasons.

Required proof:

- Broaden the strict contract to real model structured-output prompts for every role, not only runtime validation.
- Add schema version migration tests when artifact schemas evolve.

### 3. Critic Rejection and Remediation

Status: `PARTIAL`

Claim: critic rejects with comments/reasons; orchestrator creates debug/remediation tasks; chain runs again.

Current evidence:

- `critic_rejected`
- `remediation_plan_created`
- `create_reviewer_remediation_plan`
- Tauri unit/e2e tests check feedback, failure type, generated remediation tasks.

Reality:

- The remediation loop exists.
- The critic can return structured reasons.
- Critic rejection is now a mandatory contract: `review_decision = reject` must include `blocking_issues`.
- Malformed critic/coder artifacts are rejected, retried, and eventually fail with explicit contract reasons.

Required proof:

- Real-model critic smoke should prove the model follows the structured rejection contract without repair.
- Product UI should surface contract retry/failure reasons directly in Observe.

### 4. Human Review, Reward, and Experience

Status: `PARTIAL`

Claim: human approval/rejection updates experience and feeds prompt/LoRA evolution.

Current evidence:

- `crates/crytex-core/tests/feedback_loop.rs`
- Reward service tests.
- LoRA evolution tests create golden/counter examples.
- Prompt evolution can recompute fitness from experiences.

Reality:

- Review and reward data exist.
- Approved/rejected task experiences can be persisted.
- The full chain into prompt and LoRA candidate promotion is not proven.

Required proof:

- E2E: reject with human comment -> remediation task -> counter example -> prompt fitness update -> LoRA dataset eligibility.
- Diagnostics should include experience ids created by the review.

### 5. AST, Code Graph, and Impact Analysis

Status: `PARTIAL`

Claim: project code is parsed with tree-sitter, represented as an AST/code graph, and used by RAG/agents.

Current evidence:

- `crates/crytex-doc/src/lib.rs`
- `crates/crytex-doc/src/graph/*`
- `crates/crytex-doc/src/chunking.rs`
- `crates/crytex-doc/src/impact.rs`
- Tests passed: chunking, graph builder, language extractors, impact analysis.
- `crates/crytex-kernel/src/main.rs` builds a `CodeGraph`.

Reality:

- Tree-sitter chunking works.
- CodeGraph and impact analysis work at crate level.
- `chunk_code_with_graph` exists and can add `symbol_id`/`related_symbols`.
- The main `ProjectIndexer` uses graph-aware code chunking and persists `symbol_id`/`related_symbols` in vector payloads.
- `ContextAssembler` now exposes graph metadata in selected RAG evidence and prompt context headers.
- Tauri app-state proof verifies indexed Rust graph metadata reaches the agent prompt and exported `rag_context_assembled` diagnostics.

Required proof:

- Broaden graph proof beyond Rust to the other supported tree-sitter languages.
- Add product diagnostics that explain symbol-neighborhood selection, not only raw ids.

### 6. Project Indexing, Watcher, and RAG

Status: `PARTIAL`

Claim: any project is automatically indexed; watcher updates chunks; code/docs are embedded into Qdrant Edge for RAG.

Current evidence:

- `crates/crytex-core/src/indexer.rs`
- `ProjectWatcher` tests for changed/deleted files.
- `crates/crytex-storage/src/vector/edge.rs`
- Context assembler tests.
- Real smoke proves a RAG marker reaches the model.

Reality:

- Code, markdown, and HTML are indexed.
- PDF documents are routed through the document extraction path and covered by mixed-project indexing tests.
- Dense and sparse vector paths exist.
- Watcher is implemented and tested.
- Context assembly preserves before/after rerank evidence, selected chunks, retrieval sources, and graph metadata.
- Automatic indexing is present in kernel `run`, but broader app/CLI product paths need more e2e proof.

Required proof:

- Mixed project e2e from the product entrypoint: Rust + Markdown + HTML + PDF/plain document -> query -> agent prompt -> diagnostics export.
- CLI: create/open project -> automatic watcher starts -> file change appears in search without manual reindex.

### 7. Reranker

Status: `PARTIAL`

Claim: RAG uses a reranker model selected by the user.

Current evidence:

- `crates/crytex-core/src/services/reranker.rs`
- `crates/crytex-inference-onnx/src/reranker.rs`
- `create_reranker` exists in `crates/crytex-kernel/src/factory.rs`.
- `ContextAssembler::with_reranker` is covered by a unit test that proves rerank changes final context order.

Reality:

- Reranker abstraction and ONNX implementation exist.
- Kernel now wires `create_reranker` into `ContextAssembler`.
- Tauri app-state proof verifies an intentionally wrong dense order is corrected before context reaches the agent prompt.
- Exported `rag_context_assembled` diagnostics include retrieval candidates, reranked chunks, selected chunks, and `rerank_applied`.

Required proof:

- Real ONNX reranker smoke with a selected local model, including backend/model id in diagnostics.
- UI/CLI diagnostics should render before/after ranks clearly instead of exposing only raw JSON.

### 8. Context Compression / Token Optimizer

Status: `PARTIAL`

Claim: cloud/local LLM requests pass through token optimization and compression.

Current evidence:

- `crates/crytex-compress` passed 74 tests.
- `InferenceServiceImpl::with_compression`
- Tests prove messages are replaced under compression.
- Kernel builds compression pipeline with content-aware compressors.

Reality:

- Compression library is strong.
- Kernel path wires compression into inference service.
- Real Ollama/Tauri path with oversized RAG context is not proven.

Required proof:

- E2E with oversized retrieved context.
- Assert final LLM request is under budget and retains sentinel facts.
- Diagnostics should include compression ratio and original/compressed token counts.

### 9. Model Manager and HuggingFace Download

Status: `PARTIAL`

Claim: user can pick/download HF model; backend downloads, registers, and uses it.

Current evidence:

- `crates/crytex-core/src/services/model_manager.rs`
- Mocked tests for manifest/list/download/progress/recommendation.
- Kernel CLI has model list/download/show/recommend commands.

Reality:

- Manifest, registry, progress events, mocked downloader work.
- HF downloader exists through `hf-hub`.
- No real HF smoke was verified.
- No e2e proves downloaded model becomes active runtime and then executes.

Required proof:

- Non-network integration: add manifest entry -> mocked download -> registry -> runtime selection event.
- Optional real HF smoke behind env vars using a tiny/configured artifact.
- Diagnostics must show model id, local path, backend, selected config.

### 10. Runtime Optimization / mistral.rs

Status: `PARTIAL`

Claim: backend optimizes model config for user hardware.

Current evidence:

- Hardware recommendation tests passed.
- Model manager recommends quantization/context/gpu layers.
- `MistralRsBackend::new(model_path, context_size, gpu_layers)` passes settings into builders.
- CUDA preflight tests prove the app distinguishes physical NVIDIA GPU detection from CUDA toolchain readiness, discovers MSVC `cl.exe` outside PATH, and exports that through Tauri `RuntimeStatus`.

Reality:

- Current behavior is runtime config recommendation, not real model optimization or quantization.
- There is no proof that Crytex transforms a model to a better quantization.
- A real TinyLlama GGUF CUDA smoke proves the runtime can load and generate on this machine, but not that Crytex performs actual model conversion/quantization.
- On the current machine `cl.exe` is missing from normal PowerShell PATH even though RTX 5080 and `nvcc 13.2` are detected. Crytex can now discover the Build Tools compiler and run CUDA probes with `NVCC_CCBIN`; normal CUDA builds and generates after fixing the optimized chunked GDN prefill compile hang with targeted `#pragma unroll 1` on heavy `BK` loops. Direct CUDA GDN kernel correctness is proven against a CPU reference, and the tiny Qwen3-Next MoE/GDN smoke now generates on GPU. A first-pass model-family compatibility planner now prevents this class of issue from becoming per-model whack-a-mole by producing typed plans/blockers before weight load. Remaining gap: metadata-backed planning, broader model compatibility, actual automatic optimization/tuning, and performance regression coverage.

Required proof:

- Rename/clarify product language: "recommended runtime config" unless actual optimization is added.
- Integration: managed model selection uses recommended config.
- Optional real test: load a small GGUF with recommended config and generate one response.

### 11. Inference Backends

Status: `PARTIAL`

Current evidence:

- Ollama backend has a real e2e test that can pull/run a configured model.
- OpenAI and Anthropic backends have mocked HTTP tests.
- ONNX backend has embedding/reranker tests.
- Mistral backend has construction/adapter-selection tests.

Reality:

- Backend switching primitives exist.
- Cloud backends are not proven in full agent/product path.
- OpenAI/Anthropic explicitly do not support LoRA.
- Ollama does not support LoRA through current trait implementation.

Required proof:

- CLI e2e: select backend -> submit goal -> run -> diagnostics show backend/model.
- Mocked cloud provider agent-chain test.
- Explicit capability report for generation/embed/rerank/lora/hot-swap per backend.

### 12. LoRA Router

Status: `PARTIAL`

Claim: the system selects a LoRA adapter based on role/task/domain/semantic memory.

Current evidence:

- `crates/crytex-core/src/services/lora_router.rs`
- Tests for explicit payload, role registry, task kind, domain heuristic, semantic fallback.
- Kernel worker handler sets `task.lora_adapter_id`.
- Tauri app state wraps task execution with `LoraSelectingTaskExecutor`, which selects and persists a promoted adapter before delegating to real agent execution.
- `sqlite_state_uses_promoted_lora_for_next_agent_request` proves promoted `codegen-v1` reaches `InferenceRequest.lora_adapter_id` in the next real agent request.

Reality:

- Routing logic exists.
- Not proven that the selected adapter is actually applied by a backend model implementation or changes generation quality.
- Selection diagnostics are still incomplete: the request carries the id, but the diagnostics schema should show selection reason and fallback.

Required proof:

- Backend-specific integration: task -> router selects adapter -> inference backend applies adapter or returns typed unsupported evidence.
- Diagnostics: selected adapter id, reason, fallback path.

### 13. LoRA Training / Evolution

Status: `PARTIAL`

Claim: successful/failed tasks generate data; LoRA trains; benchmark proves improvement; adapter is promoted.

Current evidence:

- `LoraEvolutionService` creates examples, jobs, adapter records, validates reward/loss thresholds, indexes adapters.
- `MockLoraTrainer` writes a deterministic PEFT-like adapter directory with `adapter_config.json` and `adapter_model.safetensors`.
- Tests cover thresholds, repository effects, and a benchmark gate that either promotes the challenger or rolls it back before registration.

Reality:

- Deterministic product loop is now proven without real training.
- Training is mock in deterministic tests.
- Reward/loss threshold is not the same as proving task performance improvement.
- `BenchLoraBenchmarkGate` now runs baseline/challenger through `crytex-bench`, compares them with AB test, and accepts only a challenger winner.
- `concrete_gate_drives_lora_evolution_promotion_and_selection` proves golden examples -> deterministic trainer -> concrete benchmark gate -> promoted adapter -> inference registration -> selected adapter.
- Adapter and training-job metrics now persist `benchmark_gate` metadata: decision, reason, baseline/challenger run ids, AB winner, p-value, and pass rates.
- `LoraEvolutionServiceImpl` now emits structured `RunObserved` events on promotion and rejection, carrying training job id, task kind, adapter id, triggering task id, triggering task trace id, run id when available, metrics, and the persisted `benchmark_gate` metadata when a gate is present.
- Tauri app-state approval now calls LoRA evolution for normal work tasks: collect golden example, check threshold, and train/register when eligible.
- Tauri app-state rejection now calls LoRA evolution for normal work tasks: collect counter-example, preserve retry state, and avoid training from rejection.
- Tauri run diagnostics now exports `lora_evolution[]` entries from observable event metadata, including training job id, adapter id, triggering task id, trace id, run id, example count, AB winner, p-value, pass rates, and benchmark run ids when benchmark metadata is present.
- The SQLite-backed desktop approval path is now proven to surface a service-emitted LoRA decision in diagnostics without manual `RunObserved` injection.
- The SQLite-backed desktop approval path is now proven to carry benchmark-gate winner/p-value/pass-rate metadata to diagnostics when a gate is configured.
- The SQLite-backed desktop approval path is now proven to run the real `BenchLoraBenchmarkGate` against a held-out JSONL corpus through the default benchmark harness before promotion. This still uses a deterministic benchmark runner, so it proves policy/plumbing, not real LLM quality.
- The SQLite-backed desktop path is now proven to attach a promoted adapter id to the next real agent inference request after promotion.
- Single-file fake LoRA artifacts are now rejected before persistence/registration. The deterministic Mistral mock trainer and desktop diagnostics path use the stricter PEFT-like directory layout.
- The Mistral backend registry now enforces the same adapter layout boundary, so direct backend calls cannot bypass the evolution-service validator.
- Mistral capability reporting now distinguishes plain-model LoRA support from unsupported GGUF LoRA support.
- Core inference service consumers can request typed backend capability reports for UI/diagnostics without depending on backend-specific strings.
- Tauri runtime status and diagnostics now export backend capability truth to UI/manual testing.
- Kernel wires this gate into `LoraEvolutionService` when `lora_evolution.jsonl` exists and at least one project is available. Empty installs log a warning and start without the gate.

Required proof:

- Real trainer/model-quality proof:
  - real or semi-real trainer produces a valid adapter artifact;
  - held-out benchmark proves quality improvement without benchmark leakage.
- Real/semi-real model benchmark proof:
  - replace the deterministic benchmark runner with a real or semi-real model runner and prove that the product path promotes only when the model-quality benchmark accepts.
- Backend application proof:
  - prove a supported inference backend loads/applies the selected adapter or returns typed unsupported evidence instead of silently ignoring it.
- Then real trainer smoke when supported.

### 14. LoRA Hot Swap

Status: `NOT PROVEN`

Claim: adapter can be hot-swapped during runtime.

Current evidence:

- Trait methods `register_lora` and `swap_lora`.
- Mistral tests check active/request adapter selection metadata.
- Mistral now rejects unregistered request-level adapter ids before model load.
- Plain mistral model loading now passes registered adapter paths into `LoraModelBuilder`.
- GGUF plus locally registered LoRA now reports typed unsupported evidence instead of silently skipping adapter loading.
- Architecture already says mistral.rs path is not true free hot-swap; it may require model reload or request-level extension.

Reality:

- No proof that a loaded real model behavior changes after adapter swap.
- No proof yet that the locally produced trainer artifact is in the exact adapter format mistral.rs can load.
- Several backends do not support LoRA at all.

Required proof:

- Backend capability matrix.
- Typed `Unsupported` for backends that cannot do it.
- Real or semi-real adapter load/swap smoke for a supported backend.

### 15. Prompt Evolution

Status: `PARTIAL`

Claim: prompts mutate/evolve based on success and benchmarks.

Current evidence:

- `PromptEvolutionService` supports seed, mutate, tournament selection, recompute fitness, activate.
- Integration tests bind active prompt version to submitted tasks and agent service uses overrides.

Reality:

- Prompt record lifecycle exists.
- No benchmark/A-B promote/reject loop is wired.
- A mutated prompt can be activated manually, but "it got better" is not proven.

Required proof:

- Deterministic benchmark for prompt variants.
- Promote only if challenger improves.
- Reject if challenger regresses.
- Diagnostics: prompt version id, parent id, benchmark result id, promotion decision.

### 16. Benchmark Harness

Status: `PARTIAL`

Claim: benchmarks and A/B tests evaluate prompts/LoRA/task performance.

Current evidence:

- `crates/crytex-bench`
- Golden sets, scorers, harness persistence, AB test all pass unit tests.
- Kernel CLI can run/list/show/compare benchmarks.

Reality:

- Benchmarking exists as a tool.
- LoRA evolution now has a concrete benchmark gate implementation, kernel wiring, and deterministic promotion/selection proof.
- Prompt evolution still lacks an automatic benchmark promotion/rejection gate.

Required proof:

- Prompt evolution calls benchmark harness through a policy boundary.
- LoRA benchmark result ids and decisions are persisted in promotion/job metrics and can be surfaced through Tauri diagnostics as `lora_evolution[]`.

### 17. Tools and Sandbox

Status: `PARTIAL`

Claim: agents use safe tools and sandboxed execution.

Current evidence:

- `crates/crytex-tools`: fs, process, git, search, semantic search, sparse search.
- `PathSandbox` tests prevent path traversal/symlink escape.
- `ScanningToolService` checks prompt injection.
- `crytex-sandbox`: host, Docker, WASI backends; WASI has fuel/memory/preopen tests.

Reality:

- Tooling and sandbox primitives exist.
- Some Docker tests are external/ignored.
- Need product-level proof that real agent tool calls run inside the intended sandbox and cannot escape.

Required proof:

- E2E malicious tool attempt: path escape, network without capability, blocked prompt injection.
- Diagnostics should include tool call, capability decision, sandbox backend, exit code.

### 18. Security and Prompt Injection

Status: `PARTIAL`

Current evidence:

- Security scanner tests.
- Agent service audits security blocks with trace id.
- fs_read can wrap/block injected content.

Reality:

- Local scanners exist.
- Need e2e with malicious project document inside RAG proving it cannot override system prompt or trigger unsafe tool execution.

Required proof:

- Malicious RAG document e2e.
- Assert model/tool path refuses or quarantines injected instruction.

### 19. Observability and Diagnostics

Status: `PARTIAL`

Current evidence:

- Audit log service tests.
- Event service tests.
- Metrics and alert tests.
- Tauri diagnostics export exists and real smoke uses it.

Reality:

- Core observability exists.
- Coverage is not complete for model download, rerank, compression, LoRA, prompt evolution, benchmark promotion, and sandbox decisions.

Required proof:

- One diagnostics schema that can explain every critical decision in a run.
- Golden diagnostics tests for all backend feature paths.

### 20. Persistence

Status: `PARTIAL`

Current evidence:

- SQLite repositories for projects/tasks/prompts/LoRA/experience/benchmarks.
- Storage tests cover CRUD and some restart persistence.

Reality:

- Persistence is real.
- Migration/backward compatibility is not fully audited.

Required proof:

- Reopen test for complete app state.
- Migration fixtures for old schema/current schema.

### 21. CLI / Kernel Product Path

Status: `PARTIAL`

Claim: backend can be used and proven before UI.

Current evidence:

- Kernel has commands for prompt, project/task lifecycle, model manager, backend config, LoRA, index, run, benchmark.
- Kernel wires watcher, compression, lora router, prompt service, benchmark harness.

Reality:

- CLI is the right place to prove product readiness before UI.
- There are very few kernel-level integration tests.
- Reranker is wired into context assembly and proven in core/Tauri paths; kernel/CLI e2e proof with a real selected reranker model is still missing.

Required proof:

- CLI e2e matrix:
  1. create project;
  2. index project;
  3. submit goal;
  4. approve plan;
  5. run agents;
  6. critic reject/remediate;
  7. human approve;
  8. export diagnostics;
  9. run benchmark;
  10. prove prompt/LoRA promotion or rejection.

### 22. Agent Implementations

Status: `PARTIAL`

Claim: architect, coder, QA, security, critic, researcher, and specialized critics are real agents with structured IO and tool use.

Current evidence:

- `crates/crytex-agents/src/architect.rs`
- `crates/crytex-agents/src/coder.rs`
- `crates/crytex-agents/src/qa.rs`
- `crates/crytex-agents/src/security.rs`
- `crates/crytex-agents/src/critic.rs`
- `crates/crytex-agents/src/researcher.rs`
- `crates/crytex-agents/src/critics/*`
- `crates/crytex-agents/src/prompts.rs`
- `crates/crytex-agents/src/tooling.rs`
- Unit tests cover structured JSON parsing, markdown fence stripping, tool-call execution, prompt security blocks, specialized critic scores, and coder tool recording.

Reality:

- Agent structs and role prompts exist.
- Tooling loop exists for JSON tool calls.
- Prompt security block is injected into role prompts.
- Tests are mostly deterministic mocks, not real end-to-end model behavior.
- Agent output schemas now share core runtime validation across workflow execution and Tauri handoff diagnostics; remaining gap is schema-guided generation/prompting for real models.

Required proof:

- Typed result structs for every agent role.
- Contract tests that malformed model output is repaired/retried or rejected.
- Real local model smoke for each agent role on a tiny task.
- Diagnostics should include agent prompt id/version, raw output, parsed output, tool calls, and schema validation result.

### 23. Workflow Engine, Scheduler, and Worker Pool

Status: `PARTIAL`

Claim: tasks run according to dependency graphs/workflows with scheduling and concurrency.

Current evidence:

- `crates/crytex-core/src/services/workflow.rs`
- `crates/crytex-core/src/services/scheduler.rs`
- `crates/crytex-core/src/services/worker.rs`
- Tests cover DAG validation, cycles, TOML workflow loading, serial/parallel/conditional workflows, scheduler ordering/limits, skipped blocked tasks, worker concurrency.

Reality:

- Workflow and scheduling logic is real and well unit-tested.
- Product-level use is only partially proven. Kernel wires worker pool for run mode, but full CLI e2e is missing.
- Retry/cancellation behavior exists in task service but workflow-level retry exhaustion needs more product proof.

Required proof:

- CLI e2e with a custom workflow TOML.
- Test parallel branches producing artifacts consumed by a join/review step.
- Test cancellation/retry exhaustion with audit events.

### 24. Caching, Semantic Cache, Metrics, and Alerts

Status: `PARTIAL`

Claim: Crytex tracks performance, caches expensive operations, and emits alerts/metrics.

Current evidence:

- `crates/crytex-core/src/services/caching.rs`
- `crates/crytex-core/src/metrics.rs`
- `crates/crytex-core/src/services/alert_service.rs`
- Tests cover cached embedder hits/misses, cached vector search invalidation, metrics snapshots, persisted metrics, task/cache counters, threshold alerts.

Reality:

- Embedder/vector-store caches are implemented.
- Metrics/alerts are implemented.
- Architecture mentions a `semantic_cache` Qdrant collection, but product-level semantic response cache is not proven as an LLM response cache.

Required proof:

- E2E repeated semantic query/request that hits cache and exports saved tokens/latency.
- Diagnostics should show cache hit/miss per embed/search/LLM semantic cache decision.
- Alert stream should be included in observability export.

### 25. IDE Backend Crate

Status: `PARTIAL`

Claim: Crytex has a first-class IDE backend with LSP, editor bridge, inline suggestions, and project state sync.

Current evidence:

- `crates/crytex-ide/src/protocol.rs`
- `crates/crytex-ide/src/bridge.rs`
- `crates/crytex-ide/src/ide_service.rs`
- `crates/crytex-ide/src/lsp/*`
- Tests cover protocol serialization, LSP initialize flow, diagnostics, channel transport, and snapshot persistence from editor state.

Reality:

- The IDE backend crate is more than an empty placeholder.
- It is not yet integrated into the main Tauri/kernel product path.
- It does not prove a complete manual editing loop: list files, open file, edit file, LSP diagnostics, agent sees current IDE state, apply diff.

Required proof:

- Backend integration test: open/edit file through IDE service, persist snapshot, run LSP diagnostics, expose state to agent context.
- IPC/CLI boundary for IDE operations.
- Later UI can consume this, but backend proof should come first.

### 26. Loader Entrypoint

Status: `NOT PROVEN`

Claim: loader/entrypoint can bootstrap inference, embeddings, indexing, and storage as a packaged runtime.

Current evidence:

- `crates/crytex-loader/src/main.rs`
- No test markers were found in `crytex-loader`.

Reality:

- Loader code exists, but it is not covered by tests in the current audit.
- It should not be considered a proven product entrypoint.

Required proof:

- Add smoke/integration tests for loader config parsing and startup with mock backends.
- If loader is obsolete, document it and remove from product path to avoid confusion.

### 27. Tauri Backend Commands

Status: `PARTIAL`

Claim: Tauri command layer exposes backend capabilities to UI.

Current evidence:

- `crates/crytex-tauri/src/commands.rs`
- `crates/crytex-tauri/src/app_state.rs`
- Tauri unit tests and real Ollama e2e tests exist.

Reality:

- Tauri command layer has real backend paths and a known `StubTaskExecutor` fallback.
- Stub fallback is useful for development, but product readiness requires UI/API to clearly show when execution is stubbed.
- Some UI placeholders are backed by missing backend IPC paths.

Required proof:

- Command-level tests for every backend feature before UI claims it.
- Runtime status must expose stub vs real execution, backend, model, LoRA, prompt version, compression, RAG, reranker.

## Immediate Implementation Priorities

Priority 1: RAG correctness proof

- Product e2e: mixed project -> automatic index -> query -> rerank -> agent prompt -> exported diagnostics.
- Real ONNX reranker smoke with selected model metadata.
- Human-readable diagnostics for symbol graph selection and before/after rerank.

Priority 2: Model manager/runtime proof

- Mock HF download -> registry -> recommended config -> active runtime status.
- Optional real HF/GGUF smoke behind env vars.

Priority 3: Evolution proof

- Prompt benchmark promote/reject gate.
- LoRA deterministic benchmark promote/reject end-to-end loop with router/inference proof and diagnostics-export proof.
- Diagnostics for evolution decisions.

Priority 4: CLI product e2e

- Make kernel-level proof commands/tests so backend readiness does not depend on Tauri UI.

Priority 5: Real runtime expansion

- Ollama model switch.
- Cloud mocked full agent chain.
- Real mistral small GGUF load.
- Real LoRA only after deterministic gate is proven.
