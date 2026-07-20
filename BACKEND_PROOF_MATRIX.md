# Crytex Backend Proof Matrix

Date: 2026-07-16

Purpose: backend-first roadmap from current code to a product-ready core. UI is not the priority here. A capability is not considered real until it has deterministic tests, integration coverage, and at least one end-to-end proof where appropriate.

Detailed subsystem audit: [`BACKEND_FEATURE_AUDIT.md`](BACKEND_FEATURE_AUDIT.md).

## Latest Audit Run

Command run on 2026-07-16:

```powershell
$env:CARGO_TARGET_DIR='B:\crytex-target-audit'
cargo test -p crytex-doc -p crytex-core -p crytex-bench -p crytex-compress -p crytex-storage -p crytex-inference-mistral
cargo clean
```

Result:

- `crytex-core`: 223 unit tests passed; 5 integration tests passed in the full audit run. Later incremental RAG/rerank and LoRA benchmark-gate tests passed, raising the core unit count to 225.
- `crytex-doc`: 17 unit tests passed.
- `crytex-compress`: 74 unit tests passed.
- `crytex-storage`: 22 unit tests passed, 1 Qdrant external test ignored.
- `crytex-bench`: 14 unit tests passed in the full audit run. Later LoRA gate tests passed for challenger win, challenger loss, per-request task-kind runner creation, deterministic evolution promotion/selection through `LoraEvolutionServiceImpl`, and persisted benchmark-gate metadata in adapter/job metrics.
- `crytex-inference-mistral`: 8 unit tests passed.
- Later targeted verification passed: `cargo test -p crytex-kernel factory::tests::create_lora_evolution_service_wires_benchmark_gate_before_promotion` and `cargo check -p crytex-kernel`.
- Later LoRA observability verification passed: `cargo test -p crytex-core lora_evolution::tests` and `cargo test -p crytex-tauri build_run_diagnostics_`. The service-emitted promotion/rejection events now carry the triggering task trace id in both `Event::RunObserved.trace_id` and `metadata.trace_id`.
- Later Tauri product-path trigger verification passed: `cargo test -p crytex-tauri sqlite_state_`. Human approval of a normal work task now calls LoRA golden-example collection and training when the threshold is met; human rejection records a LoRA counter-example without breaking retry semantics; critic/review gate tasks are intentionally excluded from the LoRA dataset.
- Latest full product-path LoRA diagnostics verification passed: `sqlite_state_approval_triggered_lora_service_decision_reaches_diagnostics`, `cargo test -p crytex-tauri sqlite_state_`, `cargo test -p crytex-tauri build_run_diagnostics_`, `cargo test -p crytex-core lora_evolution::tests`, and `cargo test -p crytex-storage lora`. This proved `approve_task_review -> LoraEvolutionServiceImpl -> training job -> adapter record -> RunObservedAuditBridge -> export_run_diagnostics.lora_evolution[]` using SQLite storage and the deterministic mock trainer.
- Latest product-path benchmark metadata verification passed: `sqlite_state_approval_triggered_lora_benchmark_gate_reaches_diagnostics`, `cargo test -p crytex-tauri sqlite_state_`, and `cargo test -p crytex-tauri build_run_diagnostics_`. This proved the SQLite-backed Tauri approval path can inject a `LoraBenchmarkGate`, call it before promotion, persist winner/p-value/pass-rate metadata, and export it through diagnostics.
- Latest held-out benchmark product-path verification passed: `sqlite_state_approval_triggered_real_bench_gate_uses_held_out_corpus`, `cargo test -p crytex-tauri sqlite_state_`, `cargo test -p crytex-tauri build_run_diagnostics_`, and `cargo test -p crytex-bench lora_gate`. This proved the Tauri approval path can use the real `BenchLoraBenchmarkGate`, `DefaultBenchmarkHarness`, `ExactMatchScorer`, AB comparison, and a held-out JSONL corpus before promotion.
- Latest LoRA request-routing verification passed: `cargo test -p crytex-tauri sqlite_state_uses_promoted_lora_for_next_agent_request`. This proved that after approval-triggered promotion, the next real agent execution path attaches the selected adapter id to `InferenceRequest.lora_adapter_id`.
- Latest Mistral LoRA guard verification passed: `cargo test -p crytex-inference-mistral generate_rejects_unregistered_request_lora_adapter_before_model_load`. This proved the mistral.rs backend no longer silently accepts an unknown request-level adapter id before model load.
- Latest Mistral LoRA load-plan verification passed: `cargo test -p crytex-inference-mistral load_plan`, `cargo test -p crytex-inference-mistral`, and `cargo test -p crytex-core unsupported_inference_error_maps_to_unsupported_service_error`. This proved registered plain-model LoRA adapter paths are passed into the mistral.rs `LoraModelBuilder`, while GGUF plus locally registered LoRA is explicitly reported unsupported until that path is wired, and the unsupported evidence survives the core service boundary.
- Latest unsupported-backend LoRA verification passed: `cargo test -p crytex-inference-openai -p crytex-inference-anthropic -p crytex-inference-ollama -p crytex-inference-onnx lora_is_unsupported`. This proved OpenAI, Anthropic, Ollama, and ONNX no longer claim LoRA registration/swap success; they return typed `UnsupportedOperation`.
- Latest LoRA artifact-boundary verification passed: `cargo test -p crytex-core lora_evolution::tests`, `cargo test -p crytex-inference-mistral training::tests`, and `cargo test -p crytex-tauri sqlite_state_approval_triggered_lora_service_decision_reaches_diagnostics`. This proved `LoraEvolutionServiceImpl` rejects a single-file fake adapter before persistence/registration, requires a PEFT-like adapter directory with `adapter_config.json` and `adapter_model.safetensors`, and the deterministic Mistral mock trainer/product diagnostics path now produce and consume that layout.
- Latest Mistral registry-boundary verification passed: `cargo test -p crytex-inference-mistral`. This proved `register_lora` now rejects single-file adapters and malformed `adapter_config.json` before inserting into the backend registry, while valid PEFT-like adapter directories still feed the plain-model `LoraModelBuilder` load plan.
- Latest Mistral capability-truthfulness verification passed: `cargo test -p crytex-inference-mistral`. This proved GGUF file and GGUF-directory backends no longer advertise `"lora"` capability until GGUF LoRA is implemented, while plain model backends still advertise LoRA.
- Latest inference capability-report verification passed: `cargo test -p crytex-inference` and `cargo test -p crytex-core backend_capability_reports_are_typed`. This proved backend string capabilities now map to a typed report with `generate`, `chat`, `embed`, `rerank`, `lora`, and `hot_swap`, and `dyn InferenceService` consumers can request this report without duplicating string parsing.
- Latest Tauri capability-report export verification passed: `cargo test -p crytex-tauri sqlite_state_`, `cargo test -p crytex-tauri build_run_diagnostics_collects_trace_tasks_events_rag_and_reward_evidence`, `npm test -- --run`, and `npm run build`. This proved runtime status and run diagnostics now expose typed backend capability reports to the desktop/UI contract, including Ollama generate/chat without LoRA/hot-swap and Mistral GGUF without LoRA/hot-swap.
- Latest managed-model product-path verification passed: `cargo test -p crytex-tauri sqlite_state_adds_downloads_lists_and_activates_managed_model` and `cargo test -p crytex-tauri managed_model`. This proved the desktop-facing backend path can add a managed model, download it through the `ModelManager` trait boundary, list it as downloaded, activate it as the Mistral runtime, and expose Mistral generate/chat capabilities without manually editing the registry.
- Latest real Hugging Face downloader smoke passed: `cargo test -p crytex-core real_hf_download_persists_registry_and_reloads_as_downloaded` verified the test is ignored by default, then `cargo test -p crytex-core real_hf_download_persists_registry_and_reloads_as_downloaded -- --ignored --nocapture` downloaded `sshleifer/tiny-gpt2/config.json` through `HfHubDownloader`, copied it into the managed cache, persisted `registry.toml`, and reloaded the model as `Downloaded`.
- Latest real Hugging Face GGUF smoke passed: `cargo test -p crytex-core real_hf_tiny_gguf_download_reloads_as_mistral_runtime_candidate` verified the 83 MB network smoke is ignored by default, then `cargo test -p crytex-core real_hf_tiny_gguf_download_reloads_as_mistral_runtime_candidate -- --ignored --nocapture` downloaded `tensorblock/tiny-random-minicpm-GGUF/tiny-random-minicpm-Q2_K.gguf`, copied it into managed cache, persisted/reloaded the registry, and preserved `BackendKind::MistralRs` plus `Q2_K` metadata.
- Latest real Mistral GGUF generation smoke passed on CUDA. A slow manual smoke exists in `crytex-inference-mistral` as `real_hf_tiny_gguf_downloaded_model_generates_with_mistralrs`; it is ignored and additionally guarded by `CRYTEX_RUN_SLOW_MISTRAL_SMOKE=1`. The smoke is hardware-aware: auto mode uses detected CUDA/Metal instead of forcing `Some(0)` CPU, `CRYTEX_MISTRAL_SMOKE_DEVICE=cpu|gpu|auto` selects mode, and `CRYTEX_MISTRAL_SMOKE_GPU_LAYERS` can override layer count. Actual attempts found concrete gaps: tensorblock tiny-random GGUF files fail Candle/mistral.rs load with `unknown dtype for tensor 20`; `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` CPU generation timed out after 180 seconds. On 2026-07-17, `scripts\mistral-cuda-smoke.ps1 -TimeoutSeconds 1200 -IdleTimeoutSeconds 240` built normal CUDA without `-SkipGdnCuda`, forced GPU runtime on `NVIDIA GeForce RTX 5080`, downloaded/loaded TinyLlama GGUF, generated successfully, reported `test result: ok. 1 passed`, and cleaned 8.8 GiB.
- Latest CUDA preflight/build-probe verification passed: `cargo test -p crytex-core hardware`, `cargo test -p crytex-tauri runtime_status_includes_cuda_toolchain_preflight`, and `cargo test -p crytex-inference-mistral smoke_runtime_`. This proved Crytex now evaluates CUDA readiness as structured data (`gpu_detected`, `nvcc_available`, `msvc_cl_available`, `msvc_cl_path`, `NVCC_CCBIN`, `recommended_nvcc_ccbin`, `ready`, diagnostics), exports it through Tauri `RuntimeStatus`, and keeps the Mistral smoke runtime GPU-aware. On this machine the facts are: NVIDIA GeForce RTX 5080 / 16303 MiB / driver 596.36 is detected, `nvcc 13.2` is available, `cl.exe` is not in the current PowerShell PATH, and MSVC `cl.exe` is discoverable under Visual Studio Build Tools. `scripts/mistral-cuda-build-probe.ps1` and `scripts/mistral-cuda-smoke.ps1` set `NVCC_CCBIN`, `CL=/Zc:preprocessor /DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING`, and `CUDA_NVCC_FLAGS`; they also accept `CUDA_COMPUTE_CAP` override for diagnostics. The build probe wraps CUDA compilation with stdout/stderr logs, wall timeout, idle timeout, process command-line dump, and cleanup. Earlier production probes isolated the old blocker to `mistralrs-core src\cuda\gdn.cu`.
- Latest CUDA GDN fix on 2026-07-17 proved the blocker is specifically the optimized `chunked_gated_delta_rule_recurrence` prefill kernel, not the whole GDN translation unit. `scripts\mistral-gdn-kernel-probe.ps1 -TimeoutSeconds 120` reproduced the full-file timeout before the fix; disabling chunked recurrence compiled in 5 seconds; targeted `#pragma unroll 1` on the heavy `BK` dot-product/register loops made the real chunked kernel compile. A direct CUDA correctness test then exposed and fixed another real bug: recurrence kernels returned before `__syncthreads()` on partial V tiles, so cooperative shared-memory loads were incomplete when `v_dim < BV` or on a tail tile. The fixed kernels keep all threads in shared loads/barriers and gate only per-V computation/writes with `has_v`. Evidence: `cargo test --manifest-path patches\mistralrs-core\Cargo.toml --features cuda cuda_chunked_gdn_matches_cpu_reference -- --nocapture` passed against a CPU reference and cleaned 4.3 GiB; the default full `gdn.cu` probe passes in 62 seconds on `sm_120a`; `scripts\mistral-cuda-smoke.ps1 -TimeoutSeconds 1500 -IdleTimeoutSeconds 300` built CUDA, forced GPU runtime on `NVIDIA GeForce RTX 5080`, downloaded/loaded TinyLlama GGUF, generated successfully, reported `test result: ok. 1 passed`, and cleaned 8.8 GiB. The later Qwen3 Next MoE blocker was fixed in the local `mistralrs-quant` patch by falling back to ordinary indexed tensor matmul for CUDA `GgufMatMul` weights that are `Tensor/TensorF16` instead of `QTensor`. Evidence: `cargo test --manifest-path patches\mistralrs-quant\Cargo.toml --features cuda should_gather_unquantized_tensor_weights_without_requiring_qtensor -- --nocapture` first failed with `indexed_moe_forward is only supported for quantized tensors (QTensor)`, then passed after the fix and cleaned 1.8 GiB.
- Latest full GDN-family GPU smoke passed: `scripts\mistral-cuda-smoke.ps1 -Gdn -TimeoutSeconds 1500 -IdleTimeoutSeconds 300` built patched `mistralrs-quant` and `mistralrs-core`, forced GPU runtime on `NVIDIA GeForce RTX 5080`, ran `real_hf_tiny_qwen3_next_gdn_generates_with_mistralrs`, reported `test result: ok. 1 passed`, and cleaned 9.0 GiB. Remaining limitation: this proves the tiny Qwen3-Next GDN/MoE e2e path, not automatic production-grade model conversion/optimization or broader GDN-family performance.
- Latest runtime-status contract verification passed: `cargo test -p crytex-tauri app_state::tests` (28 passed). Tauri `RuntimeStatus` includes `compatibility_notes`; these notes now describe the narrower truth: normal CUDA generation, direct chunked GDN kernel correctness, and tiny Qwen3-Next GDN smoke are proven, while broader model-family compatibility remains a runtime diagnostic concern.
- Latest model compatibility planner proof added a core-level `ModelCompatibilityPlanner` that classifies models before weight load by format (`GGUF`, `HuggingFace`) and feature family (`dense`, `MoE`, `GDN`), then emits a typed execution strategy (`CudaFused`, `CudaWithFallback`, `Cpu`, `Metal`, `Unsupported`) with actions/warnings/blockers. Evidence: `cargo test -p crytex-core model_compatibility -- --nocapture` passed 3 tests and cleaned 3.3 GiB. This proves Qwen3-Next MoE/GDN is handled as a family requiring CUDA fallbacks, not as a one-off model patch; missing GDN CUDA support now produces a typed unsupported plan before model load. Tauri `RuntimeStatus` now carries `model_compatibility: Option<ModelCompatibilityPlan>` for managed Mistral models, and `cargo test -p crytex-tauri managed_gguf_runtime_status_reports_cuda_gdn_compatibility -- --nocapture` passed and cleaned 24.0 GiB.
- Cleanup removed 13.6 GiB in the full run; later cleanup removed 18.2 GiB, 22.5 GiB, 21.5 GiB, 7.1 GiB, 5.9 GiB, 5.9 GiB, 7.2 GiB, 3.3 GiB, 21.5 GiB, 22.7 GiB, 21.5 GiB, 3.3 GiB, 3.3 GiB, 7.3 GiB, and 7.3 GiB from `B:\crytex-target-audit`.

Important interpretation: this proves a lot of backend modules compile and their local contracts pass. It does not prove every architecture claim end-to-end.

## Rules

1. No feature is "done" because a struct, trait, or placeholder exists.
2. A feature is "proven" only when there is executable evidence:
   - unit tests for domain logic;
   - integration tests for persistence/services;
   - e2e/smoke tests for runtime behavior;
   - diagnostics/audit evidence for observability.
3. Every AI/runtime feature must expose traceable evidence: model, prompt/request, output, decision, artifact, reward, benchmark result.
4. Real-model tests may be slower, but the system must have at least one maintained non-ignored real runtime smoke per critical path.
5. After Rust verification, run `cargo clean` with `CARGO_TARGET_DIR='B:\crytex-target-audit'`.

## Current Verified Slice

The currently proven vertical slice is:

`Ollama qwen3.5:9b -> create project -> watcher/indexer -> RAG context -> submit goal -> approve plan -> generated agent chain -> critic gate -> optional remediation -> human approval -> reward -> export_run_diagnostics`

Evidence:

- `crates/crytex-tauri/tests/e2e_ollama_start_run.rs`
- `real_runtime_smoke_reports_production_agent_chain`
- `crates/crytex-tauri/src/commands.rs`
- `build_run_diagnostics_collects_trace_tasks_events_rag_and_reward_evidence`
- `build_run_diagnostics_keeps_review_gate_after_human_approval`

This proves one real vertical path. It does not prove the full product.

## Capability Matrix

Status legend:

- `PROVEN`: has meaningful unit/integration/e2e evidence for the product claim.
- `PARTIAL`: real code exists, but the full product claim is not proven.
- `NOT PROVEN`: mostly interface, mock, placeholder, or missing integration.

| Capability | Status | Current evidence | Real gap | Required proof |
| --- | --- | --- | --- |
| Goal orchestration | PARTIAL | Unit/integration tests in orchestrator, Tauri app state, real Ollama smoke | It decomposes into a serial role pipeline, but strict atomic task validation is missing | E2E matrix: approve path, reject path, retry exhaustion, malformed model output; validation that subtasks are atomic and acceptance-criteria scoped |
| Agent chain execution | PARTIAL | Real Ollama smoke proves architect/coder/qa/security/critic chain; Tauri tests verify upstream artifacts are attached; core `ArtifactContractViolation` tests enforce required role outputs; `AgentWorkflowNodeExecutor` rejects malformed coder/critic artifacts before workflow state handoff; Tauri command tests prove retry/fail/remediation diagnostics for malformed artifacts | Real model structured-output prompting is not yet guaranteed for every role; schemas are runtime-validated but not yet generated from a single JSON Schema source | Real model per-role smoke: malformed output repair/retry, valid artifact lineage, and schema-guided prompting |
| Critic rejection/remediation | PARTIAL | Smoke has `critic_rejected` and `remediation_plan_created`; tests assert feedback/failure type/target task; critic reject contract now requires `blocking_issues`; app-state remediation-chain test proves rejection feedback creates a remediation chain and returns to review | Real-model critic adherence to the structured rejection contract is not yet proven | Real critic smoke: reject includes target task, failure type, blocking issues, feedback, remediation linkage, diagnostics |
| Human review/reward | PARTIAL | Reward recorded in smoke and diagnostics; feedback loop tests exist | Need broader rejected-human path and repeated feedback loop into evolution | E2E: approve creates positive experience, reject creates counter example/debug task and affects prompt/LoRA eligibility |
| AST/chunking/code graph | PARTIAL | `crytex-doc` tree-sitter chunking, graph builder, impact tests passed; kernel builds `CodeGraph`; project indexer uses graph-aware code chunking and persists `symbol_id`/`related_symbols`; context assembler exposes graph ids in prompt/evidence; Tauri app-state proof exports graph metadata in `rag_context_assembled` diagnostics | Graph proof is Rust-focused; diagnostics still expose raw graph ids rather than a human-readable symbol-neighborhood explanation | Broaden language coverage and add product-readable graph selection rationale |
| RAG/Qdrant Edge | PARTIAL | Indexer, Edge vector store, context assembler, hybrid dense+sparse tests passed; mixed code/markdown/PDF indexing proof exists; real smoke proves marker reaches model; context evidence preserves retrieval candidates, reranked chunks, selected chunks, graph metadata, and retrieval sources | Need full product entrypoint proof from project creation through query/agent prompt/diagnostics across mixed docs; real reranker model smoke still missing | Product e2e: mixed project -> automatic index -> query -> rerank -> agent prompt -> exported diagnostics with before/after evidence |
| Reranker | PARTIAL | Trait and ONNX implementation exist; ONNX config tests exist; `ContextAssembler::with_reranker` is tested; kernel wires `create_reranker` into context assembly; Tauri app-state test proves reranker changes agent prompt order and diagnostics include retrieval candidates/reranked/selected chunks | Real ONNX reranker model path and user-selected model diagnostics are not proven | Real ONNX reranker smoke and UI/CLI diagnostics that show backend/model plus before/after ranks |
| Context compression/token optimizer | PARTIAL | `crytex-compress` passed 74 tests; inference service has compression replacement tests | Need proof in real Tauri/Ollama request path and diagnostics | E2E or integration: oversized context is compressed under budget and preserves required facts |
| Model manager/HF download | PARTIAL | ModelManager mocked download/progress/recommendation tests passed; Tauri `managed_model` tests prove add/download/list/activate through the UI-facing app-state path without manual registry edits; ignored real HF smoke downloads `sshleifer/tiny-gpt2/config.json`; ignored real GGUF smoke downloads `tiny-random-minicpm-Q2_K.gguf`, persists registry, and reloads as a Mistral runtime candidate | HF file/GGUF download is proven, but model inventory/search UI and real downloaded model generation/load are not proven | Optional real GGUF load/generate smoke after download; diagnostics for download progress/history |
| Runtime optimization | PARTIAL | Hardware recommendation tests for quantization/context/gpu layers; CUDA preflight tests; Tauri `RuntimeStatus.cuda_toolchain`; `scripts/mistral-cuda-build-probe.ps1`/`scripts/mistral-cuda-smoke.ps1` auto-discover MSVC, set CUDA/MSVC flags including `/MD`, log stdout/stderr, enforce timeout, dump live command lines, support compute-cap override, support `-SkipGdnCuda`, and clean target; `scripts/mistral-gdn-kernel-probe.ps1` isolates GDN CUDA compilation; targeted unroll control fixed the chunked GDN compile hang; direct CUDA chunked GDN test now matches CPU reference; mistral backend accepts `gpu_layers` and `context_size`; Mistral capability reporting now distinguishes GGUF vs plain model LoRA support; registry/core/Tauri runtime status and diagnostics expose typed backend capability reports; slow manual Mistral smokes exist behind env guards; normal CUDA GPU smoke proved TinyLlama GGUF generation on RTX 5080 without `-SkipGdnCuda`; GDN-family CUDA smoke proved tiny Qwen3-Next MoE/GDN generation on RTX 5080 after the `mistralrs-quant` CUDA GGUF MoE fallback fix; core compatibility planner now classifies GGUF/HF + dense/MoE/GDN and emits typed execution strategies/blockers before model load, surfaced through Tauri `RuntimeStatus.model_compatibility` | Real GPU generation, direct GDN kernel correctness, tiny GDN-family e2e, and first-pass family compatibility planning are proven, but actual model optimization/quantization is still recommendation/config, not automatic conversion/tuning; broader model-family performance is not benchmarked | Add config.json/GGUF metadata inspection, chunked prefill performance regression test, load/generate telemetry, automatic model optimization/tuning proof, and model-family compatibility reporting in UI |
| Ollama runtime | PARTIAL | Existing Tauri real smoke path with Ollama/qwen model | Need model inventory/switch e2e with a real model | Real test: list models, select model, run task, diagnostics show selected model |
| Cloud AI providers | NOT PROVEN | OpenAI/Anthropic crates and mocked HTTP tests exist; typed capability reports can expose generate/embed/lora/hot_swap support without UI-side string parsing | Product-level API key/config path and full agent path not proven | Mocked integration plus optional real provider smoke behind env vars |
| LoRA router | PARTIAL | Unit tests prove explicit/role/kind selection and semantic fallback; `sqlite_state_uses_promoted_lora_for_next_agent_request` proves a promoted adapter is selected before Tauri agent execution and reaches `InferenceRequest.lora_adapter_id` | Selection is proven at request metadata level, but backend-specific application of the adapter is not proven | Backend-specific integration: selected adapter changes or explicitly fails unsupported in the inference backend |
| LoRA training/evolution | PARTIAL | Unit tests prove experience capture, threshold, training job, adapter record, vector index, active selection; `concrete_gate_drives_lora_evolution_promotion_and_selection` proves deterministic trainer -> concrete `crytex-bench` gate -> promotion -> inference registration -> selection; adapter/job metrics persist benchmark-gate decision metadata; `LoraEvolutionServiceImpl` now emits structured trace-correlated `RunObserved` promotion/rejection events; Tauri app-state approval path triggers golden-example collection and threshold training for normal work tasks; Tauri app-state rejection path records counter-examples without breaking retry semantics; `sqlite_state_approval_triggered_lora_service_decision_reaches_diagnostics` proves the SQLite-backed desktop product path exports service-emitted LoRA decisions in diagnostics without manual event injection; `sqlite_state_approval_triggered_real_bench_gate_uses_held_out_corpus` proves product-path `BenchLoraBenchmarkGate` with held-out JSONL corpus, AB comparison, and diagnostics metadata; `sqlite_state_uses_promoted_lora_for_next_agent_request` proves the promoted adapter id is attached to the next agent request; core now rejects single-file fake adapters and requires a PEFT-like adapter directory before persistence/registration; Mistral `register_lora` independently validates the same layout before backend registry insertion | Real trainer path still needs maintained e2e; no real model-quality improvement proof yet; no backend-specific hot-swap behavior proof; held-out proof uses deterministic runner, not a real LLM; artifact layout is shape-checked but not yet proven loadable by a real model | Real trainer smoke when backend supports it; prove held-out quality improvement without leakage against a real/semi-real model; prove backend really applies selected adapter |
| LoRA hot swap | NOT PROVEN | Inference trait has `register_lora`/`swap_lora`; mistral unit tests check adapter selection metadata; mistral now rejects unknown request adapters before model load; Mistral `register_lora` rejects invalid adapter dirs before registry insertion; plain-model registered LoRA paths are wired into `LoraModelBuilder`; GGUF registered LoRA returns typed unsupported and GGUF backends no longer advertise `"lora"` capability; OpenAI/Anthropic/Ollama/ONNX return typed unsupported for LoRA | No backend-specific proof that a loaded model changes behavior; architecture says mistral currently reloads/uses adapter, not true free hot swap; deterministic mock artifact layout is now PEFT-like but still not a real trained adapter | Backend-specific integration: register adapter, swap, request uses adapter or reports unsupported clearly; real/semi-real prompt shows adapter changes output |
| LoRA benchmark improvement | PARTIAL | `BenchLoraBenchmarkGate` runs baseline/challenger through `crytex-bench`, compares with AB test, accepts only challenger winner, rejects regression, builds a runner per requested task kind, persists decision metadata, and Tauri diagnostics can surface benchmark run ids/winner/p-value/pass rates from service-emitted events; Tauri product path now has a held-out JSONL corpus proof through the real benchmark harness | Still needs real trainer/model-quality proof; deterministic runner proof is not the same as real model improvement | Real/semi-real benchmark: trained adapter improves held-out task quality without leakage |
| Prompt evolution | PARTIAL | Unit/integration tests seed/mutate/select/fitness/activate and bind active prompt version | No benchmark/A-B loop tied to task outcomes; improvement is not proven | Integration: reward experiences update fitness, benchmark challenger prompt, promote only if better |
| Prompt injection/security | PARTIAL | Security scanner and agent-service security tests passed | Need e2e proving malicious project docs do not override agent system prompt | Integration with malicious RAG document and blocked tool/write behavior |
| Bench harness | PARTIAL | `crytex-bench` harness, scorer, AB tests passed; Tauri product-path LoRA approval proof now triggers `BenchLoraBenchmarkGate` and stores benchmark evidence before promotion | Still deterministic runner, not real model-quality proof | Integration with real/semi-real model runner and held-out corpus |
| Experience/memory | PARTIAL | Reward and LoRA evolution tests write examples/experience; memory bank semantic recall tests passed | Need full chain from human review to prompt/LoRA candidate | E2E: approved/rejected task updates experience, prompt fitness, LoRA dataset eligibility |
| Agent implementations | PARTIAL | Role agents, prompts, JSON parsing, tooling tests exist; shared core artifact contracts validate architect/coder/qa/security/critic results; malformed-output retry/failure tests pass in Tauri command layer | Mostly mock-backed; real models are not yet forced through schema-guided structured output per role | Real local smoke per role with schema-conformant output and repair/retry diagnostics |
| Workflow/scheduler/worker | PARTIAL | DAG/serial/parallel/conditional workflow tests; scheduler and worker tests | Product-level CLI workflow e2e missing | CLI e2e with custom workflow, retry/cancel exhaustion, artifact join |
| Caching/metrics/alerts | PARTIAL | Cached embedder/vector tests; metrics and alert tests | Semantic response cache and diagnostics coverage not proven | Repeated request cache e2e with hit/miss, saved tokens, alert export |
| IDE backend crate | PARTIAL | LSP protocol/client/transport and editor bridge tests exist | Not integrated into main backend product path | Open/edit/diagnostics/snapshot integration exposed to agent context |
| Loader entrypoint | NOT PROVEN | `crytex-loader/src/main.rs` exists | No tests found; product role unclear | Startup/config smoke with mock backends or remove/document obsolete path |
| Observability | PARTIAL | Audit logs, events, diagnostics export, trace tests, real smoke | Need coverage for all feature domains, especially model/evolution/RAG | Diagnostics schema includes model download, rerank, compression, LoRA, prompt version, benchmark |
| Persistence/migrations | PARTIAL | SQLite repositories and storage tests passed; experience survives restart | Need migration/reopen tests for complete app state | Integration: create data with old schema/current schema, reopen, verify state |

## Backend Roadmap

### Phase B1 - Proof Inventory and Test Harness Hygiene

Goal: know exactly what is proven and make tests runnable intentionally.

Tasks:

1. Add this proof matrix as the source of truth.
2. Create test groups:
   - fast unit;
   - integration;
   - real local runtime;
   - external/network optional.
3. Normalize ignored real tests with clear env vars and reasons.
4. Add smoke report output for every real runtime test.

Exit criteria:

- One command runs fast backend tests.
- One command runs real local Ollama tests.
- Ignored tests are documented with env/model requirements.

### Phase B2 - RAG, Rerank, and Compression Proof

Goal: prove context quality, not just vector search.

Tasks:

1. Add integration test for mixed project indexing: code + markdown + PDF/plain text.
2. Add reranker test proving candidate order changes.
3. Wire reranker into the retrieval/context path used by agent execution, or explicitly document why not.
4. Add compression integration test:
   - input context exceeds budget;
   - compressed context fits budget;
   - required sentinel facts survive;
   - LLM request contains compressed context evidence.

Exit criteria:

- RAG test proves correct top context.
- Reranker test proves reranking affects final context.
- Compression test proves token budget enforcement and fact preservation.

### Phase B3 - Model Management and Runtime Selection Proof

Goal: "download/select/optimize model" becomes real backend behavior.

Tasks:

1. Add non-network integration for model manager:
   - manifest entry;
   - mock HF download;
   - registry persistence;
   - hardware recommendation;
   - runtime status update.
   Status: UI-facing add/download/list/activate path is now covered by Tauri `managed_model` tests; core registry-reload and real HF coverage still need explicit tests.
2. Add optional real HF test behind env vars with a tiny or configured artifact.
   Status: implemented as ignored tests using `sshleifer/tiny-gpt2/config.json` and `tensorblock/tiny-random-minicpm-GGUF/tiny-random-minicpm-Q2_K.gguf`.
3. Prove managed model selection emits runtime event and uses recommended config.
4. Prove Ollama model switch affects the next real run and diagnostics.

Exit criteria:

- User-facing model path can be verified without UI.
- Runtime diagnostics always show active backend/model/config.

### Phase B4 - Prompt Evolution Proof

Goal: prompt evolution must demonstrably improve or reject bad mutations.

Tasks:

1. Build a deterministic benchmark runner for prompt variants.
2. Add integration:
   - baseline prompt gets lower reward/score;
   - mutated prompt gets higher score;
   - AB/benchmark promotes challenger.
3. Add negative integration:
   - mutated prompt performs worse;
   - system rejects challenger and keeps baseline.
4. Wire prompt version id into agent execution and diagnostics.

Exit criteria:

- Prompt mutation is not just created; it is benchmarked and promoted/rejected by evidence.
- Diagnostics show prompt version used for a task.

### Phase B5 - LoRA Evolution Proof Without Real Training

Goal: prove the LoRA product loop with deterministic trainer before expensive real training.

Tasks:

1. Add fake trainer that produces deterministic adapter records.
2. Add benchmark gate:
   - baseline variant no LoRA;
   - challenger variant with LoRA;
   - challenger wins only if benchmark improves.
3. Add promotion/rollback policy.
4. Add diagnostics fields:
   - selected LoRA;
   - training job id;
   - benchmark run id;
   - promotion decision.

Exit criteria:

- End-to-end deterministic test proves LoRA evolution loop:
  experience -> training examples -> adapter -> benchmark -> promote/rollback -> router selects adapter.

### Phase B6 - Real LoRA/Hot-Swap Backend Proof

Goal: prove or explicitly bound real adapter support.

Tasks:

1. Identify which backend can actually hot-swap LoRA in-process:
   - mistral.rs;
   - candle;
   - Ollama limitations;
   - OpenAI/cloud unsupported.
2. Add backend capability reporting.
3. Add real or semi-real adapter load/swap test for supported backend.
4. For unsupported backends, return explicit `Unsupported` evidence instead of pretending.

Exit criteria:

- There is a real backend test proving hot swap, or a typed unsupported result.
- No UI/API claims hot swap works for a backend where it does not.

### Phase B7 - Full Backend Product E2E Matrix

Goal: backend alpha is proven without UI.

Required E2Es:

1. Ollama local code project happy path.
2. Ollama critic reject/remediation/human approve path.
3. RAG-only docs/PDF project path.
4. Managed model mocked download/select/run path.
5. Prompt evolution benchmark promote/reject path.
6. LoRA evolution deterministic promote/reject path.
7. Compression/token budget path.
8. Cloud provider mocked path.
9. Persistence reopen path.
10. Security prompt-injection path.

Exit criteria:

- All required E2Es pass or are explicitly marked external/ignored with env requirements.
- Diagnostics reports cover every path.
- No critical architecture feature remains "just a crate".

## Immediate Next Implementation Target

Next target: **Full-run LoRA diagnostics and real-trainer boundary**.

Reason:

- This is one of the core product moats.
- There is already a strong base: `LoraEvolutionService`, `crytex-bench`, `ABTest`, training jobs, adapter records, router.
- The deterministic loop, persisted benchmark metadata, trace-correlated service-level observability, diagnostics export schema, Tauri approval/rejection triggers, SQLite-backed approval-triggered diagnostics path, product-path benchmark-gate diagnostics path, and held-out `BenchLoraBenchmarkGate` product path are now present. The remaining proof is the real-training boundary: "a user run produces approved/rejected examples, triggers LoRA evolution when eligible, exports the service-emitted decision in diagnostics, and later a real trainer can prove model-quality improvement without changing the policy."

TDD plan:

1. DONE: add full-run diagnostics test where approval-triggered `LoraEvolutionServiceImpl` emits the LoRA decision and diagnostics exports it.
2. DONE: assert diagnostics includes training job id, adapter id, task id, trace id, run id, training example count, and accepted outcome.
3. DONE: fix desktop LoRA threshold wiring for normalized human-review scores and fix nullable SQLite LoRA persistence.
4. DONE: attach benchmark-gate proof to the same product path instead of only deterministic service/unit paths.
5. DONE: prove a held-out benchmark corpus drives the product path through the real `BenchLoraBenchmarkGate` instead of a deterministic accepting test gate.
6. DONE: prove the promoted adapter id is selected before the next Tauri agent execution and reaches `InferenceRequest.lora_adapter_id`.
7. DONE: reject single-file fake LoRA artifacts before adapter persistence/registration and make the deterministic trainer emit a PEFT-like adapter directory.
8. DONE: validate PEFT-like adapter layout at the Mistral backend registry boundary before `register_lora` mutates backend state.
9. DONE: make Mistral capability reporting stop advertising LoRA for GGUF file/directory backends until that path is implemented.
10. DONE: expose typed backend capability reports through `crytex-inference` and the core `InferenceService` default method.
11. DONE: export typed backend capability reports through Tauri runtime status and run diagnostics, with Rust and frontend contract tests.
12. NEXT: replace deterministic benchmark runner/trainer with a real or semi-real model-quality proof and prove backend-specific adapter application.
13. Verify:
   - `cargo test -p crytex-core lora`
   - `cargo test -p crytex-bench`
   - `cargo test -p crytex-tauri build_run_diagnostics_`
   - clippy for touched crates
   - cargo clean.

## Commands

Fast-ish checks:

```powershell
$env:CARGO_TARGET_DIR='B:\crytex-target-audit'
cargo test -p crytex-core -p crytex-bench -p crytex-compress -p crytex-agents
```

Tauri/local runtime smoke:

```powershell
$env:CARGO_TARGET_DIR='B:\crytex-target-audit'
cargo test -p crytex-tauri real_runtime_smoke_reports_production_agent_chain --test e2e_ollama_start_run -- --nocapture
```

Clean:

```powershell
$env:CARGO_TARGET_DIR='B:\crytex-target-audit'
cargo clean
```
