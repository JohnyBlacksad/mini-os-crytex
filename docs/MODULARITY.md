# Crytex Modularity Audit

This document is the P1 backend modularity contract. Each optional module must
degrade through a typed `ModuleCapabilityReport`; disabling a module must not
crash the process.

Primary implementation:

- `crates/crytex-core/src/capabilities.rs`
- `crates/crytex-core/src/config.rs` (`ModuleSwitchesConfig`)
- `crates/crytex-kernel/src/factory.rs`
- `crates/crytex-kernel/src/crytex_cli.rs`
- `crates/crytex-kernel/src/crytex_cli_commands.rs`
- `crates/crytex-kernel/src/crytex_proof.rs`

## Module Map

| Module | Requires | Provides | Replacement boundary | Disable behavior |
| --- | --- | --- | --- | --- |
| Core | `Persistence`, `EventService` | `TaskService`, `ProjectService`, `Orchestrator` | object-safe service traits | required, reports `ready` |
| Storage | `StorageConfig` | `Persistence`, `VectorStore`, `MetricsRepository` | object-safe repository/vector traits | required, embedded fallback for vector store |
| Agents | `InferenceService`, `ToolService`, `ContextAssembler` | `Agent`, `AgentService`, `CriticCouncil` | object-safe agent traits | required, individual capabilities can degrade |
| Inference | `BackendConfig` | `InferenceManager`, `InferenceService` | object-safe backend manager/service | no backend/cloud/CUDA reports `degraded` |
| RAG | `Embedder`, `VectorStore` | `ProjectIndexer`, `HybridRetriever`, `ContextAssembler` | object-safe embedder/vector/reranker traits | `config.modules.rag=false` reports `disabled` |
| Token economy | `TokenEstimator`, `CcrStore` | `TokenBudgetPlanner`, `SharedContext`, `ArtifactOffload`, `CompressionQualityBenchmark`, `TokenEconomyEngine` | object-safe token/CCR traits | `config.modules.token_economy=false` reports `disabled` |
| LoRA | `LoraTrainer`, `LoraBenchmarkGate` | `LoraEvolutionService`, `LoraRouter` | object-safe trainer/router/evolution traits | `config.modules.lora=false` reports `disabled` |
| Prompt Evolution | `PromptVersionRepository`, `PromptBenchmarkGate` | `PromptEvolutionService` | object-safe benchmark/repository traits | `config.modules.prompt_evolution=false` reports `disabled` |
| Bench | `BenchmarkHarness`, `Scorer` | `BenchmarkRunner`, prompt/LoRA gates | object-safe runner/scorer traits | `config.modules.bench=false` reports `disabled` |
| Sandbox | `SandboxPolicy` | `SandboxService`, `ToolService` | object-safe sandbox service traits | Docker disabled reports `degraded`; whole sandbox disabled reports `disabled` |
| CLI | `CapabilityAuditReport`, command handlers | `ProductCli`, exit policy | static clap parser plus command modules | required, proof commands isolated |

## Typed Capability Status

`CapabilityStatus` has four states:

- `ready`: module is available.
- `degraded`: module is configured partially or a lower capability is disabled.
- `disabled`: module is intentionally disabled by config.
- `unavailable`: reserved for installed-but-unusable runtime checks.

Every report includes:

- module id;
- status;
- reason;
- required traits;
- provided traits;
- object-safe traits for dynamic replacement;
- config key that disabled or degraded the module.

## Disable Switches

All switches live under `[modules]` in config:

```toml
[modules]
rag = true
reranker = true
lora = true
prompt_evolution = true
bench = true
sandbox = true
sandbox_docker = true
cloud = true
cuda = false
external_vector_db = true
token_economy = true
```

Defaults keep production behavior enabled except CUDA, which must be opted in by
runtime configuration.

## Verified Disabled Scenarios

Covered by tests:

- no reranker: `disabled_reranker_degrades_rag_with_typed_report`
- no LoRA: `disabled_lora_returns_disabled_report_without_affecting_core`
- no cloud: `disabled_cloud_degrades_only_cloud_inference_backends`
- no sandbox Docker: `disabled_docker_sandbox_keeps_sandbox_degraded`
- no CUDA: `disabled_cuda_degrades_gpu_backend_without_disabling_inference`
- no external vector DB: `disabled_external_vector_db_degrades_rag_to_embedded_store`
- factory no reranker: `create_reranker_returns_none_when_module_disabled`
- factory no external vector DB: `select_vector_store_mode_uses_embedded_when_external_vector_db_disabled`
- token economy proof: `prove-token-economy` verifies headroom, shared context,
  CCR artifact offload, token metrics, and required-fact retention without
  requiring inference, sandbox, cloud, CUDA, or external vector DB.

## CLI Split

`crates/crytex-kernel/src/main.rs` no longer owns the CLI DSL.

- `crytex_cli.rs`: product CLI facade over the stable clap contract.
- `crytex_cli_commands.rs`: current legacy runtime command enum.
- `crytex_proof.rs`: proof-only command classifier.

The remaining `main.rs` responsibility is runtime composition and command
execution. Further P-level work can migrate handlers into command modules
without changing the command type boundary again.

## SOLID Notes

- Single Responsibility: capability reporting, config switches, vector-store
  selection, CLI command definitions, and proof classification are separate
  modules.
- Open/Closed: new modules extend `ModuleId`, `trait_boundary`, and
  `module_status` without changing service traits.
- Liskov: disabled modules return typed status/`None` optional dependencies
  instead of panicking or strengthening runtime preconditions.
- Interface Segregation: plugin points remain small object-safe traits such as
  `InferenceManager`, `VectorStore`, `Reranker`, `LoraTrainer`, `Scorer`, and
  `SandboxService`.
- Dependency Inversion: high-level factory code chooses traits and capability
  modes before constructing concrete low-level implementations.
