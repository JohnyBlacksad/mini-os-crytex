# Crytex

Crytex is an autonomous, self-improving agentic CLI for project work.

It opens a project, builds a retrieval brain over every useful project artifact,
decomposes goals into a task graph, runs specialized agents, critiques their
outputs, creates remediation tasks, records experience, evolves prompts, trains
role-specific LoRA adapters, and promotes improvements only when benchmark
evidence proves that quality increased.

The backend is Rust-first and modular. Inference, storage, RAG, sandboxing,
benchmarking, prompt evolution, LoRA evolution, and CLI orchestration are
separate modules with trait boundaries so each part can be replaced, disabled,
or extended without collapsing the whole program.

## What Crytex Optimizes

- **Autonomous role improvement:** coders, analysts, QA agents, critics,
  security agents, and orchestrators learn separately.
- **Positive and negative learning:** accepted outputs teach what to do;
  rejected outputs and critic comments teach what not to repeat.
- **RAG as project memory:** code, documentation, PDFs, notes, specs, and other
  project knowledge become reranked context for every agent.
- **Kanban as backend truth:** tasks expose goal, owner role, dependency chain,
  status, review state, remediation feedback, and queue position.
- **Token economy:** context is compressed, cached, budgeted, and measured so
  the system spends tokens on useful evidence.
- **Proof before promotion:** prompts and LoRA adapters are promoted only after
  held-out benchmarks, regression checks, and safety gates pass.

## Install

Requirements:

- Rust toolchain matching the workspace `rust-version`.
- Windows, Linux, or macOS shell.
- Optional local model runtime such as Ollama.
- Optional CUDA toolchain for GPU Mistral/GGUF paths.

Build the CLI from the repository:

```powershell
cargo build -p crytex-kernel
```

The production CLI contract is documented as `crytex`. During local workspace
development the current binary target is `crytex-kernel`.

```powershell
cargo run -p crytex-kernel -- --help
```

## Quickstart

```powershell
crytex doctor --strict
crytex project open A:\Projects\my-app
crytex index run
crytex rag search "where is CSV import handled?" --rerank --explain --json
crytex prove token-economy --report-path reports\token-economy-p4.json
crytex goal submit "Add CSV import with validation, tests, and docs"
crytex plan show
crytex plan approve
crytex run start
crytex kanban watch
crytex kanban show --json
crytex diag export --run latest --out reports\latest-run.json
crytex backend-acceptance --full --json --deterministic --report-path reports\backend-acceptance.json
```

For the current development binary, replace `crytex` with:

```powershell
cargo run -p crytex-kernel -- 
```

## Architecture

```text
CLI
  -> core services
  -> project/index/RAG services
  -> agent roles
  -> tools and sandbox
  -> inference backends
  -> benchmark gates
  -> prompt and LoRA evolution
  -> storage and diagnostics
```

See [docs/MODULARITY.md](docs/MODULARITY.md) for the module map, trait
boundaries, capability statuses, disabled-module behavior, and SOLID audit.

Important crates:

- `crytex-core`: domain models, orchestration, services, task lifecycle,
  prompt/LoRA evolution, RAG assembly, metrics, events.
- `crytex-agents`: role implementations, prompts, parsers, tool loops.
- `crytex-doc`: document parsing, chunking, AST/code graph extraction.
- `crytex-compress`: token economy, compressors, CCR storage helpers.
- `crytex-storage`: SQLite and vector storage implementations.
- `crytex-inference-*`: local, cloud, ONNX, Candle, Ollama, and Mistral
  inference boundaries.
- `crytex-bench`: golden sets, scorers, benchmark harness, A/B gates.
- `crytex-sandbox` and `crytex-tools`: execution isolation and tool registry.
- `crytex-kernel`: CLI product contract and kernel command wiring.

## CLI Contract

See [docs/CLI.md](docs/CLI.md) for the full command reference, output rules,
exit codes, and examples.
See [docs/BACKEND_ACCEPTANCE.md](docs/BACKEND_ACCEPTANCE.md) for the canonical
backend acceptance harness, runtime modes, JSON artifact, and test profiles.
See [docs/RAG.md](docs/RAG.md) for the project-brain RAG pipeline, supported
formats, diagnostics, prompt-injection scanning, incremental reindex, and
crash-safe rebuild.
See [docs/KANBAN.md](docs/KANBAN.md) for the backend Kanban projection, canonical
statuses, task-card schema, movement diagnostics, watch stream, and history.
See [docs/TOKEN_ECONOMY.md](docs/TOKEN_ECONOMY.md) for headroom planning,
shared context, CCR artifact offload, token metrics, and quality-loss gates.
See [docs/ROLE_QUALITY.md](docs/ROLE_QUALITY.md) for per-role prompts,
artifact contracts, metrics, failure taxonomies, critic feedback, clean-session
handoff, and role-specific LoRA routing.
See [docs/PROMPT_EVOLUTION.md](docs/PROMPT_EVOLUTION.md) for prompt challenger
creation, benchmark-gated promotion, regression requirements, rollback,
diagnostics, and schema/format failure routing.

Every production command follows these rules:

- `--json` prints stable machine-readable output to stdout.
- Human progress, logs, and report paths go to stderr.
- Exit `0` means success.
- Exit `1` means command/config/input failure.
- Exit `2` means a proof, benchmark, doctor, or acceptance gate ran but failed.
- Exit `3` means unsupported capability.
- Exit `4` means interrupted or resumable work.

## Backend Status

The backend has strong foundations: project/task persistence, task lifecycle,
agents, RAG indexing, hybrid retrieval, reranking hooks, compression, benchmark
gates, prompt evolution, LoRA evolution, diagnostics, sandboxing, and multiple
inference backends.

Token economy now has a deterministic proof command in the development binary:

```powershell
cargo run -p crytex-kernel -- prove-token-economy --report-path reports\token-economy-p4.json
cargo run -p crytex-kernel -- prove-role-quality-contracts --report-path reports\role-quality-p6-proof.json
cargo run -p crytex-kernel -- prove-prompt-evolution --report-path reports\prompt-evolution-p7-proof.json
```

The token-economy report proves model headroom reservation, shared RAG-context reuse, CCR
offload for large artifacts, measured token savings, compression ratio, and
zero required-fact quality loss.

The role-quality report proves every production role has a system prompt,
output schema, artifact contract, metrics, failure taxonomy, benchmark fixture,
mocked smoke evidence, structured critic feedback, and role-specific LoRA
hot-swap handoff evidence.

The prompt-evolution report proves mutation creates an inactive challenger,
promotion is benchmark-gated, regression benchmark metadata is mandatory,
prompt decisions are written to diagnostics, rollback restores a previous
baseline, and schema/format failures route to Prompt Evolution before LoRA.

The production CLI contract is now fixed in code at
`crates/crytex-kernel/src/cli_contract.rs`. Existing legacy command handlers are
being migrated under the product command groups without changing the backend
service boundaries.

## Troubleshooting

- Run `crytex doctor --strict` first.
- If local model generation fails, run `crytex models prove <id>` or
  `crytex diag runtime-matrix`.
- If RAG misses context, run `crytex rag search ... --rerank --explain --json`
  and inspect selected chunks and rejection reasons.
- If a LoRA adapter does not improve quality, inspect
  `crytex lora benchmark <role> --include-negative`.
- If Windows blocks test artifacts, stop stale `cargo`, `rustc`, or test
  executable processes before cleaning the target directory.

## Development Discipline

Production changes use TDD. A backend feature is complete only when it has:

- unit tests for domain logic;
- integration tests across service boundaries;
- proof or benchmark artifacts for runtime claims;
- documented CLI behavior;
- typed errors and capability reports;
- no hidden dependency on a nonessential module.
