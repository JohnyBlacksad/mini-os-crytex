# Crytex CLI Reference

This document defines the production CLI contract for Crytex.

The current development binary target is `crytex-kernel`; the production command
name is `crytex`. Examples use `crytex`.

## Global Rules

All commands accept these global flags:

```text
--json              Emit stable JSON to stdout
--trace-id <id>     Correlate command output, traces, diagnostics, and reports
--project <id|path> Project id or path for project-scoped commands
```

Output rules:

- stdout is for primary command output.
- `--json` output must be parseable JSON and must not contain progress logs.
- stderr is for progress, warnings, diagnostics paths, and human logs.
- proof reports may be written with `--report-path` or `--out`.

Exit codes:

| Code | Meaning |
| --- | --- |
| 0 | Success |
| 1 | Command, input, config, parse, or storage error |
| 2 | Doctor, proof, benchmark, or acceptance gate ran and failed |
| 3 | Unsupported backend/module/capability |
| 4 | Interrupted or resumable operation |

Typed value domains:

| Domain | Values |
| --- | --- |
| Role | `orchestrator`, `architect`, `coder-python`, `coder-rust`, `coder-ts`, `coder-etc`, `analyst`, `researcher`, `qa`, `devops`, `security`, `critic-analyst`, `critic-coder`, `critic-researcher`, `critic-etc`, `summarizer` |
| Backend | `ollama`, `mistral`, `onnx`, `open-ai-compatible`, `anthropic`, `custom` |
| Training objective | `sft`, `preference`, `dpo`, `orpo`, `kto` |
| Task status | `backlog`, `ready`, `in-progress`, `review`, `remediation`, `done`, `failed`, `blocked` |
| Failure type | `missing-tests`, `unsafe-code`, `wrong-api`, `hallucinated-file`, `weak-analysis`, `incomplete-critique`, `bad-decomposition`, `prompt-injection`, `context-miss`, `token-budget-exceeded` |

## Command Surface

```text
crytex doctor
crytex project
crytex index
crytex rag
crytex goal
crytex plan
crytex kanban
crytex run
crytex review
crytex diag
crytex models
crytex security
crytex prompts
crytex lora
crytex evolution
crytex bench
crytex sandbox
crytex backend-acceptance
crytex prove
```

Proof-only commands live under `crytex prove ...` so the everyday CLI remains
focused on product workflows.

## `doctor`

Validate config, storage, vector stores, inference backends, model compatibility,
RAG, sandbox, token economy, Prompt Evolution, and LoRA Evolution readiness.

```powershell
crytex doctor --strict --json
crytex diag probe-runtime-matrix --json
```

Human output:

```text
Storage: ready
RAG: ready
Runtime: partial, CUDA compiler missing
LoRA Evolution: ready, no active benchmark corpus
```

JSON output includes module capability reports and blockers. CUDA/toolchain
preflight is part of doctor diagnostics: GPU-required runs fail typed checks
when `nvidia-smi` is unavailable, while optional GPU mode reports warnings and
CPU fallback instead of crashing.

## `project`

Manage projects.

```powershell
crytex project open A:\Projects\my-app
crytex project create --name my-app --path A:\Projects\my-app
crytex project list --json
crytex project status --project my-app
crytex project reopen my-app
```

Project status reports watcher state, index state, active backend, active model,
task counts, Kanban summary, and last diagnostics path.

## `index`

Build and maintain project knowledge.

```powershell
crytex index run --project my-app
crytex index status --project my-app --json
crytex index rebuild --project my-app
```

Indexing covers code, Markdown, plain text, HTML, PDFs, office documents, data
files, logs, and project notes when the corresponding parser module is enabled.

## `rag`

Search and prove retrieval quality.

```powershell
crytex rag search "where is CSV import handled?" --rerank --explain --json
crytex rag prove --fixture fixtures\mixed-docs-code --json
```

Current development binary:

```powershell
cargo run -p crytex-kernel -- rag search "where is retry policy documented?" --project-id my-project --path A:\Projects\my-app --rerank --explain --json --diagnostics-path reports\rag-search.json
cargo run -p crytex-kernel -- rag prove --fixture mixed-docs-code --report-path reports\rag-p3-proof.json
```

JSON output contains:

```json
{
  "dense_candidates": [],
  "sparse_candidates": [],
  "fused_candidates": [],
  "reranked_candidates": [],
  "selected_chunks": [],
  "selection_reasons": []
}
```

RAG is considered healthy only when agents receive selected context with source
evidence and token-budget decisions.

See [RAG.md](RAG.md) for parser coverage, diagnostics schema,
prompt-injection scanning, incremental reindex, and crash-safe rebuild behavior.

## `token-economy`

Plan headroom, share context across agents, offload large artifacts through CCR,
and prove compression does not drop required facts.

Production contract:

```powershell
crytex token-economy plan --backend ollama --model qwen3.5:9b --prompt-tokens 2000 --completion-tokens 512 --json
crytex token-economy shared-context stats --project my-app --json
crytex prove token-economy --report-path reports\token-economy-p4.json
crytex prove role-quality-contracts --report-path reports\role-quality-p6-proof.json
```

Current development binary:

```powershell
cargo run -p crytex-kernel -- prove-token-economy --backend ollama --model qwen3.5:9b --context-window 32768 --expected-completion-tokens 512 --report-path reports\token-economy-p4.json
```

The proof JSON contains model budget allocation, shared-context stats, CCR
markers for diff/log/report/tool-output artifacts, prompt/completion/saved token
metrics, compression ratio, and required-fact quality loss.

See [TOKEN_ECONOMY.md](TOKEN_ECONOMY.md) for the module contract.

## `role-quality`

Inspect and prove per-role quality contracts.

Production contract:

```powershell
crytex role-quality list --json
crytex role-quality show --role coder-python --json
crytex prove role-quality-contracts --report-path reports\role-quality-p6-proof.json
```

Current development binary:

```powershell
cargo run -p crytex-kernel -- prove-role-quality-contracts --report-path reports\role-quality-p6-proof.json
```

The proof JSON contains all production role contracts, system prompt source
evidence from `trash/agent-skills-main` and `trash/skills-main`, output schemas,
artifact fields, metrics, failure taxonomies, deterministic role benchmark
fixtures, mocked smoke status per role, structured critic feedback schema, and
role-specific LoRA hot-swap evidence.

See [ROLE_QUALITY.md](ROLE_QUALITY.md) for the module contract.

## `goal`

Submit user goals.

```powershell
crytex goal submit "Add CSV import with validation, tests, and docs"
crytex goal status --json
crytex goal list
```

Goals are decomposed by the orchestrator/architect into task graphs.

## `plan`

Inspect and decide generated plans.

```powershell
crytex plan show
crytex plan approve
crytex plan reject --comment "Split QA and security into separate tasks"
```

Plan output includes task goals, assigned roles, dependency edges, acceptance
criteria, and expected artifacts.

## `kanban`

Show canonical backend task-state projection.

```powershell
crytex kanban show --status in-progress --role coder-python --json
crytex kanban watch --status remediation
crytex kanban history --run latest --status done
```

Current development binary:

```powershell
cargo run -p crytex-kernel -- kanban show --project-id my-project --json
cargo run -p crytex-kernel -- kanban watch --project-id my-project --json --duration-seconds 30
cargo run -p crytex-kernel -- kanban history --project-id my-project --run latest --json
cargo run -p crytex-kernel -- prove-kanban-projection --report-path reports\kanban-p5-proof.json
```

Each task row includes id, title, goal, assigned role, task kind, dependency
chain, queue position, current status, critic feedback, and remediation link.

Kanban statuses are fixed as `backlog`, `ready`, `in_progress`, `review`,
`remediation`, `done`, `failed`, and `blocked`. Legacy persisted `pending`,
`completed`, and `cancelled` statuses are projected as `ready`, `done`, and
`blocked`.

See [KANBAN.md](KANBAN.md) for the projection schema, transition diagnostics,
and CLI watch/history behavior.

## `run`

Execute autonomous work.

```powershell
crytex run start
crytex run status --json
crytex run resume
crytex run cancel <run-id>
```

Every task execution records RAG evidence, prompt version, LoRA adapter,
inference backend, tool calls, artifacts, tests, critic decision, and reward.

## `review`

Inspect and record review decisions. In autonomous mode, policy and critic gates
may make review decisions without a human.

```powershell
crytex review show
crytex review approve <task-id> --score 4.8
crytex review reject <task-id> --failure-type missing-tests --comment "Missing edge-case tests"
```

Rejected outputs become negative examples. Accepted remediation output can form a
chosen/rejected preference pair for LoRA training.

## `diag`

Export diagnostics and runtime reports.

```powershell
crytex diag export --run latest --out reports\latest.json
crytex diag probe-runtime-matrix --json --report-path reports\runtime-model-matrix-p12-proof.json
crytex diag storage-recovery --json --report-path reports\storage-recovery-p14-proof.json
```

Diagnostics include runtime, task graph, Kanban transitions, RAG evidence,
prompts, LoRA adapters, tool calls, artifacts, benchmark results, and evolution
decisions. Storage recovery diagnostics additionally prove schema migration,
backup/export/import, interrupted run/training/download resume, index rebuild,
Windows CLI locking, and corrupt-adapter rejection policy.

See [STORAGE_RECOVERY.md](STORAGE_RECOVERY.md).

## `models`

Manage model inventory and runtime activation.

```powershell
crytex models list --json
crytex models add --id qwen --repo owner/model --filename model.gguf --backend mistralrs
crytex models download --id qwen --activate --backend-id local
crytex models activate --id qwen --backend-id local
crytex models prove --id qwen --backend local --report-path reports\model-runtime.json
```

Model proof reports generation evidence, compatibility strategy, CUDA/toolchain
state, and unsupported capability reasons. Runtime matrix diagnostics report
every backend as `supported`, `partial`, or `unsupported` with reasons:

- Ollama: generation/chat/embeddings/model listing/download supported; runtime LoRA unsupported.
- Mistral GGUF CPU/CUDA: generation/chat/model listing/download and Crytex LoRA training/application/hot-swap supported; embeddings/rerank delegated.
- ONNX: embeddings/rerank supported; text generation and LoRA unsupported.
- OpenAI-compatible: chat/generation/embeddings/model listing supported when the provider exposes compatible endpoints; LoRA/download/CUDA unsupported in Crytex.
- Anthropic: chat/generation supported through configured model id; embeddings/rerank/LoRA/download/CUDA unsupported.

`trash/crytex-inference-trtllm` is kept as a future optional module until it is
moved into `crates/` with CI and toolchain probes.

See [RUNTIME_MODEL_MATRIX.md](RUNTIME_MODEL_MATRIX.md) for the support contract
and references.

## `prompts`

Manage role prompts and Prompt Evolution.

```powershell
crytex prompts status --agent coder-python --json
crytex prompts propose --agent coder-python --operator inject-example --json
crytex prompts benchmark --agent coder-python --challenger <version-id> --regression-suite fixtures\prompt-regression.jsonl --json
crytex prompts promote --agent coder-python --version <version-id> --json
crytex prompts rollback --agent coder-python --to <version-id> --json
crytex prove prompt-evolution --report-path reports\prompt-evolution-p7-proof.json
```

A prompt mutation creates an inactive challenger, never a new active prompt.
Promotion requires an accepted benchmark decision and a passing regression suite.
Schema and format failures are routed to Prompt Evolution before LoRA because
they usually mean the role prompt or output contract is wrong, not that the
adapter needs new weights.

Current development binary:

```powershell
cargo run -p crytex-kernel -- prompts status --agent coder-python --json
cargo run -p crytex-kernel -- prompts propose --agent coder-python --operator inject-example --json
cargo run -p crytex-kernel -- prompts benchmark --agent coder-python --challenger <version-id> --regression-suite fixtures\prompt-regression.jsonl --json
cargo run -p crytex-kernel -- prompts promote --agent coder-python --version <version-id> --json
cargo run -p crytex-kernel -- prompts rollback --agent coder-python --to <version-id> --json
cargo run -p crytex-kernel -- prove-prompt-evolution --report-path reports\prompt-evolution-p7-proof.json
```

JSON decision output includes `decision_kind`, `accepted`, `reason`,
`baseline_score`, `challenger_score`, `regression_passed`, and `diagnostics`.

See [PROMPT_EVOLUTION.md](PROMPT_EVOLUTION.md) for the module contract.

## `lora`

Manage role-specific LoRA learning.

```powershell
crytex lora status coder-python
crytex lora dataset build coder-python --preference --json
crytex lora dataset inspect coder-python
crytex lora dataset stats coder-python
crytex prove lora-dataset --report-path reports\lora-dataset-p8-proof.json
crytex lora train coder-python --objective sft
crytex lora train coder-python --objective dpo --role coder-python
crytex lora train coder-python --objective orpo --role coder-python
crytex lora train coder-python --objective kto --role coder-python
crytex prove lora-training-objectives --report-path reports\lora-training-objectives-p9-proof.json
crytex prove lora-quality-gate --report-path reports\lora-quality-gate-p10-proof.json
crytex lora benchmark coder-python --include-negative
crytex lora prove-live coder-python
crytex lora promote coder-python <adapter-id>
crytex lora rollback coder-python
```

Current development binary:

```powershell
cargo run -p crytex-kernel -- lora dataset build coder-python --preference --json
cargo run -p crytex-kernel -- lora dataset inspect coder-python --json
cargo run -p crytex-kernel -- lora dataset stats coder-python --json
cargo run -p crytex-kernel -- lora train coder-python --objective dpo --role coder-python
cargo run -p crytex-kernel -- prove-lora-dataset --report-path reports\lora-dataset-p8-proof.json
cargo run -p crytex-kernel -- prove-lora-training-objectives --report-path reports\lora-training-objectives-p9-proof.json
cargo run -p crytex-kernel -- prove-lora-quality-gate --report-path reports\lora-quality-gate-p10-proof.json
```

LoRA dataset rows preserve role, task kind, prompt version, model id, RAG
evidence ids, accepted output, rejected output, critic feedback, failure type,
and reward. Rejected output is stored only as the negative preference side; it
is never used as an SFT target.

Dataset reports include role scoping, chosen/rejected pair counts, failure-type
balancing targets, leakage diagnostics, and low-information filtering.

See [LORA_DATASET.md](LORA_DATASET.md) and
[LORA_TRAINING_OBJECTIVES.md](LORA_TRAINING_OBJECTIVES.md), and
[LORA_QUALITY_GATE.md](LORA_QUALITY_GATE.md) for the module contracts.

Production LoRA promotion requires:

- adapter trained;
- adapter applied at runtime;
- output changed or backend diagnostics prove adapter use;
- positive benchmark improves;
- negative benchmark reduces bad patterns;
- regression benchmark passes;
- leakage and overfit checks pass.

## `evolution`

Run autonomous failure attribution across roles and choose what to improve.

```powershell
crytex evolution run --all-roles --json
crytex prove evolution-policy --report-path reports\evolution-policy-p11-proof.json
```

Current development binary:

```powershell
cargo run -p crytex-kernel -- evolution run --all-roles --json
cargo run -p crytex-kernel -- prove-evolution-policy --report-path reports\evolution-policy-p11-proof.json
```

The policy chooses among RAG fixes, prompt evolution, LoRA training, critic role
evolution, security policy, and benchmark expansion. See
[EVOLUTION_POLICY.md](EVOLUTION_POLICY.md).

## `bench`

Run and compare benchmark suites.

```powershell
crytex bench run codegen-python fixtures\codegen-python.jsonl --role coder-python
crytex bench compare baseline-run challenger-run
crytex bench show <run-id> --json
```

Benchmark results are used by Prompt Evolution and LoRA Evolution gates.

## `sandbox`

Validate execution isolation.

```powershell
crytex sandbox doctor --json
crytex sandbox prove --json --report-path reports\sandbox-security-p13-proof.json
crytex security prove --malicious-rag-fixture --json --report-path reports\security-p13-proof.json
```

Sandbox proof covers file, process, network, git, search, path traversal,
malicious RAG prompt injection, Docker/WASI/host policy states, audited tool
calls, and security-failure negative examples for the relevant role.

See [SANDBOX_SECURITY.md](SANDBOX_SECURITY.md).

## `backend-acceptance`

Run the canonical backend product acceptance suite.

```powershell
crytex backend-acceptance --full --json --deterministic --report-path reports\backend-acceptance.json
crytex backend-acceptance --full --json --runtime ollama --live-model qwen3.5:9b --live-url http://localhost:11434
crytex backend-acceptance --full --json --runtime mistral --live-model A:\models\model.gguf
```

Acceptance proves the whole backend path:

```text
doctor -> project -> index -> RAG/rerank -> goal -> plan -> Kanban
-> run -> critic -> remediation -> review -> reward
-> Prompt Evolution evidence -> LoRA dataset evidence -> diagnostics
```

The command emits one JSON proof artifact with ordered stages, nested kernel
proof, diagnostics path, benchmark evidence, prompt evolution evidence, and LoRA
evolution evidence.

## `prove`

Explicit proof-only commands:

```powershell
crytex prove kernel-e2e --full
crytex prove hf-model <id> --repo owner/model
crytex prove hf-runtime-matrix
crytex prove rag-full
crytex prove kanban-projection
crytex prove token-economy
crytex prove orchestrator-quality
crytex prove agent-swarm-lora-routing
crytex prove lora-live-e2e --role coder-python
crytex prove lora-evolution-loop --role coder-python
crytex prove lora-hot-swap
crytex prove lora-candle-learning
crytex prove lora-real-model
crytex prove lora-real-quality-gate
```

Proof commands are allowed to be expensive and may download models or run CUDA
smokes when explicitly requested.

## Complete Help Snapshot Paths

The product contract renders help for every path below:

```text
crytex
crytex doctor
crytex project
crytex project open
crytex project create
crytex project list
crytex project status
crytex project reopen
crytex index
crytex index run
crytex index status
crytex index rebuild
crytex rag
crytex rag search
crytex rag prove
crytex token-economy
crytex token-economy plan
crytex token-economy shared-context
crytex token-economy shared-context stats
crytex goal
crytex goal submit
crytex goal status
crytex goal list
crytex plan
crytex plan show
crytex plan approve
crytex plan reject
crytex kanban
crytex kanban show
crytex kanban watch
crytex kanban history
crytex run
crytex run start
crytex run status
crytex run resume
crytex run cancel
crytex review
crytex review show
crytex review approve
crytex review reject
crytex diag
crytex diag export
crytex diag runtime-matrix
crytex models
crytex models list
crytex models add
crytex models download
crytex models activate
crytex models prove
crytex prompts
crytex prompts status
crytex prompts propose
crytex prompts benchmark
crytex prompts promote
crytex prompts rollback
crytex lora
crytex lora status
crytex lora dataset
crytex lora dataset build
crytex lora dataset inspect
crytex lora dataset stats
crytex lora train
crytex lora benchmark
crytex lora promote
crytex lora rollback
crytex lora prove-live
crytex evolution
crytex evolution status
crytex evolution run
crytex bench
crytex bench run
crytex bench compare
crytex bench show
crytex sandbox
crytex sandbox doctor
crytex sandbox prove
crytex backend-acceptance
crytex prove
crytex prove kernel-e2e
crytex prove hf-model
crytex prove hf-runtime-matrix
crytex prove rag-full
crytex prove kanban-projection
crytex prove token-economy
crytex prove orchestrator-quality
crytex prove agent-swarm-lora-routing
crytex prove lora-live-e2e
crytex prove lora-evolution-loop
crytex prove lora-hot-swap
crytex prove lora-candle-learning
crytex prove lora-real-model
crytex prove lora-real-quality-gate
```

## SOLID Extension Rules

Every subsystem must obey these CLI-facing rules:

- A disabled module returns a typed capability report, not a process crash.
- A new backend is added behind an inference trait.
- A new document format is added behind a parser/chunker trait.
- A new role is added by role contract, prompt, schema, benchmark, and optional
  LoRA adapter policy.
- A new benchmark scorer is added behind the scorer trait.
- A new sandbox backend is added behind the sandbox backend trait.
- A new evolution strategy is added behind the evolution policy boundary.

The CLI must never depend on a concrete low-level implementation when a trait
boundary exists.
