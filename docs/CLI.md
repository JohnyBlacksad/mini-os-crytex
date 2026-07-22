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
| Role | `orchestrator`, `architect`, `coder-python`, `coder-rust`, `coder-typescript`, `analyst`, `researcher`, `qa`, `security`, `critic`, `summarizer` |
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
```

Human output:

```text
Storage: ready
RAG: ready
Runtime: partial, CUDA compiler missing
LoRA Evolution: ready, no active benchmark corpus
```

JSON output includes module capability reports and blockers.

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
```

Current development binary:

```powershell
cargo run -p crytex-kernel -- prove-token-economy --backend ollama --model qwen3.5:9b --context-window 32768 --expected-completion-tokens 512 --report-path reports\token-economy-p4.json
```

The proof JSON contains model budget allocation, shared-context stats, CCR
markers for diff/log/report/tool-output artifacts, prompt/completion/saved token
metrics, compression ratio, and required-fact quality loss.

See [TOKEN_ECONOMY.md](TOKEN_ECONOMY.md) for the module contract.

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

Each task row includes id, title, goal, assigned role, task kind, dependency
chain, queue position, current status, critic feedback, and remediation link.

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
crytex diag runtime-matrix --model qwen
```

Diagnostics include runtime, task graph, Kanban transitions, RAG evidence,
prompts, LoRA adapters, tool calls, artifacts, benchmark results, and evolution
decisions.

## `models`

Manage model inventory and runtime activation.

```powershell
crytex models list --json
crytex models add qwen --repo owner/model --filename model.gguf --backend mistral
crytex models download qwen
crytex models activate qwen
crytex models prove qwen --json
```

Model proof reports generation evidence, compatibility strategy, CUDA/toolchain
state, and unsupported capability reasons.

## `prompts`

Manage role prompts and Prompt Evolution.

```powershell
crytex prompts status coder-python --json
crytex prompts propose coder-python
crytex prompts benchmark coder-python
crytex prompts promote coder-python <version-id>
crytex prompts rollback coder-python
```

A prompt challenger cannot become active without benchmark evidence.

## `lora`

Manage role-specific LoRA learning.

```powershell
crytex lora status coder-python
crytex lora dataset build coder-python --preference --json
crytex lora dataset inspect coder-python
crytex lora dataset stats coder-python
crytex lora train coder-python --objective preference
crytex lora benchmark coder-python --include-negative
crytex lora prove-live coder-python
crytex lora promote coder-python <adapter-id>
crytex lora rollback coder-python
```

Production LoRA promotion requires:

- adapter trained;
- adapter applied at runtime;
- output changed or backend diagnostics prove adapter use;
- positive benchmark improves;
- negative benchmark reduces bad patterns;
- regression benchmark passes;
- leakage and overfit checks pass.

## `evolution`

Run autonomous improvement policies.

```powershell
crytex evolution status --json
crytex evolution run --role coder-python
crytex evolution run --all-roles --dry-run
```

The policy decides whether a failure should improve RAG, prompts, LoRA, security
policy, benchmark coverage, or role contracts.

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
crytex sandbox doctor
crytex sandbox prove --json
```

Sandbox proof covers file, process, network, git, Docker, WASI, and host policy
boundaries when those modules are enabled.

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
