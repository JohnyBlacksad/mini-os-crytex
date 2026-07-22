# Crytex Backend Architecture

Crytex is a Rust backend for an autonomous CLI that can open a project, build a
retrieval brain, decompose a goal, run role-specific agents, critique results,
create remediation, collect rewards, evolve prompts, train LoRA adapters, and
prove improvements with diagnostics before promotion.

The CLI is the product surface. UI can visualize state later, but backend truth
lives in storage, events, diagnostics, Kanban projection, benchmark reports, and
proof artifacts.

## Runtime Flow

```text
crytex command
  -> CLI parser and typed value enums
  -> command handler
  -> AppContext service graph
  -> project/task/storage services
  -> RAG/context assembly
  -> role workflow and tool sandbox
  -> inference backend
  -> critic/review/remediation
  -> reward, Prompt Evolution, LoRA Evolution
  -> diagnostics/proof artifact
```

Important backend commands:

```powershell
crytex doctor --strict --json
crytex project open A:\Projects\my-app
crytex index run --project my-app
crytex rag search "where is API auth documented?" --rerank --explain --json
crytex goal submit "Add CSV import with tests"
crytex plan show --json
crytex run start --json
crytex kanban show --json
crytex diag export --run latest --out reports\latest-run.json
crytex backend-acceptance --full --json --deterministic --report-path reports\backend-acceptance.json
```

## Service Boundaries

`crytex-core` owns domain models and service traits. It defines projects, tasks,
training examples, adapters, prompt versions, audit logs, workflow state,
capability reports, and recovery policies.

`crytex-storage` implements persistence through SQLite and vector stores. The
core depends on repository traits, not on concrete database code. SQLite schema
versioning is exposed by `PRAGMA user_version`; old fixtures are migrated
forward and tested.

`crytex-doc` parses project documents and code into chunks and graph metadata.
It is replaceable behind parser/chunker boundaries.

`crytex-compress` owns token economy: model budgets, shared context,
compression, CCR references, cache alignment, and quality-preservation
benchmarks.

`crytex-agents` owns role implementations and prompt assets. Roles are separate
quality contracts with output schemas and artifact obligations.

`crytex-inference-*` crates implement runtime adapters for Ollama, Mistral/GGUF,
ONNX, OpenAI-compatible providers, Anthropic, and local Candle paths. Each
backend reports capabilities instead of pretending full support.

`crytex-bench` provides golden sets, scorers, benchmark runs, A/B comparisons,
Prompt Evolution gates, and LoRA quality gates.

`crytex-sandbox` and `crytex-tools` isolate command execution and file/process/
network/git/search permissions. Tool calls are audited.

`crytex-kernel` wires the CLI to the service graph. It must remain thin: parse,
validate, call a service, emit output, and exit with the documented code.

## State And Recovery

Crytex persists durable state for projects, tasks, dependencies, artifacts,
logs, experiences, training examples, prompt versions, LoRA adapters, training
jobs, memory entries, benchmark runs, and diagnostics.

Recovery policy:

- migrations are versioned and ordered;
- backup/export/import happen before risky changes;
- interrupted `InProgress` tasks resume as `Ready`;
- review and remediation states remain visible;
- interrupted training resumes only when adapter artifacts are valid;
- partial model downloads are not registered as complete;
- RAG index rebuilds use staging and atomic swap;
- Windows writer commands use an exclusive lock policy;
- corrupt adapters cannot be promoted.

Proof command:

```powershell
crytex diag storage-recovery --json --report-path reports\storage-recovery-p14-proof.json
```

## Diagnostics

Every serious backend decision must leave evidence:

- RAG selected context and rejected candidates;
- prompt version and prompt decision;
- LoRA dataset hash, objective, adapter metadata, quality gates, rollback;
- tool calls and sandbox result;
- task movement and Kanban history;
- runtime/model capability report;
- token budget and compression metrics;
- recovery and lock decisions.

Machine-readable diagnostics go to stdout under `--json` or to a report path.
Human progress goes to stderr. Exit `2` means a proof or gate executed and
failed; exit `3` means unsupported capability; exit `4` means interrupted or
resumable work.

## Troubleshooting

- Ollama: run `crytex models list --backend ollama --json` and verify the
  daemon URL before runtime proofs.
- CUDA: run `crytex doctor --strict --json` and
  `crytex diag probe-runtime-matrix --json`; missing `nvidia-smi`, `nvcc`, or
  MSVC compiler is a typed partial capability.
- ONNX: use it for embeddings/rerank; text generation and LoRA are unsupported
  unless a backend reports otherwise.
- Windows locks: run `crytex diag storage-recovery --json`; stale lock files
  should be diagnostics, not panics.
- Model download: partial files resume and are not registered until validation
  succeeds.

## References

- [SQLite user_version](https://www.sqlite.org/pragma.html#pragma_user_version)
- [SQLite backup API](https://www.sqlite.org/backup.html)
- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [Docker Engine security](https://docs.docker.com/engine/security/)
