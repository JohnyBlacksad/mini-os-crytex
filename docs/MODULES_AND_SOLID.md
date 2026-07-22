# Crytex Modules And SOLID Contract

Crytex backend modules must be replaceable. A disabled or unsupported subsystem
returns typed capability status, not a crash. A new implementation is added
behind a trait or small service boundary, not by hard-coding another special
case in the CLI.

## Module Map

| Module | Provides | Requires |
| --- | --- | --- |
| CLI | typed command surface, JSON/human output, exit codes | service traits |
| Core | models, service traits, policies, orchestration | repositories, inference, tools |
| Storage | SQLite repositories, vector store implementations | core repository traits |
| RAG | parse, chunk, graph metadata, hybrid search, rerank evidence | doc parsers, embedder, vector/sparse stores |
| Agents | role prompts, tool loops, artifact contracts | context, inference, tools |
| Inference | generation, chat, embeddings, runtime LoRA where supported | backend config |
| Token Economy | budget planning, compression, CCR, shared context | tokenizer/model profile |
| LoRA | dataset build, training, gates, promotion, rollback | training examples, benchmark results |
| Prompt Evolution | prompt proposals, benchmark, promotion, rollback | prompt repository, benchmark gate |
| Bench | golden sets, scorers, A/B gates | inference or deterministic runner |
| Sandbox | isolated execution and permission policy | backend runner |
| Diagnostics | proof artifacts, audit logs, reports | events and repositories |

## SOLID Rules

Single Responsibility: each service has one reason to change. `RecoveryService`
decides recovery proof; `GraphStore` persists SQLite state; CLI handlers parse
and dispatch.

Open/Closed: new inference runtimes, parsers, scorers, compressors, sandbox
backends, and evolution policies are added as new implementations. Existing
command contracts remain stable.

Liskov Substitution: every trait implementation must honor the same error
contract. Unsupported capabilities return typed unsupported status; they do not
panic or silently fake success.

Interface Segregation: small traits are preferred: repository traits,
`ModelDownloader`, `ModelRegistryStore`, `LoraTrainer`, `VectorStore`,
`Reranker`, `ToolService`, and sandbox services remain focused.

Dependency Inversion: high-level orchestration depends on abstractions.
`AppContext` receives trait objects or service abstractions; concrete SQLite,
Hugging Face, Ollama, Docker, or ONNX code stays at the edge.

## CLI And Diagnostics

```powershell
crytex doctor --strict --json
crytex diag probe-runtime-matrix --json
crytex diag storage-recovery --json
crytex backend-acceptance --full --json --deterministic
```

These commands prove modules are discoverable, typed, and degradable. A disabled
reranker, disabled LoRA, missing cloud key, missing Docker daemon, missing CUDA,
or absent external vector DB should produce a capability report rather than a
process failure.

## Extension Checklist

When adding a module or backend:

- define the trait boundary first;
- add a deterministic test for disabled/unsupported behavior;
- add integration or proof coverage for the happy path;
- document CLI output and diagnostics;
- avoid leaking concrete implementation types into high-level services;
- make failure recoverable when possible;
- preserve JSON compatibility for existing CLI consumers.

## Troubleshooting

- If a module import causes startup failure when disabled, move the dependency
  behind a feature flag or factory boundary.
- If CLI code directly touches SQLite, vector DB, Docker, or a cloud SDK, push
  that access into a service implementation.
- If tests require network/CUDA by default, split them into explicit runtime or
  ignored profiles.
- If a proof cannot explain why it passed, add typed gates and evidence fields.

See also [ARCHITECTURE_BACKEND.md](ARCHITECTURE_BACKEND.md),
[RAG.md](RAG.md), [TOKEN_ECONOMY.md](TOKEN_ECONOMY.md),
[PROMPT_EVOLUTION.md](PROMPT_EVOLUTION.md), and [LORA_EVOLUTION.md](LORA_EVOLUTION.md).
