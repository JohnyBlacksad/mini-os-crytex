# Crytex Release Notes

## 0.1.0

This release candidate turns the Crytex backend into a documented CLI product
surface with proof commands for the critical autonomous workflow.

Highlights:

- product CLI contract with JSON/stdout/stderr/exit-code rules;
- backend acceptance harness;
- RAG proof with mixed document/code fixtures, rerank evidence, and diagnostics;
- token economy proof with headroom, shared context, CCR, and quality checks;
- Kanban backend projection;
- role quality contracts and role examples;
- Prompt Evolution with challenger, benchmark, regression, promotion, rollback;
- LoRA datasets with positive and negative examples;
- LoRA objectives, adapter metadata, quality gates, runtime proof, rollback;
- autonomous evolution policy;
- runtime/model matrix;
- sandbox/security proof;
- storage/recovery proof;
- release gate proof with install docs, shell completions, versioned schemas,
  performance budgets, CI scripts, full acceptance fixtures, changelog, release
  notes, and Windows/Linux binary smoke scripts.

Known operational notes:

- the workspace binary is `crytex-kernel`; packaged releases should install it
  as `crytex`;
- runtime/network/CUDA tests are explicit release profiles, not mandatory fast
  unit tests;
- Ollama runtime LoRA remains unsupported unless a model is baked outside
  Crytex;
- ONNX currently represents embeddings/rerank capability, not generation.

Troubleshooting:

- start with `crytex doctor --strict --json`;
- use `crytex diag probe-runtime-matrix --json` for CUDA/runtime issues;
- use `crytex diag storage-recovery --json` for locks, migrations, partial
  downloads, or interrupted runs;
- use `crytex security prove --malicious-rag-fixture` for indirect prompt
  injection concerns.
