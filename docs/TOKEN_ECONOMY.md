# Crytex Token Economy

This document defines the P4 token-economy backend contract.

The implementation lives in `crates/crytex-compress/src/token_economy.rs`.
It is intentionally below the kernel layer: high-level CLI/proof code depends on
the public `crytex-compress` API, while the token-economy engine depends only on
`TokenEstimator` and `CcrStore` trait objects.

## Goals

- Reserve model-specific context headroom before agents build prompts.
- Share compressed RAG context between agents so the same evidence is not paid
  for repeatedly.
- Offload large artifacts through CCR: diffs, logs, reports, and tool outputs.
- Emit measurable token metrics: prompt tokens, completion tokens, saved tokens,
  compression ratio, and quality loss.
- Prove compression quality with required-fact retention, not only byte savings.

## Headroom Planning

`TokenBudgetPlanner` accepts `ModelTokenProfile` entries keyed by backend/model.
For each request it reserves:

- requested completion tokens;
- a safety margin for tool/inference framing and prompt-cache stability;
- remaining prompt budget split across RAG, artifacts, and shared context.

The planner returns a typed `TokenBudgetAllocation`. Missing model profiles and
exhausted windows are typed errors, not panics.

## Shared Context

`SharedContext` stores:

- original context for local retrieval;
- compressed context for downstream agents;
- producer agents that reused the same context key;
- saved-token and cache-hit stats.

This is the backend primitive for avoiding repeated RAG injection across
researcher -> architect -> coder -> QA -> critic chains.

## CCR Artifact Offload

`ArtifactOffload` stores large artifacts in a `CcrStore` and emits compact
markers containing:

- artifact kind;
- original token count;
- required fact preview when detectable;
- `ccr:<key>` retrieval handle.

Supported artifact kinds:

- `Diff`
- `Log`
- `Report`
- `ToolOutput`

The original artifact remains retrievable through the CCR store, so compression
does not destroy evidence.

## Quality Benchmark

`CompressionQualityBenchmark` checks required-fact retention after compression.
A report passes only when:

- no required fact is missing;
- compressed tokens do not exceed original tokens.

The benchmark emits `CompressionQualityReport` with `missing_facts`,
`compression_ratio`, and `quality_loss`.

## CLI Proof

Development binary:

```powershell
cargo run -p crytex-kernel -- prove-token-economy --report-path reports\token-economy-p4.json
```

Production contract:

```powershell
crytex prove token-economy --report-path reports\token-economy-p4.json
```

The proof runs deterministically and does not require Ollama, CUDA, vector DB,
or cloud credentials. It verifies:

- model headroom is reserved;
- shared context saves tokens and records reuse;
- four CCR markers are emitted for diff/log/report/tool-output;
- required facts survive compression with `quality_loss = 0`;
- token savings and compression ratio are measured.

Failure exits with code `2`. Command/config errors exit with code `1`.
