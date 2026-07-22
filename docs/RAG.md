# Crytex RAG Brain

Crytex RAG is the project brain used by agents before generation. It indexes
project code and attached human documents, records graph/security metadata, and
returns explainable context under a token budget.

## Supported Inputs

The parser registry supports code, Markdown, plain text, HTML, PDF, DOCX,
XLSX, CSV, JSON, YAML, TOML, and logs.

Every parsed chunk carries source path, relative path, language or format,
text, summary, AST symbol metadata for code, related symbols, and document
security findings.

## Pipeline

```text
parse -> chunk -> graph metadata -> dense -> sparse -> hybrid -> rerank -> token budget -> selected context
```

Dense retrieval uses the configured embedder and vector store. Sparse retrieval
uses BM25 when sparse indexing is enabled and the store supports sparse vectors.
Hybrid fusion uses Reciprocal Rank Fusion by default. Rerank is optional and is
plugged through the `Reranker` trait.

## Prompt Injection Scanning

Documents are treated as untrusted data. During parsing, Crytex scans document
text for indirect prompt-injection patterns such as attempts to override prior
instructions or reveal secrets. Findings are stored under `security_findings`
and surfaced in RAG diagnostics.

## CLI

```powershell
cargo run -p crytex-kernel -- rag search "where is retry policy documented?" --project-id my-project --path A:\Projects\my-app --rerank --explain --json --diagnostics-path reports\rag-search.json
cargo run -p crytex-kernel -- rag prove --fixture mixed-docs-code --report-path reports\rag-p3-proof.json
```

## Diagnostics

`rag search --explain --json` returns and optionally writes dense candidates,
sparse candidates, fused candidates, reranked candidates, selected chunks, and
per-candidate reasons.

`rag prove` writes a JSON artifact showing indexed formats, dense/sparse hits,
rerank evidence, selected chunks, prompt-injection scan evidence, and pass/fail
gates.

## Incremental And Crash Safety

`ProjectIndexer::incremental_reindex` applies changed files through
`index_file`, removed files through `remove_file`, and returns a typed
`IncrementalReindexReport`.

`RagPipeline::recover_rebuild` distinguishes active and staging manifests.
Interrupted staging manifests are discarded without touching the active index;
completed staging manifests can be promoted atomically.

## Operational CLI, Diagnostics, And Troubleshooting

Production CLI users operate RAG as the backend project brain:

```powershell
crytex index run --project my-app
crytex index status --project my-app --json
crytex index rebuild --project my-app
crytex rag search "where is retry policy documented?" --rerank --explain --json
crytex rag prove --fixture fixtures\mixed-docs-code --json
```

When running from the workspace before packaging, use `cargo run -p
crytex-kernel --` before the command. The development `rag search` command can
write diagnostics with `--diagnostics-path reports\rag-search.json`.

Diagnostics must answer three backend questions:

- which project files were parsed and chunked;
- which dense, sparse, fused, and reranked candidates were considered;
- why the final selected chunks fit the token budget and role goal.

Every selected chunk carries source path, chunk id, content type, graph metadata
when available, token estimate, and reason. This makes failures attributable:
bad context goes to RAG/indexing, malformed output goes to Prompt Evolution,
and repeated role skill errors go to LoRA.

Troubleshooting:

- Missing PDF, DOCX, XLSX, CSV, JSON, YAML, TOML, log, Markdown, HTML, or code
  content usually means the parser capability report is degraded.
- Weak rerank results should be debugged by comparing dense, sparse, fused, and
  reranked candidate lists in the JSON diagnostics.
- Prompt-injection findings mean project documents are untrusted data; run
  `crytex security prove --malicious-rag-fixture` to verify blocking behavior.
- Interrupted index rebuilds are handled by staging manifests and atomic swap;
  run `crytex diag storage-recovery --json` to prove recovery policy.

This document intentionally treats RAG as backend infrastructure, not UI search:
agents consume evidence ids, critics inspect evidence ids, and evolution policy
decides whether quality should improve by changing retrieval, prompts, or LoRA.
