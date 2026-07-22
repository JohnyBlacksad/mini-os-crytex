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
