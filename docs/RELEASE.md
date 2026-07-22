# Crytex Release Gate

P16 defines the release gate required before Crytex can be handed to CLI users
as a production-ready backend tool.

## Required Commands

```powershell
cargo fmt --check
cargo test -p crytex-core --no-default-features
cargo test -p crytex-storage --no-default-features
cargo test -p crytex-kernel --no-default-features --bin crytex-kernel
cargo clippy -p crytex-core --no-default-features -- -D warnings
cargo clippy -p crytex-storage --no-default-features -- -D warnings
cargo clippy -p crytex-kernel --no-default-features --bin crytex-kernel -- -D warnings
cargo build --release -p crytex-kernel --no-default-features
target\release\crytex-kernel.exe doctor --strict --json
cargo run -p crytex-kernel --no-default-features -- prove-release-gate --report-path reports\release-gate-p16-proof.json
```

Linux equivalent:

```bash
cargo build --release -p crytex-kernel --no-default-features
./target/release/crytex-kernel doctor --strict --json
./scripts/smoke-linux.sh
```

Windows equivalent:

```powershell
.\scripts\smoke-windows.ps1
```

## Release Artifacts

- release build: `target/release/crytex-kernel(.exe)`;
- install docs: `docs/INSTALL.md`;
- shell completions: `completions/crytex.bash`, `completions/_crytex.ps1`,
  `completions/crytex.fish`;
- JSON schemas: `schemas/v1/*.schema.json`;
- performance budgets: `release/performance-budgets.json`;
- CI scripts: `.github/workflows/release-gate.yml`, `scripts/release-gate.*`;
- full acceptance fixtures: `fixtures/full-acceptance/project.json`;
- changelog and release notes: `CHANGELOG.md`, `docs/RELEASE_NOTES.md`;
- proof report: `reports/release-gate-p16-proof.json`.

## JSON schemas

Release JSON schemas use JSON Schema draft 2020-12 and are versioned under
`schemas/v1`. Additive fields are allowed inside `metadata` or `evidence`
objects. Breaking output changes require a new schema version directory.

## performance budgets

Performance budgets are documented in `release/performance-budgets.json`.
Budgets cover startup help, `crytex doctor --strict`, deterministic backend
acceptance, RAG proof, token economy proof, release gate proof, and binary smoke
commands. Runtime/network/CUDA tests are separate because model download and GPU
availability are machine dependent.

## doctor strict preflight

`crytex doctor --strict` is the final local preflight. It must run before
packaging and before publishing a release note. Strict failure means the release
candidate is not shippable.

## Windows And Linux Smoke

Binary smoke validates:

- binary exists and is executable;
- `--help` works;
- `doctor --strict --json` works;
- `prove-release-gate` writes a JSON report;
- release report has `passed: true`.

## Troubleshooting

- Ollama: verify daemon URL and model name before runtime acceptance.
- CUDA: check `nvidia-smi`, `nvcc`, MSVC on Windows, and runtime matrix output.
- ONNX: treat as embeddings/rerank unless future capability report says more.
- Windows locks: rerun after stale process cleanup; lock policy is documented by
  `diag storage-recovery`.
- Model download: partial files must resume and must not become active registry
  entries until validation completes.
