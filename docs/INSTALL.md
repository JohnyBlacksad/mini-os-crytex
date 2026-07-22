# Crytex Install

This document explains how to install the Crytex backend CLI for local use and
release validation. The workspace binary is currently `crytex-kernel`; released
packages install it as `crytex`.

## Build From Source

```powershell
cargo build --release -p crytex-kernel --no-default-features
```

Windows binary:

```text
target\release\crytex-kernel.exe
```

Linux binary:

```text
target/release/crytex-kernel
```

Copy or rename the binary to `crytex` on your PATH for production use.

## Preflight

Run doctor before giving the CLI to a user:

```powershell
crytex doctor --strict --json
```

During workspace development:

```powershell
target\release\crytex-kernel.exe doctor --strict --json
```

The doctor preflight checks release gate contract, storage recovery contract,
runtime matrix truth, config directories, and typed diagnostics. Failure exits
with code `2` when strict mode is enabled.

## Shell Completions

Shell completions are shipped as static release artifacts:

```text
completions/crytex.bash
completions/_crytex.ps1
completions/crytex.fish
```

Bash:

```bash
source completions/crytex.bash
```

PowerShell:

```powershell
. .\completions\_crytex.ps1
```

Fish:

```fish
cp completions/crytex.fish ~/.config/fish/completions/crytex.fish
```

## Optional Runtime Dependencies

Ollama:

```powershell
ollama serve
crytex models list --backend ollama --json
```

CUDA:

```powershell
crytex diag probe-runtime-matrix --json
```

ONNX:

Use ONNX for embeddings and rerank. Text generation and LoRA training are
unsupported unless the runtime matrix reports otherwise.

## Troubleshooting

- If the binary cannot start, run it with `--help` first.
- If `doctor --strict` fails on CUDA, inspect `diag probe-runtime-matrix`.
- If Windows reports a lock, run `crytex diag storage-recovery --json`.
- If a model download is partial, rerun the same download; it must resume and
  avoid registry promotion until validation succeeds.
