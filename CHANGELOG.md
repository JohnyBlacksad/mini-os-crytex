# Changelog

All notable Crytex backend CLI changes are recorded here.

## 0.1.0 - Release Candidate

### Added

- Root README and complete CLI documentation.
- Backend architecture, RAG, LoRA Evolution, Prompt Evolution, Token Economy,
  Modules/SOLID, Install, Release, and Release Notes docs.
- Role examples for Python coder, QA, critic, orchestrator, and analyst.
- Release gate service and CLI proof command.
- Versioned JSON schemas in `schemas/v1`.
- Shell completions for Bash, PowerShell, and Fish.
- Release performance budgets.
- Full acceptance fixture.
- Windows/Linux smoke scripts and CI release gate workflow.

### Verification

- `cargo fmt --check`
- `cargo test -p crytex-kernel --no-default-features --bin crytex-kernel`
- `cargo clippy -p crytex-kernel --no-default-features --bin crytex-kernel -- -D warnings`
- `cargo build --release -p crytex-kernel --no-default-features`
- `crytex doctor --strict --json`
- `crytex prove-release-gate`
