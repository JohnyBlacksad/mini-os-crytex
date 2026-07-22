#!/usr/bin/env bash
set -euo pipefail

REPORT_PATH="${1:-reports/release-gate-p16-proof.json}"

cargo fmt --check
cargo test -p crytex-kernel --no-default-features --bin crytex-kernel
cargo clippy -p crytex-kernel --no-default-features --bin crytex-kernel -- -D warnings
cargo build --release -p crytex-kernel --no-default-features
./target/release/crytex-kernel doctor --strict --json >/dev/null
cargo run -p crytex-kernel --no-default-features -- prove-release-gate --report-path "$REPORT_PATH" >/dev/null
./scripts/smoke-linux.sh "$REPORT_PATH"
