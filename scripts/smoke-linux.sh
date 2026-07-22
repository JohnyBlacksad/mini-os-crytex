#!/usr/bin/env bash
set -euo pipefail

BINARY="${BINARY:-./target/release/crytex-kernel}"
REPORT_PATH="${1:-reports/release-gate-p16-proof.json}"

test -x "$BINARY"
"$BINARY" --help >/dev/null
"$BINARY" doctor --strict --json >/dev/null
"$BINARY" prove-release-gate --report-path "$REPORT_PATH" >/dev/null
grep -q '"passed": true' "$REPORT_PATH"

echo "Crytex Linux binary smoke passed"
