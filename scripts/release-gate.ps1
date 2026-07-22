param(
    [string]$ReportPath = "reports\release-gate-p16-proof.json"
)

$ErrorActionPreference = "Stop"

cargo fmt --check
cargo test -p crytex-kernel --no-default-features --bin crytex-kernel
cargo clippy -p crytex-kernel --no-default-features --bin crytex-kernel -- -D warnings
cargo build --release -p crytex-kernel --no-default-features
.\target\release\crytex-kernel.exe doctor --strict --json | Out-Null
cargo run -p crytex-kernel --no-default-features -- prove-release-gate --report-path $ReportPath | Out-Null
.\scripts\smoke-windows.ps1 -ReportPath $ReportPath
