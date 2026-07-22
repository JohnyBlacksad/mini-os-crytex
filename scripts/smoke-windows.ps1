param(
    [string]$Binary = ".\target\release\crytex-kernel.exe",
    [string]$ReportPath = "reports\release-gate-p16-proof.json"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $Binary)) {
    throw "release binary not found: $Binary"
}

& $Binary --help | Out-Null
& $Binary doctor --strict --json | Out-Null
& $Binary prove-release-gate --report-path $ReportPath | Out-Null

$report = Get-Content $ReportPath -Raw | ConvertFrom-Json
if (-not $report.passed) {
    throw "release gate report failed"
}

Write-Host "Crytex Windows binary smoke passed"
