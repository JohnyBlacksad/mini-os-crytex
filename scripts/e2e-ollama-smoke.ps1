param(
    [ValidateSet("all", "inference", "tauri")]
    [string]$Suite = "all",
    [string]$Model = "qwen3.5:9b",
    [string]$OllamaUrl = "http://127.0.0.1:11434",
    [string]$TargetDir = "B:\crytex-target-audit",
    [int]$OllamaTimeoutSeconds = 900,
    [switch]$SkipPull,
    [switch]$SkipClean
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message"
}

function Invoke-OllamaJson {
    param(
        [ValidateSet("Get", "Post")]
        [string]$Method,
        [string]$Path,
        [object]$Body = $null
    )

    $uri = "$OllamaUrl$Path"
    if ($null -eq $Body) {
        return Invoke-RestMethod -Method $Method -Uri $uri -TimeoutSec $OllamaTimeoutSeconds
    }

    return Invoke-RestMethod `
        -Method $Method `
        -Uri $uri `
        -ContentType "application/json" `
        -Body ($Body | ConvertTo-Json -Depth 8) `
        -TimeoutSec $OllamaTimeoutSeconds
}

function Test-OllamaModelAvailable {
    param([object]$Tags)

    if ($null -eq $Tags.models) {
        return $false
    }

    foreach ($entry in $Tags.models) {
        if ($entry.name -eq $Model -or $entry.model -eq $Model) {
            return $true
        }
    }

    return $false
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$env:CARGO_TARGET_DIR = $TargetDir
$env:CRYTEX_E2E_OLLAMA_MODEL = $Model
$env:CRYTEX_E2E_OLLAMA_URL = $OllamaUrl

Write-Host "Crytex real Ollama E2E runner"
Write-Host "Suite:        $Suite"
Write-Host "Model:        $Model"
Write-Host "Ollama:       $OllamaUrl"
Write-Host "Cargo target: $TargetDir"
Write-Host "Skip pull:    $SkipPull"
Write-Host "Skip clean:   $SkipClean"

$startedAt = Get-Date
$exitCode = 0

Push-Location $repoRoot
try {
    Write-Step "Checking Ollama availability"
    $tags = Invoke-OllamaJson -Method Get -Path "/api/tags"

    if (-not (Test-OllamaModelAvailable -Tags $tags)) {
        if ($SkipPull) {
            throw "Ollama model '$Model' is not cached and -SkipPull was set."
        }

        Write-Step "Pulling Ollama model '$Model'"
        Invoke-OllamaJson -Method Post -Path "/api/pull" -Body @{
            name = $Model
            stream = $false
        } | Out-Null
    }

    if ($Suite -eq "all" -or $Suite -eq "inference") {
        Write-Step "Running inference Ollama smoke"
        cargo test -p crytex-inference-ollama --test e2e_ollama -- --ignored --nocapture
    }

    if ($Suite -eq "all" -or $Suite -eq "tauri") {
        Write-Step "Running Tauri real Ollama E2E suite sequentially"
        cargo test -p crytex-tauri --test e2e_ollama_start_run -- --ignored --nocapture --test-threads=1
    }
} catch {
    $exitCode = 1
    Write-Error $_
} finally {
    if (-not $SkipClean) {
        Write-Step "Cleaning cargo target artifacts"
        cargo clean
    }

    Pop-Location
}

$finishedAt = Get-Date
$elapsed = New-TimeSpan -Start $startedAt -End $finishedAt

if ($exitCode -eq 0) {
    Write-Host ""
    Write-Host "Crytex real Ollama E2E runner passed in $($elapsed.ToString())."
} else {
    Write-Host ""
    Write-Host "Crytex real Ollama E2E runner failed in $($elapsed.ToString())."
}

exit $exitCode
