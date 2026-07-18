param(
    [string]$TargetDir = "B:\crytex-target-audit",
    [int]$TimeoutSeconds = 900,
    [int]$IdleTimeoutSeconds = 180,
    [string]$ComputeCap = "",
    [string]$LogDir = "",
    [switch]$SkipGdnCuda,
    [switch]$SkipClean
)

$ErrorActionPreference = "Stop"

function Find-MsvcCl {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path -LiteralPath $vswhere) {
        $found = & $vswhere `
            -latest `
            -products * `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -find "VC\Tools\MSVC\**\bin\Hostx64\x64\cl.exe" |
            Where-Object { $_ -and (Test-Path -LiteralPath $_) } |
            Select-Object -First 1
        if ($found) {
            return $found
        }
    }

    $roots = @(
        "${env:ProgramFiles}\Microsoft Visual Studio",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio"
    )
    foreach ($root in $roots) {
        if (-not (Test-Path -LiteralPath $root)) {
            continue
        }
        $found = Get-ChildItem -LiteralPath $root -Recurse -Filter cl.exe -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -like "*\bin\Hostx64\x64\cl.exe" } |
            Select-Object -First 1
        if ($found) {
            return $found.FullName
        }
    }
    return $null
}

function Stop-CudaBuildProcesses {
    Get-Process cargo,rustc,nvcc,cl -ErrorAction SilentlyContinue |
        Stop-Process -Force -ErrorAction SilentlyContinue
}

function Stop-ProcessTree {
    param([int]$ProcessId)

    if ($ProcessId -gt 0) {
        taskkill.exe /PID $ProcessId /T /F | Out-Null
    }
}

function Invoke-CargoCleanWithRetry {
    param([int]$Attempts = 3)

    for ($i = 1; $i -le $Attempts; $i++) {
        cargo clean
        if ($LASTEXITCODE -eq 0) {
            return
        }
        Start-Sleep -Seconds (2 * $i)
    }
    throw "cargo clean failed after $Attempts attempts"
}

function Get-BuildProcessSnapshot {
    $ids = @(Get-Process cargo,rustc,nvcc,cl -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id)
    if ($ids.Count -eq 0) {
        return "No cargo/rustc/nvcc/cl processes are running."
    }
    $filter = ($ids | ForEach-Object { "ProcessId=$_" }) -join " OR "
    Get-CimInstance Win32_Process -Filter $filter |
        Select-Object ProcessId,Name,CreationDate,CommandLine |
        Format-List |
        Out-String
}

function Invoke-LoggedProcess {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList,
        [int]$TimeoutSeconds,
        [int]$IdleTimeoutSeconds,
        [string]$StdoutPath,
        [string]$StderrPath
    )

    $startedAt = Get-Date
    $lastProgressAt = Get-Date
    $lastObservedBytes = -1
    $process = Start-Process `
        -FilePath $FilePath `
        -ArgumentList $ArgumentList `
        -NoNewWindow `
        -PassThru `
        -RedirectStandardOutput $StdoutPath `
        -RedirectStandardError $StderrPath

    while (-not $process.HasExited) {
        Start-Sleep -Seconds 5
        $now = Get-Date
        $duration = [int]($now - $startedAt).TotalSeconds
        $currentBytes = 0
        foreach ($path in @($StdoutPath, $StderrPath)) {
            if (Test-Path -LiteralPath $path) {
                $currentBytes += (Get-Item -LiteralPath $path).Length
            }
        }
        if ($currentBytes -ne $lastObservedBytes) {
            $lastObservedBytes = $currentBytes
            $lastProgressAt = $now
        }
        $idleSeconds = [int]($now - $lastProgressAt).TotalSeconds
        if ($duration -ge $TimeoutSeconds -or $idleSeconds -ge $IdleTimeoutSeconds) {
            $reason = if ($duration -ge $TimeoutSeconds) {
                "TIMEOUT after $TimeoutSeconds seconds."
            } else {
                "IDLE TIMEOUT after $IdleTimeoutSeconds seconds without log output."
            }
            $processes = Get-BuildProcessSnapshot
            Stop-ProcessTree -ProcessId $process.Id
            Stop-CudaBuildProcesses
            Start-Sleep -Seconds 2
            Add-Content -LiteralPath $StderrPath -Value "`n$reason"
            Add-Content -LiteralPath $StderrPath -Value "`nLive build processes at timeout:`n$processes"
            return [pscustomobject]@{
                ExitCode = 124
                TimedOut = $true
                DurationSeconds = $duration
            }
        }
    }

    $process.WaitForExit()
    $process.Refresh()
    $exitCode = $process.ExitCode
    if ($null -eq $exitCode) {
        $hasCargoSuccess = Select-String -LiteralPath $StderrPath -SimpleMatch "Finished " -Quiet
        $hasTestExecutable = Select-String -LiteralPath $StderrPath -SimpleMatch "Executable unittests" -Quiet
        $hasError = Select-String -LiteralPath $StderrPath -Pattern "error:|LNK\d+|fatal error" -Quiet
        $exitCode = if ($hasCargoSuccess -and $hasTestExecutable -and -not $hasError) { 0 } else { 1 }
    }

    return [pscustomobject]@{
        ExitCode = $exitCode
        TimedOut = $false
        DurationSeconds = [int]((Get-Date) - $startedAt).TotalSeconds
    }
}

$cl = Find-MsvcCl
if (-not $cl) {
    throw "MSVC cl.exe was not found. Install Visual Studio Build Tools with C++ workload or run from a VS Developer PowerShell."
}

$env:CARGO_TARGET_DIR = $TargetDir
$env:NVCC_CCBIN = $cl
$env:CL = "/MD /Zc:preprocessor /DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING $env:CL".Trim()
$env:CUDA_NVCC_FLAGS = "/MD /Zc:preprocessor /DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING"
if ($ComputeCap.Trim().Length -gt 0) {
    $env:CUDA_COMPUTE_CAP = $ComputeCap
}
if ($SkipGdnCuda) {
    $env:MISTRALRS_SKIP_GDN_CUDA = "1"
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
if ($LogDir.Trim().Length -eq 0) {
    $LogDir = Join-Path (Get-Location) ".crytex-smoke-logs"
}
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$stdoutPath = Join-Path $LogDir "mistral-cuda-build-probe-$timestamp.out.log"
$stderrPath = Join-Path $LogDir "mistral-cuda-build-probe-$timestamp.err.log"

Write-Host "CUDA build probe:"
Write-Host "  CARGO_TARGET_DIR=$env:CARGO_TARGET_DIR"
Write-Host "  NVCC_CCBIN=$env:NVCC_CCBIN"
Write-Host "  CL=$env:CL"
Write-Host "  CUDA_NVCC_FLAGS=$env:CUDA_NVCC_FLAGS"
if ($env:CUDA_COMPUTE_CAP) {
    Write-Host "  CUDA_COMPUTE_CAP=$env:CUDA_COMPUTE_CAP"
}
if ($env:MISTRALRS_SKIP_GDN_CUDA) {
    Write-Host "  MISTRALRS_SKIP_GDN_CUDA=$env:MISTRALRS_SKIP_GDN_CUDA"
}
Write-Host "  TimeoutSeconds=$TimeoutSeconds"
Write-Host "  IdleTimeoutSeconds=$IdleTimeoutSeconds"
Write-Host "  stdout=$stdoutPath"
Write-Host "  stderr=$stderrPath"

try {
    $result = Invoke-LoggedProcess `
        -FilePath "cargo" `
        -ArgumentList @("test", "-p", "crytex-inference-mistral", "--features", "cuda", "--no-run") `
        -TimeoutSeconds $TimeoutSeconds `
        -IdleTimeoutSeconds $IdleTimeoutSeconds `
        -StdoutPath $stdoutPath `
        -StderrPath $stderrPath

    Write-Host "CUDA build probe finished: exit=$($result.ExitCode), timed_out=$($result.TimedOut), duration_seconds=$($result.DurationSeconds)"
    if ($result.ExitCode -ne 0) {
        throw "CUDA build probe failed. stdout=$stdoutPath stderr=$stderrPath"
    }
}
finally {
    if (-not $SkipClean) {
        Invoke-CargoCleanWithRetry
    }
}
