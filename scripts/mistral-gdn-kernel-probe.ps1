param(
    [string]$ComputeCap = "120a",
    [int]$TimeoutSeconds = 120,
    [string]$LogDir = "",
    [string[]]$Define = @(),
    [string[]]$ExtraNvccArg = @()
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

function Stop-ProcessTree {
    param([int]$ProcessId)

    if ($ProcessId -gt 0) {
        taskkill.exe /PID $ProcessId /T /F | Out-Null
    }
}

function Get-NvccProcessSnapshot {
    $names = @("nvcc", "cicc", "cl", "ptxas")
    $ids = @(Get-Process -ErrorAction SilentlyContinue |
        Where-Object { $names -contains $_.ProcessName } |
        Select-Object -ExpandProperty Id)
    if ($ids.Count -eq 0) {
        return "No nvcc/cicc/cl/ptxas processes are running."
    }

    $filter = ($ids | ForEach-Object { "ProcessId=$_" }) -join " OR "
    Get-CimInstance Win32_Process -Filter $filter |
        Select-Object ProcessId,Name,CreationDate,CommandLine |
        Format-List |
        Out-String
}

function Join-CommandLineForLog {
    param([string]$FilePath, [string[]]$ArgumentList)

    $escaped = $ArgumentList | ForEach-Object {
        if ($_ -match '\s|"' ) {
            '"' + ($_.Replace('"', '\"')) + '"'
        } else {
            $_
        }
    }
    "$FilePath $($escaped -join ' ')"
}

function Join-ProcessArguments {
    param([string[]]$ArgumentList)

    ($ArgumentList | ForEach-Object {
        if ($_ -match '\s|"' ) {
            '"' + ($_.Replace('"', '\"')) + '"'
        } else {
            $_
        }
    }) -join ' '
}

function Invoke-LoggedProcess {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList,
        [int]$TimeoutSeconds,
        [string]$StdoutPath,
        [string]$StderrPath
    )

    $command = Join-CommandLineForLog -FilePath $FilePath -ArgumentList $ArgumentList
    $command = "$command 1>`"$StdoutPath`" 2>`"$StderrPath`""
    $startedAt = Get-Date
    $process = Start-Process `
        -FilePath "cmd.exe" `
        -ArgumentList @("/d", "/s", "/c", $command) `
        -NoNewWindow `
        -PassThru

    $exited = $process.WaitForExit($TimeoutSeconds * 1000)
    if (-not $exited) {
        $snapshot = Get-NvccProcessSnapshot
        Stop-ProcessTree -ProcessId $process.Id
        $process.WaitForExit(10000) | Out-Null
        Start-Sleep -Milliseconds 500
        return [pscustomobject]@{
            ExitCode = 124
            TimedOut = $true
            DurationSeconds = [int]((Get-Date) - $startedAt).TotalSeconds
            Snapshot = $snapshot
        }
    }

    $process.WaitForExit()
    $process.Refresh()
    Start-Sleep -Milliseconds 200
    $exitCode = $process.ExitCode
    if ($null -eq $exitCode) {
        $exitCode = if (Test-Path -LiteralPath $objectPath) { 0 } else { 1 }
    }
    return [pscustomobject]@{
        ExitCode = $exitCode
        TimedOut = $false
        DurationSeconds = [int]((Get-Date) - $startedAt).TotalSeconds
        Snapshot = ""
    }
}

$cl = Find-MsvcCl
if (-not $cl) {
    throw "MSVC cl.exe was not found. Install Visual Studio Build Tools with C++ workload or run from a VS Developer PowerShell."
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$source = Join-Path $repoRoot "patches\mistralrs-core\src\cuda\gdn.cu"
if (-not (Test-Path -LiteralPath $source)) {
    throw "gdn.cu was not found at $source"
}

if ($LogDir.Trim().Length -eq 0) {
    $LogDir = Join-Path $repoRoot ".crytex-smoke-logs"
}
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$tag = if ($Define.Count -gt 0) { ($Define -join "_") } else { "full" }
$tag = $tag -replace '[^A-Za-z0-9_.-]', '_'
$objectPath = Join-Path $LogDir "gdn-$ComputeCap-$tag.obj"
$stdoutPath = Join-Path $LogDir "gdn-$ComputeCap-$tag-$timestamp.out.log"
$stderrPath = Join-Path $LogDir "gdn-$ComputeCap-$tag-$timestamp.err.log"
$cmdPath = Join-Path $LogDir "gdn-$ComputeCap-$tag-$timestamp.cmd.txt"

$env:NVCC_CCBIN = $cl
$env:CL = "/MD /Zc:preprocessor /DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING $env:CL".Trim()
$env:CUDA_NVCC_FLAGS = "/MD /Zc:preprocessor /DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING"

$args = @(
    "-gencode=arch=compute_$ComputeCap,code=sm_$ComputeCap",
    "-c",
    "--default-stream", "per-thread",
    "-std=c++17",
    "-O3",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
    "--verbose",
    "--compiler-options", "/MD /Zc:preprocessor /DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING",
    "-allow-unsupported-compiler",
    "-ccbin", $cl,
    "-D_USE_MATH_DEFINES",
    "-o", $objectPath
)

foreach ($defineName in $Define) {
    if ($defineName.Trim().Length -gt 0) {
        $args += "-D$defineName"
    }
}

$args += $ExtraNvccArg
$args += $source

Set-Content -LiteralPath $cmdPath -Value (Join-CommandLineForLog -FilePath "nvcc" -ArgumentList $args)

Write-Host "GDN kernel probe:"
Write-Host "  source=$source"
Write-Host "  compute_cap=$ComputeCap"
Write-Host "  defines=$($Define -join ',')"
Write-Host "  timeout_seconds=$TimeoutSeconds"
Write-Host "  cl=$cl"
Write-Host "  object=$objectPath"
Write-Host "  stdout=$stdoutPath"
Write-Host "  stderr=$stderrPath"
Write-Host "  command=$cmdPath"

$result = Invoke-LoggedProcess `
    -FilePath "nvcc" `
    -ArgumentList $args `
    -TimeoutSeconds $TimeoutSeconds `
    -StdoutPath $stdoutPath `
    -StderrPath $stderrPath

if ($result.TimedOut) {
    Add-Content -LiteralPath $stderrPath -Value "`nTIMEOUT after $TimeoutSeconds seconds."
    Add-Content -LiteralPath $stderrPath -Value "`nLive CUDA compiler processes at timeout:`n$($result.Snapshot)"
}

Write-Host "GDN kernel probe finished: exit=$($result.ExitCode), timed_out=$($result.TimedOut), duration_seconds=$($result.DurationSeconds)"
if ($result.ExitCode -ne 0) {
    throw "GDN kernel probe failed. stdout=$stdoutPath stderr=$stderrPath command=$cmdPath"
}
