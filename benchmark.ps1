<#
.SYNOPSIS
  RAG Toolkit GPU benchmark operator helper (Windows / PowerShell).

.DESCRIPTION
  Wraps the GPU compose stack and the dedicated benchmark-runner container.
  Reads HF_TOKEN from .env automatically (docker compose interpolation) and
  forwards it to the runner; nothing is hardcoded.

.EXAMPLE
  .\benchmark.ps1 up         # build + start stack incl. benchmark-runner
  .\benchmark.ps1 hf-verify  # verify the HuggingFace token authenticates
  .\benchmark.ps1 smoke      # verify the runner sees the GPU
  .\benchmark.ps1 bench      # run full GPU experiment + benchmark pipeline
  .\benchmark.ps1 report     # rebuild REPORT.md + AGGREGATE_REPORT.md only
  .\benchmark.ps1 shell      # interactive shell in the runner
  .\benchmark.ps1 logs       # follow logs
  .\benchmark.ps1 down       # stop + remove stack
  .\benchmark.ps1 all        # up + hf-verify + smoke + bench (default)
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet("up", "hf-verify", "smoke", "bench", "report", "shell", "logs", "ps", "down", "all")]
    [string]$Action = "all"
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$Compose = @("compose", "-f", "docker-compose.yml", "-f", "docker-compose.gpu.yml")
$Runner  = "rag-benchmark-runner"
$OutDir  = Join-Path $PSScriptRoot "experiment-outputs"

function Test-EnvFile {
    $envPath = Join-Path $PSScriptRoot ".env"
    if (-not (Test-Path $envPath)) {
        Write-Warning ".env not found. Copy .env.example to .env and set HF_TOKEN."
        return
    }
    $hasToken = Select-String -Path $envPath -Pattern '^\s*HF_TOKEN\s*=\s*hf_' -Quiet
    if (-not $hasToken) {
        Write-Warning "HF_TOKEN in .env still looks like a placeholder. Set your real token."
    }
}

function Invoke-Up {
    Test-EnvFile
    if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }
    docker @Compose up -d --build
}

function Invoke-HfVerify { docker exec -i  $Runner bash /opt/ops/hf-verify.sh }
function Invoke-Smoke    { docker exec -i  $Runner bash /opt/ops/gpu-smoke.sh }
function Invoke-Bench    { docker exec -i  $Runner bash /opt/ops/run-benchmarks.sh }
function Invoke-Report   { docker exec -i  $Runner bash /opt/ops/build-report.sh }
function Invoke-Shell    { docker exec -it $Runner bash }
function Invoke-Logs     { docker @Compose logs -f }
function Invoke-Ps       { docker @Compose ps }
function Invoke-Down     { docker @Compose down }

function Show-Outputs {
    Write-Host ""
    Write-Host "Artefacts (host): $OutDir" -ForegroundColor Cyan
    Write-Host "  per-run report : $OutDir\reports\<run_id>\REPORT.md"
    Write-Host "  aggregate      : $OutDir\reports\_aggregate\AGGREGATE_REPORT.md"
    Write-Host "  dashboard      : $OutDir\reports\<run_id>\dashboard.png"
    Write-Host "  manifest       : $OutDir\reports\<run_id>\manifest.json"
}

switch ($Action) {
    "up"        { Invoke-Up }
    "hf-verify" { Invoke-HfVerify }
    "smoke"     { Invoke-Smoke }
    "bench"     { Invoke-Bench;  Show-Outputs }
    "report"    { Invoke-Report; Show-Outputs }
    "shell"     { Invoke-Shell }
    "logs"      { Invoke-Logs }
    "ps"        { Invoke-Ps }
    "down"      { Invoke-Down }
    "all"       { Invoke-Up; Invoke-HfVerify; Invoke-Smoke; Invoke-Bench; Show-Outputs }
}
