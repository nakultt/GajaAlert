# setup.ps1
# Basic setup for this repo alone: Python deps + .env scaffolding. Does NOT
# download the (multi-GB) LLM weights or start any servers -- for those, see
# scripts\download-models.ps1 and scripts\serve-llm.ps1 (README "Setup" steps
# 3-4). Run this from anywhere; it always operates on the repo root (one
# level up from scripts\).
#
# Usage:
#   powershell -File scripts\setup.ps1
#   powershell -File scripts\setup.ps1 -IncludeYolo   # + parked YOLO/QNN extra

param(
    [switch]$IncludeYolo
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "== Gaja Alert: basic setup ==`n"

# 1. uv must already be installed (https://docs.astral.sh/uv/)
$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uv) {
    Write-Error "uv not found on PATH. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
}
Write-Host "[ok] uv found: $($uv.Source)"

# 2. Python deps
if ($IncludeYolo) {
    Write-Host "`n[run] uv sync --extra yolo"
    uv sync --extra yolo
} else {
    Write-Host "`n[run] uv sync"
    uv sync
}
if ($LASTEXITCODE -ne 0) { Write-Error "uv sync failed"; exit 1 }

# 3. .env scaffolding -- never overwrite an existing one
$EnvPath = Join-Path $RepoRoot ".env"
$EnvExamplePath = Join-Path $RepoRoot ".env.example"
if (Test-Path $EnvPath) {
    Write-Host "`n[skip] .env already exists"
} elseif (Test-Path $EnvExamplePath) {
    Copy-Item $EnvExamplePath $EnvPath
    Write-Host "`n[created] .env from .env.example -- fill in SARVAM_API_KEY before running the pipeline"
} else {
    Write-Warning ".env.example not found; create .env by hand (see README Configuration section)"
}

# 4. incidents/ output directory (gaja/incidents.py also creates this lazily,
#    but having it up front makes a fresh checkout browsable immediately)
$IncidentsDir = Join-Path $RepoRoot "incidents"
New-Item -ItemType Directory -Force $IncidentsDir | Out-Null
Write-Host "[ok] incidents/ directory ready"

# 5. Sanity-check the bundled YOLO NCNN model files are present
$YoloParam = Join-Path $RepoRoot "yolo26n_ncnn_model\model.ncnn.param"
$YoloBin   = Join-Path $RepoRoot "yolo26n_ncnn_model\model.ncnn.bin"
if ((Test-Path $YoloParam) -and (Test-Path $YoloBin)) {
    Write-Host "[ok] YOLO NCNN model files present"
} else {
    Write-Warning "YOLO NCNN model files missing under yolo26n_ncnn_model\ -- arm_server.py will fail to start without them"
}

# 6. uvx, needed to launch the Sarvam MCP server (sarvam_agent.py / sarvam_workflow.py)
$uvx = Get-Command uvx -ErrorAction SilentlyContinue
if ($uvx) {
    Write-Host "[ok] uvx found (needed for the Sarvam MCP server)"
} else {
    Write-Warning "uvx not found on PATH -- it ships with uv; reinstall uv if this is unexpected"
}

Write-Host "`n== Basic setup done. Next steps =="
Write-Host "  1. Fill in SARVAM_API_KEY in .env (and any GAJA_* overrides you need)"
Write-Host "  2. Download + serve the LLMs:  scripts\download-models.ps1  then  scripts\serve-llm.ps1"
Write-Host "  3. Run the edge server:        uv run python arm_server.py"
