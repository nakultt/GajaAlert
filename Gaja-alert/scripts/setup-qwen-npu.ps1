param(
    [string]$Bundle = "$env:USERPROFILE\Downloads\qwen3_vl_4b_instruct-geniex_qairt-w4a16-qualcomm_snapdragon_x_elite\qwen3_vl_4b_instruct-geniex_qairt-w4a16-qualcomm_snapdragon_x_elite",
    [string]$QairtHome = "$env:USERPROFILE\Downloads\v2.48.0.260626\qairt\2.48.0.260626"
)

$ErrorActionPreference = "Stop"
$Python = "$env:USERPROFILE\llm\geniex-env\Scripts\python.exe"

if (-not (Test-Path "$Bundle\metadata.json")) { throw "Missing model bundle: $Bundle" }
if (-not (Test-Path "$QairtHome\bin\aarch64-windows-msvc\genie-t2t-run.exe")) {
    throw "Missing QAIRT ARM64 SDK: $QairtHome"
}
if (-not (Test-Path $Python)) {
    $ArmPython = Get-Command python3.12-arm64, python3-arm64 -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $ArmPython) {
        throw "A native ARM64 Python is required to create $Python. Install Python 3.12 ARM64, then rerun."
    }
    & $ArmPython.Source -m venv "$env:USERPROFILE\llm\geniex-env"
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required to install GenieX into the native ARM64 environment"
}
uv pip install --python $Python --upgrade "geniex>=0.3.14"
if ($LASTEXITCODE -ne 0) { throw "GenieX installation failed" }

$env:QAIRT_HOME = $QairtHome
$env:Path = "$QairtHome\bin\aarch64-windows-msvc;$QairtHome\lib\aarch64-windows-msvc;$env:Path"
& $Python -m geniex.cli devices
if ($LASTEXITCODE -ne 0) { throw "GenieX cannot enumerate the QAIRT NPU" }

Write-Host "[ok] Qwen bundle, QAIRT, GenieX, and Qualcomm NPU are ready."
Write-Host "Start the server with: powershell -File scripts\serve-qwen-npu.ps1"
