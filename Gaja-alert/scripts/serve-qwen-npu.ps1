param(
    [string]$Bundle = "$env:USERPROFILE\Downloads\qwen3_vl_4b_instruct-geniex_qairt-w4a16-qualcomm_snapdragon_x_elite\qwen3_vl_4b_instruct-geniex_qairt-w4a16-qualcomm_snapdragon_x_elite",
    [string]$QairtHome = "$env:USERPROFILE\Downloads\v2.48.0.260626\qairt\2.48.0.260626",
    [int]$Port = 8081
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = "$env:USERPROFILE\llm\geniex-env\Scripts\python.exe"

if (-not (Test-Path "$Bundle\metadata.json")) {
    throw "Qwen QAIRT bundle not found at $Bundle"
}
if (-not (Test-Path "$QairtHome\bin\aarch64-windows-msvc\genie-t2t-run.exe")) {
    throw "QAIRT ARM64 runtime not found at $QairtHome"
}
if (-not (Test-Path $Python)) {
    throw "GenieX environment not found at $Python. Run scripts\setup-qwen-npu.ps1 first."
}

$env:GAJA_QWEN_NPU_BUNDLE = $Bundle
$env:GAJA_QWEN_NPU_PORT = "$Port"
$env:QAIRT_HOME = $QairtHome
$env:Path = "$QairtHome\bin\aarch64-windows-msvc;$QairtHome\lib\aarch64-windows-msvc;$env:Path"

Set-Location $RepoRoot
& $Python scripts\serve_qwen_npu.py
