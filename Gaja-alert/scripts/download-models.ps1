# download-models.ps1
# Downloads the Gemma 4 E4B GGUF weights + multimodal projector and the
# llama.cpp Windows-ARM64 CPU build into C:\Users\<you>\llm\ (OUTSIDE OneDrive —
# see docs\LOCAL_INFERENCE.md for why).
#
# Idempotent: skips files that already exist with a plausible size.

$ErrorActionPreference = "Stop"

$LlmRoot   = Join-Path $env:USERPROFILE "llm"
$ModelDir  = Join-Path $LlmRoot "models"
$BinDir    = Join-Path $LlmRoot "llama.cpp"
$LlamaTag  = "b9964"   # pin the release we validated on this machine

New-Item -ItemType Directory -Force $ModelDir | Out-Null
New-Item -ItemType Directory -Force $BinDir   | Out-Null

# Single-stream curl gets ~0.1-0.3 MB/s on this network and drops connections;
# aria2 with 16 parallel range requests is dramatically faster and resumes.
# Install once with: winget install aria2.aria2
$Aria = Get-Command aria2c -ErrorAction SilentlyContinue
if (-not $Aria) {
    $Aria = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\aria2.aria2*\aria2*\aria2c.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
}
if (-not $Aria) { Write-Error "aria2c not found. Run: winget install aria2.aria2"; exit 1 }
$AriaPath = if ($Aria -is [System.Management.Automation.CommandInfo]) { $Aria.Source } else { $Aria.FullName }

function Get-File($Url, $Dest, $MinBytes) {
    if ((Test-Path $Dest) -and ((Get-Item $Dest).Length -gt $MinBytes) -and -not (Test-Path "$Dest.aria2")) {
        Write-Host "[skip] $Dest already present"
        return
    }
    Write-Host "[down] $Url"
    & $AriaPath -x16 -s16 -k4M -c --retry-wait=3 --max-tries=0 `
        -d (Split-Path $Dest) -o (Split-Path $Dest -Leaf) $Url
    if ($LASTEXITCODE -ne 0) { Write-Error "download failed: $Url"; exit 1 }
}

# 1. llama.cpp native ARM64 CPU build (official ggml-org release)
$zip = Join-Path $LlmRoot "llama-$LlamaTag-bin-win-cpu-arm64.zip"
Get-File "https://github.com/ggml-org/llama.cpp/releases/download/$LlamaTag/llama-$LlamaTag-bin-win-cpu-arm64.zip" $zip 1MB
if (-not (Test-Path (Join-Path $BinDir "llama-server.exe"))) {
    Expand-Archive -Force $zip $BinDir
}

# 2. Gemma 4 E4B instruct, Q4_0 (Q4_0 specifically: llama.cpp online-repacks it
#    into ARM i8mm-optimized layouts — fastest quant on Snapdragon X CPUs)
Get-File "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_0.gguf" `
         (Join-Path $ModelDir "gemma-4-E4B-it-Q4_0.gguf") 4GB

# 3. Multimodal projector (vision + audio input)
Get-File "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/mmproj-F16.gguf" `
         (Join-Path $ModelDir "mmproj-F16.gguf") 500MB

# 4. OPTIONAL alternate model: Qwen3-VL-4B-Instruct (swap-in via serve-llm.ps1 -Model qwen)
if ($args -contains "-IncludeQwen") {
    Get-File "https://huggingface.co/unsloth/Qwen3-VL-4B-Instruct-GGUF/resolve/main/Qwen3-VL-4B-Instruct-Q4_0.gguf" `
             (Join-Path $ModelDir "Qwen3-VL-4B-Instruct-Q4_0.gguf") 2GB
    Get-File "https://huggingface.co/unsloth/Qwen3-VL-4B-Instruct-GGUF/resolve/main/mmproj-F16.gguf" `
             (Join-Path $ModelDir "qwen3-vl-4b-mmproj-F16.gguf") 300MB
}

Write-Host "`nDone. Start the server with: powershell -File scripts\serve-llm.ps1"
