# serve-llm.ps1
# Starts llama-server with Gemma 4 E4B (default) or Qwen3-VL-4B on this
# Snapdragon X Elite laptop, exposing an OpenAI-compatible API at
# http://127.0.0.1:8080/v1  (chat completions endpoint: /v1/chat/completions).
#
# Usage:
#   powershell -File scripts\serve-llm.ps1                 # Gemma 4 E4B
#   powershell -File scripts\serve-llm.ps1 -Model qwen     # Qwen3-VL-4B
#   powershell -File scripts\serve-llm.ps1 -Ctx 16384 -Port 8081
#
# Flag rationale (see docs\LOCAL_INFERENCE.md for the full story):
#   -t 8         measured optimum, NOT all 12 cores. llama-bench showed token
#                generation (what matters for short detection replies) peaks
#                at 8 threads (16.7 t/s) and *drops* at 12 (11.7 t/s) because
#                decode is memory-bandwidth-bound, not compute-bound, past
#                that point. Prompt processing keeps scaling to 12, so raise
#                -t if you have long prompts and don't care about decode speed.
#   --no-mmap    load weights into RAM so the Q4_0 -> ARM-repacked path always
#                applies; also avoids page-fault stutter on first tokens
#   -fa on       flash attention — faster prompt processing, smaller KV cache
#   --jinja      use the model's own chat template (needed for Gemma 4 tool use)
#   --mmproj     enables image (and for Gemma, audio) input over the API

param(
    [ValidateSet("gemma", "qwen")] [string]$Model = "gemma",
    [int]$Ctx  = 32768,
    [int]$Port = 8080,
    [int]$Threads = 8   # measured optimum on X1E80100, see docs/LOCAL_INFERENCE.md benchmarks
)

$LlmRoot  = Join-Path $env:USERPROFILE "llm"
$Bin      = Join-Path $LlmRoot "llama.cpp\llama-server.exe"
$ModelDir = Join-Path $LlmRoot "models"

switch ($Model) {
    "gemma" {
        $Gguf   = Join-Path $ModelDir "gemma-4-E4B-it-Q4_0.gguf"
        $Mmproj = Join-Path $ModelDir "mmproj-F16.gguf"
        $Alias  = "gemma-4-e4b"
    }
    "qwen" {
        $Gguf   = Join-Path $ModelDir "Qwen3-VL-4B-Instruct-Q4_0.gguf"
        $Mmproj = Join-Path $ModelDir "qwen3-vl-4b-mmproj-F16.gguf"
        $Alias  = "qwen3-vl-4b"
    }
}

if (-not (Test-Path $Gguf)) {
    Write-Error "Model file missing: $Gguf`nRun scripts\download-models.ps1 first."
    exit 1
}

& $Bin `
    -m $Gguf `
    --mmproj $Mmproj `
    --alias $Alias `
    -t $Threads `
    -c $Ctx `
    -fa on `
    --no-mmap `
    --jinja `
    --host 127.0.0.1 `
    --port $Port
