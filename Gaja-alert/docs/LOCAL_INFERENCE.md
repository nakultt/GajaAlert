# Local LLM inference on the Surface Laptop 7 (Snapdragon X Elite)

This documents how **Gemma 4 E4B** (and optionally **Qwen3-VL-4B-Instruct**) runs
locally on the project laptop, every decision taken, and the Windows-on-ARM64
pitfalls we hit along the way. Written 2026-07-11 on the `gemma` branch.

## Role in the Gaja-alert pipeline

Per the architecture sketch: the UNO Q board does continuous listening →
band-pass → vision → segmentation, and streams to this laptop
([arm_server.py](../arm_server.py) receives video/audio over WebSocket). The
laptop runs **Gemma 4 E4B** to *confirm the presence of an elephant* from the
segmented frames/audio and to *create reports*, then signals back (buzzer +
mobile notifications).

## Hardware

| | |
|---|---|
| Machine | Microsoft Surface Laptop, 7th Edition |
| SoC | Snapdragon X Elite **X1E80100**, 12 cores @ 3.40 GHz (no SMT) |
| RAM | 32 GB |
| GPU / NPU | Adreno X1-85 / Hexagon NPU (45 TOPS) — **not used**, see below |
| OS | Windows 11 Pro (ARM64) |

## The model

[`google/gemma-4-E4B`](https://huggingface.co/google/gemma-4-E4B) (instruct
variant): 4.5 B *effective* parameters (8 B with embeddings, MatFormer-style),
42 layers, 128 K context, **text + image + audio in → text out**, Apache 2.0.
Released April 2026. The audio encoder (~300 M) and vision encoder (~150 M)
matter for us: elephant confirmation can use both the camera frames and the
band-passed audio.

We use the community-standard GGUF conversion:
[`unsloth/gemma-4-E4B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF).

## Decisions and why

### 1. Runtime: llama.cpp, official `win-cpu-arm64` build (pinned `b9964`)

Options considered for Windows-on-ARM:

- **Ollama / LM Studio** — work on ARM64 but are wrappers around llama.cpp
  with less control over ARM-specific flags, and lag behind on brand-new model
  support (Gemma 4 fixes landed in llama.cpp right after the April release, so
  we want to control the exact build).
- **ONNX Runtime + QNN (Hexagon NPU)** — the "proper" Snapdragon path, but:
  (a) only runs models pre-converted to QNN context binaries via the
  [onnxruntime-genai Snapdragon pipeline](https://onnxruntime.ai/docs/genai/tutorials/snapdragon.html)
  or Qualcomm AI Hub — no published Gemma 4 E4B conversion exists, and its
  MatFormer/per-layer-embedding architecture makes DIY conversion a research
  project; (b) LLM-on-QNN still requires *nightly* onnxruntime builds
  (mid-2026); (c) text-only — no path for Gemma 4's vision/audio encoders,
  which we need; (d) NPU wins on power, not decode speed (decode is
  memory-bandwidth-bound; CPU i8mm already saturates it).
- **llama.cpp Hexagon NPU backend** — new, [experimental](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md),
  Windows docs exist. Requires source build with Hexagon SDK + signed HTP op
  libraries; validated on Llama-style Q4_0 models, Gemma 4 support unproven.
  **This is the phase-2 experiment** if CPU contention with the vision
  pipeline becomes a problem: same GGUF, same llama-server API, nothing
  downstream changes.
- **PyTorch/transformers** — no CUDA here, CPU-only PyTorch on ARM64 Windows
  is poorly supported and slow; many wheels (e.g. parts of the audio stack)
  simply don't exist for `win_arm64`.
- **llama.cpp** ✅ — ships an official native ARM64 Windows binary, has
  first-class Gemma 4 + multimodal (mtmd) support, and has ARM-specific
  CPU kernels (NEON dotprod / i8mm) that make the CPU the fastest option on
  this chip.

### 2. CPU, not GPU/NPU

Counter-intuitive but well established for Snapdragon X
([llama.cpp discussion #8273](https://github.com/ggml-org/llama.cpp/discussions/8273),
[#8336](https://github.com/ggml-org/llama.cpp/discussions/8336),
[Qualcomm's own blog](https://www.qualcomm.com/developer/blog/2024/04/big-performance-boost-llama-cpp-chatglm-cpp-with-windows-on-snapdragon)):
the Oryon cores with i8mm-optimized 4-bit kernels beat the Adreno GPU via
OpenCL/Vulkan for LLM decoding, and the NPU is unusable without a
QNN-converted model. There is an official `win-opencl-adreno-arm64` build if
we ever want to offload and keep the CPU free for OpenCV — benchmark before
switching, expect it to be *slower*.

### 3. Quantization: `Q4_0` (4.84 GB), not Q4_K_M

Normally Q4_K_M is the default pick. On ARM CPUs, **Q4_0 is special**:
llama.cpp *online-repacks* Q4_0 tensors at load time into interleaved layouts
(the successor of the old `Q4_0_4_8` files) that feed the NEON/i8mm matmul
kernels. K-quants don't get this path, so Q4_0 is both smaller *and*
significantly faster here. Quality loss vs Q4_K_M is minor for a
classification/report task. `Q8_0` (8.2 GB) is an easy upgrade if confirmation
accuracy looks off — RAM is not a constraint at 32 GB.

### 4. Files live in `C:\Users\<you>\llm\`, NOT in this repo folder

The repo is inside **OneDrive** (`...\OneDrive\Documents\Gaja alert`). A ~5 GB
GGUF in a OneDrive-synced folder means endless upload churn, "files on-demand"
hydration stalls, and possible file locks while the model is loaded. Models
and binaries go to `%USERPROFILE%\llm\{llama.cpp,models}`; only scripts and
docs are committed.

### 5. Server flags (`scripts/serve-llm.ps1`)

| Flag | Why |
|---|---|
| `-t 12` | All 12 physical cores, no SMT on X1E80100. If the laptop must simultaneously run OpenCV + the WebSocket server, drop to `-t 10`. |
| `--no-mmap` | Load weights fully into RAM: guarantees the Q4_0→ARM repack applies and avoids first-token page-fault stutter. Costs ~5 GB resident RAM (fine). |
| `-fa on` | Flash attention: faster prompt processing, halves KV-cache memory. |
| `-c 8192` | 8 K context is plenty for frame-confirmation prompts; raising it only grows KV cache. Model supports up to 128 K. |
| `--jinja` | Uses Gemma 4's own chat template, required for its native function-calling. |
| `--mmproj mmproj-F16.gguf` | Enables image/audio input through the mtmd pipeline. |
| `--host 127.0.0.1` | Local-only. Do not bind 0.0.0.0 unless the UNO Q needs to hit the API directly. |

## How to run

```powershell
# one-time (downloads ~6 GB)
powershell -File scripts\download-models.ps1          # add -IncludeQwen for Qwen3-VL
# start the server
powershell -File scripts\serve-llm.ps1                # or -Model qwen
```

The server exposes an **OpenAI-compatible API** at `http://127.0.0.1:8080/v1`.
From Python (works with the `openai` package or plain HTTP):

```python
import base64, requests

img_b64 = base64.b64encode(open("frame.jpg", "rb").read()).decode()
r = requests.post("http://127.0.0.1:8080/v1/chat/completions", json={
    "model": "gemma-4-e4b",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Is there an elephant in this image? Answer JSON: {\"elephant\": bool, \"confidence\": 0-1}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        ],
    }],
    "temperature": 0.1,
})
print(r.json()["choices"][0]["message"]["content"])
```

Swapping Gemma ↔ Qwen3-VL requires **no client code changes** — only the
`model`/alias differs. That was the point of standing up an OpenAI-compatible
server instead of embedding the model in-process.

## Gemma 4's "thinking" mode — disable it for this use case

Gemma 4's chat template emits a chain-of-thought `reasoning_content` pass
before the final answer whenever `enable_thinking` is truthy in the template
context. **llama-server enables this by default.** For short
detection/report prompts this is actively harmful: with `max_tokens: 300` a
real test burned ~250 tokens on reasoning and returned truncated,
unparseable JSON (`finish_reason: "length"`). Fix: pass
`"chat_template_kwargs": {"enable_thinking": false}` in the request body.
Same prompt then returned complete JSON in 2.2 s / 42 tokens
(`finish_reason: "stop"`). [scripts/test_inference.py](../scripts/test_inference.py)
sets this by default — any client code calling the API should too.

## Vision pipeline sanity check

No real elephant photo was available to test with, so
[mmproj-F16.gguf](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF) was
validated with a synthetic OpenCV-generated image (green background, dark
ellipse) sent as a base64 `image_url`. The model correctly described shape
and color in 3.9 s — confirms the multimodal (mtmd) path works end-to-end on
this ARM64 build. **Retest with a real elephant/non-elephant photo pair
before trusting confidence scores** — this only proves the plumbing works,
not detection accuracy.

## Relationship to the YOLO26-seg / QNN work in this repo

This repo also contains a separate, further-along pipeline
([webcam_inference.py](../webcam_inference.py), `yolo26n-seg.pt`,
`export_assets/yolo26_seg-qnn_context_binary-...`) that runs **YOLO26
segmentation on the Hexagon NPU via QNN**. That is the right split of work:
object-detection/segmentation models convert to QNN context binaries
cleanly and benefit from NPU offload; generative LLMs like Gemma 4 do not
(see the ONNX/QNN comparison above) and run faster on CPU today. The two
pipelines are complementary — YOLO/QNN handles the "vision → segment" stage,
Gemma/llama.cpp handles "confirm elephant + write report" using the
segmented crop and/or audio as input.

## NPU/GPU experiment: GenieX (Qualcomm's community Genie runtime)

Investigated whether we can split prefill onto the Hexagon NPU and decode
onto the Adreno GPU (as Qualcomm's own RAG research describes). Findings:

- **That exact split isn't a shipped feature.** GenieX's GGUF path only
  offers `--device hybrid` (NPU+CPU tensor scheduling via llama.cpp's
  Hexagon backend) or `--device pinned` (NPU-only). GPU isn't part of that
  scheduler. A true prefill-on-NPU/decode-on-GPU setup only exists in the
  full Genie/QAIRT stack via hand-built `genie_config.json` backend routing
  over custom-compiled QNN context binaries — no prior art for Gemma 4 E4B,
  and it's a multi-day conversion project with real risk of failure (unproven
  MatFormer architecture support).
- **The Windows CLI installer's published checksum did not match the
  binary** — `geniex-cli-setup-windows-arm64-v0.3.14.exe`, both
  `Get-FileHash` (PowerShell) and `sha256sum` (coreutils) independently
  computed `15e8fac5...`, but GitHub's own `.sha256` file says `170fd1d9...`.
  File size matched `Content-Length` exactly, ruling out a truncated
  download. Given `qualcomm/GenieX` PR #1164 moved *updates* to an S3 index
  (README still points fresh installs at GitHub Releases), this is most
  likely a stale checksum file the maintainers forgot to regenerate — but it
  wasn't verifiable, so **the `.exe` installer was deleted unused**.
  Installed via `pip install geniex` instead (PyPI's package trust chain)
  to avoid running an unverified native installer that would also touch
  system driver state.
- Model dispatch is confirmed to hit the same llama.cpp Hexagon backend
  the official docs call "experimental" — so this is a genuine experiment,
  not a guaranteed win over the CPU baseline.

**If you retry the native Windows installer** in the future: re-check the
`.sha256` file against a fresh `Get-FileHash`; if they still don't match,
report it upstream before running it — [github.com/qualcomm/GenieX/issues](https://github.com/qualcomm/GenieX/issues).

### Head-to-head result (this machine, `pip install geniex==0.3.14`, Gemma 4 E4B Q4_0, 150-token generation)

| `--device` | decode | first token | load time |
|---|---:|---:|---:|
| `cpu`    | **19.1 tok/s** | 0.4 s | 15.0 s |
| `gpu` (Adreno OpenCL) | 14.2 tok/s | 0.4 s | 49.6 s |
| `npu` (Hexagon HTP0)  | 12.3 tok/s | 1.8 s | 39.8 s |
| `hybrid` (NPU+GPU+CPU) | **crash** | — | — |

**CPU wins outright — GPU and NPU are each individually slower, and combined
("hybrid") crashes.** `hybrid` mode hit a hard assertion inside llama.cpp's
tensor-split scheduler:
`ggml-backend.cpp:1367: GGML_ASSERT(n_inputs < GGML_SCHED_MAX_SPLIT_INPUTS)
failed` — a real, reproducible bug when splitting Gemma 4's graph across
three backends (its log even says "please report this on github as an
issue"; not yet filed as of 2026-07-11). This closes the loop on the
NPU-prefill/GPU-decode question: even the parts of that idea that *do* exist
today (hybrid tensor scheduling) don't work for this model, and the
single-backend numbers confirm CPU was the right call from the start.

**Gotcha for reproducing this**: on Windows-on-ARM64, make sure the Python
interpreter itself is native ARM64, not just the venv. `uv`'s managed
Python catalog is x86_64-only on Windows (`uv python list` confirms no
aarch64 entries) — a venv built from it runs under x64 emulation and
`ctypes.CDLL()` on geniex's native library fails with
`OSError: [WinError 193] %1 is not a valid Win32 application`. Verify with
`os.environ["PROCESSOR_ARCHITECTURE"]` (not `platform.machine()`, which
reports the *host* CPU even inside an emulated process and will misleadingly
say `ARM64`). Fix: install a native interpreter —
`winget install --id Python.Python.3.12 --architecture arm64` — and point
`uv venv --python <path>` at it directly.
**This affects the main project `.venv` too** — it was found to be running
under x64 emulation, not investigated further here since it's out of scope,
but worth revisiting for OpenCV/numpy performance.

## The NPU-prefill / GPU-decode question (Gemma 4 E2B, 2026-07-11)

Requested design: prefill on Hexagon NPU, decode on Adreno GPU, CPU left
alone. Research conclusion first, evidence after: **this exact split cannot
be built for Gemma 4 with any shipped runtime today.** The blocker is
architectural, at four independent layers:

1. **Genie's config schema has no per-phase backend field.** In
   `genie_config.json`, `dialog.engine` is a single engine bound to a single
   `backend` (e.g. `QnnHtp`). Prefill (prompt-processor) and decode
   (token-generator) are separate *graphs* switched at runtime
   (`enable-graph-switching`), but both execute on that one backend.
   Verified against a real published Genie bundle config (Llama 3.1 8B) and
   the strings inside `geniex_core.dll` (has `prefill`/`decode`/`graphs`
   keys; links only `QnnHtp.dll`/`QnnSystem.dll`).
2. **QNN context binaries are backend-specific.** An HTP-compiled `.bin`
   cannot be loaded by `QnnGpu`. A GPU decode graph would need the model
   recompiled from source against the GPU backend — which is itself only a
   "preview" feature of the full QAIRT SDK on Windows-on-Snapdragon.
3. **The KV-cache wouldn't transfer anyway (Layout Incompatibility).**
   We verified this experimentally:
   - Saving and loading the KV cache on the *same* backend works perfectly (NPU -> NPU, and GPU -> GPU).
   - Attempting to load an NPU-saved KV cache file on the GPU backend fails with:
     > `state_read_data: incompatible V transposition`
     > `llama_state_load_file: error loading session file: failed to restore kv cache`
   
   **Why**: The Hexagon HTP backend repacks and transposes the KV tensors in device memory to match the Snapdragon CDSP vector architecture (HTP0 expects specific vector layout shapes), whereas the Adreno GPU OpenCL backend expects a standard layout. This makes cross-backend state loading mathematically incompatible in the current `llama.cpp` plugin.
4. **There is no Genie/QNN bundle for Gemma 4 at all.** The decisive fact:
   Qualcomm's own `release_assets.json` in
   [`qualcomm/Gemma-4-E2B-it`](https://huggingface.co/qualcomm/Gemma-4-E2B-it)
   lists exactly one asset — Google's QAT Q4_0 **GGUF** with runtime
   `geniex_llamacpp`. Even Qualcomm runs Gemma 4 through llama.cpp's Hexagon
   backend, not the Genie graph-switching stack (the MatFormer architecture
   has no QNN conversion). AI Hub's "Gemma-4-E2B-it" listing resolves to the
   same GGUF (and the AI Hub download API requires an account —
   `GenieXError(-100009): authentication required` — which we didn't need
   since the GGUF is ungated on HF).

### What WAS implemented instead: Pipeline Parallelism (HTP0 + GPUOpenCL)

While a phase-based prefill/decode split across separate backends is impossible, we can run a **hybrid pipeline configuration** by loading the model with:
`device_map="llama_cpp:HTP0,GPUOpenCL"`

Llama.cpp automatically segments the model layers:
- **Layers 0–16** are scheduled on `HTP0` (Hexagon NPU).
- **Layers 17–35** are scheduled on `GPUOpenCL` (Adreno GPU).
- The CPU is only used for fallback operations (e.g., tokenization and sampling).

We have created two dedicated scripts to run and test this configuration:
- [scripts/serve_e2b_split.py](../scripts/serve_e2b_split.py): Starts the hybrid server on port `8082`.
- [scripts/test_split_inference.py](../scripts/test_split_inference.py): A client test script that runs a validation prompt and measures speed.

#### Running and Verification:
1. Run the server:
   ```powershell
   C:\Users\qcwor\llm\geniex-env\Scripts\python.exe scripts\serve_e2b_split.py
   ```
2. Verify via the test script:
   ```powershell
   C:\Users\qcwor\llm\geniex-env\Scripts\python.exe scripts\test_split_inference.py
   ```

Note: Performance-wise, pipelining across NPU + GPU is slightly slower than running purely on the NPU (16.1 tok/s decode vs 19.4 tok/s on NPU-only), due to the overhead of copying intermediate activation tensors between the CDSP memory space and OpenCL memory space. However, it successfully reduces memory load on both individual processors and utilizes both accelerators.

Also checked (user pointer): [carrycooldude](https://github.com/carrycooldude?tab=repositories)'s
repos — his LLM-on-Snapdragon work (`QNN-On-Device-OnePlus15`: Gemma 4 2B on
Hexagon V81 via LiteRT-LM, Android) runs prefill *and* decode on the NPU,
consistent with the analysis above; `llama.cpp-X-Elite` and
`Llama-QNN-ExecuTorch` are empty repos (size 0, no commits).

## Windows-on-ARM64 gotchas encountered (running list)

1. **`sounddevice` fails to import** — no ARM64 PortAudio DLL in the wheel;
   already worked around in [arm_server.py](../arm_server.py) with a
   try/except. Same class of problem affects many wheels: check for
   `win_arm64` wheels before adding any dependency.
2. **Many "Windows" binaries are x64-only** — they run under emulation
   (slower) or crash. Always grab explicit `arm64` artifacts; verify with
   `llama-cli --version` (should say `for Windows arm64`).
3. **OneDrive + big model files** don't mix (see decision 4).
4. **No CUDA / DirectML story for llama.cpp here** — CPU is the fast path;
   don't waste time on GPU flags.
5. **Pin llama.cpp versions** — Gemma 4 support was fresh in April 2026;
   `b9964` (2026-07) is validated. Upgrades: re-run download script with a new
   tag and re-test.
6. **`llama-cli` hangs when stdin isn't a TTY**, even with `-no-cnv`: in a
   background/piped shell it can drop into its interactive REPL and loop
   printing empty `>` prompts forever instead of exiting after the one-shot
   `-p` completion (burned 38 minutes and a 300K-line log before we noticed).
   Fix: don't script against `llama-cli` for one-shot generation — use
   `llama-server` + the HTTP API instead (which is what we want for the
   pipeline anyway). If you must use `llama-cli` non-interactively, redirect
   `< /dev/null` and verify it actually exits.
7. **Slow single-stream downloads** (not ARM-specific, but hit us here): this
   network gave ~0.1–0.3 MB/s per HTTP connection to the Hugging Face CDN and
   dropped long transfers (curl exit 18). Fix: `aria2c -x16 -s16 -c` (16
   parallel range requests + resume) — this is what
   [download-models.ps1](../scripts/download-models.ps1) uses. Install via
   `winget install aria2.aria2`.

## Benchmarks (this machine, 2026-07-11, llama.cpp b9964, Q4_0, flash attn on)

| threads | prompt processing (t/s) | token generation (t/s) |
|---:|---:|---:|
| 6  | 117.6 ± 10.1 | 15.9 ± 3.9 |
| 8  | 113.9 ± 12.0 | **16.7 ± 3.0** ← best decode |
| 10 | 125.6 ± 18.3 | 12.9 ± 0.6 |
| 12 | **126.3 ± 45.8** ← best prefill | 11.7 ± 0.5 |

**Takeaway: decode (token generation) peaks at 8 threads and gets *worse*
past that** — it's memory-bandwidth-bound, so extra threads beyond the
bandwidth saturation point add scheduling contention instead of throughput.
Prompt processing is compute-bound and keeps scaling to 12 threads. Since our
elephant-confirmation prompts are short-in/short-out, generation speed is
what matters, so `serve-llm.ps1` defaults to `-t 8`. If a future workload
sends long prompts (e.g. big system prompts, multi-image batches) and cares
more about prefill latency, override with `-Threads 12`.

Reproduce with:

```powershell
& "$env:USERPROFILE\llm\llama.cpp\llama-bench.exe" -m "$env:USERPROFILE\llm\models\gemma-4-E4B-it-Q4_0.gguf" -t 6,8,10,12 -p 512 -n 128 -fa 1
```

End-to-end server smoke test (via `/v1/chat/completions`, 40-token reply,
run with the old `-t 12` default before this tuning): 2.2 s wall time,
~74 t/s prompt processing, ~23 t/s generation — single short request, so
higher than the steady-state `llama-bench` numbers above; re-run after
switching to `-t 8` if precise server-side numbers matter.
