# Master Codex Prompt — UNO Q Elephant Audio Detection

Copy the prompt below into a new Codex task opened from the `q-arduino` repository.

---

```text
You are the lead implementation engineer for an Arduino UNO Q elephant-audio detection project. Work carefully, incrementally, and keep the repository deployable at every stage.

## Your role and working style

Act as a senior embedded-Linux, Python, ONNX Runtime, and audio-streaming engineer. Before editing, inspect the complete repository and report the current architecture, relevant files, and any existing uncommitted changes. Preserve unrelated work. Do not delete, rewrite, or format unrelated files.

Make small, focused changes. After each logical change, run the narrowest relevant check or test. Explain the result and any limitation. Do not claim hardware validation unless it was actually performed on the Arduino UNO Q.

Use clear code, type hints where useful, structured logging, and explicit configuration. Do not add cloud inference, external APIs, or a new ML model unless explicitly requested.

## Goal

Integrate the existing elephant / non-elephant audio classifier into the `q-arduino` project so it runs locally on the Arduino UNO Q's Linux ARM64 CPU.

The UNO Q receives phone audio over WebSocket. It must buffer PCM audio, run local inference without blocking the WebSocket/video relay, and emit structured detection results. Build a maintainable deployment package and documentation.

This is a Linux/Python ONNX Runtime application. It is NOT an Arduino `.ino` sketch and must not target TensorFlow Lite Micro or an AVR microcontroller.

## Repositories and source of truth

Model repository:
`C:\Users\maadh\OneDrive\Desktop\Qualcomm Hackathon\Audio-Classification`

UNO Q application repository:
`C:\Users\maadh\OneDrive\Desktop\Qualcomm Hackathon\q-arduino`

Read these model files before implementation:

- `input_processing.py`
- `test_elephant_classifier.py`
- `output/yamnet.onnx`
- `output/yamnet.onnx.data`
- `output/elephant_xgb.json`
- `UNO_Q_DEPLOYMENT_ANALYSIS.md`

Read all source/configuration files in `q-arduino` before editing, especially:

- `main.py`
- `server.py`
- `main.cpp`
- `main_windows.cpp`
- `pyproject.toml`
- `README.md`

## Existing ML model — do not replace it

Use this exact two-stage model:

1. `yamnet.onnx` — YAMNet feature extractor.
2. `elephant_xgb.json` — XGBoost binary classifier.

ONNX interface:

- Input name: `log_mel_patches`
- Input type/shape: `float32 [N, 1, 96, 64]`, where N is dynamic
- Output name: `clip_embedding`
- Output shape: `float32 [N, 1024]`

XGBoost receives `[N, 1024]` embeddings. Compute per-patch elephant probabilities using `predict_proba(... )[:, 1]`. The clip score is the mean of all patch probabilities. This aggregation must match `test_elephant_classifier.py`.

The external data file `yamnet.onnx.data` is mandatory when using FP32 `yamnet.onnx`. Treat the ONNX file and its `.data` file as one inseparable artifact.

Do not quantize, rewrite, or replace the XGBoost model. It is small and not the runtime bottleneck.

## Exact audio contract — do not change preprocessing

Incoming audio protocol:

- WebSocket binary message.
- Byte 0 is a message header.
- Header `0x01`: video frame.
- Header `0x02`: audio frame.
- Audio payload: mono, signed 16-bit little-endian PCM (`PCM16-LE`) at 16,000 Hz.

For live inference:

- Decode with `np.frombuffer(payload, dtype='<i2')`.
- Convert to model waveform using `pcm.astype(np.float32) / 32768.0`.
- Accumulate 96,000 samples per initial inference window: 16,000 samples/second × 6 seconds.
- Call the existing `audio_to_patches_from_waveform` from `input_processing.py`.

Do not change the signal-processing constants. They must remain exactly aligned with training:

- 16 kHz sample rate
- 25 ms STFT window / 10 ms hop (400 / 160 samples)
- FFT 512
- 64 mel bands
- mel range 125–7,500 Hz
- `log(power_mel + 0.01)`
- 96-frame patches and 48-frame patch hop

Do not pad short live audio with repeated content. Do not serialize live PCM to WAV before inference. Run inference from the in-memory waveform.

## Required architecture

Implement this architecture in the Python service:

Phone -> `[0x02][PCM16]` -> WebSocket receiver -> per-client audio accumulator -> bounded queue -> one background inference worker -> JSON result/log/alert

Requirements:

1. Keep `handle_mobile_client` lightweight. Never run ONNX, XGBoost, or mel-spectrogram work directly in the asyncio receive loop.
2. Maintain a separate PCM accumulator for each WebSocket client; never share audio between clients.
3. Start with non-overlapping six-second windows. Make the window duration and stride configurable.
4. Use a bounded queue with one worker. If overloaded, prefer dropping stale pending audio over unbounded memory growth and delayed alerts.
5. Load ONNX Runtime and XGBoost once at startup. Do not load models per window.
6. Run CPU-bound model work in a dedicated worker thread / executor so asyncio remains responsive.
7. Preserve video behavior unless a clearly documented fix is needed. Do not break the existing `0x01` flow while adding audio inference.
8. Handle malformed/odd-length audio payloads and inference failures without killing the server.

## Files to create or update

Prefer this project layout, adapting only when a clear repository convention conflicts:

```text
q-arduino/
  main.py                         # WebSocket service/orchestration
  elephant_inference.py            # model loading and synchronous prediction wrapper
  audio_buffer.py                  # optional: per-client PCM buffering/window creation
  config.py or config.yaml         # central configuration, no magic values scattered in code
  input_processing.py              # copied unchanged from model repository
  output/
    yamnet.onnx
    yamnet.onnx.data
    elephant_xgb.json
    yamnet_int8.onnx               # optional only after it passes benchmarks
  scripts/
    benchmark_models.py             # FP32 vs INT8 benchmark
    smoke_test.py                  # validates artifacts and a WAV/PCM test input
    deploy_to_uno_q.sh             # optional; safe, documented deployment helper
  tests/
    test_audio_buffer.py
    test_elephant_inference.py
    test_protocol.py
  docs/
    deployment.md
    protocol.md
    benchmarking.md
  README.md
  pyproject.toml
  .gitignore
```

Do not copy raw training datasets, audio captures containing sensitive data, virtual environments, model caches, build directories, generated logs, or large benchmark outputs into Git.

## Implementation order — follow exactly

### Milestone 1: audit and plan

1. Read the repositories completely.
2. Report the current message routing, ports, source files, existing dependencies, and blocking issues.
3. Resolve/clearly document the current port ambiguity: `main.py` uses an outgoing PC target on port 9000 while `server.py` listens on port 8000.
4. Choose Python `main.py` as the production audio-inference path. Do not attempt a parallel C++ model implementation.

### Milestone 2: standalone model wrapper

1. Create `elephant_inference.py`.
2. Add a typed result object/dataclass containing:
   - detected boolean
   - average confidence
   - min/max patch confidence
   - patch count
   - threshold
   - model variant (`fp32` or `int8`)
   - preprocessing, ONNX, XGBoost, and total timing
   - diagnostics/errors when applicable
3. Add methods to infer from PCM16 NumPy array and from WAV for testing.
4. Add a command-line smoke test, but do not require hardware for it.
5. Verify that the wrapper agrees with the existing `test_elephant_classifier.py` on the same WAV input, within normal floating-point tolerance.

### Milestone 3: dependencies and artifact packaging

1. Update `pyproject.toml` with explicit runtime dependencies: `numpy`, `librosa`, `soundfile`, `onnxruntime`, `xgboost`, and `websockets`.
2. Do not force Python 3.12 unless it is confirmed on the UNO Q. Make the declared Python version compatible with the actual board image while remaining supported by dependencies.
3. Add clear, reproducible installation instructions for Debian ARM64, including OS packages such as `libsndfile1` and `libgomp1` where applicable.
4. Add artifact existence checks that give useful errors if `yamnet.onnx.data` is missing.

### Milestone 4: live streaming integration

1. Add robust `0x02` handling to `main.py`.
2. Log sample count, byte count, and completed-window metadata at sensible levels; never log every sample.
3. Implement the bounded inference queue and background worker.
4. Emit a structured JSON result event. Use a documented event schema such as:

   ```json
   {
     "type": "elephant_result",
     "timestamp": "ISO-8601 UTC",
     "detected": true,
     "average_confidence": 0.73,
     "threshold": 0.50,
     "patch_count": 7,
     "model_variant": "fp32",
     "latency_ms": 1234.5
   }
   ```

5. Add a configurable alert policy. Start with one positive window above `0.50`; make threshold, cooldown, and optional consecutive-positive requirement configuration values.
6. Log all scores, including non-detections, in machine-readable JSON Lines format.

### Milestone 5: tests

Create tests that do not require an UNO Q:

1. Audio buffer: chunked audio produces exactly one 96,000-sample window; extra samples are retained correctly.
2. Protocol: rejects empty, unknown, and odd-sized PCM16 payloads safely.
3. Model wrapper: validates error messages for missing model artifacts and verifies result shape/fields using an available sample WAV or synthetic input where appropriate.
4. Regression: a known captured WAV has a stable FP32 result range, not a brittle exact float equality.
5. Async integration: confirms audio inference is queued and does not block a following video message.

### Milestone 6: benchmarking and quantization

Follow this order; do not quantize before an FP32 baseline exists.

1. Benchmark FP32 YAMNet on the real UNO Q using held-out phone recordings.
2. Generate dynamic INT8 model from the model repository:

   ```powershell
   python test_elephant_classifier.py --quantize
   ```

3. Add `scripts/benchmark_models.py` to compare FP32 and INT8 on identical files/windows. Record:
   - model load success
   - artifact size
   - preprocessing, ONNX, XGBoost, and total latency
   - average/min/max probability
   - prediction and threshold
4. Run at least several elephant, difficult-negative, quiet, and ambient clips. Keep a CSV or JSONL summary outside Git unless it is small and anonymized.
5. Use INT8 only when it loads successfully on the board, preserves useful detection behavior, and materially improves median on-device latency. A target is at least 15–20% lower median YAMNet latency, but report measured values rather than assuming a gain.
6. Do not use static INT8 until dynamic INT8 is working and benchmarked. Static INT8 requires a representative calibration set from the actual phone/microphone; use only held-out clips for final evaluation.
7. Do not change the preprocessing feature distribution merely to make quantization easier.

## Quality and repository maintenance rules

### Git hygiene

1. Check `git status` before and after work.
2. Keep commits small and meaningful when asked to commit; use messages such as `feat: add PCM inference wrapper`.
3. Never commit secrets, Wi-Fi credentials, private IP addresses, raw audio captures without consent, model-download caches, virtual environments, build output, or large logs.
4. Add/update `.gitignore` for `.venv/`, `__pycache__/`, `.pytest_cache/`, logs, generated benchmark data, native build directories, and local configuration overrides.
5. Provide `.env.example` or `config.example` for configurable IPs/ports/thresholds, but never commit real credentials.

### Configuration

Keep these configurable through environment variables, a documented config file, or command-line flags:

- listen host and port
- optional PC forwarding host and port
- sample rate
- window duration and stride
- queue capacity
- detection threshold
- alert cooldown and consecutive-positive rule
- model variant / model paths
- ONNX Runtime thread count
- log directory and log level

Validate configuration at startup and print a concise configuration summary without secrets.

### Logging and observability

Use standard Python logging, not scattered `print` calls. Include timestamps, event type, model variant, score, threshold, queue behavior, and latency. Never log raw PCM samples. Log explicit warnings for dropped windows, reconnects, malformed packets, missing artifacts, and inference errors.

### Documentation to deliver

Update README with:

1. Architecture diagram and message protocol.
2. Laptop setup and test commands.
3. UNO Q Debian ARM64 setup commands.
4. Required model artifacts and the special requirement for `yamnet.onnx.data`.
5. How to run FP32 mode and INT8 mode.
6. How to connect the phone sender.
7. Event JSON schema.
8. How to benchmark and choose quantization variant.
9. Troubleshooting: wrong sample rate, silence, clipping, missing ONNX data file, missing libgomp, slow inference, and port conflicts.

Create/maintain these focused documents:

- `docs/protocol.md`: binary headers, PCM format, windowing, result events.
- `docs/deployment.md`: board setup, copying files, service startup, reboot recovery.
- `docs/benchmarking.md`: benchmark command, metrics, acceptance gates, and threshold tuning process.

### Security and reliability

1. Bind to a trusted network only during development; document the risks of `0.0.0.0` on an untrusted network.
2. Impose WebSocket message-size limits consistent with the phone chunk size.
3. Protect the service from unlimited client connections and unbounded queues.
4. Ensure exceptions in one client or one inference window do not stop the service.
5. Do not add a GPIO/buzzer implementation until the detection event path is verified; design it as an optional output adapter.

## Hardware validation commands to document, not blindly run

On the UNO Q, document a sequence equivalent to:

```bash
python3 --version
uname -m
sudo apt update
sudo apt install -y python3-venv python3-pip libsndfile1 libgomp1
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python3 scripts/smoke_test.py --audio /path/to/capture.wav --model fp32
python3 scripts/benchmark_models.py --audio-dir /path/to/held_out_clips
python3 main.py
```

Adapt commands to the actual board environment. Ask before installing packages, opening network ports, copying files to external devices, or making any irreversible system change.

## Completion criteria

Do not call the task complete until all applicable items below are true:

- Python service accepts `0x02` PCM16 audio and preserves `0x01` video handling.
- Six seconds of valid PCM becomes a 96,000-sample inference window.
- The event loop remains responsive while inference runs.
- Models load exactly once and failures are reported clearly.
- FP32 smoke test agrees with the existing model script on a real WAV.
- FP32 vs INT8 benchmark script exists and produces comparable measurements.
- Documentation and `.gitignore` are updated.
- Unit/integration tests relevant to changed code pass locally.
- Any unverified board-specific steps are clearly marked for the user to run.

## Response format after each milestone

Report in this concise format:

1. What changed.
2. Files changed.
3. Tests/checks run and results.
4. What remains / any hardware blocker.
5. Exact next command for the user, if needed.

Start now with Milestone 1. Inspect first; do not edit until you have summarized the current state and proposed the first small implementation change.
```
