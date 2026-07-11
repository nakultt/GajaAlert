# Elephant Audio Classifier on Arduino UNO Q

## Purpose

This document is the implementation plan for running the existing elephant / non-elephant classifier locally on an Arduino UNO Q. It is based on a complete source review of this repository and of `C:\Users\maadh\OneDrive\Desktop\Qualcomm Hackathon\q-arduino` on 12 July 2026.

The intended deployment is **Linux application inference on the UNO Q's Cortex-A53 CPU**. It is not an AVR Arduino-sketch or TensorFlow Lite Micro deployment.

## Executive decision

Use the Python WebSocket server in `q-arduino/main.py` as the production service. Add a Python inference worker to it. Keep the C++ relay only if it is needed for a separate video-only experiment; do not attempt to embed the Python ONNX/XGBoost pipeline inside it for the first working version.

Reasons:

* The trained runtime is already Python: NumPy, librosa, ONNX Runtime, and XGBoost.
* The model is too large and depends on operations unsuitable for a conventional Arduino microcontroller deployment.
* The UNO Q provides Debian Linux, ARM64 CPU, and sufficient RAM for the present model.
* A single Python service avoids a second network hop or IPC protocol between the relay and the classifier.

## What this repository contains

### Runtime model

The local inference pipeline is split correctly into a frozen feature extractor and a lightweight classifier:

| Artifact | Size | Role |
|---|---:|---|
| `output/yamnet.onnx` | 124,884 B | ONNX graph |
| `output/yamnet.onnx.data` | 12,845,056 B | External ONNX weights; required alongside the graph |
| `output/elephant_xgb.json` | 124,499 B | XGBoost elephant/non-elephant classifier |

`yamnet.onnx` input is `log_mel_patches` with type `float32` and shape `[N, 1, 96, 64]`, where `N` is dynamic. Its output is `clip_embedding`, shape `[N, 1024]`. The XGBoost model consumes each 1,024-element embedding and returns an elephant probability. `test_elephant_classifier.py` averages the per-patch probabilities to make the clip decision.

### Audio contract imposed by the model

`input_processing.py` reproduces the training transform. It must remain identical in deployment:

| Parameter | Required value |
|---|---:|
| PCM sample rate | 16,000 Hz |
| Channels | mono |
| PCM representation | signed 16-bit little-endian at transport boundary |
| Inference waveform type | `float32`, normally PCM divided by `32768.0` |
| STFT window / hop | 25 ms / 10 ms (400 / 160 samples) |
| FFT length | 512 |
| Mel bands | 64 |
| Mel frequency range | 125–7,500 Hz |
| Log stabilizer | `log(power_mel + 0.01)` |
| Patch dimensions | 96 x 64 mel frames |
| Patch hop | 48 frames |

Six seconds is the appropriate initial capture window. At 16 kHz it is exactly **96,000 samples** or **192,000 audio bytes** before the one-byte WebSocket message header. It produces approximately six to seven overlapping model patches.

### Training observations

`train_elephant_classifier.py` trains XGBoost on individual YAMNet patches from elephant clips and FSC22 negative clips. The ONNX export uses dynamic batch size and validates against the PyTorch embedding extractor.

Important limitation: splitting occurs at the **patch** level, not recording level. Overlapping patches from the same original recording may enter both training and test splits. Therefore offline test metrics can be optimistic. The deployment decision must be based on a separate set of phone-recorded clips that were never used for training or threshold selection.

The scripts under `audio preprocessing/` are dataset-preparation utilities. They do not belong in the live deployment path. Do not duplicate or pad a live recording; use real consecutive audio. Padding a short live segment with silence or repeating it changes the classifier's patch distribution.

## Current q-arduino code review

### `main.py`: best starting point

It listens for a mobile client on port 8000 and expects binary frames whose first byte is a type header:

* `0x01`: video.
* `0x02`: audio.

The updated code has restored the `0x02` branch, but `run_audio_inference` is only a print stub. It handles individual chunks rather than a six-second window. It also forwards only video messages to the ARM-PC socket; audio is intentionally not forwarded by this Python service.

This is the correct place to add the model, but model inference must be moved out of the asynchronous receive loop. A long inference in `handle_mobile_client` would stop the service from accepting further audio and video packets.

### `main.cpp` and `main_windows.cpp`

These contain a uWebSockets server plus an ixWebSocket client. They recognize `0x02`, but all messages are still forwarded because `ix_ws.sendBinary(...)` is outside the header branch. The comment saying audio is not forwarded is inaccurate.

The CMake files name the targets `edge_server_windows`, so they are currently Windows-oriented. `main.cpp` has a capitalized `uWebsockets/App.h` include while `main_windows.cpp` uses `uwebsockets/App.h`; this can matter on case-sensitive Linux filesystems. Neither C++ executable invokes the classifier.

### `server.py`

This is a Windows visualizer for video frames, not a UNO Q runtime: it imports OpenCV and opens a GUI window. It currently calls `threading.Thread` without importing `threading`, so it will fail as written. It only handles `0x01` video, and it listens on port 8000.

### Configuration

`pyproject.toml` currently declares only `websockets`. The model deployment needs additional pinned runtime dependencies. The UNO Q project includes `.python-version`, but the Python requirement must agree with the actual UNO Q OS. Do not assume a distribution provides Python 3.12; verify with `python3 --version` before locking dependencies.

### Network ambiguity to resolve

`main.py` connects to an ARM-PC target at `ws://<ip>:9000`, whereas `server.py` listens on 8000. Decide which process owns port 9000. For the first end-to-end audio milestone, the PC forwarder can be disabled entirely: prove phone -> UNO Q -> local detection first.

## Target design

```text
Phone microphone
  -> Binary WebSocket frame [header=0x02][PCM16-LE, mono, 16 kHz]
  -> q-arduino/main.py receiver
  -> per-client PCM accumulator (96,000 samples)
  -> bounded inference queue (one pending window maximum)
  -> one background worker
       PCM16 -> float32 waveform -> existing YAMNet preprocessing
       YAMNet ONNX Runtime -> 1024-D patch embeddings
       XGBoost -> patch probabilities -> aggregate clip probability
  -> detector policy and structured JSON event
  -> logs, optional PC forwarding, optional LED/buzzer/GPIO
```

### Why a bounded queue

The CPU may take longer to infer than a capture interval. A bounded queue prevents memory growth and stale detections. Use `Queue(maxsize=1)` initially. If the worker is busy, either discard the new completed window or keep the newest window; for a real-time alert system, keeping the newest is preferable.

### Windowing policy

Start with non-overlapping six-second windows: classify samples 0–95,999, then 96,000–191,999. This is simple and makes latency easy to measure.

After correctness is established, consider a sliding window only if needed for faster detection:

* window: six seconds;
* stride: three seconds;
* retain the last three seconds in the accumulator.

Do not use a stride shorter than inference can sustain. The model already has overlapping mel patches internally, so a three-second application-level stride is usually enough.

## Implementation phases

### Phase 0 — establish the transport contract

Deliverable: a capture log proving the phone format is correct.

1. Confirm the mobile sender emits `0x02` frames containing PCM16 little-endian, not AAC/Opus/WAV bytes.
2. Log each audio chunk's byte count. It must be even; an odd length means it cannot be PCM16 samples.
3. Convert chunks using `np.frombuffer(payload, dtype='<i2')`.
4. Count samples for a known duration. A six-second capture must contain approximately 96,000 samples. If it does not, resolve sample rate or dropped-chunk issues before touching the model.
5. Save several raw captures as WAV files during development and listen to them on a laptop. Silence, swapped endianness, clipping, or wrong sample rate must be fixed here.

### Phase 1 — run the original FP32 model offline on the UNO Q

Deliverable: one known WAV classifies locally on the board.

1. Copy `input_processing.py`, `test_elephant_classifier.py`, and all three artifacts under `output/` to one directory on the UNO Q.
2. Install Linux prerequisites: `libgomp1` (XGBoost), `libsndfile1` (soundfile), and Python build/runtime packages as required by the board image.
3. Install Python packages compatible with the board's Python version: `numpy`, `librosa`, `soundfile`, `onnxruntime`, `xgboost`, and `websockets`.
4. Run `python3 test_elephant_classifier.py --profile known.wav` on the board.
5. Record total, preprocessing, YAMNet, and XGBoost latency. This is the baseline; do not compare a later quantized run against laptop timing.

### Phase 2 — package inference as a reusable module

Deliverable: `elephant_inference.py` in `q-arduino`.

The module should:

* load ONNX Runtime and XGBoost once in a constructor;
* expose `predict_pcm16(pcm: np.ndarray) -> DetectionResult`;
* reject sample arrays that are empty, non-mono, or unexpectedly short;
* normalize PCM with `/ 32768.0`;
* call `audio_to_patches_from_waveform`, not a file-based API;
* run the ONNX input/output names exactly as exported;
* average patch probabilities consistently with `test_elephant_classifier.py`;
* return probability, max/min patch probability, patch count, timing, model variant, and diagnostics.

Do not reload either model per six-second window.

### Phase 3 — integrate the worker in `main.py`

Deliverable: streaming audio produces local detection events without disrupting video packets.

1. Initialize one accumulator per WebSocket client, not one global buffer shared between phones.
2. On `0x02`, append decoded samples. Never invoke model code directly from the receive loop.
3. When at least 96,000 samples are available, copy the window and enqueue it. Remove consumed samples according to the chosen stride.
4. Run a single dedicated worker via `asyncio.to_thread` or a `ThreadPoolExecutor(max_workers=1)`. ONNX/XGBoost work is CPU-bound and must not block asyncio.
5. Emit JSON such as `{"type":"elephant_result","probability":0.73,"detected":true,"window_seconds":6,"latency_ms":...}`. Only send after inference completes.
6. Add reconnection and error logging, while ensuring a malformed packet does not terminate the server.

### Phase 4 — detection policy

Deliverable: practical alerts rather than noisy per-window labels.

Initial rule:

* `detected = average_probability > threshold`, starting at `0.50`.

Recommended production rule after field tests:

* require two positive windows within 30 seconds, or one very strong window above a higher emergency threshold;
* add a cooldown (for example 60 seconds) after an alert;
* log every score, including non-detections, for later threshold tuning;
* keep raw audio only with user permission and a retention policy.

## Quantization strategy

### What to quantize

Quantize only YAMNet. It is the compute and storage bottleneck.

Do **not** spend effort quantizing the XGBoost JSON model. It is around 124 KB and tree evaluation is already a small fraction of end-to-end latency. Converting it to custom C++ would increase risk and provide little benefit.

Preprocessing remains float32. The log-mel values must match training; changing them to integer arithmetic is a separate model-validation project, not an initial optimization.

### Experiment A — FP32 baseline (mandatory)

This is the source of truth. Profile the original `yamnet.onnx` on the actual UNO Q with at least:

* elephant phone recordings;
* ordinary outdoor/forest ambience;
* loud negative sounds likely to cause false positives;
* silence and quiet recordings;
* at least 20–30 clips per class if time permits.

Save each result's model variant, average probability, label, and latency.

### Experiment B — dynamic INT8 (recommended first quantized version)

Dynamic quantization requires no calibration set and is the fastest way to test whether ARM CPU inference improves. From this repository:

```powershell
python test_elephant_classifier.py --quantize
```

This creates `output/yamnet_int8.onnx`. Copy that file to the Uno Q and benchmark it with:

```bash
python3 test_elephant_classifier.py --profile capture.wav
python3 test_elephant_classifier.py --quantized --profile capture.wav
```

Record model loading success, artifact size, total latency, YAMNet latency, and every predicted probability. Dynamic quantization may accelerate matrix-heavy operations but gives variable improvement for convolution-heavy networks such as YAMNet. Measure; do not assume the advertised speedup.

Use the INT8 file only if all conditions hold:

1. ONNX Runtime loads it on the UNO Q ARM64 image.
2. It reduces median YAMNet time materially (target: at least 15–20%).
3. Validation-set recall and false-positive rate remain acceptable.
4. No important field clip changes from a confident correct result to an incorrect one.

### Experiment C — static INT8 / QDQ (only after dynamic INT8)

Static quantization can quantize activations as well as weights and may improve CPU performance further. It needs a representative calibration set. Build that set from actual phone PCM captures, including quiet, loud, elephant, background, and difficult negative conditions.

Recommended approach:

1. Create 100–300 calibration clips of six seconds each, distinct from final test clips.
2. Apply the unchanged `audio_to_patches_from_waveform` function to make calibration tensors.
3. Use ONNX Runtime static quantization with QDQ format, per-channel weights, and a calibration method such as MinMax first. Quantize only supported Conv/MatMul/Gemm operations; leave fragile operations FP32.
4. Compare static INT8 against FP32 and dynamic INT8 using the same locked test corpus.
5. Accept static INT8 only if it is faster than dynamic INT8 and stays inside your accuracy/recall limits.

Static quantization is not a prerequisite for the demo. It is an optimization experiment with a real possibility of score drift.

### Accuracy gates

Define success before benchmarking:

| Gate | FP32 reference | Quantized acceptance criterion |
|---|---|---|
| Clip label agreement | FP32 results | >= 98% agreement on representative test clips |
| Elephant recall | measured on unseen clips | no more than 2 percentage-point drop |
| False positives | measured on difficult negatives | no material increase |
| Median YAMNet time | measured on UNO Q | dynamic/static version must improve it materially |
| Stability | repeated runs | no ONNX Runtime errors or memory growth |

For safety-oriented wildlife detection, preserve recall even if it means a few more alerts. The exact threshold should be selected separately for FP32 and INT8 because the probability distribution can shift slightly.

## Performance work after quantization

Profile first. In this pipeline, librosa mel extraction can be a major portion of total time on Cortex-A53. If YAMNet is no longer the bottleneck after INT8, the next improvements are:

1. Keep model sessions warm and run one dummy patch at startup.
2. Set a sensible thread count for ONNX Runtime; begin with 2–4 CPU threads and benchmark. More threads are not always faster on a small A53 system.
3. Reuse buffers and avoid WAV serialization in live inference.
4. Replace only the live preprocessing implementation with a validated SciPy/NumPy implementation if librosa dominates. Verify resulting patches and scores against `input_processing.py` before switching.
5. Use non-overlapping windows until inference is comfortably faster than capture. Then introduce a three-second stride if faster detection is required.

Avoid processing one YAMNet patch at a time: batching all patches from a six-second window is normally simpler and more efficient with 1 GB RAM.

## Deployment package

The final `q-arduino` runtime directory should contain:

```text
q-arduino/
  main.py                       # WebSocket receiver and worker orchestration
  elephant_inference.py          # model wrapper
  input_processing.py            # identical transform from this repository
  output/
    yamnet.onnx                  # keep for fallback and baseline
    yamnet.onnx.data             # mandatory with FP32 model
    yamnet_int8.onnx             # only after validation
    elephant_xgb.json
  pyproject.toml                 # complete runtime dependencies
  logs/
```

Model path handling should use paths relative to the source file, not the process's working directory.

## Definition of done

The deployment is ready for a demo when all of the following are true:

1. The UNO Q receives valid 16 kHz PCM16 audio from the phone continuously.
2. A six-second accumulated window produces exactly the same FP32 result as an equivalent WAV file on the laptop within normal floating-point tolerance.
3. Video transport remains responsive while audio inference runs.
4. All model artifacts load after a cold reboot.
5. The selected quantization variant has been benchmarked on-device and selected from evidence.
6. Alert events include timestamp, score, threshold, model variant, and latency.
7. The threshold/detection policy has been field-tested on unseen phone recordings.

## Recommended immediate next action

Implement Phase 0 and Phase 1 before any quantization work: capture a six-second PCM window from the phone, save it as a diagnostic WAV, and run the unmodified FP32 classifier on both laptop and UNO Q. Once the audio contract is proven, add the reusable inference module and background worker. Quantization then becomes a controlled benchmark instead of a debugging variable.
