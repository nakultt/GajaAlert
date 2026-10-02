# Arduino UNO Q (QRB2210) Deployment Guide

## Hardware Context

| Spec | Value |
|------|-------|
| SoC | Qualcomm Dragonwing QRB2210 |
| CPU | Quad-core Cortex-A53 @ 1.0 GHz |
| RAM | 1 GB |
| NPU | **None** (entry-tier IoT chip) |
| OS | Debian Linux (aarch64) |
| Storage | SD card via Arduino shield |

> **Key constraint:** No NPU, no GPU. All inference runs on the CPU. 
> Every millisecond of compute matters.

---

## Step 1: Profile on Your Laptop First

Before deploying anything, measure baseline inference time on x86:

```bash
# Single file with timing breakdown
python test_elephant_classifier.py --profile dataset/data/test/Roar/Roar01.wav

# Full batch with timing stats
python test_elephant_classifier.py --profile --eval dataset/data/test/
```

**Expected laptop timing (per 6s clip):**
- Preprocessing (mel-spec): ~50-100 ms
- YAMNet ONNX: ~200-400 ms
- XGBoost: ~1-5 ms
- **Total: ~300-500 ms**

**Expected UNO Q timing (Cortex-A53, ~10x slower):**
- Total: **~3-5 seconds per 6s clip**

This is acceptable — record 6s, classify in ~4s, total loop ~10s.

---

## Step 2: Quantize YAMNet ONNX to INT8

Reduces model size and speeds up inference on ARM:

```bash
python test_elephant_classifier.py --quantize
```

This creates `output/yamnet_int8.onnx`. Expected results:
- Size: 12.3 MB → ~4 MB (3x smaller)
- Speed: ~1.5-2x faster on Cortex-A53
- Accuracy: <2% drop (verify with Step 3)

---

## Step 3: Cross-Validate Quantized vs Original

```bash
# Run eval with original model
python test_elephant_classifier.py --eval --profile dataset/ > results_fp32.txt

# Run eval with quantized model
python test_elephant_classifier.py --eval --profile --quantized dataset/ > results_int8.txt

# Compare F1 scores manually
```

If F1 drops more than 2%, stay with float32. The size difference won't matter
since the UNO Q has 1 GB RAM (both models fit easily).

---

## Step 4: Deploy to UNO Q

### 4a. Files to copy to the UNO Q

```
/home/user/elephant/
├── input_processing.py          # Preprocessing module
├── test_elephant_classifier.py  # Inference script
└── output/
    ├── yamnet.onnx              # (or yamnet_int8.onnx)
    ├── yamnet.onnx.data         # External weights
    └── elephant_xgb.json        # XGBoost classifier
```

### 4b. Install Python dependencies on UNO Q

```bash
# On the UNO Q (Debian aarch64)
sudo apt update
sudo apt install python3 python3-pip libsndfile1

pip3 install onnxruntime xgboost librosa soundfile numpy
```

> **Note:** `onnxruntime` has official aarch64 wheels. If pip fails,
> try: `pip3 install onnxruntime --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime/pypi/simple/`

### 4c. Test on UNO Q

```bash
# Quick smoke test
python3 test_elephant_classifier.py --profile /path/to/test_audio.wav

# Batch test
python3 test_elephant_classifier.py --eval --profile /path/to/test_folder/
```

---

## Step 5: End-to-End Pipeline

The full pipeline on the UNO Q:

```
┌──────────────────────────────────────────────────┐
│  Arduino Shield (I2S Mic)                        │
│  → Records 6s of PCM audio                      │
│  → Saves as /sdcard/recording.wav                │
└────────────────────┬─────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────┐
│  Python Inference Daemon (runs on QRB2210)       │
│                                                  │
│  1. Watch /sdcard/ for new .wav files            │
│  2. Load audio → mel-spectrogram → patches       │
│  3. YAMNet ONNX → 1024-d embeddings             │
│  4. XGBoost → elephant probability              │
│  5. If prob > threshold → trigger alert          │
│                                                  │
│  Latency budget: ~4s for 6s clip                 │
└────────────────────┬─────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────┐
│  Alert System                                    │
│  → WebSocket to laptop visual hub                │
│  → GPIO signal to Arduino LED/buzzer             │
│  → Log to file for later analysis                │
└──────────────────────────────────────────────────┘
```

### Example Daemon Script

Create `elephant_daemon.py` on the UNO Q:

```python
#!/usr/bin/env python3
"""
elephant_daemon.py — Watches for new WAV files and classifies them.
Runs continuously on the Arduino UNO Q.
"""
import os, sys, time, glob

# Add the project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from input_processing import audio_to_patches
import onnxruntime as ort
from xgboost import XGBClassifier
import numpy as np

# Config
WATCH_DIR = "/sdcard/"       # Where Arduino saves recordings
THRESHOLD = 0.5              # Tune this based on your test results
POLL_INTERVAL = 1.0          # Check for new files every 1 second
ONNX_MODEL = "output/yamnet.onnx"  # or yamnet_int8.onnx
XGB_MODEL = "output/elephant_xgb.json"

# Load models once at startup
print("Loading models...")
yamnet = ort.InferenceSession(ONNX_MODEL, providers=["CPUExecutionProvider"])
clf = XGBClassifier()
clf.load_model(XGB_MODEL)
print("Models loaded. Watching for audio files...")

processed = set()

while True:
    wav_files = glob.glob(os.path.join(WATCH_DIR, "*.wav"))
    
    for wav in wav_files:
        if wav in processed:
            continue
        
        try:
            print(f"\n[{time.strftime('%H:%M:%S')}] New file: {os.path.basename(wav)}")
            t0 = time.perf_counter()
            
            patches = audio_to_patches(wav)
            embeddings = yamnet.run(["clip_embedding"], {"log_mel_patches": patches})[0]
            probs = clf.predict_proba(embeddings)[:, 1]
            avg_prob = float(np.mean(probs))
            
            elapsed = (time.perf_counter() - t0) * 1000
            
            if avg_prob > THRESHOLD:
                print(f"  🐘 ELEPHANT DETECTED! Confidence: {avg_prob:.2%} ({elapsed:.0f}ms)")
                # TODO: Trigger alert (WebSocket, GPIO, etc.)
            else:
                print(f"  ✓ Non-elephant. Confidence: {avg_prob:.2%} ({elapsed:.0f}ms)")
            
            processed.add(wav)
            
        except Exception as e:
            print(f"  Error processing {wav}: {e}")
            processed.add(wav)
    
    time.sleep(POLL_INTERVAL)
```

---

## Performance Tips for Cortex-A53

1. **Use INT8 quantization** — saves ~40% inference time on ARM
2. **Process only 1 patch at a time** if memory is tight (unlikely with 1 GB)
3. **Set OMP threads:**
   ```bash
   export OMP_NUM_THREADS=4  # Use all 4 Cortex-A53 cores
   ```
4. **Disable librosa's cache** to save disk I/O:
   ```python
   os.environ["LIBROSA_CACHE_DIR"] = ""
   ```
5. **Pre-warm the model** — first inference is always slower due to JIT compilation

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `onnxruntime` won't install on aarch64 | Use `pip install onnxruntime` — official ARM64 wheels exist since v1.14 |
| `librosa` is slow on ARM | The bottleneck is FFT. Consider `scipy.fft` backend or reduce `n_fft` |
| XGBoost segfaults | Ensure you have `libgomp1`: `sudo apt install libgomp1` |
| Model files too large for SD card | Use INT8 quantized model (~4 MB vs 12 MB) |
| Inference too slow (>10s) | Profile with `--profile`, consider reducing patch overlap |
