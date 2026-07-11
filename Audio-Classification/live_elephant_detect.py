#!/usr/bin/env python3
"""
live_elephant_detect.py — 24x7 live microphone elephant-sound detector.

Continuously captures audio from the default input device, runs it through the
same YAMNet -> XGBoost pipeline used by test_elephant_classifier.py, and reports
whether an elephant sound was detected. Every decision is printed to the terminal
and appended to a log file.

Detection logic (important)
---------------------------
Elephant calls (trumpet / roar / rumble) are SHORT events — often a single
0.96 s YAMNet patch inside a multi-second window. Averaging patch probabilities
therefore dilutes a strong 1 s elephant call down below threshold and misses it.
So this detector scores each patch and pools with **max** ("fire if ANY patch
looks like an elephant"), which is the correct rule for event detection.

To suppress momentary false positives, a real detection must PERSIST: the
detector requires M of the last K windows to fire (temporal debounce) before it
declares a confirmed detection. A genuine elephant vocalisation lasts longer than
one hop; an isolated spurious spike does not.

Pipeline
--------
- Audio captured at 16 kHz mono (YAMNet's native rate) via sounddevice — no
  resampling, no temporary .wav round-trip.
- Rolling window (default 3.0 s) slides forward by a hop (default 1.0 s).
- Capture thread and inference loop are decoupled by a queue so slow inference
  never drops microphone data.
- Runs forever until Ctrl+C, resilient to audio overflows / inference errors.

Usage
-----
    python live_elephant_detect.py
    python live_elephant_detect.py --threshold 0.5 --pooling max
    python live_elephant_detect.py --debounce-m 2 --debounce-k 3
    python live_elephant_detect.py --list-devices
    python live_elephant_detect.py --device 2 --save-detections

Requirements
------------
    pip install onnxruntime xgboost librosa soundfile numpy sounddevice
"""

import argparse
import datetime as _dt
import json
import os
import queue
import sys
import threading
from collections import deque

import numpy as np

# Local pipeline (same directory)
from input_processing import (
    audio_to_patches_from_waveform,
    diagnose_audio,
    SAMPLE_RATE,
)

# --- Paths --------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
YAMNET_ONNX = os.path.join(OUTPUT_DIR, "yamnet.onnx")
YAMNET_QUANTIZED = os.path.join(OUTPUT_DIR, "yamnet_int8.onnx")
XGB_MODEL = os.path.join(OUTPUT_DIR, "elephant_xgb.json")
METADATA_PATH = os.path.join(OUTPUT_DIR, "model_metadata.json")

LOG_PATH = os.path.join(SCRIPT_DIR, "elephant_detections.log")
DETECTIONS_DIR = os.path.join(SCRIPT_DIR, "detections")


# --- Aggregation (for clip-level retrained models, if ever present) -----------

def aggregate_embeddings(embeddings: np.ndarray, strategy: str) -> np.ndarray:
    """Aggregate N patch embeddings into one clip-level feature vector."""
    mean_emb = np.mean(embeddings, axis=0)
    if strategy == "mean":
        return mean_emb
    max_emb = np.max(embeddings, axis=0)
    if strategy == "mean_max":
        return np.concatenate([mean_emb, max_emb])
    std_emb = np.std(embeddings, axis=0)
    return np.concatenate([mean_emb, max_emb, std_emb])


def load_metadata_defaults():
    """Return (threshold, aggregation, pooling) from model_metadata.json."""
    threshold, aggregation, pooling = 0.5, None, "max"
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r") as f:
                meta = json.load(f)
            threshold = float(meta.get("selected_threshold", threshold))
            aggregation = meta.get("aggregation_strategy", aggregation)
            pooling = meta.get("probability_pooling", pooling)
        except Exception:
            pass
    return threshold, aggregation, pooling


# --- Model loading ------------------------------------------------------------

def load_models(use_quantized=False):
    import onnxruntime as ort
    from xgboost import XGBClassifier

    onnx_path = YAMNET_QUANTIZED if use_quantized else YAMNET_ONNX
    if not os.path.exists(onnx_path):
        print(f"ERROR: YAMNet ONNX not found at {onnx_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(XGB_MODEL):
        print(f"ERROR: XGBoost model not found at {XGB_MODEL}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading YAMNet ONNX: {onnx_path}")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    print(f"Loading XGBoost:     {XGB_MODEL}")
    clf = XGBClassifier()
    clf.load_model(XGB_MODEL)
    return session, clf


# --- Inference on an in-memory window -----------------------------------------

def classify_window(waveform, session, clf, threshold, aggregation, pooling="max"):
    """
    Classify a mono 16 kHz waveform.

    pooling:
        "max"  — fire if the strongest patch exceeds threshold (event detection).
        "mean" — fire if the mean patch probability exceeds threshold (legacy).

    Returns dict (score, max_prob, mean_prob, prediction, ...) or None if the
    window produced no patches.
    """
    patches = audio_to_patches_from_waveform(waveform)
    if patches.shape[0] == 0:
        return None

    embeddings = session.run(["clip_embedding"], {"log_mel_patches": patches})[0]

    if aggregation is not None:
        # Clip-level retrained model path (single aggregated feature vector).
        feats = aggregate_embeddings(embeddings, aggregation).reshape(1, -1)
        prob = float(clf.predict_proba(feats)[0, 1])
        max_prob = mean_prob = prob
        votes = 1 if prob > threshold else 0
        n_patches = 1
    else:
        # Per-patch model: probability per 0.96 s patch.
        patch_probs = clf.predict_proba(embeddings)[:, 1]
        max_prob = float(np.max(patch_probs))
        mean_prob = float(np.mean(patch_probs))
        votes = int(np.sum(patch_probs > threshold))
        n_patches = int(embeddings.shape[0])

    score = max_prob if pooling == "max" else mean_prob
    return {
        "prediction": "Elephant" if score > threshold else "Non-Elephant",
        "score": score,
        "max_prob": max_prob,
        "mean_prob": mean_prob,
        "num_patches": n_patches,
        "votes_above_threshold": votes,
    }


# --- Logging ------------------------------------------------------------------

def log_line(text, log_path):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def save_detection_wav(waveform, ts):
    import soundfile as sf
    os.makedirs(DETECTIONS_DIR, exist_ok=True)
    fname = f"elephant_{ts.strftime('%Y%m%d_%H%M%S_%f')[:-3]}.wav"
    path = os.path.join(DETECTIONS_DIR, fname)
    sf.write(path, waveform, SAMPLE_RATE)
    return path


# --- Device helpers -----------------------------------------------------------

def list_devices():
    import sounddevice as sd
    print(sd.query_devices())


# --- Main capture + inference loop --------------------------------------------

def run(args):
    import sounddevice as sd

    threshold, aggregation, pooling = load_metadata_defaults()
    if args.threshold is not None:
        threshold = args.threshold
    if args.aggregation is not None:
        aggregation = args.aggregation
    if args.pooling is not None:
        pooling = args.pooling

    session, clf = load_models(use_quantized=args.quantized)

    window_samples = int(args.window * SAMPLE_RATE)
    hop_samples = int(args.hop * SAMPLE_RATE)
    block_samples = hop_samples  # deliver audio one hop at a time

    ring = np.zeros(window_samples, dtype=np.float32)
    audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=64)
    stop_event = threading.Event()

    def audio_callback(indata, frames, time_info, status):
        if status:
            log_line(f"[{_dt.datetime.now().isoformat(timespec='seconds')}] "
                     f"AUDIO STATUS: {status}", args.log)
        try:
            audio_q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass

    banner = (
        f"[{_dt.datetime.now().isoformat(timespec='seconds')}] "
        f"START live elephant detection | rate={SAMPLE_RATE}Hz "
        f"window={args.window}s hop={args.hop}s threshold={threshold} "
        f"pooling={pooling} debounce={args.debounce_m}/{args.debounce_k} "
        f"aggregation={aggregation or 'per_patch'} "
        f"device={args.device if args.device is not None else 'default'}"
    )
    print(banner)
    print("Listening 24x7. Press Ctrl+C to stop.\n")
    log_line(banner, args.log)

    # Temporal debounce state.
    fire_history = deque(maxlen=args.debounce_k)
    in_detection = False

    infer_count = 0
    detect_events = 0
    samples_since_infer = 0

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=block_samples,
            device=args.device,
            callback=audio_callback,
        ):
            while not stop_event.is_set():
                try:
                    block = audio_q.get(timeout=1.0)
                except queue.Empty:
                    continue

                n = len(block)
                if n >= window_samples:
                    ring[:] = block[-window_samples:]
                    samples_since_infer += window_samples
                else:
                    ring[:-n] = ring[n:]
                    ring[-n:] = block
                    samples_since_infer += n

                if samples_since_infer < hop_samples:
                    continue
                samples_since_infer = 0

                window = ring.copy()
                ts = _dt.datetime.now()
                stamp = ts.isoformat(timespec="milliseconds")

                # Skip near-silent windows (saves compute, avoids noise triggers).
                diag = diagnose_audio(window)
                if diag["rms"] < 0.0015:
                    fire_history.append(False)
                    continue

                try:
                    result = classify_window(
                        window, session, clf, threshold, aggregation, pooling
                    )
                except Exception as e:
                    log_line(f"[{stamp}] INFERENCE ERROR: {e}", args.log)
                    continue
                if result is None:
                    continue

                infer_count += 1
                fired = result["prediction"] == "Elephant"
                fire_history.append(fired)
                score = result["score"]

                # Confirmed detection = M of the last K windows fired.
                confirmed = sum(fire_history) >= args.debounce_m

                # Per-window line (compact). Green when this window fired.
                inst = (f"[{stamp}] {'FIRE' if fired else 'idle'} "
                        f"score={score:.1%} max={result['max_prob']:.1%} "
                        f"votes={result['votes_above_threshold']}/{result['num_patches']} "
                        f"rms={diag['rms']:.3f}")
                print(f"\033[92m{inst}\033[0m" if fired else inst)
                if args.log_all:
                    log_line(inst, args.log)

                # Edge-triggered confirmed-detection events.
                if confirmed and not in_detection:
                    in_detection = True
                    detect_events += 1
                    line = (f"[{stamp}] *** ELEPHANT DETECTED (confirmed) *** "
                            f"score={score:.1%} "
                            f"{sum(fire_history)}/{len(fire_history)} recent windows fired")
                    print(f"\033[1;92m{line}\033[0m")
                    log_line(line, args.log)
                    if args.save_detections:
                        try:
                            wav_path = save_detection_wav(window, ts)
                            log_line(f"[{stamp}]   saved -> {wav_path}", args.log)
                        except Exception as e:
                            log_line(f"[{stamp}]   save failed: {e}", args.log)
                elif not confirmed and in_detection:
                    in_detection = False
                    line = f"[{stamp}] --- detection cleared ---"
                    print(line)
                    log_line(line, args.log)

    except KeyboardInterrupt:
        pass
    finally:
        end = (f"[{_dt.datetime.now().isoformat(timespec='seconds')}] "
               f"STOP live detection | windows_classified={infer_count} "
               f"confirmed_detections={detect_events}")
        print("\n" + end)
        log_line(end, args.log)


def build_parser():
    p = argparse.ArgumentParser(
        description="24x7 live microphone elephant-sound detector "
                    "(YAMNet + XGBoost, max-pooling + debounce).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--threshold", type=float, default=None,
                   help="Detection threshold on the pooled patch score "
                        "(default: metadata or 0.5)")
    p.add_argument("--pooling", choices=["max", "mean"], default=None,
                   help="Patch-probability pooling (default: metadata or 'max')")
    p.add_argument("--aggregation", choices=["mean", "mean_max", "mean_max_std"],
                   default=None,
                   help="Clip-level feature aggregation (only for a retrained "
                        "clip-level model; leave unset for the per-patch model)")
    p.add_argument("--window", type=float, default=3.0,
                   help="Analysis window length in seconds (default: 3.0)")
    p.add_argument("--hop", type=float, default=1.0,
                   help="Seconds of new audio between classifications (default: 1.0)")
    p.add_argument("--debounce-m", type=int, default=2,
                   help="Windows that must fire out of the last K (default: 2)")
    p.add_argument("--debounce-k", type=int, default=3,
                   help="Sliding window count for debounce (default: 3)")
    p.add_argument("--device", type=int, default=None,
                   help="Input device index (see --list-devices)")
    p.add_argument("--list-devices", action="store_true",
                   help="List audio input devices and exit")
    p.add_argument("--quantized", action="store_true",
                   help="Use quantized INT8 YAMNet model")
    p.add_argument("--save-detections", action="store_true",
                   help="Save a WAV of every confirmed detection under ./detections/")
    p.add_argument("--log", default=LOG_PATH,
                   help=f"Log file path (default: {LOG_PATH})")
    p.add_argument("--log-all", action="store_true",
                   help="Log every window, not only confirmed detections")
    return p


def main():
    args = build_parser().parse_args()
    if args.list_devices:
        list_devices()
        return
    if args.hop > args.window:
        print("ERROR: --hop must be <= --window", file=sys.stderr)
        sys.exit(1)
    if args.debounce_m > args.debounce_k:
        print("ERROR: --debounce-m must be <= --debounce-k", file=sys.stderr)
        sys.exit(1)
    run(args)


if __name__ == "__main__":
    main()
