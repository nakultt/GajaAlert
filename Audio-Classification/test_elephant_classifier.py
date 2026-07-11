#!/usr/bin/env python3
"""
test_elephant_classifier.py  -  Local Elephant / Non-Elephant Audio Classifier

Runs the full YAMNet -> XGBoost pipeline entirely locally.
No PyTorch, no Colab, no cloud dependencies.

Usage:
    # Single file
    python test_elephant_classifier.py audio.wav

    # Batch folder
    python test_elephant_classifier.py dataset/test/

    # With profiling
    python test_elephant_classifier.py --profile audio.wav

    # With custom threshold
    python test_elephant_classifier.py --threshold 0.65 audio.wav

    # Clip-level aggregation (matches retrained model)
    python test_elephant_classifier.py --aggregation mean_max_std audio.wav

    # Batch test against known datasets (auto-detects labels from folder names)
    python test_elephant_classifier.py --eval dataset/

    # Use quantized model
    python test_elephant_classifier.py --quantized audio.wav

Requirements:
    pip install onnxruntime xgboost librosa soundfile numpy
"""

import argparse
import csv
import json
import os
import sys
import time
import glob
import numpy as np

# Local import  -  must be in the same directory
from input_processing import (
    audio_to_patches,
    audio_to_patches_from_waveform,
    load_audio,
    diagnose_audio,
    SAMPLE_RATE,
)


# --- Paths to model artifacts ---------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

YAMNET_ONNX = os.path.join(OUTPUT_DIR, "yamnet.onnx")
XGB_MODEL = os.path.join(OUTPUT_DIR, "elephant_xgb.json")
YAMNET_QUANTIZED = os.path.join(OUTPUT_DIR, "yamnet_int8.onnx")

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
METADATA_PATH = os.path.join(OUTPUT_DIR, "model_metadata.json")


# --- Aggregation (matches train_binary_detector.py) --------------------------

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


def load_default_threshold() -> float:
    """Load threshold from model_metadata.json if available, else 0.5."""
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r") as f:
                meta = json.load(f)
            t = meta.get("selected_threshold", 0.5)
            return float(t)
        except Exception:
            pass
    return 0.5


def load_default_aggregation() -> str | None:
    """Load aggregation strategy from model_metadata.json if available."""
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r") as f:
                meta = json.load(f)
            return meta.get("aggregation_strategy")
        except Exception:
            pass
    return None


# --- Model Loading --------------------------------------------------------------

def load_models(use_quantized=False):
    """Load YAMNet ONNX and XGBoost classifier."""
    import onnxruntime as ort
    from xgboost import XGBClassifier

    onnx_path = YAMNET_QUANTIZED if use_quantized else YAMNET_ONNX
    if not os.path.exists(onnx_path):
        if use_quantized:
            print(f"Quantized model not found at {onnx_path}")
            print("Run: python test_elephant_classifier.py --quantize  to create it")
            sys.exit(1)
        print(f"ERROR: YAMNet ONNX not found at {onnx_path}")
        print(f"Expected model artifacts in: {OUTPUT_DIR}")
        sys.exit(1)

    if not os.path.exists(XGB_MODEL):
        print(f"ERROR: XGBoost model not found at {XGB_MODEL}")
        sys.exit(1)

    print(f"Loading YAMNet ONNX: {onnx_path}")
    yamnet_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    print(f"Loading XGBoost:     {XGB_MODEL}")
    clf = XGBClassifier()
    clf.load_model(XGB_MODEL)

    return yamnet_session, clf


# --- Single File Inference -------------------------------------------------------

def predict_file(path, yamnet_session, clf, threshold=0.5, verbose=True,
                 profile=False, aggregation=None):
    """
    Run full inference on a single audio file.

    Args:
        aggregation: None for legacy patch-level averaging,
                     or 'mean'/'mean_max'/'mean_max_std' for clip-level aggregation.

    Returns dict with prediction, confidence, per-patch details, timing.
    """
    filename = os.path.basename(path)

    # Timing
    t0 = time.perf_counter()

    # Load and diagnose
    waveform = load_audio(path)
    t_load = time.perf_counter()

    diag = diagnose_audio(waveform, filename)
    if diag["issues"] and verbose:
        for issue in diag["issues"]:
            print(f"  ! {issue}")

    # Preprocess: waveform -> log-mel patches
    patches = audio_to_patches(path)
    t_preprocess = time.perf_counter()

    if patches.shape[0] == 0:
        if verbose:
            print(f"  {filename}: too short to produce patches, skipping.")
        return None

    # YAMNet inference: patches -> embeddings
    embeddings = yamnet_session.run(
        ["clip_embedding"],
        {"log_mel_patches": patches}
    )[0]  # shape: (N, 1024)
    t_yamnet = time.perf_counter()

    total_patches = embeddings.shape[0]

    if aggregation is not None:
        # ── Clip-level aggregation (retrained model) ────────────────────
        clip_features = aggregate_embeddings(embeddings, aggregation)
        clip_features_2d = clip_features.reshape(1, -1)  # (1, D)
        clip_prob = float(clf.predict_proba(clip_features_2d)[0, 1])
        t_xgb = time.perf_counter()

        avg_conf = clip_prob
        max_conf = clip_prob
        min_conf = clip_prob
        prediction = "Elephant" if clip_prob > threshold else "Non-Elephant"
        votes_above = 1 if clip_prob > threshold else 0
        mode_label = f"clip-level ({aggregation})"
    else:
        # ── Legacy: per-patch probability averaging ─────────────────────
        patch_probs = clf.predict_proba(embeddings)[:, 1]  # P(elephant) per patch
        t_xgb = time.perf_counter()

        avg_conf = float(np.mean(patch_probs))
        max_conf = float(np.max(patch_probs))
        min_conf = float(np.min(patch_probs))
        prediction = "Elephant" if avg_conf > threshold else "Non-Elephant"
        votes_above = int(np.sum(patch_probs > threshold))
        mode_label = "patch-level avg (legacy)"

    result = {
        "filename": filename,
        "path": path,
        "prediction": prediction,
        "avg_confidence": avg_conf,
        "max_confidence": max_conf,
        "min_confidence": min_conf,
        "num_patches": total_patches,
        "votes_above_threshold": votes_above,
        "duration_s": diag["duration_s"],
        "issues": diag["issues"],
        "aggregation": aggregation or "patch_avg",
        "timing": {
            "load_ms": (t_load - t0) * 1000,
            "preprocess_ms": (t_preprocess - t_load) * 1000,
            "yamnet_ms": (t_yamnet - t_preprocess) * 1000,
            "xgb_ms": (t_xgb - t_yamnet) * 1000,
            "total_ms": (t_xgb - t0) * 1000,
        },
    }

    if verbose:
        print(f"  File:        {filename}")
        print(f"  Duration:    {diag['duration_s']:.2f}s -> {total_patches} patch(es)")
        print(f"  Mode:        {mode_label}")
        print(f"  Prediction:  {prediction}")
        print(f"  Confidence:  {avg_conf:.2%}")
        if aggregation is None:
            print(f"  Patch range: {min_conf:.2%} - {max_conf:.2%}")
            print(f"  Votes:       {votes_above}/{total_patches} patches above {threshold}")

        if profile:
            t = result["timing"]
            print(f"  -- Timing --")
            print(f"     Load:       {t['load_ms']:7.1f} ms")
            print(f"     Preprocess: {t['preprocess_ms']:7.1f} ms")
            print(f"     YAMNet:     {t['yamnet_ms']:7.1f} ms")
            print(f"     XGBoost:    {t['xgb_ms']:7.1f} ms")
            print(f"     TOTAL:      {t['total_ms']:7.1f} ms")

    return result


# --- Batch Processing ------------------------------------------------------------

def find_audio_files(path):
    """Recursively find all audio files under a path."""
    if os.path.isfile(path):
        return [path]

    files = []
    for root, dirs, filenames in os.walk(path):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS:
                files.append(os.path.join(root, f))
    return sorted(files)


def infer_ground_truth(filepath):
    """
    Try to infer ground truth label from the file path.
    Looks for 'elephant', 'roar', 'rumble', 'trumpet' in parent dirs.
    Returns 'Elephant', 'Non-Elephant', or 'Unknown'.
    """
    path_lower = filepath.lower().replace("\\", "/")

    elephant_keywords = ["elephant", "roar", "rumble", "trumpet"]
    for keyword in elephant_keywords:
        if keyword in path_lower:
            return "Elephant"

    # Known negative dataset folders
    negative_indicators = ["esc50", "esc-50", "urbansound", "urban-sound", "fsc22"]
    for neg in negative_indicators:
        if neg in path_lower:
            return "Non-Elephant"

    return "Unknown"


def batch_test(input_path, yamnet_session, clf, threshold=0.5, profile=False,
               eval_mode=False, aggregation=None):
    """Run inference on all audio files in a folder."""
    files = find_audio_files(input_path)
    if not files:
        print(f"No audio files found in: {input_path}")
        return []

    print(f"\nFound {len(files)} audio files in: {input_path}")
    print("=" * 80)

    results = []
    for i, fpath in enumerate(files):
        print(f"\n[{i+1}/{len(files)}] ", end="")
        result = predict_file(fpath, yamnet_session, clf, threshold,
                             verbose=True, profile=profile,
                             aggregation=aggregation)
        if result:
            if eval_mode:
                result["ground_truth"] = infer_ground_truth(fpath)
            results.append(result)

    # Summary
    print("\n" + "=" * 80)
    print(f"BATCH SUMMARY  -  {len(results)} files processed")
    print("=" * 80)

    elephant_count = sum(1 for r in results if r["prediction"] == "Elephant")
    non_elephant_count = len(results) - elephant_count
    print(f"  Elephant:     {elephant_count}")
    print(f"  Non-Elephant: {non_elephant_count}")

    if profile and results:
        avg_total = np.mean([r["timing"]["total_ms"] for r in results])
        avg_yamnet = np.mean([r["timing"]["yamnet_ms"] for r in results])
        avg_xgb = np.mean([r["timing"]["xgb_ms"] for r in results])
        print(f"\n  Avg timing per file:")
        print(f"    Total:   {avg_total:.1f} ms")
        print(f"    YAMNet:  {avg_yamnet:.1f} ms")
        print(f"    XGBoost: {avg_xgb:.1f} ms")

    # Evaluation metrics (when ground truth is known)
    if eval_mode:
        labeled = [r for r in results if r.get("ground_truth") != "Unknown"]
        if labeled:
            tp = sum(1 for r in labeled if r["ground_truth"] == "Elephant" and r["prediction"] == "Elephant")
            tn = sum(1 for r in labeled if r["ground_truth"] == "Non-Elephant" and r["prediction"] == "Non-Elephant")
            fp = sum(1 for r in labeled if r["ground_truth"] == "Non-Elephant" and r["prediction"] == "Elephant")
            fn = sum(1 for r in labeled if r["ground_truth"] == "Elephant" and r["prediction"] == "Non-Elephant")

            total = tp + tn + fp + fn
            accuracy = (tp + tn) / total if total else 0
            precision = tp / (tp + fp) if (tp + fp) else 0
            recall = tp / (tp + fn) if (tp + fn) else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

            print(f"\n  -- Evaluation (threshold={threshold}) --")
            print(f"    Labeled files: {len(labeled)}")
            print(f"    TP={tp}  FP={fp}  TN={tn}  FN={fn}")
            print(f"    Accuracy:  {accuracy:.4f}")
            print(f"    Precision: {precision:.4f}")
            print(f"    Recall:    {recall:.4f}")
            print(f"    F1 Score:  {f1:.4f}")

            # Threshold sweep
            print(f"\n  -- Threshold Sweep --")
            print(f"    {'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
            for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                tp_t = sum(1 for r in labeled if r["ground_truth"] == "Elephant" and r["avg_confidence"] > t)
                fp_t = sum(1 for r in labeled if r["ground_truth"] == "Non-Elephant" and r["avg_confidence"] > t)
                fn_t = sum(1 for r in labeled if r["ground_truth"] == "Elephant" and r["avg_confidence"] <= t)
                prec_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) else 0
                rec_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) else 0
                f1_t = 2 * prec_t * rec_t / (prec_t + rec_t) if (prec_t + rec_t) else 0
                print(f"    {t:<12.1f} {prec_t:<12.4f} {rec_t:<12.4f} {f1_t:<12.4f}")

    # Save CSV
    csv_path = os.path.join(SCRIPT_DIR, "test_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["filename", "prediction", "avg_confidence", "max_confidence",
                  "min_confidence", "num_patches", "votes", "duration_s"]
        if eval_mode:
            header.append("ground_truth")
        if profile:
            header.extend(["total_ms", "yamnet_ms", "xgb_ms"])
        writer.writerow(header)

        for r in results:
            row = [r["filename"], r["prediction"],
                   f"{r['avg_confidence']:.4f}", f"{r['max_confidence']:.4f}",
                   f"{r['min_confidence']:.4f}", r["num_patches"],
                   r["votes_above_threshold"], f"{r['duration_s']:.2f}"]
            if eval_mode:
                row.append(r.get("ground_truth", "Unknown"))
            if profile:
                row.extend([f"{r['timing']['total_ms']:.1f}",
                           f"{r['timing']['yamnet_ms']:.1f}",
                           f"{r['timing']['xgb_ms']:.1f}"])
            writer.writerow(row)

    print(f"\n  Results saved to: {csv_path}")
    return results


# --- Quantization ----------------------------------------------------------------

def quantize_yamnet():
    """Quantize yamnet.onnx to INT8 for edge deployment."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    if not os.path.exists(YAMNET_ONNX):
        print(f"ERROR: Source model not found: {YAMNET_ONNX}")
        sys.exit(1)

    output_path = YAMNET_QUANTIZED
    print(f"Quantizing {YAMNET_ONNX} -> {output_path}")

    quantize_dynamic(
        model_input=YAMNET_ONNX,
        model_output=output_path,
        weight_type=QuantType.QInt8,
    )

    orig_size = os.path.getsize(YAMNET_ONNX) / (1024 * 1024)
    # Check for external data file
    data_file = YAMNET_ONNX + ".data"
    if os.path.exists(data_file):
        orig_size += os.path.getsize(data_file) / (1024 * 1024)

    quant_size = os.path.getsize(output_path) / (1024 * 1024)
    quant_data = output_path + ".data"
    if os.path.exists(quant_data):
        quant_size += os.path.getsize(quant_data) / (1024 * 1024)

    print(f"  Original size:  {orig_size:.2f} MB")
    print(f"  Quantized size: {quant_size:.2f} MB")
    print(f"  Compression:    {orig_size/quant_size:.1f}x")
    print(f"\nSaved to: {output_path}")
    print("Use --quantized flag to run inference with the quantized model.")


# --- Main ------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Elephant / Non-Elephant Audio Classifier (YAMNet + XGBoost)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_elephant_classifier.py audio.wav
  python test_elephant_classifier.py dataset/test/ --eval --profile
  python test_elephant_classifier.py --threshold 0.65 recording.mp3
  python test_elephant_classifier.py --quantize
  python test_elephant_classifier.py --quantized audio.wav
        """
    )
    parser.add_argument("input", nargs="?", help="Audio file or folder path")
    parser.add_argument("--threshold", type=float, default=None,
                       help="Classification threshold (default: auto from model_metadata.json or 0.5)")
    parser.add_argument("--aggregation", type=str, default=None,
                       choices=["mean", "mean_max", "mean_max_std"],
                       help="Clip-level aggregation strategy (matches retrained model)")
    parser.add_argument("--profile", action="store_true",
                       help="Show per-stage timing breakdown")
    parser.add_argument("--eval", action="store_true",
                       help="Auto-detect ground truth from folder names and compute metrics")
    parser.add_argument("--quantize", action="store_true",
                       help="Quantize yamnet.onnx to INT8 and exit")
    parser.add_argument("--quantized", action="store_true",
                       help="Use quantized INT8 model for inference")

    args = parser.parse_args()

    # Quantize mode
    if args.quantize:
        quantize_yamnet()
        return

    if not args.input:
        parser.print_help()
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"ERROR: Path not found: {args.input}")
        sys.exit(1)

    # Resolve aggregation and threshold from metadata if not specified
    aggregation = args.aggregation
    if aggregation is None:
        aggregation = load_default_aggregation()
        if aggregation:
            print(f"[Auto] Using aggregation from model_metadata.json: {aggregation}")

    threshold = args.threshold
    if threshold is None:
        threshold = load_default_threshold()
        print(f"[Auto] Using threshold from model_metadata.json: {threshold}")

    # Load models
    yamnet_session, clf = load_models(use_quantized=args.quantized)

    # Run inference
    if os.path.isfile(args.input):
        print(f"\n{'='*60}")
        result = predict_file(
            args.input, yamnet_session, clf,
            threshold=threshold, verbose=True, profile=args.profile,
            aggregation=aggregation
        )
        print(f"{'='*60}")
    else:
        batch_test(
            args.input, yamnet_session, clf,
            threshold=threshold, profile=args.profile, eval_mode=args.eval,
            aggregation=aggregation
        )


if __name__ == "__main__":
    main()
