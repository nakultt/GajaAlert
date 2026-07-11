#!/usr/bin/env python3
"""
train_binary_detector.py — Retrain YAMNet + XGBoost elephant/non-elephant classifier.

Colab-optimized: auto-installs deps, mounts Drive, handles all I/O via prints.
Run in Colab:  !python train_binary_detector.py
Run locally:   python train_binary_detector.py --manifest dataset/manifest.jsonl

Key fixes over original train_elephant_classifier.py:
  1. Uses input_processing.py for BOTH training and inference (no torch_audioset)
  2. Splits by recording group, not individual YAMNet patches
  3. Clip-level aggregated embeddings (mean / mean+max / mean+max+std)
  4. Threshold selected on validation data with recall >= 0.85 constraint
  5. No augmented/duplicate data leaking across splits
  6. Exports model metadata with artifact hashes for reproducibility
"""

import os
import sys
import json
import hashlib
import argparse
import warnings
import time
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

# ── Colab auto-setup ────────────────────────────────────────────────────────────

def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

def colab_setup():
    if not is_colab():
        return
    print("[Colab] Installing dependencies...")
    os.system("pip install -q numpy librosa soundfile onnxruntime xgboost scikit-learn")
    from google.colab import drive
    if not os.path.ismount("/content/drive"):
        drive.mount("/content/drive")

colab_setup()

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    os.system("pip install onnxruntime")
    import onnxruntime as ort

try:
    from xgboost import XGBClassifier
except ImportError:
    os.system("pip install xgboost")
    from xgboost import XGBClassifier

try:
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, confusion_matrix, precision_recall_curve)
except ImportError:
    os.system("pip install scikit-learn")
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, confusion_matrix, precision_recall_curve)


# ── Constants ───────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "training_results")
YAMNET_ONNX = os.path.join(OUTPUT_DIR, "yamnet.onnx")
XGB_OUTPUT = os.path.join(OUTPUT_DIR, "elephant_xgb.json")

AGGREGATION_STRATEGIES = ["mean", "mean_max", "mean_max_std"]

# Minimum recall constraint for threshold selection
MIN_RECALL_CONSTRAINT = 0.85


# ── Embedding extraction ───────────────────────────────────────────────────────

def load_yamnet_session() -> ort.InferenceSession:
    """Load YAMNet ONNX session."""
    if not os.path.exists(YAMNET_ONNX):
        print(f"[ERROR] YAMNet ONNX not found at: {YAMNET_ONNX}")
        print("  Run train_elephant_classifier.py first to export the ONNX model,")
        print("  or copy yamnet.onnx + yamnet.onnx.data to output/")
        sys.exit(1)

    print(f"[Model] Loading YAMNet ONNX: {YAMNET_ONNX}")
    session = ort.InferenceSession(YAMNET_ONNX, providers=["CPUExecutionProvider"])

    # Warm up
    dummy = np.random.randn(1, 1, 96, 64).astype(np.float32)
    session.run(["clip_embedding"], {"log_mel_patches": dummy})
    print("[Model] YAMNet loaded and warmed up.")
    return session


def extract_embeddings_for_file(
    path: str,
    yamnet_session: ort.InferenceSession,
) -> np.ndarray | None:
    """
    Extract YAMNet embeddings for one audio file using deployment preprocessing.
    Returns shape (N, 1024) or None on failure.
    """
    # Import from the same directory
    sys.path.insert(0, SCRIPT_DIR)
    from input_processing import load_audio, audio_to_patches_from_waveform

    try:
        waveform = load_audio(path)
        patches = audio_to_patches_from_waveform(waveform)

        if patches.shape[0] == 0:
            return None

        embeddings = yamnet_session.run(
            ["clip_embedding"],
            {"log_mel_patches": patches}
        )[0]  # shape: (N, 1024)

        return embeddings
    except Exception as e:
        print(f"  [WARN] Failed to process {path}: {e}")
        return None


def aggregate_embeddings(embeddings: np.ndarray, strategy: str) -> np.ndarray:
    """
    Aggregate N patch embeddings into one clip-level feature vector.

    Args:
        embeddings: shape (N, 1024)
        strategy: one of 'mean', 'mean_max', 'mean_max_std'

    Returns:
        feature vector of shape (D,)
    """
    mean_emb = np.mean(embeddings, axis=0)          # (1024,)

    if strategy == "mean":
        return mean_emb

    max_emb = np.max(embeddings, axis=0)             # (1024,)

    if strategy == "mean_max":
        return np.concatenate([mean_emb, max_emb])   # (2048,)

    std_emb = np.std(embeddings, axis=0)             # (1024,)
    return np.concatenate([mean_emb, max_emb, std_emb])  # (3072,)


# ── Dataset loading ────────────────────────────────────────────────────────────

def load_manifest(manifest_path: str) -> list[dict]:
    """Load the JSONL manifest."""
    entries = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def extract_all_embeddings(
    entries: list[dict],
    yamnet_session: ort.InferenceSession,
    cache_path: str = "dataset/embeddings_cache.npz",
) -> dict:
    """
    Extract embeddings for all files in the manifest.
    Caches results to avoid re-processing.

    Returns: dict mapping path -> embeddings array (N, 1024)
    """
    # Check cache
    if os.path.exists(cache_path):
        print(f"[Cache] Loading cached embeddings from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        cache = dict(data["cache"].item())
        # Validate cache covers all entries
        missing = [e["path"] for e in entries if e["path"] not in cache]
        if not missing:
            print(f"[Cache] All {len(cache)} files found in cache.")
            return cache
        else:
            print(f"[Cache] {len(missing)} files missing from cache, re-extracting those.")
    else:
        cache = {}
        missing = [e["path"] for e in entries]

    # Extract missing
    print(f"[Extract] Processing {len(missing)} audio files...")
    t_start = time.time()

    for i, path in enumerate(missing):
        embs = extract_embeddings_for_file(path, yamnet_session)
        if embs is not None:
            cache[path] = embs
        else:
            print(f"  [SKIP] {os.path.basename(path)}: no patches extracted")

        if (i + 1) % 50 == 0 or (i + 1) == len(missing):
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(missing) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(missing)}] {rate:.1f} files/s, ETA {eta:.0f}s")

    # Save cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, cache=cache)
    print(f"[Cache] Saved {len(cache)} embeddings to {cache_path}")

    return cache


def prepare_features(
    entries: list[dict],
    embeddings_cache: dict,
    strategy: str,
    split: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Prepare feature matrix and labels for a given split and aggregation strategy.

    Returns: (X, y, filenames)
    """
    X_list = []
    y_list = []
    filenames = []

    for entry in entries:
        if entry["split"] != split:
            continue
        path = entry["path"]
        if path not in embeddings_cache:
            continue

        embs = embeddings_cache[path]
        feat = aggregate_embeddings(embs, strategy)
        X_list.append(feat)
        y_list.append(1 if entry["label"] == "elephant" else 0)
        filenames.append(os.path.basename(path))

    if not X_list:
        return np.array([]), np.array([]), []

    return np.array(X_list), np.array(y_list), filenames


# ── Training ───────────────────────────────────────────────────────────────────

def select_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_recall: float = MIN_RECALL_CONSTRAINT,
) -> dict:
    """
    Select optimal threshold by maximizing F1 with recall >= min_recall constraint.
    Also reports threshold sweep.
    """
    thresholds = np.arange(0.05, 0.96, 0.01)
    sweep = []

    best_f1 = -1
    best_threshold = 0.5
    best_metrics = {}

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if np.sum(y_pred) == 0 and np.sum(y_true) > 0:
            # All predicted negative but there are positives
            prec, rec, f1 = 0.0, 0.0, 0.0
        elif np.sum(y_true) == 0:
            prec = 1.0 if np.sum(y_pred) == 0 else 0.0
            rec = 1.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        else:
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

        sweep.append({
            "threshold": round(float(t), 2),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        })

        # Select best F1 with recall constraint
        if rec >= min_recall and f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)
            best_metrics = {"precision": prec, "recall": rec, "f1": f1}

    # If no threshold meets recall constraint, use the one with highest recall
    if best_f1 < 0:
        print(f"  [WARN] No threshold achieves recall >= {min_recall}")
        best_entry = max(sweep, key=lambda s: (s["recall"], s["f1"]))
        best_threshold = best_entry["threshold"]
        best_metrics = {
            "precision": best_entry["precision"],
            "recall": best_entry["recall"],
            "f1": best_entry["f1"],
        }

    return {
        "selected_threshold": round(best_threshold, 2),
        "selected_metrics": {k: round(v, 4) for k, v in best_metrics.items()},
        "sweep": sweep,
    }


def train_and_evaluate(
    train_X: np.ndarray, train_y: np.ndarray,
    val_X: np.ndarray, val_y: np.ndarray,
    strategy: str,
) -> dict:
    """Train XGBoost and evaluate on validation set for one aggregation strategy."""
    print(f"\n{'─'*60}")
    print(f"Training with aggregation: {strategy}")
    print(f"  Train: {len(train_y)} clips ({np.sum(train_y==1)} elephant, {np.sum(train_y==0)} non-elephant)")
    print(f"  Val:   {len(val_y)} clips ({np.sum(val_y==1)} elephant, {np.sum(val_y==0)} non-elephant)")
    print(f"  Feature dim: {train_X.shape[1]}")

    # Compute class weight
    n_neg = np.sum(train_y == 0)
    n_pos = np.sum(train_y == 1)
    scale_pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

    # Train
    clf = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        early_stopping_rounds=20,
        verbosity=0,
    )

    t_start = time.time()
    clf.fit(
        train_X, train_y,
        eval_set=[(val_X, val_y)],
        verbose=False,
    )
    train_time = time.time() - t_start
    print(f"  Training time: {train_time:.1f}s")
    print(f"  Best iteration: {clf.best_iteration}")

    # Predict probabilities on validation
    val_probs = clf.predict_proba(val_X)[:, 1]

    # Threshold selection
    threshold_result = select_threshold(val_y, val_probs)
    threshold = threshold_result["selected_threshold"]
    print(f"  Selected threshold: {threshold}")
    print(f"  Metrics at threshold: {threshold_result['selected_metrics']}")

    # Confusion matrix
    val_preds = (val_probs >= threshold).astype(int)
    cm = confusion_matrix(val_y, val_preds)
    print(f"  Confusion matrix (val):")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")

    # Also evaluate at default 0.5
    val_preds_05 = (val_probs >= 0.5).astype(int)
    acc_05 = accuracy_score(val_y, val_preds_05)
    prec_05 = precision_score(val_y, val_preds_05, zero_division=0)
    rec_05 = recall_score(val_y, val_preds_05, zero_division=0)
    f1_05 = f1_score(val_y, val_preds_05, zero_division=0)
    print(f"  Metrics at 0.5: acc={acc_05:.4f} prec={prec_05:.4f} rec={rec_05:.4f} f1={f1_05:.4f}")

    result = {
        "strategy": strategy,
        "feature_dim": int(train_X.shape[1]),
        "train_samples": int(len(train_y)),
        "val_samples": int(len(val_y)),
        "train_class_dist": {"elephant": int(n_pos), "non_elephant": int(n_neg)},
        "scale_pos_weight": round(scale_pos_weight, 2),
        "best_iteration": int(clf.best_iteration),
        "train_time_s": round(train_time, 1),
        "threshold_selection": threshold_result,
        "metrics_at_selected_threshold": threshold_result["selected_metrics"],
        "metrics_at_05": {
            "accuracy": round(acc_05, 4),
            "precision": round(prec_05, 4),
            "recall": round(rec_05, 4),
            "f1": round(f1_05, 4),
        },
        "confusion_matrix": cm.tolist(),
        "val_prob_distribution": {
            "elephant_mean": round(float(np.mean(val_probs[val_y == 1])), 4) if np.sum(val_y == 1) > 0 else None,
            "elephant_std": round(float(np.std(val_probs[val_y == 1])), 4) if np.sum(val_y == 1) > 0 else None,
            "non_elephant_mean": round(float(np.mean(val_probs[val_y == 0])), 4) if np.sum(val_y == 0) > 0 else None,
            "non_elephant_std": round(float(np.std(val_probs[val_y == 0])), 4) if np.sum(val_y == 0) > 0 else None,
        },
    }

    return result, clf


def sha256_of_file(path: str) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Retrain elephant binary detector with clip-level aggregation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # After running build_manifest.py:
  python train_binary_detector.py

  # Custom manifest:
  python train_binary_detector.py --manifest my_dataset/manifest.jsonl

  # Only evaluate, don't retrain:
  python train_binary_detector.py --eval-only
        """,
    )
    parser.add_argument("--manifest", type=str, default="dataset/manifest.jsonl",
                        help="Path to manifest.jsonl (default: dataset/manifest.jsonl)")
    parser.add_argument("--cache", type=str, default="dataset/embeddings_cache.npz",
                        help="Embeddings cache path")
    parser.add_argument("--strategies", type=str, nargs="+",
                        default=AGGREGATION_STRATEGIES,
                        choices=AGGREGATION_STRATEGIES,
                        help="Aggregation strategies to evaluate")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only evaluate cached embeddings, don't extract new ones")
    args = parser.parse_args()

    os.chdir(SCRIPT_DIR)

    print("=" * 70)
    print("TRAIN BINARY ELEPHANT DETECTOR")
    print("  Clip-level aggregated embeddings + XGBoost")
    print("  Using input_processing.py for both training and inference")
    print("=" * 70)

    # ── Load manifest ───────────────────────────────────────────────────────
    if not os.path.exists(args.manifest):
        print(f"[ERROR] Manifest not found: {args.manifest}")
        print("  Run build_manifest.py first.")
        sys.exit(1)

    entries = load_manifest(args.manifest)
    print(f"\n[Data] Loaded {len(entries)} entries from {args.manifest}")

    # Print split distribution
    split_counts = Counter(e["split"] for e in entries)
    label_counts = Counter(e["label"] for e in entries)
    print(f"  Splits: {dict(split_counts)}")
    print(f"  Labels: {dict(label_counts)}")

    # ── Extract embeddings ──────────────────────────────────────────────────
    yamnet_session = load_yamnet_session()
    embeddings_cache = extract_all_embeddings(entries, yamnet_session, args.cache)

    successful = sum(1 for e in entries if e["path"] in embeddings_cache)
    print(f"\n[Data] Embeddings available for {successful}/{len(entries)} files")

    # ── Train all strategies ────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []
    best_model = None
    best_f1 = -1
    best_strategy = None
    best_clf = None

    for strategy in args.strategies:
        train_X, train_y, train_files = prepare_features(
            entries, embeddings_cache, strategy, "train")
        val_X, val_y, val_files = prepare_features(
            entries, embeddings_cache, strategy, "val")

        if len(train_X) == 0 or len(val_X) == 0:
            print(f"\n[SKIP] {strategy}: insufficient data (train={len(train_X)}, val={len(val_X)})")
            continue

        result, clf = train_and_evaluate(train_X, train_y, val_X, val_y, strategy)
        all_results.append(result)

        # Track best by F1, with recall tiebreaker
        f1 = result["metrics_at_selected_threshold"]["f1"]
        recall = result["metrics_at_selected_threshold"]["recall"]
        score = (f1, recall)  # tiebreaker

        if score > (best_f1, 0):
            best_f1 = f1
            best_strategy = strategy
            best_clf = clf
            best_model = result

    if best_clf is None:
        print("\n[ERROR] No model trained successfully!")
        sys.exit(1)

    # ── Evaluate best model on test set ─────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"BEST STRATEGY: {best_strategy}")
    print(f"{'='*70}")

    test_X, test_y, test_files = prepare_features(
        entries, embeddings_cache, best_strategy, "test")

    if len(test_X) > 0:
        test_probs = best_clf.predict_proba(test_X)[:, 1]
        threshold = best_model["threshold_selection"]["selected_threshold"]
        test_preds = (test_probs >= threshold).astype(int)

        test_acc = accuracy_score(test_y, test_preds)
        test_prec = precision_score(test_y, test_preds, zero_division=0)
        test_rec = recall_score(test_y, test_preds, zero_division=0)
        test_f1 = f1_score(test_y, test_preds, zero_division=0)
        test_cm = confusion_matrix(test_y, test_preds)

        print(f"\nTest set evaluation (threshold={threshold}):")
        print(f"  Accuracy:  {test_acc:.4f}")
        print(f"  Precision: {test_prec:.4f}")
        print(f"  Recall:    {test_rec:.4f}")
        print(f"  F1:        {test_f1:.4f}")
        print(f"  Confusion matrix:")
        print(f"    TN={test_cm[0,0]}  FP={test_cm[0,1]}")
        print(f"    FN={test_cm[1,0]}  TP={test_cm[1,1]}")

        # Per-file results
        print(f"\n  Per-file predictions:")
        for fname, prob, true_label in zip(test_files, test_probs, test_y):
            pred = "ELE" if prob >= threshold else "NEG"
            true = "ELE" if true_label == 1 else "NEG"
            match = "✓" if pred == true else "✗"
            print(f"    {match} {fname:30s}  prob={prob:.4f}  pred={pred}  true={true}")

        # Probability separation check
        if np.sum(test_y == 1) > 0 and np.sum(test_y == 0) > 0:
            ele_probs = test_probs[test_y == 1]
            neg_probs = test_probs[test_y == 0]
            print(f"\n  Probability separation:")
            print(f"    Elephant:     [{np.min(ele_probs):.4f} - {np.max(ele_probs):.4f}] "
                  f"mean={np.mean(ele_probs):.4f}")
            print(f"    Non-elephant: [{np.min(neg_probs):.4f} - {np.max(neg_probs):.4f}] "
                  f"mean={np.mean(neg_probs):.4f}")
            overlap = np.max(neg_probs) > np.min(ele_probs)
            print(f"    Overlap: {'YES ⚠' if overlap else 'NO ✓'}")

        best_model["test_metrics"] = {
            "accuracy": round(test_acc, 4),
            "precision": round(test_prec, 4),
            "recall": round(test_rec, 4),
            "f1": round(test_f1, 4),
            "confusion_matrix": test_cm.tolist(),
            "n_samples": int(len(test_y)),
        }
    else:
        print("\n[WARN] No test data available.")

    # ── Export best model ───────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Exporting best model...")

    best_clf.save_model(XGB_OUTPUT)
    print(f"  Saved XGBoost model: {XGB_OUTPUT}")

    # ── Save metrics ────────────────────────────────────────────────────────
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    metrics = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "best_strategy": best_strategy,
        "all_strategies": all_results,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  Saved metrics: {metrics_path}")

    # ── Save model metadata ─────────────────────────────────────────────────
    manifest_hash = hashlib.sha256(
        open(args.manifest, "rb").read()).hexdigest()

    artifact_hashes = {}
    for name in ["yamnet.onnx", "yamnet.onnx.data", "elephant_xgb.json"]:
        path = os.path.join(OUTPUT_DIR, name)
        if os.path.exists(path):
            artifact_hashes[name] = sha256_of_file(path)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest_hash": manifest_hash,
        "preprocessing": "input_processing.py (librosa, no torch_audioset)",
        "aggregation_strategy": best_strategy,
        "feature_dim": best_model["feature_dim"],
        "class_mapping": {"0": "non-elephant", "1": "elephant"},
        "xgboost_params": {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "scale_pos_weight": best_model["scale_pos_weight"],
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "best_iteration": best_model["best_iteration"],
        },
        "selected_threshold": best_model["threshold_selection"]["selected_threshold"],
        "validation_metrics": best_model["metrics_at_selected_threshold"],
        "test_metrics": best_model.get("test_metrics"),
        "artifact_hashes": artifact_hashes,
        "val_prob_distribution": best_model["val_prob_distribution"],
    }

    metadata_path = os.path.join(RESULTS_DIR, "model_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  Saved metadata: {metadata_path}")

    # Also copy metadata next to the model for deployment
    deploy_metadata_path = os.path.join(OUTPUT_DIR, "model_metadata.json")
    with open(deploy_metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # ── Final summary ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Best strategy:   {best_strategy}")
    print(f"  Feature dim:     {best_model['feature_dim']}")
    print(f"  Threshold:       {metadata['selected_threshold']}")
    print(f"  Val F1:          {metadata['validation_metrics']['f1']}")
    print(f"  Val Recall:      {metadata['validation_metrics']['recall']}")
    print(f"  Val Precision:   {metadata['validation_metrics']['precision']}")
    if best_model.get("test_metrics"):
        print(f"  Test F1:         {best_model['test_metrics']['f1']}")
        print(f"  Test Recall:     {best_model['test_metrics']['recall']}")
    print(f"\n  Model:    {XGB_OUTPUT}")
    print(f"  Metadata: {metadata_path}")
    print(f"  Metrics:  {metrics_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
