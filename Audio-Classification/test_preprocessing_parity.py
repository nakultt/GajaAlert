#!/usr/bin/env python3
"""
test_preprocessing_parity.py — Verify input_processing.py matches torch_audioset.

Colab-optimized: auto-installs deps.
Run:  python test_preprocessing_parity.py [optional_wav_path]

Tests:
  1. Synthetic waveform: compares patches from both pipelines
  2. (Optional) Real WAV: compares patches, ONNX embeddings, XGBoost probs
  3. Reports PASS/FAIL with numeric tolerances

Since we retrain using input_processing.py exclusively (Option B from the plan),
this test is diagnostic — it quantifies the gap but does not block retraining.
"""

import os
import sys
import argparse
import warnings
import time

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
    os.system("pip install -q librosa soundfile numpy onnxruntime xgboost "
              "torch torchaudio")
    os.system("pip install -q git+https://github.com/w-hc/torch_audioset.git")

colab_setup()

import numpy as np

# ── Test harness ────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOLERANCE_PATCH_MAX_ABS = 0.5   # log-mel values can differ due to filterbank impl
TOLERANCE_EMBED_MAX_ABS = 0.1   # ONNX embedding tolerance
TOLERANCE_PROB_MAX_ABS = 0.15   # XGBoost probability tolerance


def generate_deterministic_waveform(duration_s: float = 6.0, seed: int = 42) -> np.ndarray:
    """Generate a reproducible test waveform: 440 Hz tone + noise."""
    rng = np.random.RandomState(seed)
    sr = 16000
    n_samples = int(sr * duration_s)
    t = np.linspace(0, duration_s, n_samples, dtype=np.float32)
    tone = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    noise = 0.05 * rng.randn(n_samples).astype(np.float32)
    waveform = tone + noise
    return waveform


def test_deployment_pipeline(waveform: np.ndarray) -> dict:
    """Run waveform through input_processing.py (deployment path)."""
    sys.path.insert(0, SCRIPT_DIR)
    from input_processing import audio_to_patches_from_waveform
    patches = audio_to_patches_from_waveform(waveform)
    return {"patches": patches, "source": "input_processing.py (librosa)"}


def test_training_pipeline(waveform: np.ndarray) -> dict | None:
    """Run waveform through torch_audioset (training path)."""
    try:
        import torch
        from torch_audioset.data.torch_input_processing import WaveformToInput
    except ImportError:
        print("[SKIP] torch_audioset not installed — cannot compare training pipeline.")
        print("       Install with: pip install git+https://github.com/w-hc/torch_audioset.git")
        return None

    wti = WaveformToInput()
    waveform_tensor = torch.from_numpy(waveform).unsqueeze(0).float()
    patches_tensor = wti(waveform_tensor, 16000)
    patches = patches_tensor.numpy()
    return {"patches": patches, "source": "torch_audioset.WaveformToInput"}


def compare_patches(deploy: dict, train: dict) -> dict:
    """Compare patch arrays from both pipelines."""
    dp = deploy["patches"]
    tp = train["patches"]

    results = {
        "deploy_shape": list(dp.shape),
        "train_shape": list(tp.shape),
        "shape_match": dp.shape == tp.shape,
        "deploy_n_patches": dp.shape[0],
        "train_n_patches": tp.shape[0],
    }

    if dp.shape == tp.shape:
        diff = np.abs(dp - tp)
        results["max_abs_diff"] = float(np.max(diff))
        results["mean_abs_diff"] = float(np.mean(diff))
        results["median_abs_diff"] = float(np.median(diff))
        results["std_abs_diff"] = float(np.std(diff))
        results["pct_within_01"] = float(np.mean(diff < 0.1) * 100)
        results["pct_within_001"] = float(np.mean(diff < 0.01) * 100)

        # Check correlation per patch
        correlations = []
        for i in range(dp.shape[0]):
            d_flat = dp[i].flatten()
            t_flat = tp[i].flatten()
            corr = np.corrcoef(d_flat, t_flat)[0, 1]
            correlations.append(corr)
        results["mean_correlation"] = float(np.mean(correlations))
        results["min_correlation"] = float(np.min(correlations))

        results["pass_tolerance"] = results["max_abs_diff"] < TOLERANCE_PATCH_MAX_ABS
    else:
        # Different shapes — compare what we can
        min_patches = min(dp.shape[0], tp.shape[0])
        if min_patches > 0:
            diff = np.abs(dp[:min_patches] - tp[:min_patches])
            results["max_abs_diff_overlap"] = float(np.max(diff))
            results["mean_abs_diff_overlap"] = float(np.mean(diff))
        results["pass_tolerance"] = False

    return results


def compare_embeddings(deploy_patches: np.ndarray, train_patches: np.ndarray) -> dict | None:
    """Compare ONNX embeddings from both patch sets."""
    try:
        import onnxruntime as ort
    except ImportError:
        return None

    onnx_path = os.path.join(SCRIPT_DIR, "output", "yamnet.onnx")
    if not os.path.exists(onnx_path):
        print(f"[SKIP] YAMNet ONNX not found at {onnx_path}")
        return None

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    deploy_emb = session.run(["clip_embedding"],
                             {"log_mel_patches": deploy_patches})[0]
    train_emb = session.run(["clip_embedding"],
                            {"log_mel_patches": train_patches})[0]

    results = {
        "deploy_shape": list(deploy_emb.shape),
        "train_shape": list(train_emb.shape),
    }

    min_n = min(deploy_emb.shape[0], train_emb.shape[0])
    if min_n > 0:
        diff = np.abs(deploy_emb[:min_n] - train_emb[:min_n])
        results["max_abs_diff"] = float(np.max(diff))
        results["mean_abs_diff"] = float(np.mean(diff))
        results["pass_tolerance"] = results["max_abs_diff"] < TOLERANCE_EMBED_MAX_ABS
    else:
        results["pass_tolerance"] = False

    return results


def compare_xgb_probs(deploy_patches: np.ndarray, train_patches: np.ndarray) -> dict | None:
    """Compare XGBoost probabilities from both patch sets."""
    try:
        import onnxruntime as ort
        from xgboost import XGBClassifier
    except ImportError:
        return None

    onnx_path = os.path.join(SCRIPT_DIR, "output", "yamnet.onnx")
    xgb_path = os.path.join(SCRIPT_DIR, "output", "elephant_xgb.json")
    if not os.path.exists(onnx_path) or not os.path.exists(xgb_path):
        return None

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    clf = XGBClassifier()
    clf.load_model(xgb_path)

    deploy_emb = session.run(["clip_embedding"],
                             {"log_mel_patches": deploy_patches})[0]
    train_emb = session.run(["clip_embedding"],
                            {"log_mel_patches": train_patches})[0]

    deploy_probs = clf.predict_proba(deploy_emb)[:, 1]
    train_probs = clf.predict_proba(train_emb)[:, 1]

    deploy_avg = float(np.mean(deploy_probs))
    train_avg = float(np.mean(train_probs))

    results = {
        "deploy_avg_prob": deploy_avg,
        "train_avg_prob": train_avg,
        "prob_difference": abs(deploy_avg - train_avg),
        "deploy_patch_probs": [round(float(p), 4) for p in deploy_probs],
        "train_patch_probs": [round(float(p), 4) for p in train_probs[:len(deploy_probs)]],
        "pass_tolerance": abs(deploy_avg - train_avg) < TOLERANCE_PROB_MAX_ABS,
    }
    return results


def run_parity_test(wav_path: str | None = None):
    """Run the full parity test suite."""
    print("=" * 70)
    print("PREPROCESSING PARITY TEST")
    print("=" * 70)
    print(f"Purpose: Compare input_processing.py (deployment) vs torch_audioset (training)")
    print(f"Decision: We retrain using input_processing.py exclusively (Option B).")
    print(f"          This test is diagnostic — it quantifies the gap.\n")

    # ── Test 1: Synthetic waveform ──────────────────────────────────────────
    print("─── Test 1: Synthetic Waveform (6s, 440Hz + noise) ───")
    waveform = generate_deterministic_waveform()
    print(f"  Waveform: {waveform.shape}, range [{waveform.min():.3f}, {waveform.max():.3f}]")

    deploy = test_deployment_pipeline(waveform)
    print(f"  Deploy patches: {deploy['patches'].shape}")

    train = test_training_pipeline(waveform)
    if train is not None:
        print(f"  Train patches:  {train['patches'].shape}")
        results = compare_patches(deploy, train)

        print(f"\n  Patch comparison:")
        print(f"    Shape match:     {results['shape_match']}")
        if results.get("max_abs_diff") is not None:
            print(f"    Max abs diff:    {results['max_abs_diff']:.6f}")
            print(f"    Mean abs diff:   {results['mean_abs_diff']:.6f}")
            print(f"    Mean correlation: {results['mean_correlation']:.6f}")
            print(f"    % within 0.1:    {results['pct_within_01']:.1f}%")
            print(f"    % within 0.01:   {results['pct_within_001']:.1f}%")
        print(f"    PASS (tol={TOLERANCE_PATCH_MAX_ABS}): "
              f"{'✓' if results['pass_tolerance'] else '✗'}")

        # Compare embeddings if models available
        emb_results = compare_embeddings(deploy["patches"], train["patches"])
        if emb_results:
            print(f"\n  Embedding comparison:")
            print(f"    Max abs diff:  {emb_results['max_abs_diff']:.6f}")
            print(f"    Mean abs diff: {emb_results['mean_abs_diff']:.6f}")
            print(f"    PASS (tol={TOLERANCE_EMBED_MAX_ABS}): "
                  f"{'✓' if emb_results['pass_tolerance'] else '✗'}")

        xgb_results = compare_xgb_probs(deploy["patches"], train["patches"])
        if xgb_results:
            print(f"\n  XGBoost probability comparison:")
            print(f"    Deploy avg prob: {xgb_results['deploy_avg_prob']:.4f}")
            print(f"    Train avg prob:  {xgb_results['train_avg_prob']:.4f}")
            print(f"    Difference:      {xgb_results['prob_difference']:.4f}")
            print(f"    PASS (tol={TOLERANCE_PROB_MAX_ABS}): "
                  f"{'✓' if xgb_results['pass_tolerance'] else '✗'}")
    else:
        print("  [SKIP] torch_audioset not available — deployment-only test.")

    # ── Test 2: Real WAV (if provided) ──────────────────────────────────────
    if wav_path and os.path.exists(wav_path):
        print(f"\n─── Test 2: Real WAV ({os.path.basename(wav_path)}) ───")
        sys.path.insert(0, SCRIPT_DIR)
        from input_processing import load_audio
        real_waveform = load_audio(wav_path)
        print(f"  Waveform: {real_waveform.shape}, {real_waveform.shape[0]/16000:.2f}s")

        deploy_real = test_deployment_pipeline(real_waveform)
        print(f"  Deploy patches: {deploy_real['patches'].shape}")

        train_real = test_training_pipeline(real_waveform)
        if train_real is not None:
            print(f"  Train patches:  {train_real['patches'].shape}")
            results_real = compare_patches(deploy_real, train_real)

            print(f"\n  Patch comparison:")
            print(f"    Shape match:     {results_real['shape_match']}")
            if results_real.get("max_abs_diff") is not None:
                print(f"    Max abs diff:    {results_real['max_abs_diff']:.6f}")
                print(f"    Mean correlation: {results_real['mean_correlation']:.6f}")
            print(f"    PASS: {'✓' if results_real['pass_tolerance'] else '✗'}")

            xgb_real = compare_xgb_probs(deploy_real["patches"], train_real["patches"])
            if xgb_real:
                print(f"\n  XGBoost probability comparison:")
                print(f"    Deploy: {xgb_real['deploy_avg_prob']:.4f}")
                print(f"    Train:  {xgb_real['train_avg_prob']:.4f}")
                print(f"    Diff:   {xgb_real['prob_difference']:.4f}")

    # ── Test 3: Deployment pipeline self-consistency ────────────────────────
    print(f"\n─── Test 3: Deployment Pipeline Self-Consistency ───")
    print("  Running deployment pipeline twice on same input...")
    w1 = generate_deterministic_waveform(seed=99)
    d1 = test_deployment_pipeline(w1)
    d2 = test_deployment_pipeline(w1)
    diff = np.max(np.abs(d1["patches"] - d2["patches"]))
    consistent = diff == 0.0
    print(f"  Max diff between runs: {diff:.2e}")
    print(f"  Deterministic: {'✓ PASS' if consistent else '✗ FAIL'}")

    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    print("Since we retrain using input_processing.py exclusively,")
    print("train/inference mismatch is eliminated by construction.")
    print("The parity test above is informational only.")
    if train is None:
        print("\nTo run the full comparison, install torch_audioset:")
        print("  pip install git+https://github.com/w-hc/torch_audioset.git")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Test preprocessing parity.")
    parser.add_argument("wav", nargs="?", default=None,
                        help="Optional WAV file for real-audio comparison")
    args = parser.parse_args()
    run_parity_test(args.wav)


if __name__ == "__main__":
    main()
