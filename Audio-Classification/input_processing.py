"""
input_processing.py — Standalone audio preprocessing for YAMNet inference.

Replicates torch_audioset's WaveformToInput transform using only librosa + numpy.
No PyTorch dependency required at inference time.

YAMNet Mel-Spectrogram Parameters (from AudioSet / torch_audioset):
  - Sample rate: 16000 Hz
  - STFT window: 25 ms (400 samples)
  - STFT hop:    10 ms (160 samples)
  - Mel bands:   64
  - Mel range:   125 Hz – 7500 Hz
  - Patch size:  96 frames × 64 mels (0.96 seconds per patch)
  - Patch hop:   48 frames (0.48 seconds)
  - Amplitude:   log-scaled (stabilised with +0.01 offset, matching TF YAMNet)

Usage:
    from input_processing import audio_to_patches
    patches = audio_to_patches("path/to/audio.wav")
    # patches.shape = (N, 1, 96, 64)  float32, ready for yamnet.onnx
"""

import numpy as np
import librosa
import soundfile as sf
import warnings
import os


# ─── YAMNet Constants ───────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
STFT_WINDOW_SECONDS = 0.025      # 25 ms → 400 samples
STFT_HOP_SECONDS = 0.010         # 10 ms → 160 samples
MEL_BANDS = 64
MEL_MIN_HZ = 125.0
MEL_MAX_HZ = 7500.0
LOG_OFFSET = 0.01                # Matches TF YAMNet's stabilisation constant
PATCH_FRAMES = 96                # 0.96 s per patch
PATCH_HOP_FRAMES = 48            # 0.48 s hop between patches

# Derived
STFT_WINDOW_SAMPLES = int(SAMPLE_RATE * STFT_WINDOW_SECONDS)   # 400
STFT_HOP_SAMPLES = int(SAMPLE_RATE * STFT_HOP_SECONDS)         # 160
FFT_LENGTH = 512             # next power of 2 from 400, matches torch_audioset


# ─── Core Functions ─────────────────────────────────────────────────────────────

def load_audio(path: str) -> np.ndarray:
    """
    Load an audio file, convert to mono float32 at 16 kHz.
    Handles .wav, .mp3, .flac, .ogg, .m4a via soundfile + librosa fallback.
    
    Returns:
        waveform: np.ndarray float32, shape (num_samples,), range roughly [-1, 1]
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="PySoundFile failed.*")
        warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
        
        try:
            data, sr = sf.read(path)
        except Exception:
            # Fallback for mp3/m4a/ogg that libsndfile can't handle
            data, sr = librosa.load(path, sr=None, mono=False)
            if data.ndim == 2:
                data = data.T  # librosa returns (channels, samples), we want (samples, channels)
    
    # Convert to mono
    if data.ndim > 1:
        data = data.mean(axis=1 if data.shape[1] <= data.shape[0] else 0)
    
    data = data.astype(np.float32)
    
    # Resample to 16 kHz if needed
    if sr != SAMPLE_RATE:
        data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)
    
    # Clip to valid range
    data = np.clip(data, -1.0, 1.0)
    
    return data


def waveform_to_log_mel(waveform: np.ndarray) -> np.ndarray:
    """
    Compute log-mel spectrogram matching YAMNet's parameters.
    
    Args:
        waveform: float32 mono audio at 16 kHz, shape (num_samples,)
    
    Returns:
        log_mel: float32 array, shape (num_frames, 64)
    """
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=SAMPLE_RATE,
        n_fft=FFT_LENGTH,
        hop_length=STFT_HOP_SAMPLES,
        win_length=STFT_WINDOW_SAMPLES,
        n_mels=MEL_BANDS,
        fmin=MEL_MIN_HZ,
        fmax=MEL_MAX_HZ,
        power=2.0,           # Power spectrogram — matches torch_audioset
    )
    
    # Log-scale with offset (matches TF YAMNet)
    log_mel = np.log(mel_spec.T + LOG_OFFSET)  # Transpose to (frames, mels)
    
    return log_mel.astype(np.float32)


def log_mel_to_patches(log_mel: np.ndarray) -> np.ndarray:
    """
    Frame a log-mel spectrogram into overlapping patches for YAMNet.
    
    Args:
        log_mel: float32 array, shape (num_frames, 64)
    
    Returns:
        patches: float32 array, shape (N, 1, 96, 64) — ready for ONNX
        Returns empty array (0, 1, 96, 64) if audio is too short.
    """
    num_frames = log_mel.shape[0]
    
    if num_frames < PATCH_FRAMES:
        # Audio is too short for even one patch — pad with zeros
        padded = np.zeros((PATCH_FRAMES, MEL_BANDS), dtype=np.float32)
        padded[:num_frames, :] = log_mel
        return padded[np.newaxis, np.newaxis, :, :]  # (1, 1, 96, 64)
    
    patches = []
    start = 0
    while start + PATCH_FRAMES <= num_frames:
        patch = log_mel[start:start + PATCH_FRAMES, :]  # (96, 64)
        patches.append(patch)
        start += PATCH_HOP_FRAMES
    
    patches = np.array(patches, dtype=np.float32)  # (N, 96, 64)
    patches = patches[:, np.newaxis, :, :]          # (N, 1, 96, 64)
    
    return patches


def audio_to_patches(path: str) -> np.ndarray:
    """
    End-to-end: audio file → YAMNet-ready patches.
    
    Args:
        path: path to any audio file
    
    Returns:
        patches: float32 array, shape (N, 1, 96, 64)
    """
    waveform = load_audio(path)
    log_mel = waveform_to_log_mel(waveform)
    patches = log_mel_to_patches(log_mel)
    return patches


def audio_to_patches_from_waveform(waveform: np.ndarray) -> np.ndarray:
    """
    Waveform (already loaded, mono, 16kHz) → YAMNet-ready patches.
    
    Args:
        waveform: float32 array at 16 kHz, shape (num_samples,)
    
    Returns:
        patches: float32 array, shape (N, 1, 96, 64)
    """
    log_mel = waveform_to_log_mel(waveform)
    patches = log_mel_to_patches(log_mel)
    return patches


# ─── Audio Diagnostics ──────────────────────────────────────────────────────────

def diagnose_audio(waveform: np.ndarray, filename: str = "") -> dict:
    """
    Check for common audio quality issues.
    Returns a dict of diagnostics.
    """
    duration_s = len(waveform) / SAMPLE_RATE
    rms = np.sqrt(np.mean(waveform ** 2))
    peak = np.max(np.abs(waveform))
    clipped_samples = np.sum(np.abs(waveform) >= 0.999)
    clipped_pct = clipped_samples / len(waveform) * 100
    
    issues = []
    if duration_s < 0.5:
        issues.append(f"VERY SHORT ({duration_s:.2f}s) — may produce unreliable results")
    if rms < 0.001:
        issues.append(f"SILENT/NEAR-SILENT (RMS={rms:.6f}) — likely no useful audio")
    if clipped_pct > 1.0:
        issues.append(f"CLIPPED ({clipped_pct:.1f}% samples at max) — distorted audio")
    if peak < 0.01:
        issues.append(f"EXTREMELY QUIET (peak={peak:.4f}) — may need amplification")
    
    return {
        "filename": filename or "unknown",
        "duration_s": duration_s,
        "rms": rms,
        "peak": peak,
        "clipped_pct": clipped_pct,
        "issues": issues,
        "ok": len(issues) == 0
    }


# ─── Self-Test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("input_processing.py — Self-test")
    print(f"  STFT window: {STFT_WINDOW_SAMPLES} samples ({STFT_WINDOW_SECONDS*1000:.0f} ms)")
    print(f"  STFT hop:    {STFT_HOP_SAMPLES} samples ({STFT_HOP_SECONDS*1000:.0f} ms)")
    print(f"  Mel bands:   {MEL_BANDS} ({MEL_MIN_HZ}-{MEL_MAX_HZ} Hz)")
    print(f"  Patch size:  {PATCH_FRAMES} frames × {MEL_BANDS} mels")
    print(f"  Patch hop:   {PATCH_HOP_FRAMES} frames")
    
    # Test with synthetic 6-second audio
    duration = 6.0
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), dtype=np.float32)
    test_waveform = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    
    log_mel = waveform_to_log_mel(test_waveform)
    print(f"\n  6s tone -> log-mel shape: {log_mel.shape}")  # Expected: (~375, 64)
    
    patches = log_mel_to_patches(log_mel)
    print(f"  6s tone -> patches shape: {patches.shape}")    # Expected: (N, 1, 96, 64)
    print(f"  Number of patches: {patches.shape[0]}")
    print(f"  Patch value range: [{patches.min():.3f}, {patches.max():.3f}]")
    
    diag = diagnose_audio(test_waveform, "synthetic_440hz_6s")
    print(f"\n  Diagnostics: {diag}")
    
    # Test with very short audio (edge case)
    short_waveform = np.zeros(8000, dtype=np.float32)  # 0.5s
    short_patches = audio_to_patches_from_waveform(short_waveform)
    print(f"\n  0.5s silence -> patches shape: {short_patches.shape}")
    
    print("\n  Self-test passed.")
