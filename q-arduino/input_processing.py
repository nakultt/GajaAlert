"""Standalone audio preprocessing for YAMNet inference.

Vendored from github.com/Team-Highest/Audio-Classification (input_processing.py,
master branch) so the elephant-sound embeddings computed here numerically match
how audio/elephant_xgb.json was trained. Trimmed to what the streaming
classifier needs -- no file-loading helpers, only the in-memory waveform path.

YAMNet Mel-Spectrogram Parameters (from AudioSet / torch_audioset):
  - Sample rate: 16000 Hz
  - STFT window: 25 ms (400 samples)
  - STFT hop:    10 ms (160 samples)
  - Mel bands:   64
  - Mel range:   125 Hz - 7500 Hz
  - Patch size:  96 frames x 64 mels (0.96 seconds per patch)
  - Patch hop:   48 frames (0.48 seconds)
  - Amplitude:   log-scaled (stabilised with +0.01 offset, matching TF YAMNet)
"""

import numpy as np
import librosa

SAMPLE_RATE = 16000
STFT_WINDOW_SECONDS = 0.025      # 25 ms -> 400 samples
STFT_HOP_SECONDS = 0.010         # 10 ms -> 160 samples
MEL_BANDS = 64
MEL_MIN_HZ = 125.0
MEL_MAX_HZ = 7500.0
LOG_OFFSET = 0.01                # Matches TF YAMNet's stabilisation constant
PATCH_FRAMES = 96                # 0.96 s per patch
PATCH_HOP_FRAMES = 48            # 0.48 s hop between patches

STFT_WINDOW_SAMPLES = int(SAMPLE_RATE * STFT_WINDOW_SECONDS)   # 400
STFT_HOP_SAMPLES = int(SAMPLE_RATE * STFT_HOP_SECONDS)         # 160
FFT_LENGTH = 512             # next power of 2 from 400, matches torch_audioset


def waveform_to_log_mel(waveform: np.ndarray) -> np.ndarray:
    """Compute log-mel spectrogram matching YAMNet's parameters.

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
        power=2.0,           # Power spectrogram -- matches torch_audioset
    )
    log_mel = np.log(mel_spec.T + LOG_OFFSET)  # Transpose to (frames, mels)
    return log_mel.astype(np.float32)


def log_mel_to_patches(log_mel: np.ndarray) -> np.ndarray:
    """Frame a log-mel spectrogram into overlapping patches for YAMNet.

    Args:
        log_mel: float32 array, shape (num_frames, 64)

    Returns:
        patches: float32 array, shape (N, 1, 96, 64) -- ready for ONNX.
        Returns a single zero-padded patch if audio is too short.
    """
    num_frames = log_mel.shape[0]

    if num_frames < PATCH_FRAMES:
        padded = np.zeros((PATCH_FRAMES, MEL_BANDS), dtype=np.float32)
        padded[:num_frames, :] = log_mel
        return padded[np.newaxis, np.newaxis, :, :]  # (1, 1, 96, 64)

    patches = []
    start = 0
    while start + PATCH_FRAMES <= num_frames:
        patches.append(log_mel[start:start + PATCH_FRAMES, :])  # (96, 64)
        start += PATCH_HOP_FRAMES

    patches = np.array(patches, dtype=np.float32)  # (N, 96, 64)
    patches = patches[:, np.newaxis, :, :]          # (N, 1, 96, 64)
    return patches


def audio_to_patches_from_waveform(waveform: np.ndarray) -> np.ndarray:
    """Waveform (already loaded, mono, 16kHz) -> YAMNet-ready patches.

    Args:
        waveform: float32 array at 16 kHz, shape (num_samples,)

    Returns:
        patches: float32 array, shape (N, 1, 96, 64)
    """
    log_mel = waveform_to_log_mel(waveform)
    return log_mel_to_patches(log_mel)
