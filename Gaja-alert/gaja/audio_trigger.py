"""Low-frequency band-energy trigger for elephant rumbles.

numpy-only (no scipy): every hop we take an FFT of the last window and compare
the energy in the elephant band (10-250 Hz by default) against total energy.
A ratio is self-normalizing against unknown phone-mic gain and distance.
"""

import logging
import time
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("gaja.trigger")

_IDLE, _COOLDOWN = 0, 1


@dataclass
class TriggerEvent:
    timestamp: float
    ratio: float
    rms: float


class BandTrigger:
    def __init__(self, cfg):
        self.cfg = cfg
        self.window_n = int(cfg.sample_rate * cfg.fft_window_s)
        self.hop_n = int(cfg.sample_rate * cfg.hop_s)
        self._buf = np.zeros(self.window_n, dtype=np.int16)
        self._filled = 0            # samples seen so far (saturates at window_n)
        self._since_hop = 0
        self._hann = np.hanning(self.window_n)
        freqs = np.fft.rfftfreq(self.window_n, d=1.0 / cfg.sample_rate)
        self._band = (freqs >= cfg.band_low_hz) & (freqs <= cfg.band_high_hz)
        self._total = freqs >= cfg.band_low_hz  # exclude DC and sub-band drift
        self._state = _IDLE
        self._hits = 0
        self._fired_at = 0.0

    def analyze_window(self, window: np.ndarray) -> tuple[float, float]:
        """Pure per-window analysis: returns (band_energy_ratio, rms)."""
        x = window.astype(np.float64)
        x -= x.mean()
        rms = float(np.sqrt(np.mean(x * x)))
        power = np.abs(np.fft.rfft(x * self._hann)) ** 2
        total = float(power[self._total].sum())
        if total <= 0.0:
            return 0.0, rms
        return float(power[self._band].sum()) / total, rms

    def feed(self, chunk: np.ndarray) -> TriggerEvent | None:
        """Feed a PCM chunk; returns a TriggerEvent when the detector fires."""
        event = None
        pos = 0
        while pos < len(chunk):
            take = min(len(chunk) - pos, self.hop_n - self._since_hop)
            part = chunk[pos:pos + take]
            self._buf = np.roll(self._buf, -take)
            self._buf[-take:] = part
            self._filled = min(self._filled + take, self.window_n)
            self._since_hop += take
            pos += take
            if self._since_hop >= self.hop_n:
                self._since_hop = 0
                if self._filled >= self.window_n:
                    ev = self._check(self._buf)
                    event = event or ev
        return event

    def _check(self, window: np.ndarray) -> TriggerEvent | None:
        cfg = self.cfg
        ratio, rms = self.analyze_window(window)
        log.debug("hop ratio=%.3f rms=%.0f state=%s hits=%d",
                  ratio, rms, "COOLDOWN" if self._state else "IDLE", self._hits)
        if self._state == _COOLDOWN:
            if time.time() - self._fired_at >= cfg.cooldown_s and ratio < cfg.ratio_off:
                self._state = _IDLE
                self._hits = 0
            return None
        if rms < cfg.min_rms:
            self._hits = 0
            return None
        if ratio >= cfg.ratio_on:
            self._hits += 1
            if self._hits >= cfg.consecutive_windows:
                self._state = _COOLDOWN
                self._fired_at = time.time()
                self._hits = 0
                log.info("TRIGGER fired: ratio=%.3f rms=%.0f", ratio, rms)
                return TriggerEvent(self._fired_at, ratio, rms)
        else:
            self._hits = 0
        return None
