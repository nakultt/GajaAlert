"""Streaming elephant-sound classifier: YAMNet (ONNX) embeddings + XGBoost.

Runs on the UNO Q itself, fed directly from the Q-Mobile app's mic stream
(0x02 chunks arriving over the phone<->UNO-Q websocket in main.py) -- no wav
files, no separate mic capture. Uses the same models Gaja alert/arm_server.py
expects a 0x04 event to be backed by (audio/yamnet.onnx + audio/elephant_xgb.json
at the repo root, one level up from q-arduino/).

State machine mirrors Gaja alert/gaja/audio_trigger.py's BandTrigger
(IDLE/COOLDOWN, consecutive-hits-to-fire, hysteresis on release) so the two
trigger implementations in this codebase read the same way, even though one
is a frequency-band heuristic and this one is a trained classifier.
"""

import logging
import os
import time
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
from xgboost import XGBClassifier

from input_processing import SAMPLE_RATE, audio_to_patches_from_waveform

log = logging.getLogger("qarduino.audio_classifier")

_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "audio")
YAMNET_ONNX = os.path.join(_MODEL_DIR, "yamnet.onnx")
XGB_MODEL = os.path.join(_MODEL_DIR, "elephant_xgb.json")

WINDOW_S = 2.0              # rolling audio buffer fed to the classifier each hop
HOP_S = 1.0                 # how often we run inference
PROB_ON = 0.5                # fire threshold (matches the sibling repo's default)
PROB_OFF = 0.35               # must drop below this (with cooldown elapsed) to re-arm
CONSECUTIVE_WINDOWS = 2       # hops above PROB_ON before actually firing
COOLDOWN_S = 60.0

_IDLE, _COOLDOWN = 0, 1


@dataclass
class TriggerEvent:
    timestamp: float
    confidence: float


class ElephantAudioClassifier:
    def __init__(self):
        log.info("Loading YAMNet ONNX from %s", YAMNET_ONNX)
        self._yamnet = ort.InferenceSession(YAMNET_ONNX, providers=["CPUExecutionProvider"])
        log.info("Loading XGBoost classifier from %s", XGB_MODEL)
        self._clf = XGBClassifier()
        self._clf.load_model(XGB_MODEL)

        self.window_n = int(SAMPLE_RATE * WINDOW_S)
        self.hop_n = int(SAMPLE_RATE * HOP_S)
        self._buf = np.zeros(self.window_n, dtype=np.int16)
        self._filled = 0
        self._since_hop = 0
        self._state = _IDLE
        self._hits = 0
        self._fired_at = 0.0

    def _predict(self, window: np.ndarray) -> float:
        waveform = window.astype(np.float32) / 32768.0
        patches = audio_to_patches_from_waveform(waveform)
        embeddings = self._yamnet.run(["clip_embedding"], {"log_mel_patches": patches})[0]
        probs = self._clf.predict_proba(embeddings)[:, 1]
        return float(np.mean(probs))

    def feed(self, chunk: np.ndarray) -> "TriggerEvent | None":
        """Feed a mono int16 PCM chunk; returns a TriggerEvent when the
        classifier fires. Mirrors BandTrigger.feed's hop-accumulation loop."""
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

    def feed_bytes(self, payload: bytes) -> "TriggerEvent | None":
        """Convenience wrapper for raw little-endian int16 PCM bytes off the
        websocket (main.py's 0x02 payload)."""
        chunk = np.frombuffer(payload, dtype=np.int16)
        return self.feed(chunk)

    def _check(self, window: np.ndarray) -> "TriggerEvent | None":
        prob = self._predict(window)
        log.debug("hop prob=%.3f state=%s hits=%d",
                   prob, "COOLDOWN" if self._state else "IDLE", self._hits)
        if self._state == _COOLDOWN:
            if time.time() - self._fired_at >= COOLDOWN_S and prob < PROB_OFF:
                self._state = _IDLE
                self._hits = 0
            return None
        if prob >= PROB_ON:
            self._hits += 1
            if self._hits >= CONSECUTIVE_WINDOWS:
                self._state = _COOLDOWN
                self._fired_at = time.time()
                self._hits = 0
                log.info("AUDIO TRIGGER fired: confidence=%.3f", prob)
                return TriggerEvent(self._fired_at, prob)
        else:
            self._hits = 0
        return None
