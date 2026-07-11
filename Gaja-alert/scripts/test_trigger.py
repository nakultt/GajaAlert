"""Unit checks for the audio band trigger (no hardware needed).

Usage:  uv run python scripts/test_trigger.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaja.audio_trigger import BandTrigger
from gaja.config import Config


def tone(freqs, seconds, sr=16000, amp=3000):
    t = np.arange(int(sr * seconds)) / sr
    x = sum(np.sin(2 * np.pi * f * t) for f in freqs)
    x = x / max(len(freqs), 1) * amp
    return x.astype(np.int16)


def noise(seconds, sr=16000, amp=3000):
    rng = np.random.default_rng(42)
    return (rng.standard_normal(int(sr * seconds)) * amp / 3).astype(np.int16)


def feed_chunks(trig, samples, chunk=1600):
    events = []
    for i in range(0, len(samples), chunk):
        ev = trig.feed(samples[i:i + chunk])
        if ev:
            events.append(ev)
    return events


def check(name, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return ok


def main():
    cfg = Config.load()
    results = []

    # 1. Elephant-band tone (60+120 Hz) fires
    trig = BandTrigger(cfg)
    events = feed_chunks(trig, tone([60, 120], 4.0))
    results.append(check("low-frequency rumble tone fires", len(events) == 1))
    if events:
        print(f"        ratio={events[0].ratio:.3f} rms={events[0].rms:.0f}")

    # 2. High-frequency content does not fire
    trig = BandTrigger(cfg)
    events = feed_chunks(trig, tone([500, 1500, 3000], 4.0))
    results.append(check("high-frequency tone does not fire", len(events) == 0))

    # 3. White noise does not fire (broadband -> low band ratio)
    trig = BandTrigger(cfg)
    events = feed_chunks(trig, noise(4.0))
    results.append(check("white noise does not fire", len(events) == 0))

    # 4. Silence does not fire (min_rms gate)
    trig = BandTrigger(cfg)
    events = feed_chunks(trig, np.zeros(16000 * 4, dtype=np.int16))
    results.append(check("silence does not fire", len(events) == 0))

    # 5. Sustained tone fires exactly once, then cooldown holds
    trig = BandTrigger(cfg)
    events = feed_chunks(trig, tone([60, 120], 10.0))
    results.append(check("sustained tone fires exactly once (cooldown)", len(events) == 1))

    # 6. Cooldown expires -> can fire again
    cfg2 = Config.load()
    cfg2.cooldown_s = 0.5
    trig = BandTrigger(cfg2)
    e1 = feed_chunks(trig, tone([60, 120], 3.0))
    time.sleep(0.6)
    feed_chunks(trig, np.zeros(16000, dtype=np.int16))  # quiet hop to release
    e2 = feed_chunks(trig, tone([60, 120], 3.0))
    results.append(check("fires again after cooldown", len(e1) == 1 and len(e2) == 1))

    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
