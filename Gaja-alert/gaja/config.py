"""Pipeline configuration with optional gaja.json overrides."""

import dataclasses
import json
import logging
from dataclasses import dataclass

log = logging.getLogger("gaja.config")


@dataclass
class Config:
    # audio trigger
    sample_rate: int = 16000
    fft_window_s: float = 1.0
    hop_s: float = 0.5
    # Elephant rumble fundamentals are ~10-40 Hz but phone mics roll off below
    # ~50-100 Hz, so the band leans on low harmonics up to 250 Hz.
    band_low_hz: float = 10.0
    band_high_hz: float = 250.0
    ratio_on: float = 0.55
    ratio_off: float = 0.35
    min_rms: float = 150.0
    consecutive_windows: int = 2
    cooldown_s: float = 60.0
    # vision
    frames_per_check: int = 2
    frame_gap_s: float = 0.7
    confirm_confidence: float = 0.6
    # llm endpoints: vision = llama.cpp E4B+mmproj, text = geniex NPU/GPU split E2B
    vision_llm_base: str = "http://127.0.0.1:8080"
    text_llm_base: str = "http://127.0.0.1:8082"
    llm_timeout_s: float = 120.0
    llm_retries: int = 2
    # servers
    sensor_port: int = 9000
    alert_port: int = 9001
    # output
    incidents_dir: str = "incidents"
    location_name: str = "Gaja camp perimeter"

    @classmethod
    def load(cls, path: str = "gaja.json") -> "Config":
        cfg = cls()
        try:
            with open(path, encoding="utf-8") as f:
                overrides = json.load(f)
        except FileNotFoundError:
            return cfg
        except (OSError, json.JSONDecodeError) as e:
            log.warning("Could not read %s (%s); using defaults", path, e)
            return cfg
        known = {f.name for f in dataclasses.fields(cls)}
        for key, value in overrides.items():
            if key in known:
                setattr(cfg, key, value)
            else:
                log.warning("Ignoring unknown config key %r in %s", key, path)
        return cfg
