"""Pipeline configuration with optional gaja.json and .env overrides."""

import dataclasses
import json
import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

log = logging.getLogger("gaja.config")

# Loaded once at import time so every module that reads os.environ (e.g.
# sarvam_workflow.py's SARVAM_API_KEY) sees .env values without needing its
# own load_dotenv() call, as long as it imports gaja.config first.
load_dotenv()


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
    # YOLO-triggered VLM verification (arm_server.py)
    vlm_poll_interval_s: float = 1.5
    # Audio-triggered observation window (arm_server.py): after a UNO Q audio
    # trigger, collect audio + YOLO evidence for this long before deciding;
    # window_cooldown_s then blocks a new window so one ongoing event can't
    # produce duplicate incidents.
    audio_window_s: float = 13.0
    window_cooldown_s: float = 60.0
    # Languages sarvam_agent.py guarantees translate+TTS for on a confirmed
    # incident, regardless of what the tool-calling agent decides on its own.
    sarvam_languages: list = dataclasses.field(default_factory=lambda: ["hi", "ta"])
    # Keep agent turns low because tool calls/results consume the Qwen QAIRT
    # bundle's fixed 4096-token context even when results are truncated.
    sarvam_agent_max_turns: int = 4
    # Vision, text, and MCP decisions all use the hardware-compiled Qwen
    # QAIRT VLM bundle on the Hexagon NPU.
    vision_llm_base: str = "http://127.0.0.1:8081"
    text_llm_base: str = "http://127.0.0.1:8081"
    tool_llm_base: str = "http://127.0.0.1:8081"
    llm_timeout_s: float = 500.0
    llm_retries: int = 2
    # servers
    sensor_port: int = 9000
    alert_port: int = 9001
    dashboard_port: int = 9002
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
            overrides = {}
        except (OSError, json.JSONDecodeError) as e:
            log.warning("Could not read %s (%s); using defaults", path, e)
            overrides = {}
        known = {f.name for f in dataclasses.fields(cls)}
        for key, value in overrides.items():
            if key in known:
                setattr(cfg, key, value)
            else:
                log.warning("Ignoring unknown config key %r in %s", key, path)
        cfg._apply_env_overrides()
        return cfg

    def _apply_env_overrides(self):
        """GAJA_<FIELD> environment variables (e.g. from .env) win over
        gaja.json — deployment-specific values (ports, LLM endpoints) live
        per-machine and shouldn't require editing a committed file."""
        for f in dataclasses.fields(self):
            raw = os.environ.get(f"GAJA_{f.name.upper()}")
            if raw is None:
                continue
            current = getattr(self, f.name)
            try:
                if isinstance(current, bool):
                    setattr(self, f.name, raw.strip().lower() in ("1", "true", "yes", "on"))
                elif isinstance(current, int):
                    setattr(self, f.name, int(raw))
                elif isinstance(current, float):
                    setattr(self, f.name, float(raw))
                else:
                    setattr(self, f.name, raw)
            except ValueError:
                log.warning("Ignoring invalid GAJA_%s=%r", f.name.upper(), raw)
