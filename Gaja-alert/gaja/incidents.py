"""Incident logging: JSONL record per trigger plus the frames sent to Gemma."""

import json
import logging
import os
import time
import uuid

log = logging.getLogger("gaja.incidents")


class IncidentLog:
    def __init__(self, cfg):
        self.dir = cfg.incidents_dir
        self.path = os.path.join(self.dir, "incidents.jsonl")

    @staticmethod
    def new_id() -> str:
        return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]

    def record(self, incident: dict, frames: list[bytes] | None = None):
        """Append one incident line; save frames under incidents/<id>/."""
        try:
            os.makedirs(self.dir, exist_ok=True)
            names = []
            if frames:
                frame_dir = os.path.join(self.dir, incident["id"])
                os.makedirs(frame_dir, exist_ok=True)
                for i, jpeg in enumerate(frames):
                    name = f"frame_{i}.jpg"
                    with open(os.path.join(frame_dir, name), "wb") as f:
                        f.write(jpeg)
                    names.append(name)
            incident["frames"] = names
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(incident, ensure_ascii=False) + "\n")
            log.info("Recorded incident %s (%s)", incident["id"], incident.get("status"))
        except OSError as e:
            log.error("Failed to record incident: %s", e)
