"""Incident pipeline worker: trigger -> frames -> Gemma confirm -> alert -> log.

Runs in its own thread so slow LLM calls never block the WebSocket receive
loop. Re-triggers during a running incident coalesce into one pending event
(and are mostly absorbed by the trigger cooldown anyway).
"""

import datetime
import json
import logging
import threading
import time

from .audio_trigger import TriggerEvent

log = logging.getLogger("gaja.pipeline")


class Pipeline(threading.Thread):
    def __init__(self, cfg, llm, frame_source, alert_sink, incident_log):
        """frame_source() -> bytes | None (latest JPEG); alert_sink(bytes) sends 0x03."""
        super().__init__(daemon=True, name="gaja-pipeline")
        self.cfg = cfg
        self.llm = llm
        self.frame_source = frame_source
        self.alert_sink = alert_sink
        self.log = incident_log
        self._event = threading.Event()
        self._pending: TriggerEvent | None = None

    def notify_trigger(self, ev: TriggerEvent):
        self._pending = ev
        self._event.set()

    def run(self):
        log.info("Pipeline worker started")
        while True:
            self._event.wait()
            self._event.clear()
            ev = self._pending
            if ev is None:
                continue
            try:
                self.handle_incident(ev)
            except Exception:
                log.exception("Incident handling failed")

    def handle_incident(self, ev: TriggerEvent):
        cfg = self.cfg
        incident_id = self.log.new_id()
        when = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        base = {
            "id": incident_id,
            "timestamp": when,
            "trigger": {"ratio": round(ev.ratio, 3), "rms": round(ev.rms, 1)},
        }
        log.info("Incident %s: audio trigger ratio=%.2f", incident_id, ev.ratio)

        frames = []
        for i in range(cfg.frames_per_check):
            if i:
                time.sleep(cfg.frame_gap_s)
            jpeg = self.frame_source()
            if jpeg is not None:
                frames.append(jpeg)
        if not frames:
            log.warning("Incident %s: no video frames available", incident_id)
            self.log.record({**base, "status": "no_video"})
            return

        det = self.llm.detect_elephant(frames)
        if det is None:
            self.log.record({**base, "status": "llm_down"}, frames)
            return
        detection = {"elephant": det.elephant, "confidence": det.confidence,
                     "notes": det.notes}
        if not (det.elephant and det.confidence >= cfg.confirm_confidence):
            log.info("Incident %s: rejected by vision (elephant=%s conf=%.2f)",
                     incident_id, det.elephant, det.confidence)
            self.log.record({**base, "status": "rejected", "detection": detection},
                            frames)
            return

        alert = self.llm.generate_alert(det, ev.ratio, when)
        payload = {
            "type": "elephant_alert",
            "id": incident_id,
            "timestamp": when,
            "confidence": det.confidence,
            "report": alert.report,
            "alerts": alert.alerts,
            "location": cfg.location_name,
        }
        self.alert_sink(json.dumps(payload, ensure_ascii=False).encode())
        log.info("Incident %s: ALERT broadcast (fallback=%s)", incident_id, alert.fallback)
        self.log.record({**base, "status": "alerted", "detection": detection,
                         "alert": payload, "alert_fallback": alert.fallback}, frames)
