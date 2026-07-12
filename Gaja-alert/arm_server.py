"""Gaja-alert edge server.

Two trigger sources feed the same verify/report/alert pipeline:
  - video: sensor phone streams JPEG frames (0x01) to ws://<laptop>:9000;
    local YOLO flags an elephant, starting continuous VLM polling.
  - audio: the UNO Q audio classifier (q-arduino/audio_classifier.py, YAMNet
    + XGBoost) sends a detection event (0x04 JSON) on the same :9000 socket,
    opening a cfg.audio_window_s observation window that collects both the
    audio confidence and the live YOLO state; video frames only reach the
    vision LLM on hops where YOLO currently sees an elephant.
Either source triggers a Qwen QAIRT/NPU vision confirm (:8081) against live
frames; once confirmed, the same NPU model generates the report/alert and
Sarvam MCP translates/TTS's it (guaranteed for
cfg.sarvam_languages, on top of whatever the tool-calling agent itself
decides), and the result broadcasts as 0x03 + JSON to receiver phones
connected on ws://<laptop>:9001. An audio trigger that the window can't
corroborate with YOLO+VLM is logged as "unconfirmed" but never broadcast.
A shared lock (_incident_lock) stops the video and audio paths from ever
double-filing a report for the same real sighting.
"""

import asyncio
import json
import logging
import queue
import threading
import time
import datetime

import cv2
import numpy as np
import websockets

from gaja.config import Config
from gaja.dashboard import start_dashboard
from gaja.incidents import IncidentLog
from gaja.llm import GemmaClient
from sarvam_agent import run_sarvam_agent
import ncnn


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("gaja.server")

cfg = Config.load()

# YOLO local detection (real-time overlay on the display window)
MODEL_PARAM_PATH = "yolo26n_ncnn_model/model.ncnn.param"
MODEL_BIN_PATH = "yolo26n_ncnn_model/model.ncnn.bin"
CONFIDENCE = 0.40
INPUT_W, INPUT_H = 640, 640

CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush"
]

net = ncnn.Net()
net.load_param(MODEL_PARAM_PATH)
net.load_model(MODEL_BIN_PATH)

# Queues to pass data from the async websocket thread to processing threads
video_queue = queue.Queue(maxsize=2)

# Latest raw JPEG for the pipeline (no re-encode, no queue contention)
_frame_lock = threading.Lock()
_latest_frame: bytes | None = None


def _store_frame(payload: bytes):
    global _latest_frame
    with _frame_lock:
        _latest_frame = payload


def frame_source() -> bytes | None:
    with _frame_lock:
        return _latest_frame


# Receiver phones connected on the alert port; owned by the asyncio loop
receivers: set = set()
_loop: asyncio.AbstractEventLoop | None = None


def send_alert(payload: bytes):
    """Broadcast 0x03 + JSON to all receiver phones (called from pipeline thread)."""
    if _loop is None:
        log.error("Alert not sent: server loop not running")
        return
    targets = list(receivers)
    if not targets:
        log.warning("No receiver phones connected; alert only logged")
    for ws in targets:
        try:
            asyncio.run_coroutine_threadsafe(ws.send(b"\x03" + payload), _loop)
        except Exception as e:
            log.error("Failed to queue alert for a receiver: %s", e)


# 1. Pipeline wiring
llm = GemmaClient(cfg)
incident_log = IncidentLog(cfg)


# 2. WEBSOCKET SERVERS
async def sensor_handler(websocket):
    log.info("Sensor client connected")
    try:
        async for message in websocket:
            if not isinstance(message, bytes) or len(message) == 0:
                continue

            header = message[0]
            payload = message[1:]

            if header == 0x01:  # Video
                _store_frame(payload)
                # If queue is full, drop the OLDEST frame to make room for the NEWEST
                if video_queue.full():
                    try:
                        video_queue.get_nowait()
                    except queue.Empty:
                        pass
                video_queue.put(payload)
            elif header == 0x02:
                pass  # phone-mic audio: not used as a trigger (see UNO Q audio classifier, 0x04)
            elif header == 0x04:  # Arduino UNO Q audio-classifier detection event
                try:
                    info = json.loads(payload.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    log.warning("Malformed audio-trigger payload: %r", payload[:100])
                    continue
                confidence = float(info.get("confidence", 0.0))
                sound_type = str(info.get("sound_type", "elephant_vocalization"))
                threading.Thread(target=handle_audio_trigger, args=(confidence, sound_type),
                                  daemon=True, name="gaja-audio-trigger").start()
            else:
                log.warning("Unknown header: %s", header)
    except websockets.exceptions.ConnectionClosed:
        log.info("Sensor client disconnected")


async def receiver_handler(websocket):
    log.info("Receiver phone connected (%d total)", len(receivers) + 1)
    receivers.add(websocket)
    try:
        async for _ in websocket:
            pass  # receivers only listen
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        receivers.discard(websocket)
        log.info("Receiver phone disconnected (%d left)", len(receivers))


async def run_servers():
    global _loop
    _loop = asyncio.get_running_loop()
    async with (
        websockets.serve(sensor_handler, "0.0.0.0", cfg.sensor_port),
        websockets.serve(receiver_handler, "0.0.0.0", cfg.alert_port),
    ):
        log.info("Sensor ingest on :%d, alert receivers on :%d",
                 cfg.sensor_port, cfg.alert_port)
        await asyncio.Future()  # run forever


def start_asyncio_servers():
    asyncio.run(run_servers())


# Continuous VLM verification, started when YOLO first sees an elephant and
# closed as soon as YOLO stops seeing one. Verification and the detailed
# report both need image input; scripts/serve_qwen_npu.py converts incoming
# OpenAI data URLs into GenieX image inputs for the QAIRT vision encoder.
# Report/alert text and MCP decisions use the same resident NPU model.
_elephant_present = threading.Event()
_verifier_thread: threading.Thread | None = None

# Latest YOLO box confidence for an "elephant" detection, read by both the
# video verifier loop and the audio observation window below.
_yolo_lock = threading.Lock()
_yolo_confidence = 0.0


def _set_yolo_confidence(score: float):
    global _yolo_confidence
    with _yolo_lock:
        _yolo_confidence = score


def get_yolo_confidence() -> float:
    with _yolo_lock:
        return _yolo_confidence


# Guards the report/alert/Sarvam/broadcast path so the video-only verifier
# loop and the audio observation window can never both file a report for the
# same real sighting -- whichever gets there first wins, the other is a no-op.
_incident_lock = threading.Lock()


def _grab_frames(n: int, gap_s: float) -> list:
    """Same latest-frame-with-spacing approach as the old audio pipeline
    (gaja/pipeline.py) — spaced grabs off the single-slot frame_source()."""
    frames = []
    for i in range(n):
        if i:
            time.sleep(gap_s)
        jpeg = frame_source()
        if jpeg is not None:
            frames.append(jpeg)
    return frames


def _severity(audio_confidence: float | None, yolo_confidence: float) -> str:
    """Deterministic severity rule for a *confirmed* incident (only called
    from _handle_verified, so this is never "low" -- that's reserved for
    audio-only unconfirmed events, assigned where those are logged)."""
    audio_ok = audio_confidence is not None and audio_confidence >= 0.75
    yolo_ok = yolo_confidence >= 0.75
    if (audio_confidence is None or audio_ok) and yolo_ok:
        return "high"
    return "medium"


def _handle_verified(det, source: str = "video", audio_confidence: float | None = None,
                      yolo_confidence: float | None = None, detection_time: str | None = None):
    """Runs once per sighting, right after VLM verification succeeds.
    `source` is which trigger started this check ("video" YOLO sighting or
    "audio" UNO Q classifier event) — recorded for the dashboard/log only,
    the report/alert/broadcast pipeline itself is identical either way.

    Guarded by _incident_lock (non-blocking): the video verifier loop and the
    audio observation window both funnel here, and only one report should
    ever be produced for the same real sighting -- whichever gets here first
    wins, the other call is dropped as a duplicate."""
    if not _incident_lock.acquire(blocking=False):
        log.info("Incident already being reported (source=%s ignored — duplicate)", source)
        return
    try:
        incident_id = IncidentLog.new_id()
        now = datetime.datetime.now().astimezone()
        when_iso = now.isoformat(timespec="seconds")
        when_display = now.strftime("%I:%M %p")
        yolo_conf = yolo_confidence if yolo_confidence is not None else get_yolo_confidence()

        frames = _grab_frames(cfg.frames_per_check, cfg.frame_gap_s)
        description = llm.generate_detailed_report(frames) if frames else None
        alert = llm.generate_alert(det, ratio=1.0, when=when_display)
        log.info("Incident report: %s", description or det.notes)
        log.info("Alert generated (fallback=%s): %s", alert.fallback, alert.report)

        observation = (description or det.notes or "")[:800]
        incident_text = (
            f"Elephant confirmed near {cfg.location_name} at {when_display} "
            f"(confidence {det.confidence:.2f}, trigger={source}).\n"
            f"Detailed observation: {observation}\n"
            f"Public alert: {alert.report}"
        )
        sarvam_result = None
        try:
            sarvam_result = asyncio.run(run_sarvam_agent(cfg, llm, incident_text, incident_id))
        except Exception:
            log.exception("Sarvam agent failed")

        # Sarvam's own translations are the multi-language alert text sent to
        # receiver phones when available; Gemma's en/hi/ta (already generated
        # above) are the fallback if the Sarvam MCP call failed or translated
        # nothing, so receivers still get an alert either way.
        if sarvam_result and sarvam_result.translations:
            alerts_for_broadcast = dict(sarvam_result.translations)
            alerts_for_broadcast.setdefault("en", alert.alerts.get("en", alert.report))
        else:
            alerts_for_broadcast = dict(alert.alerts)

        broadcast_payload = {
            "type": "elephant_alert",
            "id": incident_id,
            "timestamp": when_iso,
            "confidence": det.confidence,
            "trigger_source": source,
            "report": description or det.notes or alert.report,
            "alerts": alerts_for_broadcast,
            "location": cfg.location_name,
            "source": "audio+video" if source == "audio" else "video_only",
            "detected_sound_type": "elephant_vocalization" if source == "audio" else None,
            "elephant_visibility": True,
            "detection_time": detection_time or when_iso,
            "audio_confidence": audio_confidence,
            "yolo_confidence": yolo_conf,
            "verification_status": "confirmed",
            "event_severity": _severity(audio_confidence, yolo_conf),
            "notification_message": alerts_for_broadcast,
        }
        send_alert(json.dumps(broadcast_payload, ensure_ascii=False).encode())
        log.info("Incident %s: ALERT broadcast (source=%s)", incident_id, source)

        incident_log.record({
            "id": incident_id,
            "timestamp": when_iso,
            "status": "alerted",
            "trigger_source": source,
            "detection": {"elephant": det.elephant, "confidence": det.confidence, "notes": det.notes},
            "report": description or det.notes,
            "alert": {"report": alert.report, "alerts": alert.alerts, "fallback": alert.fallback,
                       "location": cfg.location_name},
            "sarvam": {
                "summary": sarvam_result.summary if sarvam_result else "",
                "translations": sarvam_result.translations if sarvam_result else {},
                "audio_files": sarvam_result.audio_files if sarvam_result else {},
                "final_message": sarvam_result.final_message if sarvam_result else "",
            },
            "broadcast": broadcast_payload,
        }, frames)
    finally:
        _incident_lock.release()


def _vlm_check(source: str):
    """Grab current frames and run the vision LLM once. Returns the Detection,
    or None if there were no frames or the vision server was unreachable.
    Shared by the continuous YOLO verifier loop and the one-shot audio
    trigger below so both go through the identical confirm step."""
    frames = _grab_frames(cfg.frames_per_check, cfg.frame_gap_s)
    if not frames:
        return None
    det = llm.detect_elephant(frames)
    if det is None:
        log.error("VLM verification (%s): vision server (%s) unreachable",
                   source, cfg.vision_llm_base)
        return None
    log.info("VLM check (%s): elephant=%s confidence=%.2f notes=%s",
              source, det.elephant, det.confidence, det.notes)
    return det


def verifier_loop():
    """Background thread: keeps polling live frames through the vision LLM
    while YOLO still sees an elephant. Verifies (and reports/alerts) once
    per sighting, but keeps watching — and logging re-checks — until the
    elephant leaves frame, at which point the thread exits on its own."""
    log.info("VLM verification started")
    verified = False
    while _elephant_present.is_set():
        det = _vlm_check("video")
        if det is None:
            time.sleep(cfg.vlm_poll_interval_s)
            continue
        if not verified and det.elephant and det.confidence >= cfg.confirm_confidence:
            verified = True
            log.info("Elephant VERIFIED by VLM (confidence=%.2f)", det.confidence)
            _handle_verified(det, source="video")
        time.sleep(cfg.vlm_poll_interval_s)
    log.info("Elephant left frame — VLM verification closing")


# Audio-triggered observation window for the UNO Q audio classifier (0x04 on
# the sensor socket, see sensor_handler / q-arduino/audio_classifier.py). An
# audio trigger alone doesn't generate a report: it opens a cfg.audio_window_s
# window during which both the audio evidence and the live YOLO state are
# collected. Video frames are only pulled and only sent to the vision LLM on
# hops where YOLO currently sees an elephant -- so a false-positive audio
# trigger with nothing in frame never reaches the LLM at all. If YOLO+VLM
# corroborate before the window closes, _handle_verified runs the shared
# report/alert/Sarvam/broadcast path; otherwise the incident is logged as
# unconfirmed and nothing is broadcast to receiver phones.
_window_lock = threading.Lock()
_window_open = False
_window_audio_confidence = 0.0
_window_sound_type = "elephant_vocalization"
_last_window_end = 0.0


def handle_audio_trigger(confidence: float, sound_type: str = "elephant_vocalization"):
    global _window_open, _window_audio_confidence, _window_sound_type
    with _window_lock:
        if _window_open:
            _window_audio_confidence = max(_window_audio_confidence, confidence)
            log.info("Audio trigger (confidence=%.2f) merged into the open observation window",
                      confidence)
            return
        if time.time() - _last_window_end < cfg.window_cooldown_s:
            log.info("Audio trigger (confidence=%.2f) ignored: window cooldown active", confidence)
            return
        _window_open = True
        _window_audio_confidence = confidence
        _window_sound_type = sound_type
    threading.Thread(target=_run_observation_window, daemon=True,
                      name="gaja-audio-window").start()


def _run_observation_window():
    """Runs for cfg.audio_window_s collecting audio + YOLO evidence, then
    either hands off to _handle_verified (confirmed) or logs an unconfirmed
    incident (no broadcast — avoids paging receivers on audio-only noise)."""
    global _window_open, _last_window_end
    start = time.time()
    window_started_at = datetime.datetime.now().astimezone()
    log.info("Audio-triggered observation window open for %.0fs", cfg.audio_window_s)

    yolo_confirmed = False
    max_yolo_conf = 0.0
    confirmed_det = None
    while time.time() - start < cfg.audio_window_s:
        if _elephant_present.is_set():
            yolo_confirmed = True
            max_yolo_conf = max(max_yolo_conf, get_yolo_confidence())
            if confirmed_det is None:
                det = _vlm_check("audio")
                if det is not None and det.elephant and det.confidence >= cfg.confirm_confidence:
                    confirmed_det = det
                    break  # confirmed early — no need to keep polling the rest of the window
        time.sleep(cfg.vlm_poll_interval_s)

    with _window_lock:
        audio_conf = _window_audio_confidence
        sound_type = _window_sound_type
        _window_open = False
        _last_window_end = time.time()

    if confirmed_det is not None:
        log.info("Elephant VERIFIED by VLM (source=audio confidence=%.2f)", confirmed_det.confidence)
        _handle_verified(confirmed_det, source="audio", audio_confidence=audio_conf,
                          yolo_confidence=max_yolo_conf,
                          detection_time=window_started_at.isoformat(timespec="seconds"))
        return

    log.info("Audio trigger not corroborated by YOLO+vision within the %.0fs window",
              cfg.audio_window_s)
    incident_id = IncidentLog.new_id()
    incident_log.record({
        "id": incident_id,
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "unconfirmed",
        "trigger_source": "audio",
        "source": "audio+video" if yolo_confirmed else "audio_only",
        "detected_sound_type": sound_type,
        "elephant_visibility": yolo_confirmed,
        "detection_time": window_started_at.isoformat(timespec="seconds"),
        "audio_confidence": audio_conf,
        "yolo_confidence": max_yolo_conf,
        "verification_status": "unconfirmed",
        "event_severity": "low",
        "notification_message": None,
    })


def start_verifier_if_needed():
    """Fires when local YOLO detects an elephant in a display frame."""
    global _verifier_thread
    _elephant_present.set()
    if _verifier_thread is None or not _verifier_thread.is_alive():
        _verifier_thread = threading.Thread(target=verifier_loop, daemon=True,
                                            name="gaja-vlm-verify")
        _verifier_thread.start()


def display_loop():
    """Main-thread OpenCV display (GUI requires main thread on Windows)."""
    try:
        cv2.namedWindow("Edge Video Stream", cv2.WINDOW_NORMAL)
    except cv2.error as e:
        log.warning("No display available (%s); running headless", e)
        while True:
            time.sleep(3600)

    log.info("Waiting for frames...")
    while True:
        try:
            payload = video_queue.get(timeout=0.1)
            np_arr = np.frombuffer(payload, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                display = frame.copy()
                h, w = frame.shape[:2]

                # ---------- Preprocess ----------
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (INPUT_W, INPUT_H))
                img = img.astype(np.float32) / 255.0
                img = np.ascontiguousarray(np.transpose(img, (2, 0, 1)))

                # ---------- Inference ----------
                with net.create_extractor() as ex:
                    ex.input("in0", ncnn.Mat(img).clone())
                    _, out0 = ex.extract("out0")

                    arr = np.array(out0).T  # shape: (8400, 84)

                    boxes = arr[:, :4]
                    scores = np.max(arr[:, 4:], axis=1)
                    class_ids = np.argmax(arr[:, 4:], axis=1)

                    mask = scores > CONFIDENCE
                    boxes = boxes[mask]
                    scores = scores[mask]
                    class_ids = class_ids[mask]

                    elephant_found = False

                    if len(boxes) > 0:
                        x = boxes[:, 0] - boxes[:, 2] / 2
                        y = boxes[:, 1] - boxes[:, 3] / 2
                        bw = boxes[:, 2]
                        bh = boxes[:, 3]
                        boxes_xywh = np.stack((x, y, bw, bh), axis=1)

                        indices = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(), CONFIDENCE, 0.45)

                        for idx in np.array(indices).reshape(-1):
                            idx = int(idx)
                            box = boxes_xywh[idx]
                            class_id = class_ids[idx]
                            score = scores[idx]
                            label = CLASSES[class_id]

                            x1 = int(box[0] * w / INPUT_W)
                            y1 = int(box[1] * h / INPUT_H)
                            x2 = int((box[0] + box[2]) * w / INPUT_W)
                            y2 = int((box[1] + box[3]) * h / INPUT_H)

                            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(
                                display,
                                f"{label} {score:.2f}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0),
                                2,
                            )

                            if label == "elephant":
                                elephant_found = True
                                _set_yolo_confidence(float(score))
                                start_verifier_if_needed()

                    if not elephant_found:
                        _elephant_present.clear()

                cv2.imshow("Edge Video Stream", display)

        except queue.Empty:
            pass
        except Exception:
            log.exception("Frame processing error")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    for name, base in (("vision", cfg.vision_llm_base), ("text", cfg.text_llm_base)):
        if not llm.healthy(base):
            log.warning("%s LLM at %s is not responding — start it before an incident",
                        name, base)

    start_dashboard(cfg)
    threading.Thread(target=start_asyncio_servers, daemon=True).start()
    display_loop()
