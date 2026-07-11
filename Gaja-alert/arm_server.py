"""Gaja-alert edge server.

Sensor phone streams to ws://<laptop>:9000 (0x01 = JPEG frame, 0x02 = 16kHz
mono int16 PCM). The audio feeds a low-frequency band trigger; on a trigger
the pipeline grabs the latest frames, confirms with Gemma vision (:8080),
generates a trilingual report/alert (:8082 NPU/GPU split), and broadcasts
0x03 + JSON to receiver phones connected on ws://<laptop>:9001.
"""

import asyncio
import logging
import queue
import threading
import time

import cv2
import numpy as np
import websockets

from gaja.audio_trigger import BandTrigger
from gaja.config import Config
from gaja.incidents import IncidentLog
from gaja.llm import GemmaClient
from gaja.pipeline import Pipeline

# Attempt to load sounddevice (often fails on Windows ARM64 due to missing DLLs)
try:
    import sounddevice as sd
    AUDIO_ENABLED = True
except OSError as e:
    print(f"[Warning] Audio playback disabled: {e}")
    AUDIO_ENABLED = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("gaja.server")

cfg = Config.load()

# Queues to pass data from the async websocket thread to processing threads
video_queue = queue.Queue(maxsize=2)
audio_queue = queue.Queue(maxsize=100)

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


# Pipeline wiring
trigger = BandTrigger(cfg)
llm = GemmaClient(cfg)
pipeline = Pipeline(cfg, llm, frame_source, send_alert, IncidentLog(cfg))


# 1. AUDIO THREAD: single consumer — optional playback + trigger detection
def audio_consumer_thread():
    stream = None
    if AUDIO_ENABLED:
        try:
            # Android sends 16kHz, Mono, 16-bit PCM
            stream = sd.OutputStream(samplerate=cfg.sample_rate, channels=1, dtype='int16')
            stream.start()
            log.info("Audio playback started")
        except Exception as e:
            log.warning("Audio playback unavailable: %s", e)
            stream = None
    while True:
        chunk = audio_queue.get()
        if stream is not None:
            try:
                stream.write(chunk)
            except Exception as e:
                log.error("Audio playback error: %s", e)
                stream = None
        try:
            ev = trigger.feed(chunk)
            if ev is not None:
                pipeline.notify_trigger(ev)
        except Exception:
            log.exception("Trigger processing error")


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
            elif header == 0x02:  # Audio
                audio_data = np.frombuffer(payload, dtype=np.int16)
                if not audio_queue.full():
                    audio_queue.put(audio_data)
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
                cv2.imshow("Edge Video Stream", frame)
        except queue.Empty:
            pass

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    for name, base in (("vision", cfg.vision_llm_base), ("text", cfg.text_llm_base)):
        if not llm.healthy(base):
            log.warning("%s LLM at %s is not responding — start it before an incident",
                        name, base)

    threading.Thread(target=audio_consumer_thread, daemon=True).start()
    threading.Thread(target=start_asyncio_servers, daemon=True).start()
    pipeline.start()
    display_loop()
