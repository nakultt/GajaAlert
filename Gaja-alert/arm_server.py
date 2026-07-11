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
import ncnn

# Attempt to load sounddevice (often fails on Windows ARM64 due to missing DLLs)
try:
    import sounddevice as sd
    AUDIO_ENABLED = True
except OSError as e:
    print(f"[Warning] Audio playback disabled: {e}")
    AUDIO_ENABLED = False
import queue

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


def run_vlm(image):
    """Placeholder: fires when local YOLO detects an elephant in a display frame."""
    log.info("VLM Activated (image shape=%s)", image.shape)


def display_loop():
    """Main-thread OpenCV display (GUI requires main thread on Windows)."""
    try:
        cv2.namedWindow("Edge Video Stream", cv2.WINDOW_NORMAL)
    except cv2.error as e:
        log.warning("No display available (%s); running headless", e)
        while True:
            time.sleep(3600)

    first_detection_frame = None
    vlm_active = False

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
                                if first_detection_frame is None:
                                    first_detection_frame = frame.copy()
                                if not vlm_active:
                                    vlm_active = True
                                    run_vlm(first_detection_frame)

                    if not elephant_found:
                        first_detection_frame = None
                        vlm_active = False

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

    threading.Thread(target=audio_consumer_thread, daemon=True).start()
    threading.Thread(target=start_asyncio_servers, daemon=True).start()
    pipeline.start()
    display_loop()
