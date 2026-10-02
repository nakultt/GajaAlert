"""Q-Mobile gateway: local audio inference and low-latency Gaja video relay.

Wire format from Q-Mobile:
    0x01 + encoded video frame
    0x02 + 16 kHz, mono, little-endian int16 PCM

Video is never decoded on the Arduino, so OpenCV is not required.  Slow
consumers always receive the newest video frame rather than an old backlog.
"""

import asyncio
import datetime
import json
import os
import sys

import websockets

from audio_classifier import ElephantAudioClassifier


ARM_PC_IP = "localhost"
ARM_PC_PORT = int(os.getenv("ARM_PC_PORT", "9000"))
ARDUINO_PORT = int(os.getenv("ARDUINO_PORT", "8000"))

# websockets defaults to only 1 MiB. Phone video payloads can be considerably
# larger; keep a finite ceiling so a bad client cannot exhaust Arduino memory.
MAX_MOBILE_MESSAGE = int(os.getenv("MAX_MOBILE_MESSAGE", str(64 * 1024 * 1024)))
AUDIO_QUEUE_CHUNKS = int(os.getenv("AUDIO_QUEUE_CHUNKS", "32"))
ARM_SEND_TIMEOUT = float(os.getenv("ARM_SEND_TIMEOUT", "1.0"))

audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=AUDIO_QUEUE_CHUNKS)
alert_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=8)
outbound_ready = asyncio.Event()
latest_video: bytes | None = None


def replace_latest_video(message: bytes) -> None:
    """Keep exactly one frame: latency is more important than completeness."""
    global latest_video
    latest_video = message
    outbound_ready.set()


def put_realtime(queue: asyncio.Queue[bytes], item: bytes) -> None:
    """Non-blocking bounded enqueue, dropping the oldest stale item if full."""
    if queue.full():
        try:
            queue.get_nowait()
            queue.task_done()
        except asyncio.QueueEmpty:
            pass
    queue.put_nowait(item)


async def next_outbound_message() -> bytes:
    """Alerts have priority; video is a single replaceable latest-frame slot."""
    global latest_video
    while True:
        try:
            return alert_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        if latest_video is not None:
            message, latest_video = latest_video, None
            return message

        # Clear and check again before waiting to avoid a lost wake-up.
        outbound_ready.clear()
        if not alert_queue.empty() or latest_video is not None:
            outbound_ready.set()
            continue
        await outbound_ready.wait()


async def arm_connection_loop() -> None:
    """Maintain one ARM connection and one writer (WebSocket sends cannot race)."""
    uri = f"ws://{ARM_PC_IP}:{ARM_PC_PORT}"
    while True:
        try:
            print(f"[Arduino Network] Connecting to Gaja at {uri}...")
            async with websockets.connect(
                uri,
                compression=None,
                max_size=None,
                max_queue=2,
                write_limit=256 * 1024,
                ping_interval=20,
                ping_timeout=20,
            ) as socket:
                print("[Arduino Network] Connected to Gaja.")
                while True:
                    message = await next_outbound_message()
                    await asyncio.wait_for(socket.send(message), ARM_SEND_TIMEOUT)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Any frame that was being sent is intentionally discarded. Keeping
            # it would replay stale video after reconnect.
            print(f"[Arduino Network] Gaja connection lost ({exc}); retrying...")
            await asyncio.sleep(2)


async def audio_inference_loop(classifier: ElephantAudioClassifier) -> None:
    """Run the CPU-heavy model sequentially without blocking network I/O."""
    while True:
        payload = await audio_queue.get()
        try:
            event = await asyncio.to_thread(classifier.feed_bytes, payload)
            if event is None:
                continue
            when = datetime.datetime.fromtimestamp(event.timestamp).astimezone().isoformat(
                timespec="seconds"
            )
            body = json.dumps(
                {
                    "event": "audio_elephant",
                    "confidence": round(event.confidence, 3),
                    "sound_type": "elephant_vocalization",
                    "timestamp": when,
                },
                separators=(",", ":"),
            ).encode()
            put_realtime(alert_queue, b"\x04" + body)
            outbound_ready.set()
            print(f"[Arduino Audio] Elephant detected ({event.confidence:.2f}).")
        except Exception as exc:
            # A malformed chunk or model error must not kill audio processing.
            print(f"[Arduino Audio] Inference error: {exc}")
        finally:
            audio_queue.task_done()


async def handle_mobile_client(websocket) -> None:
    print("[Arduino Server] Q-Mobile connected.")
    try:
        async for message in websocket:
            if not isinstance(message, bytes) or not message:
                continue
            header = message[0]
            if header == 0x01:
                # Forward encoded bytes unchanged; no OpenCV/decode on Arduino.
                replace_latest_video(message)
            elif header == 0x02:
                payload = message[1:]
                if payload and len(payload) % 2 == 0:
                    put_realtime(audio_queue, payload)
                else:
                    print("[Arduino Audio] Dropped malformed PCM chunk.")
            else:
                print(f"[Arduino Server] Unknown message header 0x{header:02x}.")
    except websockets.exceptions.ConnectionClosed as exc:
        print(f"[Arduino Server] Q-Mobile disconnected ({exc.code}).")
    finally:
        print("[Arduino Server] Q-Mobile connection closed.")


async def main() -> None:
    print("[Arduino Audio] Loading elephant classifier...")
    classifier = await asyncio.to_thread(ElephantAudioClassifier)
    print("[Arduino Audio] Classifier ready.")

    arm_task = asyncio.create_task(arm_connection_loop(), name="gaja-writer")
    audio_task = asyncio.create_task(audio_inference_loop(classifier), name="audio-inference")
    try:
        print(
            f"[Arduino Server] Listening on 0.0.0.0:{ARDUINO_PORT} "
            f"(max payload {MAX_MOBILE_MESSAGE // (1024 * 1024)} MiB)"
        )
        async with websockets.serve(
            handle_mobile_client,
            "0.0.0.0",
            ARDUINO_PORT,
            compression=None,
            max_size=MAX_MOBILE_MESSAGE,
            max_queue=2,
            write_limit=256 * 1024,
            ping_interval=20,
            ping_timeout=20,
        ):
            await asyncio.Future()
    finally:
        arm_task.cancel()
        audio_task.cancel()
        await asyncio.gather(arm_task, audio_task, return_exceptions=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ARM_PC_IP = sys.argv[1]
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down.")
