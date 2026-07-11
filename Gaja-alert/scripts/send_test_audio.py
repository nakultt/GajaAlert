"""Fake sensor phone + fake receiver phone for end-to-end pipeline testing.

Streams a low-frequency test tone (or a WAV file) as 0x02 audio chunks and a
JPEG as repeated 0x01 frames to the sensor port, while a receiver connection
on the alert port prints any 0x03 alert that comes back.

Usage:
    uv run python scripts/send_test_audio.py [image.jpg] [audio.wav]

    image.jpg  frame to stream (default: a synthetic gray test image)
    audio.wav  16kHz mono 16-bit WAV (default: synthesized 60+120 Hz rumble)

Requires arm_server.py running. The receiver side of this script is also the
reference implementation for the mobile receiver app (ws://<laptop>:9001,
messages are 0x03 + UTF-8 JSON).
"""

import asyncio
import json
import sys
import wave

import numpy as np

SENSOR_URL = "ws://127.0.0.1:9000"
ALERT_URL = "ws://127.0.0.1:9001"
SR = 16000
CHUNK = 1600  # 100 ms of audio


def load_audio(path: str | None) -> np.ndarray:
    if path:
        with wave.open(path, "rb") as w:
            assert w.getframerate() == SR and w.getnchannels() == 1, \
                "need 16kHz mono WAV"
            return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    t = np.arange(SR * 8) / SR
    x = (np.sin(2 * np.pi * 60 * t) + np.sin(2 * np.pi * 120 * t)) / 2 * 3000
    return x.astype(np.int16)


def load_image(path: str | None) -> bytes:
    if path:
        return open(path, "rb").read()
    import cv2
    img = np.full((480, 640, 3), 100, dtype=np.uint8)
    cv2.putText(img, "GAJA TEST FRAME", (120, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    return cv2.imencode(".jpg", img)[1].tobytes()


async def sensor(audio: np.ndarray, jpeg: bytes):
    import websockets
    async with websockets.connect(SENSOR_URL, max_size=None) as ws:
        print(f"[sensor] connected to {SENSOR_URL}, streaming "
              f"{len(audio)/SR:.0f}s of audio + frames...")
        pos = 0
        while pos < len(audio):
            chunk = audio[pos:pos + CHUNK]
            await ws.send(b"\x02" + chunk.tobytes())
            if pos % (CHUNK * 5) == 0:  # a frame every ~0.5 s
                await ws.send(b"\x01" + jpeg)
            pos += CHUNK
            await asyncio.sleep(CHUNK / SR)  # real-time pacing
        print("[sensor] done streaming; keeping frames flowing for the pipeline...")
        for _ in range(120):  # keep latest-frame fresh while Gemma thinks
            await ws.send(b"\x01" + jpeg)
            await asyncio.sleep(1.0)


async def receiver(done: asyncio.Event):
    import websockets
    async with websockets.connect(ALERT_URL, max_size=None) as ws:
        print(f"[receiver] connected to {ALERT_URL}, waiting for alerts...")
        async for msg in ws:
            if isinstance(msg, bytes) and msg[:1] == b"\x03":
                alert = json.loads(msg[1:].decode("utf-8"))
                print("\n[receiver] ALERT RECEIVED:")
                print(json.dumps(alert, indent=2, ensure_ascii=False))
                done.set()
                return


async def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    wav_path = sys.argv[2] if len(sys.argv) > 2 else None
    audio = load_audio(wav_path)
    jpeg = load_image(img_path)

    done = asyncio.Event()
    recv_task = asyncio.create_task(receiver(done))
    sensor_task = asyncio.create_task(sensor(audio, jpeg))
    try:
        await asyncio.wait_for(done.wait(), timeout=300)
        print("[main] end-to-end alert delivery confirmed")
    except asyncio.TimeoutError:
        print("[main] no alert within 5 min (check server logs / incidents.jsonl)")
    finally:
        sensor_task.cancel()
        recv_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
