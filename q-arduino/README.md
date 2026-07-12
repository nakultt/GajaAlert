# q-arduino

Receives encoded video and 16 kHz mono PCM audio from Q-Mobile over WebSocket.
Video (`0x01`) is relayed unchanged to Gaja; it is not decoded and OpenCV is
not used. Audio (`0x02`) runs through the local YAMNet/XGBoost elephant model,
which emits compact alert messages (`0x04`) to Gaja.

```powershell
uv run python main.py <gaja-ip>
```

The gateway is designed for live data: audio inference runs outside the
network event loop, only the newest unsent video frame is retained, and all
Gaja traffic uses one writer. Incoming payloads default to a 64 MiB maximum.
The limits can be configured with `MAX_MOBILE_MESSAGE`, `AUDIO_QUEUE_CHUNKS`,
`ARM_PC_PORT`, `ARDUINO_PORT`, and `ARM_SEND_TIMEOUT` environment variables.
