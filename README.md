# GajaAlert

GajaAlert is an edge-based elephant early-warning system. It combines a field phone, an Arduino UNO Q edge gateway, an AI server, and Android alert receivers. The system can use audio and video evidence to confirm a sighting, record an incident, and broadcast an alert.

The system has five components: the AI server, the edge gateway, audio-model training, and two Android apps. Each lives in its own directory with its own setup and development workflow.

## Architecture

```text
Field phone: q-mobile (Sensor mode)
  camera frames: 0x01 + JPEG ─┐
  microphone:    0x02 + PCM  ─┴─ WebSocket :8000 ──► q-arduino on UNO Q
                                                        │
                         video: forward latest frame ──┤
                         audio: YAMNet + XGBoost ──────┘
                                                        │ WebSocket :9000
                                                        ▼
                                             Gaja-alert AI server
                                  YOLO video detection + observation window
                                  Qwen vision confirmation on supported NPU
                                  incident report + optional Sarvam translation/TTS
                                  incident records and evidence frames
                                                        │
                                  alert: 0x03 + JSON ───┴─ WebSocket :9001
                                                        ▼
                                  notify receiver and q-mobile Receiver mode
                                  on-screen alert + Android notification
```

### Data flow

1. **Capture:** `q-mobile` can run in Sensor mode. It streams JPEG camera frames and 16 kHz mono signed 16-bit PCM microphone chunks to the UNO Q.
2. **Edge audio detection and relay:** `q-arduino` classifies audio locally with YAMNet embeddings and an XGBoost classifier. It forwards the newest video frame to the AI server and sends a compact event when its audio trigger fires. The gateway does not decode video.
3. **Confirm and report:** `Gaja-alert` runs the server-side detection and incident workflow. YOLO sightings and audio-triggered observation windows can be checked with the configured vision model. Confirmed incidents are recorded with evidence; translation and speech generation can use Sarvam when configured.
4. **Notify:** The server broadcasts confirmed alerts to receivers. `notify` is a dedicated receiver app; `q-mobile` can also switch into Receiver mode. Receivers show the alert and raise a local Android notification.

An audio trigger alone is not necessarily a broadcast alert: the server's observation workflow requires corroborating vision evidence before confirming and sending it. See the detailed server documentation for current behavior and configuration.

## Projects

| Directory | Role | Main technologies |
|---|---|---|
| [`Gaja-alert/`](Gaja-alert/) | AI server, vision detection and confirmation, incident workflow, alert broadcast, and configuration | Python, WebSockets, YOLO, Qwen/QAIRT, optional Sarvam |
| [`q-arduino/`](q-arduino/) | UNO Q gateway; receives phone streams, runs edge audio classification, and relays video and detections | Python, ONNX Runtime, YAMNet, XGBoost |
| [`Audio-Classification/`](Audio-Classification/) | Audio classifier training, evaluation, preprocessing, and deployment material | Python, YAMNet, XGBoost, ONNX |
| [`q-mobile/`](q-mobile/) | Android app for sensor capture and optional alert-receiver use | Kotlin, Jetpack Compose, CameraX, OkHttp WebSockets |
| [`notify/`](notify/) | Dedicated Android alert receiver | Kotlin, Jetpack Compose, WebSockets, Android notifications |

The `q-arduino/models/` directory contains the inference artifacts used by the gateway. `Audio-Classification/` contains the training and evaluation project; its large source datasets are not included and must be obtained separately as described in its README.

## Network protocol

| Port | Direction | Message |
|---|---|---|
| `8000` | `q-mobile` → UNO Q | `0x01` + JPEG video frame; `0x02` + 16 kHz mono PCM audio chunk |
| `9000` | UNO Q → `Gaja-alert` | `0x01` + relayed JPEG frame; `0x04` + JSON audio-detection event |
| `9001` | `Gaja-alert` → receiver apps | `0x03` + JSON confirmed incident alert |

The JSON alert includes incident metadata, report text, confidence and any available language-specific messages. See [`Gaja-alert/README.md`](Gaja-alert/README.md) for the full payload, server configuration, model setup, and end-to-end test instructions.

## Getting started

The components run on different devices and use different toolchains, so set up each one separately. A typical deployment is:

1. Set up and start the AI server by following [`Gaja-alert/README.md`](Gaja-alert/README.md). Configure its environment and model endpoints for the target machine.
2. Install the UNO Q gateway and its Python dependencies from [`q-arduino/`](q-arduino/). Start it with the AI server's reachable address, for example `python main.py 192.168.1.20`.
3. Build and install [`q-mobile/`](q-mobile/) on the field phone. Select **Sensor**, enter the UNO Q address, grant camera and microphone permissions, and connect.
4. Build and install [`notify/`](notify/) on receiver phones, or use Receiver mode in `q-mobile`. Enter the AI server address and listen for alerts.
5. To retrain or evaluate the audio model, follow [`Audio-Classification/README.md`](Audio-Classification/README.md), including its dataset setup instructions.

Use the per-project READMEs for exact dependencies, device requirements, environment variables, build steps, and troubleshooting. Keep credentials such as `SARVAM_API_KEY` in local environment files; do not commit secrets.

## Repository layout

```text
GajaAlert/
├── Gaja-alert/              # AI server and incident workflow
├── q-arduino/               # UNO Q streaming gateway and edge audio inference
├── Audio-Classification/    # classifier training and evaluation
├── q-mobile/                # sensor and receiver Android app
└── notify/                  # dedicated receiver Android app
```
