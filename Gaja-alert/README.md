# Gaja Alert

Edge elephant early-warning system running fully on-device on a Snapdragon X Elite (Windows on ARM64, 32 GB RAM). Video and audio both arrive at an Arduino UNO Q from the field phone (Q-Mobile app: camera + mic); the UNO Q relays video on to this AI PC while running a local YAMNet+XGBoost elephant-sound classifier ([q-arduino/audio_classifier.py](../q-arduino/audio_classifier.py)) itself. A confirmed audio trigger opens a 13s observation window that collects both audio and live YOLO evidence before a Gemma vision model confirms; a report is then written and alerts in the field's languages are generated via **Sarvam AI** (translation + TTS, guaranteed for `sarvam_languages`) and broadcast to nearby receiver phones ([gajanotify/notify](https://github.com/Team-Highest/notify)).

## Architecture

```
Q-Mobile app (camera + mic)  ──ws:8000──►  UNO Q (QRB2210)
                                              │  0x01 video → relayed as-is
                                              │  0x02 audio → local YAMNet+XGBoost
                                              │     classifier (q-arduino/audio_classifier.py)
                                              ▼
                                        ws://laptop:9000
                                              │  0x01 = JPEG frame
                                              │  0x04 = {"event":"audio_elephant","confidence":...,"sound_type":...}
                                              ▼
arm_server.py: YOLO sighting ──────┬── UNO Q audio trigger
      │ video: continuous VLM polling while YOLO sees an elephant (verifier_loop)
      │ audio: opens a cfg.audio_window_s (13s) observation window — video frames
      │        only reach the vision LLM on hops where YOLO also confirms an elephant
      ▼
_vlm_check() / verifier_loop() / _run_observation_window()      [arm_server.py]
      ├─► Gemma 4 E4B + mmproj (llama.cpp, CPU)  http://127.0.0.1:8080
      │      vision confirm: {"elephant", "confidence", "notes"}
      ▼ confirmed (both paths funnel through the same _incident_lock-guarded step,
      │            so a video and audio confirmation of the same sighting can't double-file)
_handle_verified()
      ├─► Gemma 4 E2B split (geniex, NPU prefill + GPU decode)  http://127.0.0.1:8082
      │      detailed report + fallback en/hi/ta alert   (falls back to :8080, then template)
      ├─► Sarvam MCP agent: summarize, translate, TTS             [sarvam_agent.py]
      │      translations become the broadcast alert text; sarvam_languages (default hi/ta)
      │      are *guaranteed* to be translated + spoken even if the tool-calling agent itself
      │      didn't call those tools
      ├─► incidents/incidents.jsonl + frames                     [gaja/incidents.py]
      ▼
receiver phones   ws://laptop:9001   receive 0x03 + UTF-8 JSON alert

An audio trigger the window can't corroborate with YOLO+VLM within audio_window_s is logged
as "unconfirmed" (status/verification_status) but never broadcast — no paging on audio-only
false positives, nothing silently lost either.
```

## WebSocket protocol

| Port | Direction | Header | Payload |
|------|-----------|--------|---------|
| 9000 | sensor phone → laptop | `0x01` | JPEG frame |
| 9000 | sensor phone → laptop | `0x02` | 16 kHz mono int16 PCM chunk (received, not used as a trigger here — the UNO Q classifies this locally and reports `0x04` instead) |
| 9000 | UNO Q → laptop | `0x04` | UTF-8 JSON audio-detection event: `{"event": "audio_elephant", "confidence": 0.0-1.0, "sound_type": "elephant_vocalization", "timestamp": "..."}` |
| 9001 | laptop → receivers | `0x03` | UTF-8 JSON alert (below) |

```json
{
  "type": "elephant_alert",
  "id": "20260711-214501-a3f2",
  "timestamp": "2026-07-11T21:45:01+05:30",
  "confidence": 0.92,
  "trigger_source": "audio",
  "report": "…detailed incident report…",
  "alerts": { "en": "…", "hi": "…", "ta": "…" },
  "location": "Gaja camp perimeter",
  "source": "audio+video",
  "detected_sound_type": "elephant_vocalization",
  "elephant_visibility": true,
  "detection_time": "2026-07-11T21:45:01+05:30",
  "audio_confidence": 0.85,
  "yolo_confidence": 0.91,
  "verification_status": "confirmed",
  "event_severity": "high",
  "notification_message": { "en": "…", "hi": "…", "ta": "…" }
}
```

`alerts` (and its alias `notification_message`) holds whatever languages got translated for this incident — `sarvam_languages` (`gaja/config.py`, default `hi`/`ta`) are *guaranteed* to be present; the tool-calling Sarvam agent may add more on top. `en` always falls back to Gemma's own English alert if Sarvam didn't return one. `source` is `"audio+video"` for a window-confirmed audio trigger or `"video_only"` for a pure YOLO sighting with no audio corroboration; `event_severity` is `"high"` when both audio and YOLO confidence are ≥0.75, else `"medium"` (confirmed incidents only — unconfirmed audio triggers are logged, not broadcast, with `event_severity: "low"`). `scripts/send_test_audio.py` doubles as a reference receiver implementation and also fires a simulated `0x04` trigger to exercise the observation window.

## Setup

1. **Python deps** (uses [uv](https://docs.astral.sh/uv/)):
   ```powershell
   uv sync                # pipeline only
   uv sync --extra yolo   # + parked YOLO/QNN experiments
   ```
2. **Secrets and ports**: copy `.env.example` to `.env` and fill in `SARVAM_API_KEY` (adjust `GAJA_*` ports/LLM endpoints only if you're not using the defaults). `.env` is gitignored — never commit it. See [Configuration](#configuration).
3. **Models + llama.cpp** (downloads to `%USERPROFILE%\llm`, outside OneDrive):
   ```powershell
   powershell -File scripts\download-models.ps1
   ```
4. **Start the LLM servers** (separate terminals):
   ```powershell
   powershell -File scripts\serve-llm.ps1                        # :8080 vision (E4B+mmproj, CPU)
   C:\Users\<you>\llm\geniex-env\Scripts\python.exe scripts\serve_e2b_split.py   # :8082 text (NPU+GPU)
   ```
5. **Run the edge server**:
   ```powershell
   uv run python arm_server.py
   ```
   Missing LLM servers only log warnings — the server keeps running and records incidents with status `llm_down`.
6. **Point the sensors at the laptop**:
   - Q-Mobile app: connect to the UNO Q (camera → `0x01`, mic → `0x02`, `q-arduino/main.py` on `:8000`).
   - UNO Q (`q-arduino/`): `python main.py <laptop-ip>` — loads `audio_classifier.py` (YAMNet + XGBoost, models at `audio/` in the repo root) once at startup, classifies the Q-Mobile mic stream locally, relays video frames straight through, and reports confirmed audio triggers to the laptop as `0x04`.
   - Receiver phone(s): [gajanotify/notify](https://github.com/Team-Highest/notify) — enter the laptop's IP, tap "Listen for Alerts".

## Configuration

Two layers, both optional:

- **`.env`** (copy from `.env.example`) — secrets and machine/deployment-specific values: `SARVAM_API_KEY`, and `GAJA_<FIELD>` overrides for any `gaja/config.py` field (e.g. `GAJA_SENSOR_PORT`, `GAJA_VISION_LLM_BASE`). These win over `gaja.json` since they're meant to vary per machine.
- **`gaja.json`** in the repo root — behavior/tuning knobs you'd actually want to commit or hand-tune per site, overriding any field of `gaja/config.py` (`Config`), e.g.:
  ```json
  { "confirm_confidence": 0.5, "location_name": "North paddy fence" }
  ```

Key vision-confirm knob to tune in the field: `confirm_confidence` (minimum Gemma vision confidence before an incident is reported/alerted). `vlm_poll_interval_s` controls how often the verifier re-checks frames while YOLO still sees an elephant. `audio_window_s` (default 13s) controls how long an audio trigger's observation window stays open collecting YOLO/vision evidence before being logged unconfirmed; `window_cooldown_s` (default 60s) blocks a new window from opening right after one closes. `sarvam_languages` (default `["hi", "ta"]`) are the languages translate+TTS are guaranteed to run for on a confirmed incident.

## Testing without an elephant

```powershell
uv run python scripts/test_inference.py img.jpg  # vision server (:8080) alone
uv run python scripts/test_split_inference.py    # text server (:8082) alone
uv run python scripts/send_test_audio.py [image.jpg] [audio.wav]
                                                 # fake sensor + receiver end-to-end
```

Pass a real elephant photo as `image.jpg` to `send_test_audio.py` to get a confirmed alert; incidents land in `incidents/incidents.jsonl` with the frames. `send_test_audio.py` now also fires a simulated `0x04` audio trigger right after connecting, so a real elephant photo exercises the audio-triggered observation window (`source: "audio+video"`) rather than only the video-only YOLO path. To send a raw `0x04` event yourself, see the WebSocket protocol table above.

## Repo map

- `arm_server.py` — entry point: WS servers, YOLO display loop, VLM confirm, audio-triggered observation window, report/alert/broadcast wiring
- `gaja/` — `config.py` (`.env`/`gaja.json`), `llm.py` (Gemma clients), `incidents.py`, `dashboard.py`
- `sarvam_agent.py` / `sarvam_workflow.py` — Sarvam MCP translation/TTS (`sarvam_agent.py`'s agent loop always tops up `sarvam_languages` even if the LLM didn't call those tools itself)
- `scripts/` — model download/serving + test clients
- `../q-arduino/` — UNO Q side: `main.py` (Q-Mobile relay + audio trigger reporting), `audio_classifier.py` (streaming YAMNet+XGBoost classifier), `input_processing.py` (YAMNet mel-spectrogram preprocessing); models live in `../audio/` at the repo root
- `web/index.html` — chat/telemetry UI served by `serve_e2b_split.py` at `/`
- `docs/LOCAL_INFERENCE.md` — engineering log: why llama.cpp CPU won for the LLM, NPU/GPU split findings, ARM64 gotchas
- **Parked / not wired into the live pipeline:** `gaja/audio_trigger.py` + `gaja/pipeline.py` (the earlier phone-mic band-energy trigger, superseded by the UNO Q classifier), `webcam_inference.py`, `yolo26n-seg.*`, `export_assets/` (YOLO26-seg QNN/NPU — worked in daylight, failed at night, kept for reference), `scripts/serve_e2b_npu.py`, `scripts/test_trigger.py` (tests the parked band trigger)

## Sibling repos

This pipeline is one of four repos that make up the full system:

- [q-arduino/Audio-Classification](https://github.com/Team-Highest/Audio-Classification) — UNO Q audio elephant classifier (YAMNet + XGBoost); `live_elephant_detect.py` reports confirmed detections here over `0x04`.
- [q-arduino/q-mobile](https://github.com/Team-Highest/q-mobile) — sensor phone app (camera → `0x01`, mic → `0x02`) that can also act as a receiver.
- [gajanotify/notify](https://github.com/Team-Highest/notify) — dedicated receiver app: listens on `:9001`, shows the full incident report and per-language alerts, posts a notification with sound.
