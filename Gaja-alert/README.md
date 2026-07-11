# Gaja Alert

Edge elephant early-warning system running fully on-device on a Snapdragon X Elite (Windows on ARM64, 32 GB RAM). A phone acts as the field sensor (mic + camera); the laptop detects low-frequency elephant rumbles in the audio, confirms with a Gemma vision model, writes an incident report, and broadcasts alerts in **English, Hindi, and Tamil** to nearby receiver phones.

## Architecture

```
sensor phone (mic+cam, replaces Arduino sensors)
      │  ws://laptop:9000   0x01=JPEG frame, 0x02=16kHz mono int16 PCM
      ▼
arm_server.py ──► BandTrigger (numpy FFT, 10–250 Hz band-energy ratio,
      │            hysteresis + cooldown)                      [gaja/audio_trigger.py]
      │ trigger
      ▼
Pipeline worker thread                                          [gaja/pipeline.py]
      │ grabs latest frames
      ├─► Gemma 4 E4B + mmproj (llama.cpp, CPU)  http://127.0.0.1:8080
      │      vision confirm: {"elephant", "confidence", "notes"}
      ├─► Gemma 4 E2B split (geniex, NPU prefill + GPU decode)  http://127.0.0.1:8082
      │      report + alerts {"en","hi","ta"}   (falls back to :8080, then template)
      ├─► incidents/incidents.jsonl + frames                   [gaja/incidents.py]
      ▼
receiver phones   ws://laptop:9001   receive 0x03 + UTF-8 JSON alert
```

## WebSocket protocol

| Port | Direction | Header | Payload |
|------|-----------|--------|---------|
| 9000 | sensor → laptop | `0x01` | JPEG frame |
| 9000 | sensor → laptop | `0x02` | 16 kHz mono int16 PCM chunk |
| 9001 | laptop → receivers | `0x03` | UTF-8 JSON alert (below) |

```json
{
  "type": "elephant_alert",
  "id": "20260711-214501-a3f2",
  "timestamp": "2026-07-11T21:45:01+05:30",
  "confidence": 0.92,
  "report": "…2-3 sentence incident report…",
  "alerts": { "en": "…", "hi": "…", "ta": "…" },
  "location": "Gaja camp perimeter"
}
```

`scripts/send_test_audio.py` doubles as the reference receiver implementation for the mobile team.

## Setup

1. **Python deps** (uses [uv](https://docs.astral.sh/uv/)):
   ```powershell
   uv sync                # pipeline only
   uv sync --extra yolo   # + parked YOLO/QNN experiments
   ```
2. **Models + llama.cpp** (downloads to `%USERPROFILE%\llm`, outside OneDrive):
   ```powershell
   powershell -File scripts\download-models.ps1
   ```
3. **Start the LLM servers** (separate terminals):
   ```powershell
   powershell -File scripts\serve-llm.ps1                        # :8080 vision (E4B+mmproj, CPU)
   C:\Users\<you>\llm\geniex-env\Scripts\python.exe scripts\serve_e2b_split.py   # :8082 text (NPU+GPU)
   ```
4. **Run the edge server**:
   ```powershell
   uv run python arm_server.py
   ```
   Missing LLM servers only log warnings — the server keeps running and records incidents with status `llm_down`.

## Configuration

Optional `gaja.json` in the repo root overrides any field of `gaja/config.py` (`Config`), e.g.:

```json
{ "ratio_on": 0.5, "min_rms": 100, "location_name": "North paddy fence" }
```

Key trigger knobs to tune in the field: `band_low_hz`/`band_high_hz` (default 10–250 Hz — phone mics roll off below ~50–100 Hz so we lean on rumble harmonics), `ratio_on`/`ratio_off` (band-energy hysteresis), `min_rms` (silence gate), `cooldown_s`. Run with DEBUG logging to see per-hop `(ratio, rms)` values.

## Testing without an elephant

```powershell
uv run python scripts/test_trigger.py            # trigger unit checks (tones/noise/silence/cooldown)
uv run python scripts/test_inference.py img.jpg  # vision server (:8080) alone
uv run python scripts/test_split_inference.py    # text server (:8082) alone
uv run python scripts/send_test_audio.py [image.jpg] [audio.wav]
                                                 # fake sensor + receiver end-to-end
```

The last one streams a synthesized 60+120 Hz "rumble" and frames to :9000 and prints the 0x03 alert arriving on :9001. Pass a real elephant photo as `image.jpg` to get a confirmed alert; incidents land in `incidents/incidents.jsonl` with the frames.

## Repo map

- `arm_server.py` — entry point: WS servers, display, wiring
- `gaja/` — pipeline package (config, audio trigger, LLM client, pipeline, incident log)
- `scripts/` — model download/serving + test clients
- `web/index.html` — chat/telemetry UI served by `serve_e2b_split.py` at `/`
- `docs/LOCAL_INFERENCE.md` — engineering log: why llama.cpp CPU won for the LLM, NPU/GPU split findings, ARM64 gotchas
- **Parked experiments (not in the pipeline):** `webcam_inference.py`, `yolo26n-seg.*`, `export_assets/` (YOLO26-seg QNN/NPU — worked in daylight, failed at night, kept for reference), `scripts/serve_e2b_npu.py`
