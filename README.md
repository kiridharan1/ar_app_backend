# YOLOv11n Live Detection Backend

Backend service that exposes real-time YOLOv11n detections over Socket.IO. It loads a custom Ultralytics model, handles frame ingest from AR/vision clients, and streams detections plus live server stats back to each connected client.

## Features

- Async Socket.IO server built on `uvicorn`/ASGI.
- Configurable confidence/IoU/presets via `config.py`.
- Automatic frame decoding, optional resizing, and detection filtering.
- Per-client telemetry: processing latency, detection counts, server uptime.
- Healthful logging and runtime stats endpoint (`get_stats` event).

## Requirements

Install Python 3.10+ and the packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key dependencies: `ultralytics`, `python-socketio[asyncio_server]`, `uvicorn`, `opencv-python`, `numpy`, `aiohttp`.

## Configuration

All tunables live in `config.py`. Every value is imported by `server.py`, so changing it here updates runtime behavior without hunting through the codebase.

- `CONFIDENCE_THRESHOLD`, `IOU_THRESHOLD`, `MAX_DETECTIONS`, `AGNOSTIC_NMS` → passed into every YOLO inference call.
- `IMAGE_SIZE`, `FRAME_MAX_DIM` → define dummy model validation size and optional client-frame resizing.
- `DEVICE`, `HALF_PRECISION`, `VERBOSE_INFERENCE` → control Ultralytics runtime (CPU vs CUDA, FP16, log verbosity).
- `SERVER_HOST`, `SERVER_PORT`, `CORS_ORIGINS` → configure the Socket.IO server and UVicorn host binding.
- `MODEL_PATH` → path used when `YOLODetectionServer.load_model()` initializes the weights.
- `SHOW_CONFIDENCE`, `SHOW_CLASS_ID`, `BBOX_COLOR`, `BBOX_THICKNESS`, `FONT_SCALE` → echoed back in the `display_config` portion of every `detections` payload.
- `PERFORMANCE_LOG_INTERVAL`, `LOG_SEPARATOR_LENGTH` → affect server logging cadence and formatting.

Four presets (`PRESET_HIGH_ACCURACY`, `PRESET_BALANCED`, `PRESET_HIGH_RECALL`, `PRESET_VERY_HIGH_ACCURACY`) can override the main thresholds by setting `ACTIVE_PRESET`. `_apply_preset_config()` rewrites `CONFIDENCE_THRESHOLD`, `IOU_THRESHOLD`, and `MAX_DETECTIONS` using the chosen preset before the server starts listening.

## Running the Server

1. Place your trained YOLO weights at the path referenced by `MODEL_PATH` (default `model/final_model.pt`).
2. Start the server:

   ```bash
   python server.py
   ```

3. The server logs include applied preset details, device info, and the listening host/port (default `0.0.0.0:3000`).

On startup, `YOLODetectionServer` loads the model, performs a dry-run inference, and begins accepting Socket.IO connections.

## Socket.IO Events

| Event           | Direction      | Payload (summary)                                                                         |
| --------------- | -------------- | ----------------------------------------------------------------------------------------- |
| `connect`       | server→client  | Welcome info (`model_loaded`, `device`, uptime).                                          |
| `frame`         | client→server  | `{ "image": "<base64>", "options": { "min_confidence": 0.5, "allowed_classes": [0,1] } }` |
| `detections`    | server→client  | Detection list, processing time, frame info, server stats, display config.                |
| `get_stats`     | client↔server  | Returns aggregate stats, model metadata, effective config values.                         |
| `ping` / `pong` | bi-directional | Simple heartbeat with timestamps.                                                         |
| `error`         | server→client  | Emitted when frame validation or inference fails.                                         |

Detections are arrays of `{ bbox, confidence, class_id, class_name }`.

## Project Structure

```text
ar_app_backend/
├── config.py        # All runtime settings & presets
├── server.py        # YOLODetectionServer and ASGI bootstrap
├── utils.py         # Frame decoding, resizing, detection helpers
├── model/           # Drop your YOLO weight file here
└── requirements.txt # Python dependencies
```
