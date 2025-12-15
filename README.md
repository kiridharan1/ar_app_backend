# AR Backend Production Service

This project provides a lightweight backend service for real‑time AR object detection using a YOLO model.  
It exposes:

- A **REST API** to ingest frames and fetch the latest detection.
- A **Socket.IO** channel to push detection updates to connected clients.
- A **CLI script** for running offline inference on images, folders, or videos.

The backend is built with `aiohttp`, `python-socketio`, `opencv-python`, `torch`, and `ultralytics`.

---

## Project Structure

- `server.py` – Async web server (REST + Socket.IO) for real‑time detection.
- `run_inference.py` – Command‑line script to run YOLO inference on files/webcam.
- `config.py` – Central configuration (model path, thresholds, ports, etc.).
- `model/`
  - `final_model.pt` – Trained YOLO model checkpoint.
  - `run_tracking.py` – (Optional) additional tracking utilities.
- `requirements.txt` – Python dependencies for the project.

---

## Python Environment Setup

1. **Install Python**
   - Use Python **3.9–3.11**.
   - Verify:  
     ```bash
     python --version
     ```

2. **Create and activate a virtual environment**
   From the project root (`ar_backend_production`):

   ```bash
   python -m venv venv
   ```

   - Git Bash:
     ```bash
     source venv/Scripts/activate
     ```
   - PowerShell:
     ```powershell
     venv\Scripts\Activate.ps1
     ```
   - cmd:
     ```cmd
     venv\Scripts\activate.bat
     ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   > For GPU acceleration, install the appropriate `torch` build for your CUDA version from the official PyTorch website, then (if needed) reinstall `ultralytics`.

---

## Configuration

All configuration lives in `config.py` and can be overridden using environment variables.

Key settings:

- **Model / classes**
  - `MODEL_PATH` (env: `MODEL_PATH`) – default: `model/final_model.pt`
  - `CLASS_NAMES` – mapping of class IDs to labels.

- **Detection / tracking**
  - `LOCK_CONF` (env: `LOCK_CONF`) – default: `0.45`
  - `SMOOTH_ALPHA` (env: `SMOOTH_ALPHA`) – default: `0.45`
  - `YOLO_MISS_LIMIT` (env: `YOLO_MISS_LIMIT`) – default: `12`
  - `FLOW_MAX_FRAMES` (env: `FLOW_MAX_FRAMES`) – default: `30`
  - `MAX_MOVE_RATIO` (env: `MAX_MOVE_RATIO`) – default: `0.65`
  - `FEATURE_COUNT` (env: `FEATURE_COUNT`) – default: `60`

- **Server**
  - `SOCKET_PORT` (env: `SOCKET_PORT`) – default: `5000`
  - Logging level via env: `LOG_LEVEL` (default `INFO`, can be `DEBUG`, `WARNING`, etc.).

### Setting environment variables (examples)

- Git Bash:
  ```bash
  export MODEL_PATH="model/final_model.pt"
  export SOCKET_PORT=5000
  export LOG_LEVEL=INFO
  ```

- PowerShell:
  ```powershell
  $env:MODEL_PATH="model/final_model.pt"
  $env:SOCKET_PORT="5000"
  $env:LOG_LEVEL="INFO"
  ```

---

## Running the Server

From the project root, with the virtual environment activated:

```bash
python server.py
```

The server starts an `aiohttp` app and Socket.IO server, listening on:

- Host: `0.0.0.0`
- Port: `SOCKET_PORT` (default `5000`)

### REST Endpoints

- **POST `/api/frame`**
  - Body (JSON):
    ```json
    {
      "image": "<base64-encoded JPEG/PNG frame>"
    }
    ```
  - Response (JSON):
    ```json
    {
      "detected": true,
      "class_id": 0,
      "class_name": "ARJUNA",
      "bbox": [x1, y1, x2, y2],
      "timestamp": 1730000000.123
    }
    ```
    or, if nothing is locked:
    ```json
    {
      "detected": false,
      "class_id": null,
      "class_name": null,
      "bbox": null,
      "timestamp": 1730000000.456
    }
    ```

- **GET `/api/detection/latest`**
  - Returns the latest detection object (same structure as above).

### Socket.IO Channel

- Namespace: default (`/`)
- Event: `"detection"`
- Payload: same detection object as the REST responses.

Example (JavaScript client):

```js
const socket = io("http://<SERVER_HOST>:<SOCKET_PORT>");

socket.on("connect", () => {
  console.log("Connected", socket.id);
});

socket.on("detection", (data) => {
  console.log("Detection update:", data);
});
```

---

## Running Offline Inference

Use `run_inference.py` to run YOLO on images, folders, videos, or webcam:

```bash
python run_inference.py --source <path_or_index> [--save] [--model PATH] [--conf 0.25] [--iou 0.5]
```

Examples:

- Single image and save result:
  ```bash
  python run_inference.py --source images/portrait.jpg --save
  ```

- Folder of images:
  ```bash
  python run_inference.py --source test_files/images/ --save
  ```

- Video file:
  ```bash
  python run_inference.py --source test_files/videos/video-3.mp4 --save
  ```

- Webcam (index 0):
  ```bash
  python run_inference.py --source 0
  ```

By default, `--model` uses `MODEL_PATH` from `config.py`, keeping CLI and server consistent.

---

