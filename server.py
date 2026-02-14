import asyncio
import base64
import time
import os
import logging

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import socketio
from aiohttp import web

from config import (
    CLASS_NAMES,
    MODEL_PATH,
    LOCK_CONF,
    SMOOTH_ALPHA,
    YOLO_MISS_LIMIT,
    FLOW_MAX_FRAMES,
    MAX_MOVE_RATIO,
    FEATURE_COUNT,
    SOCKET_PORT,
)


# ===================== LOGGING ==================== #

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("ar-backend")

# ===================== GLOBAL STATE =============== #

state_lock = asyncio.Lock()

prev_gray = None
locked_box = None
locked_cls = None
features = None

yolo_miss = 0
flow_frames = 0

latest_detection = {
    "detected": False,
    "class_id": None,
    "class_name": None,
    "bbox": None,
    "timestamp": None,
}

# ===================== LOAD MODEL ================= #

logger.info("Loading YOLO model from %s ...", MODEL_PATH)
# Use CUDA if available; fall back to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH).to(device)
logger.info("Model loaded on %s", device)

# ===================== SOCKET.IO ================== #

sio = socketio.AsyncServer(cors_allowed_origins="*")


# Simple CORS middleware for the REST API routes
@web.middleware
async def cors_middleware(request, handler):
    # Handle preflight
    if request.method == "OPTIONS":
        resp = web.Response(status=200)
    else:
        resp = await handler(request)

    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    return resp


app = web.Application(middlewares=[cors_middleware])

sio.attach(app)


@sio.event
async def connect(sid, environ):
    logger.info("Client connected: %s", sid)


@sio.event
async def disconnect(sid):
    logger.info("Client disconnected: %s", sid)


# ===================== HELPERS ==================== #


def smooth_box(prev, new, alpha=SMOOTH_ALPHA):
    if prev is None:
        return new
    return prev * alpha + new * (1 - alpha)


def extract_features(gray, box):
    x1, y1, x2, y2 = box.astype(int)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    pts = cv2.goodFeaturesToTrack(
        roi,
        maxCorners=FEATURE_COUNT,
        qualityLevel=0.02,
        minDistance=7,
        blockSize=7,
    )

    if pts is None:
        return None

    pts[:, 0, 0] += x1
    pts[:, 0, 1] += y1
    return pts


# ===================== API ======================== #


async def ingest_frame(request):
    global prev_gray, locked_box, locked_cls, features
    global yolo_miss, flow_frames, latest_detection

    data = await request.json()
    if "image" not in data:
        return web.json_response({"error": "image missing"}, status=400)

    img_bytes = base64.b64decode(data["image"])
    img_np = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    if frame is None:
        return web.json_response({"error": "invalid image"}, status=400)

    async with state_lock:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        yolo_updated = False

        # ---------------- YOLO ----------------
        with torch.no_grad():
            # Force smaller inference size for speed; adjust if needed
            result = model(frame, conf=0.4, iou=0.5,
                           imgsz=640, device=device)[0]

        if result.boxes is not None and len(result.boxes) > 0:
            best = None
            best_score = 0

            for b in result.boxes:
                conf = float(b.conf[0])
                if conf < LOCK_CONF:
                    continue

                x1, y1, x2, y2 = b.xyxy[0]
                score = (x2 - x1) * (y2 - y1) * conf
                if score > best_score:
                    best_score = score
                    best = b

            if best is not None:
                new_box = np.array(best.xyxy[0], dtype=np.float32)
                locked_box = smooth_box(locked_box, new_box)
                locked_cls = int(best.cls[0])

                yolo_miss = 0
                flow_frames = 0
                features = extract_features(gray, locked_box)
                yolo_updated = True
        else:
            yolo_miss += 1

        # ---------------- FLOW ----------------
        if (
            not yolo_updated
            and locked_box is not None
            and prev_gray is not None
            and features is not None
        ):
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                gray,
                features,
                None,
                winSize=(21, 21),
                maxLevel=3,
            )

            good_new = new_pts[status == 1]
            good_old = features[status == 1]

            if len(good_new) >= 8:
                dx = np.median(good_new[:, 0] - good_old[:, 0])
                dy = np.median(good_new[:, 1] - good_old[:, 1])

                w = locked_box[2] - locked_box[0]
                h = locked_box[3] - locked_box[1]

                if abs(dx) / w < MAX_MOVE_RATIO and abs(dy) / h < MAX_MOVE_RATIO:
                    locked_box += np.array([dx, dy, dx, dy])
                    features = good_new.reshape(-1, 1, 2)
                    flow_frames += 1

        # ---------------- DROP ----------------
        if yolo_miss > YOLO_MISS_LIMIT or flow_frames > FLOW_MAX_FRAMES:
            locked_box = None
            locked_cls = None
            features = None
            yolo_miss = 0
            flow_frames = 0

        # ---------------- OUTPUT ----------------
        ts = time.time()

        if locked_box is not None:
            x1, y1, x2, y2 = [int(v) for v in locked_box.astype(int)]
            latest_detection = {
                "detected": True,
                "class_id": locked_cls,
                "class_name": CLASS_NAMES[locked_cls],
                "bbox": [x1, y1, x2, y2],
                "timestamp": ts,
            }
        else:
            latest_detection = {
                "detected": False,
                "class_id": None,
                "class_name": None,
                "bbox": None,
                "timestamp": ts,
            }

        await sio.emit("detection", latest_detection)
        prev_gray = gray

    return web.json_response(latest_detection)


async def index(request):
    """Main entry point route providing service status and info."""
    return web.json_response({
        "status": "online",
        "service": "AR Detection Backend",
        "version": "1.0.0",
        "endpoints": {
            "health": "/",
            "frame_ingestion": "/api/frame",
            "latest_detection": "/api/detection/latest"
        }
    })


async def get_latest(request):
    return web.json_response(latest_detection)


# ===================== ROUTES ===================== #

app.router.add_get("/", index)
app.router.add_post("/api/frame", ingest_frame)
app.router.add_get("/api/detection/latest", get_latest)

# ===================== ENTRY ====================== #

if __name__ == "__main__":
    logger.info("Async Socket.IO + API starting on port %s", SOCKET_PORT)
    web.run_app(app, host="0.0.0.0", port=SOCKET_PORT)
