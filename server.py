import asyncio
import base64
import time
import os
import logging
import gc
import psutil  # For memory monitoring

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
    ENABLE_ENHANCEMENT,
    IDLE_TIMEOUT_SECONDS,
    GC_CLEANUP_INTERVAL,
)


# ===================== LOGGING ==================== #

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("ar-backend")

# ===================== GLOBAL STATE =============== #

state_lock = asyncio.Lock()
start_time = time.time()
last_frame_time = 0.0

prev_gray = None
locked_box = None
locked_cls = None
features = None

yolo_miss = 0
flow_frames = 0
frame_count = 0  # Track total frames processed for periodic cleanup

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

# Warm up the model to avoid first-frame latency
logger.info("Warming up model...")
dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
with torch.no_grad():
    _ = model(dummy_frame, verbose=False)
logger.info("Model warm-up complete")

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


def enhance_image_quality(frame):
    """
    Enhance image quality for better detection with compressed images.
    Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) and sharpening.
    
    Args:
        frame: Input BGR image
        
    Returns:
        Enhanced BGR image
    """
    try:
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel (brightness)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Merge back
        enhanced_lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # Optional: Sharpen (helps with JPEG compression blur)
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        # Blend original and sharpened (70/30)
        result = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)

        return result
    except Exception as e:
        logger.warning("Image enhancement failed: %s. Using original frame.", e)
        return frame


# ===================== API ======================== #


async def ingest_frame(request):
    """
    Process incoming frame for object detection.
    
    Args:
        request: aiohttp request containing base64-encoded image
        
    Returns:
        JSON response with detection results
    """
    global prev_gray, locked_box, locked_cls, features
    global yolo_miss, flow_frames, latest_detection, last_frame_time, frame_count

    start_total = time.time()
    
    try:
        # Parse request
        data = await request.json()
        if "image" not in data:
            return web.json_response({"error": "image missing"}, status=400)

        # 1. Decode
        t0 = time.time()
        try:
            img_bytes = base64.b64decode(data["image"])
            img_np = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error("Image decode error: %s", e)
            return web.json_response({"error": "invalid image encoding"}, status=400)
            
        decode_time = (time.time() - t0) * 1000

        if frame is None:
            return web.json_response({"error": "invalid image"}, status=400)
        
        h, w = frame.shape[:2]
        current_time = time.time()
        
        # Increment frame counter for periodic cleanup
        frame_count += 1

        # 2. Enhance (optional)
        enhance_time = 0
        if ENABLE_ENHANCEMENT:
            t0 = time.time()
            frame = enhance_image_quality(frame)
            enhance_time = (time.time() - t0) * 1000

        # Prepare detection result to emit (will be populated in lock)
        detection_to_emit = None
        
        async with state_lock:
            # Check for idle timeout
            if last_frame_time > 0 and (current_time - last_frame_time) > IDLE_TIMEOUT_SECONDS:
                logger.info("Idle timeout reached. Clearing state.")
                prev_gray = None
                locked_box = None
                locked_cls = None
                features = None
                yolo_miss = 0
                flow_frames = 0
            
            last_frame_time = current_time
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            yolo_updated = False

            # 3. YOLO Inference
            t0 = time.time()
            try:
                with torch.no_grad():
                    # Use configured LOCK_CONF instead of hardcoded 0.1
                    result = model(frame, conf=LOCK_CONF, iou=0.45,
                                   imgsz=416, device=device, verbose=False)[0]
            except Exception as e:
                logger.error("YOLO inference error: %s", e)
                yolo_time = (time.time() - t0) * 1000
                flow_time = 0
                # Continue with previous state
            else:
                yolo_time = (time.time() - t0) * 1000

                if result.boxes is not None and len(result.boxes) > 0:
                    cx, cy = w / 2, h / 2
                    detected_items = []
                    best = None
                    best_score = 0
                    
                    for b in result.boxes:
                        conf = float(b.conf[0])
                        cls_id = int(b.cls[0])
                        name = CLASS_NAMES.get(cls_id, f"Unknown_{cls_id}")
                        
                        x1, y1, x2, y2 = b.xyxy[0]
                        bx, by = (x1 + x2) / 2, (y1 + y2) / 2
                        
                        dist_to_center = np.sqrt((bx - cx)**2 + (by - cy)**2)
                        max_dist = np.sqrt(cx**2 + cy**2)
                        center_score = 1.0 - (dist_to_center / max_dist)
                        
                        current_score = (conf * 0.7) + (center_score * 0.3)
                        detected_items.append((name, conf, center_score, current_score))
                        
                        if current_score > best_score:
                            best_score = current_score
                            best = b

                    if detected_items:
                        # Log all candidates found in this frame
                        log_msg = " | ".join([f"{n}: c={c:.2f}, s={s:.2f}" for n, c, cs, s in detected_items])
                        logger.info("Candidates: %s", log_msg)

                    if best is not None:
                        new_box = best.xyxy[0].cpu().numpy().astype(np.float32)
                        locked_box = smooth_box(locked_box, new_box)
                        locked_cls = int(best.cls[0])

                        yolo_miss = 0
                        flow_frames = 0
                        features = extract_features(gray, locked_box)
                        yolo_updated = True
                else:
                    yolo_miss += 1

            # 4. Optical Flow
            flow_time = 0
            if (
                not yolo_updated
                and locked_box is not None
                and prev_gray is not None
                and features is not None
            ):
                t0 = time.time()
                try:
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

                        box_w = locked_box[2] - locked_box[0]
                        box_h = locked_box[3] - locked_box[1]

                        if abs(dx) / box_w < MAX_MOVE_RATIO and abs(dy) / box_h < MAX_MOVE_RATIO:
                            locked_box += np.array([dx, dy, dx, dy])
                            features = good_new.reshape(-1, 1, 2)
                            flow_frames += 1
                except Exception as e:
                    logger.warning("Optical flow error: %s", e)
                    
                flow_time = (time.time() - t0) * 1000

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

            # Copy for emission outside lock
            detection_to_emit = latest_detection.copy()
            prev_gray = gray

        # Emit outside the lock to prevent blocking
        try:
            await sio.emit("detection", detection_to_emit)
        except Exception as e:
            logger.error("Socket.IO emit error: %s", e)

        total_time = (time.time() - start_total) * 1000
        
        det_status = f"DETECTED: {latest_detection['class_name']}" if latest_detection['detected'] else "NO DETECTION"
        logger.info(
            "Perf: Total=%.1fms [Decode=%.1fms, Enhance=%.1fms, YOLO=%.1fms, Flow=%.1fms] %s",
            total_time, decode_time, enhance_time, yolo_time, flow_time, det_status
        )

        if latest_detection['detected']:
            logger.info("Latest Detected Object name is : %s", latest_detection['class_name'])
        else:
            logger.info("Latest Detected Object name is : No Object Detected")

        # Periodic memory cleanup every GC_CLEANUP_INTERVAL frames
        if frame_count % GC_CLEANUP_INTERVAL == 0:
            # Clear PyTorch cache
            if device == "cuda":
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            # Log memory usage
            process = psutil.Process()
            mem_info = process.memory_info()
            logger.info(
                "Memory cleanup (frame %d): RSS=%.1fMB, VMS=%.1fMB",
                frame_count,
                mem_info.rss / 1024 / 1024,
                mem_info.vms / 1024 / 1024
            )

        # Cleanup temporary objects to help garbage collector
        del frame, gray, img_np, img_bytes
        if 'result' in locals():
            del result

        return web.json_response(latest_detection)
        
    except Exception as e:
        logger.exception("Unexpected error in ingest_frame")
        return web.json_response({"error": "Internal server error"}, status=500)


async def index(request):
    """Main entry point route providing service status and info."""
    uptime = time.time() - start_time
    return web.json_response({
        "status": "online",
        "service": "AR Detection Backend",
        "version": "1.0.1",
        "device": device,
        "uptime_seconds": round(uptime, 2),
        "config": {
            "lock_conf": LOCK_CONF,
            "yolo_miss_limit": YOLO_MISS_LIMIT,
            "flow_max_frames": FLOW_MAX_FRAMES,
            "enhancement_enabled": ENABLE_ENHANCEMENT
        },
        "endpoints": {
            "health": "/",
            "frame_ingestion": "/api/frame",
            "latest_detection": "/api/detection/latest",
            "mobile_view": "/public"
        }
    })


async def get_latest(request):
    return web.json_response(latest_detection)


async def mobile_view(request):
    """Serve the mobile view HTML page."""
    return web.FileResponse("mobileview.html")


# ===================== ROUTES ===================== #

app.router.add_get("/", index)
app.router.add_post("/api/frame", ingest_frame)
app.router.add_get("/api/detection/latest", get_latest)
app.router.add_get("/public", mobile_view)

# ===================== ENTRY ====================== #

if __name__ == "__main__":
    logger.info("Async Socket.IO + API starting on port %s", SOCKET_PORT)
    web.run_app(app, host="0.0.0.0", port=SOCKET_PORT)
