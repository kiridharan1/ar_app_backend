import asyncio
import base64
import time
import os
import logging
import gc
import uuid  # For generating unique session IDs
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

start_time = time.time()
frame_count = 0  # Track total frames processed for periodic cleanup
frame_count_lock = asyncio.Lock()  # Only for frame counter

# Per-client session state (keyed by session_id from client)
client_sessions = {}
session_lock = asyncio.Lock()  # Only for session dict access

# Map client IP addresses to session IDs (for auto-assignment)
client_ip_to_session = {}


class ClientSession:
    """Per-client detection state for parallel processing."""
    def __init__(self):
        self.prev_gray = None
        self.locked_box = None
        self.locked_cls = None
        self.features = None
        self.yolo_miss = 0
        self.flow_frames = 0
        self.last_frame_time = 0.0
        self.latest_detection = {
            "detected": False,
            "class_id": None,
            "class_name": None,
            "bbox": None,
            "timestamp": None,
        }
        self.lock = asyncio.Lock()  # Per-client lock


async def get_or_create_session(session_id: str, client_ip: str = None) -> tuple:
    """
    Get existing session or create new one.
    
    Args:
        session_id: Session ID from client (or None)
        client_ip: Client IP address for auto-assignment
        
    Returns:
        Tuple of (ClientSession, actual_session_id_used)
    """
    async with session_lock:
        # If no session_id provided, try to use IP-based mapping or create new
        if not session_id or session_id == "default":
            if client_ip and client_ip in client_ip_to_session:
                # Reuse existing session for this IP
                session_id = client_ip_to_session[client_ip]
                logger.debug("Reusing session %s for IP %s", session_id, client_ip)
            else:
                # Generate new unique session ID
                session_id = str(uuid.uuid4())
                if client_ip:
                    client_ip_to_session[client_ip] = session_id
                logger.info("Auto-generated session ID: %s for IP: %s", session_id, client_ip or "unknown")
        
        # Create session if it doesn't exist
        if session_id not in client_sessions:
            client_sessions[session_id] = ClientSession()
            logger.info("Created new session: %s", session_id)
        
        return client_sessions[session_id], session_id


async def cleanup_idle_sessions():
    """Background task to remove idle sessions (prevents memory leaks)."""
    while True:
        await asyncio.sleep(300)  # Check every 5 minutes
        
        current_time = time.time()
        sessions_to_remove = []
        
        async with session_lock:
            for session_id, session in client_sessions.items():
                # Remove sessions idle for > 1 hour
                if current_time - session.last_frame_time > 3600:
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                del client_sessions[session_id]
                # Also remove from IP mapping
                for ip, sid in list(client_ip_to_session.items()):
                    if sid == session_id:
                        del client_ip_to_session[ip]
                logger.info("Removed idle session: %s", session_id)
            
            if sessions_to_remove:
                logger.info("Cleaned up %d idle sessions", len(sessions_to_remove))


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
    global frame_count

    start_total = time.time()
    
    try:
        # Parse request with connection error handling
        try:
            data = await request.json()
        except ConnectionResetError:
            logger.warning("Connection reset by peer while reading request body")
            return web.json_response({"error": "Connection reset by client"}, status=499)
        except asyncio.TimeoutError:
            logger.warning("Request timeout while reading request body")
            return web.json_response({"error": "Request timeout"}, status=408)
        except Exception as e:
            logger.error("Error reading request body: %s", e)
            return web.json_response({"error": "Invalid request body"}, status=400)
        
        if "image" not in data:
            return web.json_response({"error": "image missing"}, status=400)
        
        # Get real client IP (handles proxies)
        client_ip = request.headers.get('X-Forwarded-For', request.remote)
        if isinstance(client_ip, str):
            # X-Forwarded-For can be "client, proxy1, proxy2"
            client_ip = client_ip.split(',')[0].strip()
            # Remove port if present (e.g., "192.168.1.1:12345" -> "192.168.1.1")
            client_ip = client_ip.split(':')[0]
        
        # Get or create session for this client
        # If frontend sends session_id, use it; otherwise auto-generate based on IP
        session_id_from_client = data.get("session_id")
        session, actual_session_id = await get_or_create_session(session_id_from_client, client_ip)

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
        
        # Increment global frame counter for periodic cleanup
        async with frame_count_lock:
            frame_count += 1
            current_frame_count = frame_count

        # 2. Enhance (optional)
        enhance_time = 0
        if ENABLE_ENHANCEMENT:
            t0 = time.time()
            frame = enhance_image_quality(frame)
            enhance_time = (time.time() - t0) * 1000

        # Prepare detection result to emit (will be populated in session lock)
        detection_to_emit = None
        
        async with session.lock:
            # Check for idle timeout
            if session.last_frame_time > 0 and (current_time - session.last_frame_time) > IDLE_TIMEOUT_SECONDS:
                logger.info("Idle timeout reached for session %s. Clearing state.", actual_session_id)
                session.prev_gray = None
                session.locked_box = None
                session.locked_cls = None
                session.features = None
                session.yolo_miss = 0
                session.flow_frames = 0
            
            session.last_frame_time = current_time
            
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
                        session.locked_box = smooth_box(session.locked_box, new_box)
                        session.locked_cls = int(best.cls[0])

                        session.yolo_miss = 0
                        session.flow_frames = 0
                        session.features = extract_features(gray, session.locked_box)
                        yolo_updated = True
                else:
                    session.yolo_miss += 1

            # 4. Optical Flow
            flow_time = 0
            if (
                not yolo_updated
                and session.locked_box is not None
                and session.prev_gray is not None
                and session.features is not None
            ):
                t0 = time.time()
                try:
                    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                        session.prev_gray,
                        gray,
                        session.features,
                        None,
                        winSize=(21, 21),
                        maxLevel=3,
                    )

                    good_new = new_pts[status == 1]
                    good_old = session.features[status == 1]

                    if len(good_new) >= 8:
                        dx = np.median(good_new[:, 0] - good_old[:, 0])
                        dy = np.median(good_new[:, 1] - good_old[:, 1])

                        box_w = session.locked_box[2] - session.locked_box[0]
                        box_h = session.locked_box[3] - session.locked_box[1]

                        if abs(dx) / box_w < MAX_MOVE_RATIO and abs(dy) / box_h < MAX_MOVE_RATIO:
                            session.locked_box += np.array([dx, dy, dx, dy])
                            session.features = good_new.reshape(-1, 1, 2)
                            session.flow_frames += 1
                except Exception as e:
                    logger.warning("Optical flow error: %s", e)
                    
                flow_time = (time.time() - t0) * 1000

            # ---------------- DROP ----------------
            if session.yolo_miss > YOLO_MISS_LIMIT or session.flow_frames > FLOW_MAX_FRAMES:
                session.locked_box = None
                session.locked_cls = None
                session.features = None
                session.yolo_miss = 0
                session.flow_frames = 0

            # ---------------- OUTPUT ----------------
            ts = time.time()

            if session.locked_box is not None:
                x1, y1, x2, y2 = [int(v) for v in session.locked_box.astype(int)]
                session.latest_detection = {
                    "detected": True,
                    "class_id": session.locked_cls,
                    "class_name": CLASS_NAMES[session.locked_cls],
                    "bbox": [x1, y1, x2, y2],
                    "timestamp": ts,
                }
            else:
                session.latest_detection = {
                    "detected": False,
                    "class_id": None,
                    "class_name": None,
                    "bbox": None,
                    "timestamp": ts,
                }

            # Copy for emission outside lock
            detection_to_emit = session.latest_detection.copy()
            session.prev_gray = gray

        # Emit outside the lock to prevent blocking
        try:
            await sio.emit("detection", detection_to_emit)
        except Exception as e:
            logger.error("Socket.IO emit error: %s", e)

        total_time = (time.time() - start_total) * 1000
        
        det_status = f"DETECTED: {detection_to_emit['class_name']}" if detection_to_emit['detected'] else "NO DETECTION"
        logger.info(
            "Perf: Total=%.1fms [Decode=%.1fms, Enhance=%.1fms, YOLO=%.1fms, Flow=%.1fms] %s",
            total_time, decode_time, enhance_time, yolo_time, flow_time, det_status
        )

        if detection_to_emit['detected']:
            logger.info("Latest Detected Object name is : %s", detection_to_emit['class_name'])
        else:
            logger.info("Latest Detected Object name is : No Object Detected")

        # Periodic memory cleanup every GC_CLEANUP_INTERVAL frames
        if current_frame_count % GC_CLEANUP_INTERVAL == 0:
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
                current_frame_count,
                mem_info.rss / 1024 / 1024,
                mem_info.vms / 1024 / 1024
            )

        # Cleanup temporary objects to help garbage collector
        del frame, gray, img_np, img_bytes
        if 'result' in locals():
            del result

        return web.json_response(detection_to_emit)
        
    except Exception as e:
        logger.exception("Unexpected error in ingest_frame")
        return web.json_response({"error": "Internal server error"}, status=500)


async def index(request):
    """Main entry point route providing service status and info."""
    uptime = time.time() - start_time
    
    # Get active session count
    async with session_lock:
        active_sessions = len(client_sessions)
    
    return web.json_response({
        "status": "online",
        "service": "AR Detection Backend",
        "version": "1.0.2",
        "device": device,
        "uptime_seconds": round(uptime, 2),
        "active_sessions": active_sessions,
        "config": {
            "lock_conf": LOCK_CONF,
            "yolo_miss_limit": YOLO_MISS_LIMIT,
            "flow_max_frames": FLOW_MAX_FRAMES,
            "enhancement_enabled": ENABLE_ENHANCEMENT,
            "idle_timeout_seconds": IDLE_TIMEOUT_SECONDS
        },
        "endpoints": {
            "health": "/",
            "frame_ingestion": "/api/frame",
            "session_stats": "/api/sessions",
            "latest_detection": "/api/detection/latest",
            "mobile_view": "/public"
        }
    })


async def session_stats(request):
    """Get statistics about active sessions."""
    async with session_lock:
        current_time = time.time()
        stats = {
            "total_sessions": len(client_sessions),
            "sessions": []
        }
        
        for session_id, session in client_sessions.items():
            idle_time = current_time - session.last_frame_time if session.last_frame_time > 0 else 0
            stats["sessions"].append({
                "id": session_id[:8] + "..." if len(session_id) > 8 else session_id,
                "idle_seconds": round(idle_time, 1),
                "has_detection": session.locked_box is not None,
                "last_detected_class": CLASS_NAMES.get(session.locked_cls) if session.locked_cls is not None else None
            })
        
        return web.json_response(stats)


async def get_latest(request):
    """Get latest detection from default session (for backward compatibility)."""
    async with session_lock:
        # Try to get from default session first
        if "default" in client_sessions:
            session = client_sessions["default"]
            return web.json_response(session.latest_detection)
        
        # If no default session, return from any active session
        if client_sessions:
            # Get the most recently active session
            most_recent_session = max(
                client_sessions.values(),
                key=lambda s: s.last_frame_time
            )
            return web.json_response(most_recent_session.latest_detection)
        
        # No active sessions
        return web.json_response({
            "detected": False,
            "class_id": None,
            "class_name": None,
            "bbox": None,
            "timestamp": None,
            "message": "No active sessions"
        })


async def mobile_view(request):
    """Serve the mobile view HTML page."""
    return web.FileResponse("mobileview.html")


# ===================== ROUTES ===================== #

app.router.add_get("/", index)
app.router.add_post("/api/frame", ingest_frame)
app.router.add_get("/api/sessions", session_stats)
app.router.add_get("/api/detection/latest", get_latest)
app.router.add_get("/public", mobile_view)

# ===================== ENTRY ====================== #

if __name__ == "__main__":
    logger.info("Async Socket.IO + API starting on port %s", SOCKET_PORT)
    
    # Start background cleanup task
    loop = asyncio.get_event_loop()
    loop.create_task(cleanup_idle_sessions())
    logger.info("Background session cleanup task started")
    
    web.run_app(app, host="0.0.0.0", port=SOCKET_PORT)
