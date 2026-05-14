import os
from typing import Dict

"""
Central configuration for the backend service.

All tunable parameters and paths live here and can be overridden
via environment variables for different environments (dev/stage/prod).
"""


# ===================== MODEL / CLASSES ===================== #

CLASS_NAMES: Dict[int, str] = {
    0: "ARJUNA",
    1: "CAT",
    2: "LION",
    3: "NANDHI",
    4: "Charger",
    5: "golf-ball",
}

MODEL_PATH: str = os.getenv(
    "MODEL_PATH", os.path.join("model", "final_model.pt")
)


# ===================== DETECTION PARAMS ==================== #

LOCK_CONF: float = float(os.getenv("LOCK_CONF", "0.25"))
# Lower = more responsive (less smoothing)
SMOOTH_ALPHA: float = float(os.getenv("SMOOTH_ALPHA", "0.1"))
# Drop immediately if not detected
YOLO_MISS_LIMIT: int = int(os.getenv("YOLO_MISS_LIMIT", "0"))

# Disable optical flow for instant switching
FLOW_MAX_FRAMES: int = int(os.getenv("FLOW_MAX_FRAMES", "0"))
MAX_MOVE_RATIO: float = float(os.getenv("MAX_MOVE_RATIO", "0.65"))
FEATURE_COUNT: int = int(os.getenv("FEATURE_COUNT", "60"))

# Image enhancement (can be disabled for performance)
ENABLE_ENHANCEMENT: bool = os.getenv(
    "ENABLE_ENHANCEMENT", "false").lower() == "true"

# Idle timeout to clear state (seconds)
IDLE_TIMEOUT_SECONDS: int = int(os.getenv("IDLE_TIMEOUT_SECONDS", "30"))

# Garbage collection interval (frames)
GC_CLEANUP_INTERVAL: int = int(os.getenv("GC_CLEANUP_INTERVAL", "200"))


# ===================== SERVER CONFIG ======================= #

SOCKET_PORT: int = int(os.getenv("SOCKET_PORT", "5000"))
