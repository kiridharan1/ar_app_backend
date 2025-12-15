from ultralytics import YOLO
import cv2
import numpy as np
import argparse
import os

# ===================== CONFIG ===================== #

CLASS_NAMES = {
    0: "ARJUNA",
    1: "CAT",
    2: "LION",
    3: "NANDHI",
}

LOCK_CONF = 0.45
# Lower alpha → less smoothing → faster, more responsive box updates
SMOOTH_ALPHA = 0.45

YOLO_MISS_LIMIT = 12
FLOW_MAX_FRAMES = 30
MAX_MOVE_RATIO = 0.65
FEATURE_COUNT = 60

# ================================================= #


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
        blockSize=7
    )

    if pts is None:
        return None

    pts[:, 0, 0] += x1
    pts[:, 0, 1] += y1
    return pts


def dominant_box(boxes):
    best = None
    best_score = 0
    for b in boxes:
        conf = float(b.conf[0])
        if conf < LOCK_CONF:
            continue
        x1, y1, x2, y2 = b.xyxy[0]
        score = (x2 - x1) * (y2 - y1) * conf
        if score > best_score:
            best_score = score
            best = b
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--model", default=os.path.join("model", "final_model.pt"))
    parser.add_argument("--imgsz", type=int, nargs=2, default=[960, 960])
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print("[INFO] Loading YOLO model...")
    model = YOLO(args.model)
    print("[INFO] Model loaded")

    prev_gray = None
    locked_box = None
    locked_cls = None
    features = None

    yolo_miss = 0
    flow_frames = 0

    results = model.predict(
        source=args.source,
        imgsz=tuple(args.imgsz),
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        stream=True,
        show=False
    )

    print("[INFO] Fast & smooth tracking started")

    for frame_id, r in enumerate(results):
        if r.orig_img is None:
            continue

        frame = r.orig_img.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        yolo_updated = False

        # ---------------- YOLO STEP ----------------
        if r.boxes is not None and len(r.boxes) > 0:
            dom = dominant_box(r.boxes)
            if dom is not None:
                new_box = np.array(dom.xyxy[0], dtype=np.float32)
                locked_box = smooth_box(locked_box, new_box, SMOOTH_ALPHA)
                locked_cls = int(dom.cls[0])

                yolo_miss = 0
                flow_frames = 0

                if features is None or flow_frames % 10 == 0:
                    features = extract_features(gray, locked_box)

                yolo_updated = True

        else:
            yolo_miss += 1

        # ---------------- OPTICAL FLOW ----------------
        if not yolo_updated and locked_box is not None and prev_gray is not None and features is not None:
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, features, None,
                winSize=(21, 21),
                maxLevel=3
            )

            good_new = new_pts[status == 1]
            good_old = features[status == 1]

            if len(good_new) >= 8:
                dx = np.median(good_new[:, 0] - good_old[:, 0])
                dy = np.median(good_new[:, 1] - good_old[:, 1])

                w = locked_box[2] - locked_box[0]
                h = locked_box[3] - locked_box[1]

                if abs(dx) / w < MAX_MOVE_RATIO and abs(dy) / h < MAX_MOVE_RATIO:
                    delta = np.array([dx, dy, dx, dy])
                    # Lower smoothing here as well for snappier optical-flow updates
                    locked_box = smooth_box(locked_box, locked_box + delta, 0.55)
                    features = good_new.reshape(-1, 1, 2)
                    flow_frames += 1

        # ---------------- DROP LOGIC ----------------
        if yolo_miss > YOLO_MISS_LIMIT or flow_frames > FLOW_MAX_FRAMES:
            locked_box = None
            locked_cls = None
            features = None
            yolo_miss = 0
            flow_frames = 0

        # ---------------- DRAW ----------------
        if locked_box is not None:
            x1, y1, x2, y2 = locked_box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                frame,
                f"{CLASS_NAMES[locked_cls]} [LOCKED]",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        prev_gray = gray.copy()

        cv2.imshow("Fast Smooth Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    print("[INFO] Shutdown cleanly")


if __name__ == "__main__":
    main()



# from ultralytics import YOLO
# import cv2
# import numpy as np
# import argparse
# import os
# import time

# # ===================== CONFIG ===================== #

# CLASS_NAMES = {
#     0: "ARJUNA",
#     1: "CAT",
#     2: "LION",
#     3: "NANDHI",
# }

# LOCK_CONF = 0.45          # lock / keep tracking
# DISPLAY_CONF = 0.60       # label confidence display
# YOLO_MISS_LIMIT = 8       # tolerate YOLO flicker
# FLOW_MAX_FRAMES = 15      # survive camera motion
# MAX_MOVE_RATIO = 0.45     # allow large motion
# FEATURE_COUNT = 100

# # ================================================= #


# def extract_features(gray, box):
#     x1, y1, x2, y2 = box.astype(int)
#     roi = gray[y1:y2, x1:x2]
#     if roi.size == 0:
#         return None

#     pts = cv2.goodFeaturesToTrack(
#         roi,
#         maxCorners=FEATURE_COUNT,
#         qualityLevel=0.01,
#         minDistance=5
#     )

#     if pts is None:
#         return None

#     pts[:, 0, 0] += x1
#     pts[:, 0, 1] += y1
#     return pts


# def dominant_box(boxes):
#     best = None
#     best_score = 0

#     for b in boxes:
#         cls = int(b.cls[0])
#         conf = float(b.conf[0])
#         if conf < LOCK_CONF:
#             continue

#         x1, y1, x2, y2 = b.xyxy[0]
#         area = (x2 - x1) * (y2 - y1)
#         score = area * conf

#         if score > best_score:
#             best_score = score
#             best = b

#     return best


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--source", required=True)
#     parser.add_argument("--model", default=os.path.join("model", "final_model.pt"))
#     parser.add_argument("--imgsz", type=int, nargs=2, default=[960, 960])
#     parser.add_argument("--conf", type=float, default=0.4)
#     parser.add_argument("--iou", type=float, default=0.5)
#     parser.add_argument("--device", default="cpu")
#     args = parser.parse_args()

#     print("[INFO] Loading YOLO model...")
#     model = YOLO(args.model)
#     print("[INFO] Model loaded")

#     prev_gray = None
#     locked_box = None
#     locked_cls = None
#     features = None

#     yolo_miss = 0
#     flow_frames = 0

#     results = model.predict(
#         source=args.source,
#         imgsz=tuple(args.imgsz),
#         conf=args.conf,
#         iou=args.iou,
#         device=args.device,
#         stream=True,
#         show=False
#     )

#     print("[INFO] Live tracking started")

#     for frame_id, r in enumerate(results):
#         if r.orig_img is None:
#             continue

#         frame = r.orig_img.copy()
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         updated = False

#         # ---------------- YOLO STEP ----------------
#         if r.boxes is not None and len(r.boxes) > 0:
#             print(f"[FRAME {frame_id}] [YOLO] Raw detections: {len(r.boxes)}")

#             dom = dominant_box(r.boxes)
#             if dom is not None:
#                 cls = int(dom.cls[0])
#                 conf = float(dom.conf[0])
#                 box = np.array(dom.xyxy[0], dtype=np.float32)

#                 # Update lock
#                 locked_box = box
#                 locked_cls = cls
#                 yolo_miss = 0
#                 flow_frames = 0

#                 features = extract_features(gray, locked_box)

#                 print(f"[FRAME {frame_id}] YOLO LOCK → {CLASS_NAMES[cls]} (conf={conf:.2f})")
#                 updated = True
#             else:
#                 yolo_miss += 1
#                 print(f"[FRAME {frame_id}] [YOLO] No dominant box passed threshold")

#         else:
#             yolo_miss += 1
#             print(f"[FRAME {frame_id}] [YOLO] No detections")

#         # ---------------- OPTICAL FLOW STEP ----------------
#         if not updated and locked_box is not None and prev_gray is not None and features is not None:
#             new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
#                 prev_gray, gray, features, None,
#                 winSize=(31, 31), maxLevel=4
#             )

#             good_new = new_pts[status == 1]
#             good_old = features[status == 1]

#             if len(good_new) >= 10:
#                 dx = np.mean(good_new[:, 0] - good_old[:, 0])
#                 dy = np.mean(good_new[:, 1] - good_old[:, 1])

#                 w = locked_box[2] - locked_box[0]
#                 h = locked_box[3] - locked_box[1]

#                 rx = abs(dx) / w
#                 ry = abs(dy) / h

#                 if rx < MAX_MOVE_RATIO and ry < MAX_MOVE_RATIO:
#                     locked_box += np.array([dx, dy, dx, dy])
#                     features = good_new.reshape(-1, 1, 2)
#                     flow_frames += 1

#                     print(f"[FRAME {frame_id}] FLOW TRACK ({flow_frames}) dx={dx:.2f}, dy={dy:.2f}")
#                 else:
#                     print(f"[FRAME {frame_id}] FLOW motion too large → ignore")

#             if yolo_miss > YOLO_MISS_LIMIT or flow_frames > FLOW_MAX_FRAMES:
#                 print(f"[FRAME {frame_id}] TRACK LOST → DROP")
#                 locked_box = None
#                 locked_cls = None
#                 features = None
#                 yolo_miss = 0
#                 flow_frames = 0

#         # ---------------- DRAW ----------------
#         if locked_box is not None:
#             x1, y1, x2, y2 = locked_box.astype(int)
#             label = f"{CLASS_NAMES[locked_cls]} [LOCKED]"
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
#             cv2.putText(frame, label, (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#         else:
#             print(f"[FRAME {frame_id}] [DRAW] No locked box")

#         prev_gray = gray.copy()

#         cv2.imshow("Live Detection (Production)", frame)
#         if cv2.waitKey(1) & 0xFF == 27:
#             break

#     cv2.destroyAllWindows()
#     print("[INFO] Shutdown cleanly")


# if __name__ == "__main__":
#     main()
