from ultralytics import YOLO
import cv2
import argparse
import os

from config import MODEL_PATH as DEFAULT_MODEL_PATH

# python run_inference.py --source images/portrait.jpg --save
# python run_inference.py --source test_files/images/ --save
# python run_inference.py --source test_files/videos/video-3.mp4 --model model/final_model.pt --conf 0.65 --iou 0.5 --max-det 100 --save
# python run_inference.py --source 0

def main():
    parser = argparse.ArgumentParser(description="YOLO11 Inference Script (Portrait Safe)")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to image / folder / video file or webcam index (0)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs=2,
        default=[960, 640],  # (height, width) -> portrait-safe
        help="Inference image size as: height width"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the YOLO model (.pt file)"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold for NMS"
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=100,
        help="Maximum number of detections per image"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save output images/videos"
    )
    args = parser.parse_args()

    # Path to your trained model
    MODEL_PATH = args.model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    # Load model
    model = YOLO(MODEL_PATH)

    print("Model loaded successfully")
    print(f"Source: {args.source}")
    print(f"Confidence: {args.conf}")
    print(f"Image size (HxW): {args.imgsz}")

    # Run inference
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=tuple(args.imgsz),   # IMPORTANT: tuple enables portrait inference
        rect=True,                 # Preserve aspect ratio
        save=args.save,
        iou=args.iou,
        max_det=args.max_det,
        show=not args.save,        # Show window if not saving
        device="cpu"               # GPU if available
    )

    # Optional: print detections
    for i, r in enumerate(results):
        if r.boxes is not None:
            print(f"\nFrame {i}:")
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                print(f"  Class: {cls}, Conf: {conf:.2f}, Box: {xyxy}")

    print("\nInference complete.")

if __name__ == "__main__":
    main()
