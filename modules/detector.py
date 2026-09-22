from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "plate_model.pt"

model = YOLO(str(MODEL_PATH))


def detect_plates(image):
    results = model(image, verbose=False)

    detections = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])

            detections.append({
                "box": (x1, y1, x2, y2),
                "confidence": confidence
            })

    return detections