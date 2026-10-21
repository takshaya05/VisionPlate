from pathlib import Path

import cv2
import numpy as np


class PlateDetector:
    """YOLO-based plate detector with safe model-loading failures."""

    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.model = None
        self.status = "Not initialized"
        self.error = None
        self._load_model()

    def _load_model(self):
        if not self.model_path.is_file():
            self.status = f"Model file not found: {self.model_path.name}"
            return

        try:
            from ultralytics import YOLO

            self.model = YOLO(str(self.model_path))
            self.status = f"Loaded model: {self.model_path.name}"
            self.error = None

        except Exception as exc:
            self.model = None
            self.error = str(exc)
            self.status = f"Model loading failed: {exc}"

    @property
    def available(self):
        return self.model is not None

    def detect(self, image, confidence=0.25):
        """
        Detect plates in an RGB uint8 NumPy image.

        Returns bounding boxes in [x1, y1, x2, y2] format.
        """
        if not self.available:
            return []

        if (
            image is None
            or not isinstance(image, np.ndarray)
            or image.size == 0
        ):
            return []

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if image.ndim != 3 or image.shape[2] != 3:
            self.error = "Expected an RGB or grayscale image."
            return []

        image = np.clip(image, 0, 255).astype(np.uint8)
        confidence = float(np.clip(confidence, 0.01, 0.99))

        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            predictions = self.model.predict(
                source=image_bgr,
                conf=confidence,
                verbose=False,
            )

            detections = []
            if not predictions:
                return detections

            result = predictions[0]
            boxes = getattr(result, "boxes", None)

            if boxes is None:
                return detections

            height, width = image.shape[:2]

            for box in boxes:
                try:
                    coords = (
                        box.xyxy[0]
                        .detach()
                        .cpu()
                        .numpy()
                        .tolist()
                    )
                    score = float(box.conf[0].detach().cpu().item())
                    class_id = int(box.cls[0].detach().cpu().item())

                    x1 = max(
                        0, min(int(round(coords[0])), width - 1)
                    )
                    y1 = max(
                        0, min(int(round(coords[1])), height - 1)
                    )
                    x2 = max(
                        0, min(int(round(coords[2])), width)
                    )
                    y2 = max(
                        0, min(int(round(coords[3])), height)
                    )

                    if x2 <= x1 or y2 <= y1:
                        continue

                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": score,
                        "class_id": class_id,
                    })

                except (
                    ValueError,
                    TypeError,
                    IndexError,
                    AttributeError,
                ):
                    continue

            self.error = None
            return detections

        except Exception as exc:
            self.error = str(exc)
            self.status = f"Inference failed: {exc}"
            return []

    def reload(self):
        """Reload the model from disk."""
        self.model = None
        self.error = None
        self._load_model()
        return self.available