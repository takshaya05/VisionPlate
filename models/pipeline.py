import time
import cv2
import numpy as np
from models.plate_validation import validate_indian_plate


class RecognitionPipeline:
    def __init__(self, detector, enhancer, ocr):
        self.detector = detector
        self.enhancer = enhancer
        self.ocr = ocr

    @staticmethod
    def _validate_image(image):
        if (
            image is None
            or not isinstance(image, np.ndarray)
            or image.size == 0
        ):
            raise ValueError("Input image is empty or invalid.")

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "Input image must be RGB or grayscale."
            )

        return np.clip(image, 0, 255).astype(np.uint8)

    @staticmethod
    def _crop(image, bbox, padding=0.04):
        height, width = image.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)

        pad_x = int((x2 - x1) * padding)
        pad_y = int((y2 - y1) * padding)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(width, x2 + pad_x)
        y2 = min(height, y2 + pad_y)

        crop = image[y1:y2, x1:x2]

        return crop.copy() if crop.size else None

    @staticmethod
    def _annotate(image, detections, plates):
        output = image.copy()
        height, width = output.shape[:2]

        for index, detection in enumerate(detections):
            x1, y1, x2, y2 = detection["bbox"]
            confidence = float(detection["confidence"])

            x1 = max(0, min(int(x1), width - 1))
            y1 = max(0, min(int(y1), height - 1))
            x2 = max(0, min(int(x2), width))
            y2 = max(0, min(int(y2), height))

            if x2 <= x1 or y2 <= y1:
                continue

            plate = plates[index] if index < len(plates) else {}
            text = plate.get("text") or "PLATE"
            label = f"{text} | {confidence:.2f}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.55
            thickness = 2

            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, scale, thickness
            )

            label_y = max(text_height + 10, y1)

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (56, 232, 255),
                2,
            )

            cv2.rectangle(
                output,
                (x1, max(0, label_y - text_height - 10)),
                (
                    min(width - 1, x1 + text_width + 10),
                    min(height - 1, label_y + baseline - 4),
                ),
                (36, 26, 52),
                -1,
            )

            cv2.putText(
                output,
                label,
                (x1 + 5, max(text_height, label_y - 5)),
                font,
                scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        return output

    def _recognize(self, image, detection, adaptive):
        crop = self._crop(image, detection["bbox"])

        if crop is None:
            return {
                "text": "",
                "ocr_confidence": 0.0,
                "ocr_engine": "Unavailable",
                "crop": None,
                "valid": False,
                "validation_message": "Plate crop could not be extracted.",
                "validation_format": None,
            }

        if adaptive:
            try:
                enhanced_crop, _ = self.enhancer.prepare(
                    crop,
                    enhance=True,
                    night_vision=False,
                    adaptive=True,
                )
                ocr_result = self.ocr.read_variants(
                    [crop, enhanced_crop]
                )
            except Exception:
                ocr_result = self.ocr.read(crop)
        else:
            ocr_result = self.ocr.read(crop)

        validation = validate_indian_plate(
            ocr_result.get("text", "")
        )

        return {
            "text": validation["normalized"],
            "ocr_confidence": float(
                ocr_result.get("confidence", 0.0)
            ),
            "ocr_engine": ocr_result.get("engine", "Unknown"),
            "crop": crop,
            "valid": validation["valid"],
            "validation_message": validation["message"],
            "validation_format": validation["format"],
        }

    def run(
        self,
        image,
        confidence=0.25,
        enhance=True,
        night_vision=False,
        adaptive=True,
    ):
        started = time.perf_counter()
        original = self._validate_image(image)

        enhancement_start = time.perf_counter()

        try:
            enhanced, enhancement_info = self.enhancer.prepare(
                original,
                enhance=enhance,
                night_vision=night_vision,
                adaptive=adaptive,
            )

        except Exception as exc:
            enhanced = original.copy()
            metrics = self.enhancer.metrics(original)

            enhancement_info = {
                "applied": [],
                "enhancement_applied": False,
                "night_vision_applied": False,
                "error": str(exc),
                "quality_before": metrics,
                "quality_after": metrics,
            }

        enhancement_ms = (
            time.perf_counter() - enhancement_start
        ) * 1000

        detection_start = time.perf_counter()
        model_available = bool(self.detector.available)
        model_status = self.detector.status

        if model_available:
            detection_image = enhanced
            detections = self.detector.detect(
                enhanced, confidence
            )

            if (
                adaptive
                and not detections
                and not np.array_equal(original, enhanced)
            ):
                alternate = self.detector.detect(
                    original, confidence
                )

                if alternate:
                    detections = alternate
                    detection_image = original
                    model_status += " | Original-image fallback used"

        else:
            detections = []
            detection_image = enhanced
            model_status = (
                self.detector.status
                or "Model unavailable; detection was skipped."
            )

        detection_ms = (
            time.perf_counter() - detection_start
        ) * 1000

        ocr_start = time.perf_counter()
        plates = []

        for detection in detections:
            try:
                plates.append(
                    self._recognize(
                        detection_image,
                        detection,
                        adaptive,
                    )
                )
            except Exception as exc:
                plates.append({
                    "text": "",
                    "ocr_confidence": 0.0,
                    "ocr_engine": "Error",
                    "crop": None,
                    "valid": False,
                    "validation_message": f"OCR failed: {exc}",
                    "validation_format": None,
                })

        ocr_ms = (time.perf_counter() - ocr_start) * 1000

        annotated = self._annotate(
            detection_image,
            detections,
            plates,
        )

        processing_ms = (time.perf_counter() - started) * 1000

        return {
            "original": original,
            "enhanced": enhanced,
            "annotated": annotated,
            "detections": detections,
            "plates": plates,
            "quality": self.enhancer.metrics(original),
            "enhancement_info": enhancement_info,
            "enhancement_applied": enhancement_info.get(
                "enhancement_applied", False
            ),
            "night_vision_applied": enhancement_info.get(
                "night_vision_applied", False
            ),
            "adaptive": bool(adaptive),
            "model_available": model_available,
            "model_status": model_status,
            "detection_ms": detection_ms,
            "enhancement_ms": enhancement_ms,
            "ocr_ms": ocr_ms,
            "processing_ms": processing_ms,
        }