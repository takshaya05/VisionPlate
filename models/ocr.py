import re
import cv2
import numpy as np

from models.plate_validation import validate_indian_plate


class PlateOCR:
    """Multi-engine license plate OCR with preprocessing and fallbacks."""

    ALPHANUMERIC = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    def __init__(self, languages=None, use_gpu=False):
        self.languages = languages or ["en"]
        self.use_gpu = bool(use_gpu)

        self.reader = None
        self.tesseract = None
        self.status = "Not initialized"
        self.errors = []

        self._initialize()

    def _initialize(self):
        self.errors = []
        statuses = []

        try:
            import easyocr

            self.reader = easyocr.Reader(
                self.languages,
                gpu=self.use_gpu,
                verbose=False,
            )
            statuses.append("EasyOCR available")

        except Exception as exc:
            self.reader = None
            self.errors.append(f"EasyOCR initialization: {exc}")
            statuses.append("EasyOCR unavailable")

        try:
            import pytesseract

            pytesseract.get_tesseract_version()
            self.tesseract = pytesseract
            statuses.append("Tesseract available")

        except Exception as exc:
            self.tesseract = None
            self.errors.append(f"Tesseract initialization: {exc}")
            statuses.append("Tesseract unavailable")

        self.status = " | ".join(statuses)

        if self.reader is None and self.tesseract is None:
            self.status += " | No OCR engine is available"

    @property
    def available(self):
        return self.reader is not None or self.tesseract is not None

    @staticmethod
    def _prepare_crop(crop):
        """Validate, normalize, and upscale a plate crop."""
        if (
            crop is None
            or not isinstance(crop, np.ndarray)
            or crop.size == 0
        ):
            raise ValueError("A non-empty plate crop is required.")

        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)

        if crop.ndim != 3 or crop.shape[2] != 3:
            raise ValueError(
                "Expected an RGB or grayscale plate crop."
            )

        crop = np.clip(crop, 0, 255).astype(np.uint8)

        height, width = crop.shape[:2]

        if height < 40 or width < 40:
            raise ValueError("Plate crop is too small for OCR.")

        # Upscale without changing the aspect ratio.
        if height < 100:
            scale = min(4.0, 100.0 / height)
            crop = cv2.resize(
                crop,
                (
                    max(1, int(round(width * scale))),
                    max(1, int(round(height * scale))),
                ),
                interpolation=cv2.INTER_CUBIC,
            )

        return crop

    @staticmethod
    def _clean_text(text):
        """Keep only uppercase letters and digits."""
        return re.sub(
            r"[^A-Z0-9]",
            "",
            str(text or "").upper(),
        )

    @staticmethod
    def _make_variants(crop):
        """
        Generate alternate RGB crops for OCR.

        All returned variants have three channels.
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

        # Improve local contrast without changing the original.
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )
        enhanced = clahe.apply(gray)

        # Mild sharpening; avoid aggressive filtering.
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
        sharpened = cv2.addWeighted(
            enhanced, 1.5, blurred, -0.5, 0
        )

        _, otsu = cv2.threshold(
            enhanced,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        adaptive = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )

        # Keep the number of variants small to control processing time.
        grayscale_rgb = cv2.cvtColor(
            enhanced, cv2.COLOR_GRAY2RGB
        )
        sharpened_rgb = cv2.cvtColor(
            sharpened, cv2.COLOR_GRAY2RGB
        )
        otsu_rgb = cv2.cvtColor(
            otsu, cv2.COLOR_GRAY2RGB
        )
        adaptive_rgb = cv2.cvtColor(
            adaptive, cv2.COLOR_GRAY2RGB
        )

        return [
            ("original", crop),
            ("contrast", grayscale_rgb),
            ("sharpened", sharpened_rgb),
            ("otsu", otsu_rgb),
            ("adaptive", adaptive_rgb),
        ]

    def _easyocr_read(self, crop):
        if self.reader is None:
            return None

        try:
            results = self.reader.readtext(
                crop,
                detail=1,
                paragraph=False,
                allowlist=self.ALPHANUMERIC,
            )

            candidates = []

            for item in results or []:
                if len(item) < 3:
                    continue

                text = self._clean_text(item[1])

                try:
                    confidence = float(item[2])
                except (ValueError, TypeError):
                    confidence = 0.0

                if text:
                    candidates.append({
                        "text": text,
                        "confidence": float(
                            np.clip(confidence, 0.0, 1.0)
                        ),
                        "engine": "EasyOCR",
                    })

            return max(
                candidates,
                key=self._candidate_score,
                default=None,
            )

        except Exception as exc:
            self.errors.append(f"EasyOCR inference: {exc}")
            return None

    def _tesseract_read(self, crop):
        if self.tesseract is None:
            return None

        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

            config = (
                "--oem 3 --psm 7 "
                "-c tessedit_char_whitelist="
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )

            data = self.tesseract.image_to_data(
                gray,
                config=config,
                output_type=self.tesseract.Output.DICT,
            )

            tokens = []
            confidences = []

            for raw_text, raw_confidence in zip(
                data.get("text", []),
                data.get("conf", []),
            ):
                text = self._clean_text(raw_text)

                if not text:
                    continue

                tokens.append(text)

                try:
                    confidence = float(raw_confidence)
                    if confidence >= 0:
                        confidences.append(confidence / 100.0)
                except (ValueError, TypeError):
                    pass

            text = "".join(tokens)

            if not text:
                return None

            return {
                "text": text,
                "confidence": float(
                    np.mean(confidences) if confidences else 0.0
                ),
                "engine": "Tesseract",
            }

        except Exception as exc:
            self.errors.append(f"Tesseract inference: {exc}")
            return None

    @staticmethod
    def _candidate_score(candidate):
        """
        Rank candidates using format validity and OCR confidence.

        This is a heuristic, not a calibrated probability of correctness.
        """
        text = candidate.get("text", "")
        validation = validate_indian_plate(text)

        confidence = float(candidate.get("confidence", 0.0))
        confidence = float(np.clip(confidence, 0.0, 1.0))

        # A common Indian registration number has around 8-10
        # alphanumeric characters, but formats can vary.
        plausible_length = 8 <= len(text) <= 10

        # Format validity is useful, but should not overwhelm OCR quality.
        return (
            0.45 * int(validation["valid"])
            + 0.35 * confidence
            + 0.20 * int(plausible_length)
        )

    def _read_one(self, crop):
        """Run both engines on several image variants."""
        try:
            prepared = self._prepare_crop(crop)
        except (ValueError, TypeError):
            return None

        candidates = []

        for _, variant in self._make_variants(prepared):
            easy_result = self._easyocr_read(variant)
            tess_result = self._tesseract_read(variant)

            if easy_result and easy_result.get("text"):
                candidates.append(easy_result)

            if tess_result and tess_result.get("text"):
                candidates.append(tess_result)

        # Avoid choosing an empty or malformed candidate.
        candidates = [
            candidate
            for candidate in candidates
            if candidate.get("text")
        ]

        return max(
            candidates,
            key=self._candidate_score,
            default=None,
        )

    def read(self, crop):
        """Recognize text from a single plate crop."""
        result = self._read_one(crop)

        if result is None:
            return {
                "text": "",
                "confidence": 0.0,
                "engine": "Unavailable",
            }

        return result

    def read_variants(self, crops):
        """
        Compare multiple supplied crops.

        Each crop is itself evaluated using multiple OCR variants.
        """
        candidates = []

        for crop in crops or []:
            result = self._read_one(crop)

            if result and result.get("text"):
                candidates.append(result)

        return max(
            candidates,
            key=self._candidate_score,
            default={
                "text": "",
                "confidence": 0.0,
                "engine": "Unavailable",
            },
        )
