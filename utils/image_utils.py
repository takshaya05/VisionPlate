import os
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageOps


def load_image(source):
    """
    Load an image from bytes, a file-like object, a path,
    a PIL image, or a NumPy array.

    Returns an RGB uint8 NumPy array.
    """
    if isinstance(source, np.ndarray):
        image = source

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected an RGB or grayscale image.")

        return np.clip(image, 0, 255).astype(np.uint8)

    if isinstance(source, Image.Image):
        return np.asarray(
            ImageOps.exif_transpose(source).convert("RGB"),
            dtype=np.uint8,
        ).copy()

    if isinstance(source, (bytes, bytearray, memoryview)):
        image_bytes = bytes(source)

    elif isinstance(source, (str, os.PathLike)):
        with open(source, "rb") as file:
            image_bytes = file.read()

    elif hasattr(source, "read"):
        image_bytes = source.read()

    else:
        raise TypeError("Unsupported image source.")

    if not image_bytes:
        raise ValueError("The uploaded image is empty.")

    if isinstance(image_bytes, str):
        raise TypeError("The source must return binary image data.")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            return np.asarray(
                ImageOps.exif_transpose(image).convert("RGB"),
                dtype=np.uint8,
            ).copy()

    except Exception as pil_error:
        encoded = np.frombuffer(image_bytes, dtype=np.uint8)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        if decoded is None:
            raise ValueError(
                f"Unable to decode the image: {pil_error}"
            ) from pil_error

        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def image_quality_metrics(image):
    """Calculate descriptive image quality metrics."""
    if (
        image is None
        or not isinstance(image, np.ndarray)
        or image.size == 0
    ):
        raise ValueError("A valid image is required.")

    if image.ndim == 2:
        gray = image.astype(np.uint8)

    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(
            np.clip(image, 0, 255).astype(np.uint8),
            cv2.COLOR_RGB2GRAY,
        )

    else:
        raise ValueError("Expected an RGB or grayscale image.")

    return {
        "brightness": float(np.mean(gray)),
        "contrast": float(np.std(gray)),
        "sharpness": float(
            cv2.Laplacian(gray, cv2.CV_64F).var()
        ),
        "width": int(gray.shape[1]),
        "height": int(gray.shape[0]),
    }


def image_to_bytes(image, extension=".png"):
    """Encode an RGB image as downloadable bytes."""
    if (
        image is None
        or not isinstance(image, np.ndarray)
        or image.size == 0
    ):
        raise ValueError("A valid image is required.")

    extension = str(extension).lower()

    if extension not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        extension = ".png"

    if image.ndim == 2:
        encoded_image = image.astype(np.uint8)

    elif image.ndim == 3 and image.shape[2] == 3:
        encoded_image = cv2.cvtColor(
            np.clip(image, 0, 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        )

    else:
        raise ValueError("Expected an RGB or grayscale image.")

    success, encoded = cv2.imencode(extension, encoded_image)

    if not success:
        raise ValueError(f"Could not encode image as {extension}.")

    return encoded.tobytes()


def draw_detections(image, detections, labels=None):
    """Draw bounding boxes and labels on an RGB image."""
    if (
        image is None
        or not isinstance(image, np.ndarray)
        or image.size == 0
    ):
        raise ValueError("A valid image is required.")

    if image.ndim == 2:
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 3:
        output = image.copy()
    else:
        raise ValueError("Expected an RGB or grayscale image.")

    labels = labels or []
    height, width = output.shape[:2]

    for index, detection in enumerate(detections or []):
        try:
            x1, y1, x2, y2 = map(int, detection["bbox"])
            confidence = float(detection.get("confidence", 0.0))

            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

            if x2 <= x1 or y2 <= y1:
                continue

            label = (
                str(labels[index])
                if index < len(labels)
                else f"Plate {confidence:.2f}"
            )

            cv2.rectangle(
                output, (x1, y1), (x2, y2), (56, 232, 255), 2
            )

            text_y = max(22, y1 - 8)

            cv2.putText(
                output,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        except (KeyError, TypeError, ValueError):
            continue

    return output


def make_comparison(original, enhanced, labels=True):
    """Build a side-by-side original/enhanced comparison."""

    def normalize(image):
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError(
                "Comparison images must be non-empty arrays."
            )

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "Comparison images must be RGB or grayscale."
            )

        return np.clip(image, 0, 255).astype(np.uint8)

    original = normalize(original)
    enhanced = normalize(enhanced)

    target_height = max(
        original.shape[0], enhanced.shape[0], 1
    )

    def resize_height(image):
        height, width = image.shape[:2]
        new_width = max(
            1, int(round(width * target_height / height))
        )

        interpolation = (
            cv2.INTER_AREA
            if target_height < height
            else cv2.INTER_CUBIC
        )

        return cv2.resize(
            image,
            (new_width, target_height),
            interpolation=interpolation,
        )

    original = resize_height(original)
    enhanced = resize_height(enhanced)

    gap = 8
    separator = np.full(
        (target_height, gap, 3), 20, dtype=np.uint8
    )

    comparison = np.concatenate(
        [original, separator, enhanced], axis=1
    )

    if labels:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.45, min(0.8, target_height / 700))
        thickness = 2

        cv2.rectangle(
            comparison,
            (0, 0),
            (min(original.shape[1], 230), 36),
            (15, 24, 43),
            -1,
        )
        cv2.putText(
            comparison,
            "ORIGINAL",
            (12, 24),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

        right_start = original.shape[1] + gap

        cv2.rectangle(
            comparison,
            (right_start, 0),
            (
                min(comparison.shape[1] - 1, right_start + 250),
                36,
            ),
            (32, 22, 55),
            -1,
        )
        cv2.putText(
            comparison,
            "ENHANCED",
            (right_start + 12, 24),
            font,
            scale,
            (56, 232, 255),
            thickness,
            cv2.LINE_AA,
        )

    return comparison