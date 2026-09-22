import easyocr
import re


reader = easyocr.Reader(["en"], gpu=False)


def read_plate(plate_crop):
    if plate_crop is None or plate_crop.size == 0:
        return None, 0.0

    results = reader.readtext(
        plate_crop,
        detail=1,
        paragraph=False,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    if not results:
        return None, 0.0

    best_result = max(results, key=lambda result: result[2])

    text = re.sub(r"[^A-Z0-9]", "", best_result[1].upper())
    confidence = float(best_result[2])

    if not text:
        return None, 0.0

    return text, confidence