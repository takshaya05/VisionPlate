def process_image(image, detect_plates, read_plate):
    if image is None or image.size == 0:
        return []

    detections = detect_plates(image)

    results = []

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]

        height, width = image.shape[:2]

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))

        plate_crop = image[y1:y2, x1:x2]

        if plate_crop.size == 0:
            continue

        plate_text, ocr_confidence = read_plate(plate_crop)

        results.append({
            "plate_text": plate_text,
            "detection_confidence": float(detection["confidence"]),
            "ocr_confidence": float(ocr_confidence),
            "box": (x1, y1, x2, y2),
            "crop": plate_crop
        })

    return results