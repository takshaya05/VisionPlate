import os
import warnings

warnings.filterwarnings(
    "ignore",
    message="torch.quantize_per_tensor.*deprecated"
)

warnings.filterwarnings(
    "ignore",
    message="'pin_memory' argument is set as true but no accelerator is found"
)

import cv2

from modules.detector import detect_plates
from modules.ocr import read_plate
from modules.processor import process_image


IMAGE_PATH = "assets/test_images/test.png"
RESULT_PATH = "assets/results/test_result.png"


def main():
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        print(f"\nError: Unable to load image: {IMAGE_PATH}")
        raise SystemExit(1)

    results = process_image(
        image,
        detect_plates,
        read_plate
    )

    output = image.copy()

    print("\n" + "=" * 55)
    print("                VISIONPLATE ANPR")
    print("=" * 55)
    print(f"Image           : {IMAGE_PATH}")
    print(f"Image Size      : {image.shape[1]} x {image.shape[0]}")

    if not results:
        print("\nStatus          : No license plate detected")

    else:
        print(f"Status          : {len(results)} plate(s) detected")
        print("-" * 55)

        for index, result in enumerate(results, 1):
            x1, y1, x2, y2 = result["box"]

            plate_text = result.get("plate_text") or "Not recognized"

            detection_confidence = float(
                result.get("detection_confidence", 0.0)
            )

            ocr_confidence = float(
                result.get("ocr_confidence", 0.0)
            )

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                4
            )

            label = f"{plate_text} | OCR {ocr_confidence:.0%}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.65
            thickness = 2

            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                font,
                font_scale,
                thickness
            )

            label_x = x1
            label_y = y1 - 10

            if label_y - text_height - baseline < 0:
                label_y = y1 + text_height + 10

            cv2.rectangle(
                output,
                (
                    label_x,
                    label_y - text_height - baseline
                ),
                (
                    label_x + text_width + 12,
                    label_y + 5
                ),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                output,
                label,
                (label_x + 6, label_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )

            print(f"\nPlate {index}")
            print(f"Plate Number    : {plate_text}")
            print(f"YOLO Confidence : {detection_confidence:.2%}")
            print(f"OCR Confidence  : {ocr_confidence:.2%}")
            print(f"Bounding Box    : {result['box']}")

    os.makedirs(
        os.path.dirname(RESULT_PATH),
        exist_ok=True
    )

    success = cv2.imwrite(
        RESULT_PATH,
        output
    )

    if success:
        print("\n" + "-" * 55)
        print(f"Result saved    : {RESULT_PATH}")
    else:
        print("\nError: Failed to save result image.")

    print("=" * 55)


if __name__ == "__main__":
    main()