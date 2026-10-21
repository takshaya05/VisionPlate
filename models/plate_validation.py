import re

INDIAN_PLATE_PATTERNS = [
    re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$"),
    re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$"),
    re.compile(r"^[A-Z]{2}TEMP[0-9]{1,6}$"),
]

INDIAN_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL", "DN",
    "GA", "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD",
    "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ",
    "SK", "TN", "TR", "TS", "UK", "UP", "WB",
}


def normalize_plate(text):
    """Normalize OCR text to uppercase alphanumeric characters."""
    text = str(text or "").upper().strip()
    return re.sub(r"[^A-Z0-9]", "", text)


def validate_indian_plate(text):
    """
    Check whether text matches one of the supported Indian plate formats.

    Returns:
        dict with valid, normalized, message, and format.
    """
    normalized = normalize_plate(text)

    if not normalized:
        return {
            "valid": False,
            "normalized": "",
            "message": "No registration text recognized.",
            "format": None,
        }

    for index, pattern in enumerate(INDIAN_PLATE_PATTERNS):
        if not pattern.fullmatch(normalized):
            continue

        if index == 0 and normalized[:2] not in INDIAN_STATE_CODES:
            return {
                "valid": False,
                "normalized": normalized,
                "message": (
                    "Pattern matched, but the state or union-territory "
                    "code is not recognized."
                ),
                "format": "standard-like",
            }

        labels = {
            0: "standard-like Indian registration",
            1: "Bharat Series (BH)",
            2: "temporary registration-like",
        }

        return {
            "valid": True,
            "normalized": normalized,
            "message": "Matches a supported registration pattern.",
            "format": labels[index],
        }

    return {
        "valid": False,
        "normalized": normalized,
        "message": "Does not match a supported Indian registration pattern.",
        "format": None,
    }


def is_valid_indian_plate(text):
    """Return True when text matches a supported format."""
    return validate_indian_plate(text)["valid"]