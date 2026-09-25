import streamlit as st
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
from PIL import Image
import torch
import re
import hashlib

st.set_page_config(
    page_title="VisionPlate – Smart Plate Recognition",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(0,217,255,0.08), transparent 28%),
        radial-gradient(circle at 90% 15%, rgba(104,117,255,0.09), transparent 30%),
        radial-gradient(circle at 50% 90%, rgba(38,215,199,0.06), transparent 28%),
        #070b12;
    color: #eef5ff;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid rgba(94,211,255,0.14) !important;
    border-radius: 20px !important;
    background:
        linear-gradient(
            145deg,
            rgba(18,28,44,0.88),
            rgba(9,15,25,0.94)
        ) !important;
    box-shadow:
        0 10px 30px rgba(0,0,0,0.18),
        0 0 18px rgba(0,217,255,0.035),
        inset 0 1px 0 rgba(255,255,255,0.025) !important;
    padding: 5px !important;
    margin-bottom: 20px !important;
    transition: all 0.25s ease;
}

[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(0,217,255,0.26) !important;
    box-shadow:
        0 14px 36px rgba(0,0,0,0.24),
        0 0 24px rgba(0,217,255,0.055),
        inset 0 1px 0 rgba(255,255,255,0.035) !important;
}

.stButton > button {
    width: 100%;
    border-radius: 12px;
    border: 1px solid rgba(0,217,255,0.24);
    background: linear-gradient(135deg, #102234, #132d43);
    color: #eaf9ff;
    font-weight: 600;
    min-height: 44px;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border-color: #00d9ff;
    color: white;
    background: linear-gradient(135deg, #123149, #173d5b);
    box-shadow: 0 0 20px rgba(0,217,255,0.12);
    transform: translateY(-1px);
}

.stButton > button:focus {
    box-shadow: 0 0 0 2px rgba(0,217,255,0.18);
}

button[aria-label="🚀 Start Now"] {
    width: 220px !important;
    margin: 0 auto !important;
    display: block !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(10,16,27,0.72);
    padding: 7px;
    border-radius: 15px;
    border: 1px solid rgba(94,211,255,0.10);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 10px 20px;
    color: #9eafc4;
}

.stTabs [aria-selected="true"] {
    background: rgba(0,217,255,0.10);
    color: #72eaff !important;
}

[data-testid="stFileUploader"] {
    border-radius: 16px;
}

[data-testid="stMetric"] {
    background: rgba(13,24,38,0.65);
    border: 1px solid rgba(94,211,255,0.12);
    border-radius: 15px;
    padding: 12px;
}

[data-testid="stMetricValue"] {
    color: #72eaff;
}

[data-testid="stImage"] {
    border-radius: 15px;
    overflow: hidden;
}

.stAlert {
    border-radius: 14px;
}

.hero-title {
    font-size: clamp(42px, 6vw, 78px);
    line-height: 0.98;
    font-weight: 800;
    letter-spacing: -3px;
    margin: 12px 0;
    background: linear-gradient(
        90deg,
        #f2fbff,
        #72eaff,
        #8f86ff
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-tagline {
    font-size: 22px;
    font-weight: 600;
    color: #72eaff;
    margin-bottom: 14px;
}

.hero-description {
    color: #a9b9cc;
    font-size: 16px;
    line-height: 1.8;
    max-width: 720px;
}

.section-kicker {
    color: #72eaff;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

.section-title {
    font-size: 30px;
    font-weight: 750;
    color: #f3f8ff;
    margin: 5px 0 4px;
}

.section-description {
    color: #8fa2b8;
    line-height: 1.7;
}

.card-text {
    color: #91a4ba;
    line-height: 1.65;
    font-size: 14px;
}

.feature-icon {
    font-size: 30px;
    margin-bottom: 5px;
}

.about-intro {
    color: #a9b9cc;
    line-height: 1.8;
    font-size: 15px;
}

.tech-name {
    font-size: 16px;
    font-weight: 700;
    color: #eaf7ff;
}

.tech-desc {
    color: #899cb2;
    font-size: 13px;
    line-height: 1.5;
}

.plate-result {
    font-size: 34px;
    font-weight: 800;
    letter-spacing: 2px;
    color: #72eaff;
    text-align: center;
    padding: 12px 0;
}

.result-label {
    color: #8195ac;
    font-size: 12px;
    text-align: center;
    text-transform: uppercase;
    letter-spacing: 1.2px;
}

.camera-note {
    color: #8fa2b8;
    font-size: 13px;
    text-align: center;
    padding: 5px;
}

.footer-brand {
    font-size: 18px;
    font-weight: 800;
    color: #eaf7ff;
}

.footer-tagline {
    color: #72eaff;
    font-size: 13px;
    margin-top: -5px;
}

.footer-text {
    color: #778ba2;
    font-size: 12px;
    line-height: 1.6;
}

.small-gap {
    height: 10px;
}

.hero-points {
    color: #91a4ba;
    font-size: 14px;
    line-height: 1.9;
    margin-top: 12px;
}

.about-badge {
    color: #72eaff;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

[data-testid="stMarkdownContainer"] p {
    margin-bottom: 0.45rem;
}

hr {
    border-color: rgba(94,211,255,0.10) !important;
}
</style>
""", unsafe_allow_html=True)

if "section" not in st.session_state:
    st.session_state.section = "Home"

if "camera_enabled" not in st.session_state:
    st.session_state.camera_enabled = False

if "camera_key" not in st.session_state:
    st.session_state.camera_key = 0

if "image_key" not in st.session_state:
    st.session_state.image_key = 0

if "image_signature" not in st.session_state:
    st.session_state.image_signature = None

if "image_result" not in st.session_state:
    st.session_state.image_result = None

if "image_detections" not in st.session_state:
    st.session_state.image_detections = []

if "camera_result" not in st.session_state:
    st.session_state.camera_result = None

if "camera_detections" not in st.session_state:
    st.session_state.camera_detections = []

if "video_key" not in st.session_state:
    st.session_state.video_key = 0


@st.cache_resource
def load_model():
    return YOLO("plate_model.pt")


@st.cache_resource
def load_reader():
    return easyocr.Reader(
        ["en"],
        gpu=torch.cuda.is_available()
    )


try:
    model = load_model()
    reader = load_reader()
    model_ready = True
except Exception as e:
    model_ready = False
    model_error = str(e)


def clean_plate(text):
    text = str(text).upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def correct_letter(char):
    replacements = {
        "0": "O",
        "1": "I",
        "2": "Z",
        "5": "S",
        "6": "G",
        "8": "B"
    }

    return replacements.get(char, char)


def correct_digit(char):
    replacements = {
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "S": "5",
        "B": "8",
        "G": "6"
    }

    return replacements.get(char, char)


def correct_segment(segment, segment_type):
    result = []

    for char in segment:

        if segment_type == "letter":

            if char in "0123456789":
                char = correct_letter(char)

        elif segment_type == "digit":

            if char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                char = correct_digit(char)

        result.append(char)

    return "".join(result)


def validate_indian_plate_raw(text):
    text = clean_plate(text)

    patterns = [
        r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$",
        r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$"
    ]

    return any(
        re.fullmatch(pattern, text)
        for pattern in patterns
    )


def generate_indian_plate_candidates(text):
    text = clean_plate(text)

    if not text:
        return []

    candidates = []

    possible_lengths = [
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 1),
        (2, 2),
        (2, 3)
    ]

    for district_length, series_length in possible_lengths:

        expected_length = (
            2
            + district_length
            + series_length
            + 4
        )

        if len(text) != expected_length:
            continue

        state_end = 2
        district_end = state_end + district_length
        series_end = district_end + series_length

        state_part = text[:state_end]

        district_part = text[
            state_end:district_end
        ]

        series_part = text[
            district_end:series_end
        ]

        number_part = text[
            series_end:series_end + 4
        ]

        corrected_state = correct_segment(
            state_part,
            "letter"
        )

        corrected_district = correct_segment(
            district_part,
            "digit"
        )

        corrected_series = correct_segment(
            series_part,
            "letter"
        )

        corrected_number = correct_segment(
            number_part,
            "digit"
        )

        candidate = (
            corrected_state
            + corrected_district
            + corrected_series
            + corrected_number
        )

        if validate_indian_plate_raw(candidate):
            candidates.append(candidate)

    return list(dict.fromkeys(candidates))


def normalize_bharat_plate(text):
    text = clean_plate(text)

    if len(text) != 10:
        return ""

    first_part = correct_segment(
        text[:2],
        "digit"
    )

    bh_part = correct_segment(
        text[2:4],
        "letter"
    )

    number_part = correct_segment(
        text[4:8],
        "digit"
    )

    series_part = correct_segment(
        text[8:],
        "letter"
    )

    candidate = (
        first_part
        + bh_part
        + number_part
        + series_part
    )

    if re.fullmatch(
        r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$",
        candidate
    ):
        return candidate

    return ""


def normalize_indian_plate(text):
    text = clean_plate(text)

    bharat_candidate = normalize_bharat_plate(text)

    if bharat_candidate:
        return bharat_candidate

    if validate_indian_plate_raw(text):
        return text

    candidates = generate_indian_plate_candidates(text)

    if candidates:
        return candidates[0]

    return text


def validate_indian_plate(text):
    normalized = normalize_indian_plate(text)

    return validate_indian_plate_raw(normalized)


def prepare_ocr_candidates(crop):

    if crop is None or crop.size == 0:
        return []

    if len(crop.shape) == 3:
        gray = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray = crop.copy()

    height, width = gray.shape[:2]

    if width < 500:
        scale = 4.0
    elif width < 900:
        scale = 3.0
    else:
        scale = 2.0

    resized = cv2.resize(
        gray,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    denoised = cv2.bilateralFilter(
        resized,
        9,
        75,
        75
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(denoised)

    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    sharpened = cv2.filter2D(
        enhanced,
        -1,
        sharpen_kernel
    )

    otsu = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7
    )

    return [
        resized,
        enhanced,
        sharpened,
        otsu,
        adaptive
    ]


def read_plate(crop):

    if crop is None or crop.size == 0:
        return "", 0.0

    candidates = prepare_ocr_candidates(crop)

    best_text = ""
    best_confidence = 0.0
    best_score = -1.0
    best_valid = False

    for candidate in candidates:

        try:

            results = reader.readtext(
                candidate,
                decoder="beamsearch",
                beamWidth=10,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                detail=1,
                paragraph=False,
                min_size=5,
                contrast_ths=0.05,
                adjust_contrast=0.6,
                text_threshold=0.55,
                low_text=0.25,
                link_threshold=0.25,
                canvas_size=2560,
                mag_ratio=1.2,
                add_margin=0.12
            )

            if not results:
                continue

            individual_results = []

            for item in results:

                if len(item) < 3:
                    continue

                raw_text = clean_plate(item[1])
                confidence = float(item[2])

                if raw_text:
                    individual_results.append(
                        (
                            item,
                            raw_text,
                            confidence
                        )
                    )

            combined_candidates = []

            for item, raw_text, confidence in individual_results:

                combined_candidates.append(
                    (
                        raw_text,
                        confidence
                    )
                )

            if len(individual_results) > 1:

                sorted_results = sorted(
                    individual_results,
                    key=lambda x: x[0][0][0][0]
                )

                combined_text = "".join(
                    item[1]
                    for item in sorted_results
                )

                combined_confidence = sum(
                    item[2]
                    for item in sorted_results
                ) / len(sorted_results)

                combined_candidates.append(
                    (
                        combined_text,
                        combined_confidence
                    )
                )

            for raw_text, confidence in combined_candidates:

                normalized = normalize_indian_plate(
                    raw_text
                )

                valid = validate_indian_plate_raw(
                    normalized
                )

                score = confidence

                if valid:
                    score += 0.50

                if 9 <= len(normalized) <= 10:
                    score += 0.12

                if len(raw_text) >= 8:
                    score += 0.05

                if (
                    score > best_score
                    or
                    (
                        valid
                        and not best_valid
                    )
                ):

                    best_score = score
                    best_text = normalized
                    best_confidence = confidence
                    best_valid = valid

        except Exception:
            continue

    if not best_text:
        return "", 0.0

    return (
        best_text,
        best_confidence
    )


def draw_detection(
    image,
    box,
    plate,
    confidence
):

    x1, y1, x2, y2 = map(
        int,
        box
    )

    output = image.copy()

    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        (0, 217, 255),
        3
    )

    label = (
        f"{plate if plate else 'Plate'} "
        f"| {confidence:.0%}"
    )

    text_y = max(
        y1 - 12,
        25
    )

    cv2.rectangle(
        output,
        (x1, text_y - 27),
        (
            x1 + max(
                180,
                len(label) * 12
            ),
            text_y + 3
        ),
        (10, 25, 40),
        -1
    )

    cv2.putText(
        output,
        label,
        (x1 + 6, text_y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (114, 234, 255),
        2,
        cv2.LINE_AA
    )

    return output


def process_frame(image):

    if image is None:
        return None, []

    if isinstance(
        image,
        Image.Image
    ):
        image = np.array(image)

    if (
        len(image.shape) == 3
        and image.shape[2] == 4
    ):
        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGBA2RGB
        )

    rgb_image = image.copy()

    results = model(
        rgb_image,
        conf=0.25,
        verbose=False
    )

    output = rgb_image.copy()
    detections = []

    for result in results:

        boxes = result.boxes

        if boxes is None:
            continue

        for box in boxes:

            coordinates = (
                box.xyxy[0]
                .cpu()
                .numpy()
            )

            detection_confidence = float(
                box.conf[0]
                .cpu()
                .numpy()
            )

            x1, y1, x2, y2 = map(
                int,
                coordinates
            )

            x1 = max(
                0,
                x1
            )

            y1 = max(
                0,
                y1
            )

            x2 = min(
                rgb_image.shape[1],
                x2
            )

            y2 = min(
                rgb_image.shape[0],
                y2
            )

            if (
                x2 <= x1
                or y2 <= y1
            ):
                continue

            plate_width = x2 - x1
            plate_height = y2 - y1

            padding_x = max(
                5,
                int(plate_width * 0.06)
            )

            padding_y = max(
                5,
                int(plate_height * 0.10)
            )

            crop_x1 = max(
                0,
                x1 - padding_x
            )

            crop_y1 = max(
                0,
                y1 - padding_y
            )

            crop_x2 = min(
                rgb_image.shape[1],
                x2 + padding_x
            )

            crop_y2 = min(
                rgb_image.shape[0],
                y2 + padding_y
            )

            crop = rgb_image[
                crop_y1:crop_y2,
                crop_x1:crop_x2
            ]

            bgr_crop = cv2.cvtColor(
                crop,
                cv2.COLOR_RGB2BGR
            )

            plate_text, ocr_confidence = read_plate(
                bgr_crop
            )

            display_confidence = (
                ocr_confidence
                if plate_text
                else detection_confidence
            )

            output = draw_detection(
                output,
                (
                    x1,
                    y1,
                    x2,
                    y2
                ),
                plate_text,
                display_confidence
            )

            detections.append({
                "plate": plate_text,
                "detection_confidence": detection_confidence,
                "ocr_confidence": ocr_confidence,
                "valid": validate_indian_plate(
                    plate_text
                )
            })

    return output, detections


def show_detection_results(
    detections
):

    if not detections:

        st.warning(
            "No number plate detected."
        )

        return

    st.markdown(
        "### Detection Results"
    )

    total = len(detections)

    recognized = sum(
        1
        for d in detections
        if d["plate"]
    )

    valid = sum(
        1
        for d in detections
        if d["valid"]
    )

    m1, m2, m3 = st.columns(
        3,
        gap="large"
    )

    with m1:

        st.metric(
            "Plates Detected",
            total
        )

    with m2:

        st.metric(
            "Text Recognized",
            recognized
        )

    with m3:

        st.metric(
            "Valid Indian Plates",
            valid
        )

    st.markdown(
        "<div class='small-gap'></div>",
        unsafe_allow_html=True
    )

    for index, detection in enumerate(
        detections,
        1
    ):

        with st.container(
            border=True
        ):

            c1, c2, c3 = st.columns(
                [1.4, 1, 1],
                gap="large"
            )

            with c1:

                st.markdown(
                    f"<div class='result-label'>Plate {index}</div>",
                    unsafe_allow_html=True
                )

                plate = (
                    detection["plate"]
                    or "Not Recognized"
                )

                st.markdown(
                    f"<div class='plate-result'>{plate}</div>",
                    unsafe_allow_html=True
                )

            with c2:

                st.metric(
                    "OCR Confidence",
                    f"{detection['ocr_confidence']:.1%}"
                )

            with c3:

                if detection["valid"]:

                    st.success(
                        "Valid Indian Plate"
                    )

                elif detection["plate"]:

                    st.warning(
                        "Format Not Confirmed"
                    )

                else:

                    st.error(
                        "Text Not Recognized"
                    )


def reset_image():

    st.session_state.image_key += 1
    st.session_state.image_signature = None
    st.session_state.image_result = None
    st.session_state.image_detections = []


def reset_camera():

    st.session_state.camera_key += 1
    st.session_state.camera_result = None
    st.session_state.camera_detections = []


def close_camera():

    st.session_state.camera_enabled = False
    st.session_state.camera_key += 1
    st.session_state.camera_result = None
    st.session_state.camera_detections = []


with st.container():

    header_left, header_right = st.columns(
        [2.8, 1],
        gap="large"
    )

    with header_left:

        st.markdown(
            "<div style='font-size:24px;font-weight:800;color:#eaf7ff;'>🚘 VisionPlate</div>",
            unsafe_allow_html=True
        )

        st.caption(
            "Smart Plate Recognition"
        )

    with header_right:

        st.markdown(
            "<div style='height:8px'></div>",
            unsafe_allow_html=True
        )

        n1, n2, n3 = st.columns(
            3,
            gap="small"
        )

        with n1:

            if st.button(
                "Home",
                key="nav_home"
            ):

                st.session_state.section = "Home"
                st.rerun()

        with n2:

            if st.button(
                "About",
                key="nav_about"
            ):

                st.session_state.section = "About"
                st.rerun()

        with n3:

            if st.button(
                "Dashboard",
                key="nav_dashboard"
            ):

                st.session_state.section = "Dashboard"
                st.rerun()


st.divider()


if not model_ready:

    st.error(
        "VisionPlate model could not be loaded. "
        "Make sure plate_model.pt is available in the project folder."
    )

    with st.expander(
        "Model Error"
    ):

        st.code(
            model_error
        )


if st.session_state.section == "Home":

    left, right = st.columns(
        [1.65, 1],
        gap="large"
    )

    with left:

        st.markdown(
            "<div class='hero-title'>VISIONPLATE</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            "<div class='hero-tagline'>Smart Plate Recognition</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class='hero-description'>
            VisionPlate is an intelligent Automatic Number Plate Recognition
            system that detects vehicle number plates and extracts plate
            information using YOLO, EasyOCR, OpenCV and Python.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            "<div class='small-gap'></div>",
            unsafe_allow_html=True
        )

        if st.button(
            "🚀 Start Now",
            key="home_start"
        ):

            st.session_state.section = "Dashboard"
            st.rerun()

    with right:

        with st.container(
            border=True
        ):

            st.markdown(
                "<div style='text-align:center;font-size:100px;padding:20px 0;'>🚘</div>",
                unsafe_allow_html=True
            )

            st.markdown(
                "<div style='text-align:center;font-size:19px;font-weight:700;color:#eaf7ff;'>AI Vehicle Recognition</div>",
                unsafe_allow_html=True
            )

            st.markdown(
                "<div style='text-align:center;color:#8296ad;font-size:13px;padding:8px 20px;'>Detect &nbsp;•&nbsp; Read &nbsp;•&nbsp; Validate &nbsp;•&nbsp; Analyze</div>",
                unsafe_allow_html=True
            )

    st.markdown(
        "<div style='height:25px'></div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='section-kicker'>WHY VISIONPLATE</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='section-title'>Intelligent Plate Recognition</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='section-description'>A simple interface for fast and automated vehicle plate detection.</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='small-gap'></div>",
        unsafe_allow_html=True
    )

    f1, f2, f3 = st.columns(
        3,
        gap="large"
    )

    with f1:

        with st.container(
            border=True
        ):

            st.markdown(
                "### 🎯 Accurate Detection"
            )

            st.markdown(
                "<div class='card-text'>YOLO identifies vehicle number plate regions from images and captured frames.</div>",
                unsafe_allow_html=True
            )

    with f2:

        with st.container(
            border=True
        ):

            st.markdown(
                "### 🔤 OCR Recognition"
            )

            st.markdown(
                "<div class='card-text'>EasyOCR extracts readable characters from detected number plates.</div>",
                unsafe_allow_html=True
            )

    with f3:

        with st.container(
            border=True
        ):

            st.markdown(
                "### ⚡Fast Processing"
            )

            st.markdown(
                "<div class='card-text'>OpenCV preprocessing improves image quality before OCR recognition.</div>",
                unsafe_allow_html=True
            )


elif st.session_state.section == "About":

    st.title(
        "About VisionPlate"
    )

    st.caption(
        "Smart Plate Recognition using computer vision and OCR."
    )

    st.markdown(
        "<div style='height:10px'></div>",
        unsafe_allow_html=True
    )

    with st.container(
        border=True
    ):

        st.markdown(
            "### What is VisionPlate?"
        )

        st.markdown(
            """
            <div class='about-intro'>
            VisionPlate is an Automatic Number Plate Recognition system
            designed mainly for Indian vehicle registration plates. It
            detects number plate regions, extracts characters using OCR,
            and checks the recognized text against common Indian plate
            formats.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        "### ✨ Key Features"
    )

    with st.container(
        border=True
    ):

        feature_columns = st.columns(
            2,
            gap="large"
        )

        features = [
            (
                "Automatic Plate Detection",
                "Detects number plate regions using YOLO."
            ),
            (
                "OCR Recognition",
                "Extracts characters using EasyOCR with an alphanumeric allowlist."
            ),
            (
                "Image Processing",
                "Uses OpenCV preprocessing and image enhancement before OCR."
            ),
            (
                "Video Processing",
                "Supports vehicle plate detection from uploaded videos."
            ),
            (
                "Camera Recognition",
                "Captures vehicle images through the device camera."
            ),
            (
                "Indian Plate Validation",
                "Checks recognized text against common Indian registration plate formats."
            )
        ]

        for index, (
            title,
            description
        ) in enumerate(features):

            with feature_columns[index % 2]:

                st.markdown(
                    f"### {title}"
                )

                st.markdown(
                    f"<div class='card-text'>{description}</div>",
                    unsafe_allow_html=True
                )

    st.markdown(
        "### 🛠️ Technology Stack"
    )

    with st.container(
        border=True
    ):

        technologies = [
            (
                "Python",
                "Core programming and application logic"
            ),
            (
                "YOLO",
                "Number plate object detection"
            ),
            (
                "EasyOCR",
                "Optical character recognition"
            ),
            (
                "OpenCV",
                "Image processing and enhancement"
            ),
            (
                "Streamlit",
                "Interactive web interface"
            ),
            (
                "PyTorch",
                "Deep learning model execution"
            )
        ]

        tech_columns = st.columns(
            3,
            gap="large"
        )

        for index, (
            name,
            description
        ) in enumerate(technologies):

            with tech_columns[index % 3]:

                st.markdown(
                    f"<div class='tech-name'>{name}</div>",
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"<div class='tech-desc'>{description}</div>",
                    unsafe_allow_html=True
                )


elif st.session_state.section == "Dashboard":

    st.title(
        "VisionPlate Dashboard"
    )

    st.caption(
        "Upload an image, process a video, or capture a vehicle plate using your device camera."
    )

    if not model_ready:

        st.warning(
            "The detection model is not available. Add plate_model.pt to continue."
        )

    tab_image, tab_video, tab_camera = st.tabs(
        [
            "🖼️ Image",
            "🎥 Video",
            "📷 Camera"
        ]
    )

    with tab_image:

        with st.container(
            border=True
        ):

            st.markdown(
                "### Upload Vehicle Image"
            )

            st.caption(
                "Supported formats: JPG, JPEG, PNG"
            )

            image_file = st.file_uploader(
                "Choose an image",
                type=[
                    "jpg",
                    "jpeg",
                    "png"
                ],
                key=f"image_uploader_{st.session_state.image_key}",
                label_visibility="collapsed"
            )

        if image_file is not None:

            file_bytes = image_file.getvalue()

            current_signature = hashlib.md5(
                file_bytes
            ).hexdigest()

            if (
                st.session_state.image_signature is not None
                and
                st.session_state.image_signature != current_signature
            ):

                st.session_state.image_result = None
                st.session_state.image_detections = []

            st.session_state.image_signature = current_signature

            original_image = Image.open(
                image_file
            ).convert("RGB")

            st.markdown(
                "<div style='height:8px'></div>",
                unsafe_allow_html=True
            )

            input_col, output_col = st.columns(
                2,
                gap="large"
            )

            with input_col:

                with st.container(
                    border=True
                ):

                    st.markdown(
                        "### Input Image"
                    )

                    st.image(
                        original_image,
                        use_container_width=True
                    )

            with output_col:

                with st.container(
                    border=True
                ):

                    st.markdown(
                        "### Detection Output"
                    )

                    if st.session_state.image_result is not None:

                        st.image(
                            st.session_state.image_result,
                            use_container_width=True
                        )

                    else:

                        st.info(
                            "Click Process Image to detect the number plate."
                        )

            st.markdown(
                "<div style='height:8px'></div>",
                unsafe_allow_html=True
            )

            action1, action2 = st.columns(
                2,
                gap="large"
            )

            with action1:

                if st.button(
                    "🔍 Process Image",
                    key="process_image",
                    disabled=not model_ready
                ):

                    with st.spinner(
                        "Detecting and reading number plate..."
                    ):

                        result_image, detections = process_frame(
                            original_image
                        )

                    st.session_state.image_result = result_image
                    st.session_state.image_detections = detections

                    st.rerun()

            with action2:

                if st.button(
                    "↻ Reset",
                    key="reset_image"
                ):

                    reset_image()
                    st.rerun()

            if st.session_state.image_detections:

                st.markdown(
                    "<div style='height:8px'></div>",
                    unsafe_allow_html=True
                )

                show_detection_results(
                    st.session_state.image_detections
                )

    with tab_video:

        with st.container(
            border=True
        ):

            st.markdown(
                "### Upload Video"
            )

            st.caption(
                "Upload a video containing vehicles for plate detection."
            )

            video_file = st.file_uploader(
                "Choose a video",
                type=[
                    "mp4",
                    "avi",
                    "mov",
                    "mkv"
                ],
                key=f"video_uploader_{st.session_state.video_key}",
                label_visibility="collapsed"
            )

        if video_file is not None:

            video_bytes = video_file.getvalue()

            with st.container(
                border=True
            ):

                st.markdown(
                    "### Video Preview"
                )

                st.video(
                    video_bytes
                )

            if st.button(
                "↻ Reset Video",
                key="reset_video"
            ):

                st.session_state.video_key += 1
                st.rerun()

            st.info(
                "Video preview is available. Frame-by-frame automatic processing "
                "can be connected to the YOLO pipeline when required."
            )

    with tab_camera:

        if not st.session_state.camera_enabled:

            with st.container(
                border=True
            ):

                st.markdown(
                    "### Camera Recognition"
                )

                st.markdown(
                    "<div class='camera-note'>Your camera remains inactive until you click the button below.</div>",
                    unsafe_allow_html=True
                )

                st.markdown(
                    "<div style='height:10px'></div>",
                    unsafe_allow_html=True
                )

                if st.button(
                    "📷 Open Camera",
                    key="open_camera"
                ):

                    st.session_state.camera_enabled = True
                    st.rerun()

        else:

            with st.container(
                border=True
            ):

                st.markdown(
                    "### Camera Capture"
                )

                st.caption(
                    "Capture a vehicle image using your device camera."
                )

                captured_image = st.camera_input(
                    "Capture Image",
                    key=f"camera_capture_{st.session_state.camera_key}",
                    label_visibility="collapsed"
                )

                camera_actions_1, camera_actions_2 = st.columns(
                    2,
                    gap="large"
                )

                with camera_actions_1:

                    if st.button(
                        "✕ Close Camera",
                        key="close_camera"
                    ):

                        close_camera()
                        st.rerun()

                with camera_actions_2:

                    if st.button(
                        "↻ Reset Camera",
                        key="reset_camera"
                    ):

                        reset_camera()
                        st.rerun()

            if captured_image is not None:

                camera_image = Image.open(
                    captured_image
                ).convert("RGB")

                input_col, output_col = st.columns(
                    2,
                    gap="large"
                )

                with input_col:

                    with st.container(
                        border=True
                    ):

                        st.markdown(
                            "### Captured Image"
                        )

                        st.image(
                            camera_image,
                            use_container_width=True
                        )

                with output_col:

                    with st.container(
                        border=True
                    ):

                        st.markdown(
                            "### Detection Output"
                        )

                        if st.session_state.camera_result is not None:

                            st.image(
                                st.session_state.camera_result,
                                use_container_width=True
                            )

                        else:

                            st.info(
                                "Click Process Camera to detect the plate."
                            )

                st.markdown(
                    "<div style='height:8px'></div>",
                    unsafe_allow_html=True
                )

                if st.button(
                    "🔍 Process Camera",
                    key="process_camera",
                    disabled=not model_ready
                ):

                    with st.spinner(
                        "Processing captured image..."
                    ):

                        result_image, detections = process_frame(
                            camera_image
                        )

                    st.session_state.camera_result = result_image
                    st.session_state.camera_detections = detections

                    st.rerun()

                if st.session_state.camera_detections:

                    st.markdown(
                        "<div style='height:8px'></div>",
                        unsafe_allow_html=True
                    )

                    show_detection_results(
                        st.session_state.camera_detections
                    )


st.divider()

with st.container():

    footer_1, footer_2, footer_3 = st.columns(
        [1.4, 1, 1],
        gap="large"
    )

    with footer_1:

        st.markdown(
            "<div class='footer-brand'>🚘 VisionPlate</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            "<div class='footer-tagline'>Smart Plate Recognition</div>",
            unsafe_allow_html=True
        )

    with footer_2:

        st.markdown(
            "<div class='footer-text'>© 2026 VisionPlate<br>All rights reserved</div>",
            unsafe_allow_html=True
        )

    with footer_3:

        st.markdown(
            "<div class='footer-text'>+91 XXXXX XXXXX<br>visionplate@mail.com</div>",
            unsafe_allow_html=True
        )