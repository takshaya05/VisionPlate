from pathlib import Path
from collections import OrderedDict
import json
import time
import tempfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from models.detector import PlateDetector
from models.enhancement import ImageEnhancer
from models.ocr import PlateOCR
from models.pipeline import RecognitionPipeline
from utils.image_utils import load_image, image_to_bytes


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "plate_model.pt"
DEFAULT_CONFIDENCE = 0.25

st.set_page_config(
    page_title="VisionPlate",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #090b12;
    --surface: #111522;
    --surface2: #171c2c;
    --ink: #f4f6ff;
    --muted: #9aa3bb;
    --line: #252c40;
    --blue: #7185ff;
    --purple: #a17aff;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: radial-gradient(ellipse at 80% 0%, #202044 0%, #090b12 45%);
    color: var(--ink);
}

#MainMenu, footer, header {
    visibility: hidden;
}

.block-container {
    max-width: 1440px;
    padding: 1.2rem 4.5% 3rem;
}

[data-testid="stHeader"] {
    background: transparent;
}

.top-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 4px 0 18px;
}

.brand-symbol {
    width: 44px;
    height: 44px;
    border-radius: 14px;
    background: linear-gradient(135deg, #7185ff, #a17aff);
    display: flex;
    justify-content: center;
    align-items: center;
    color: white;
    font-size: 25px;
    box-shadow: 0 8px 25px #7185ff30;
}

.brand-name {
    font-family: Manrope, sans-serif;
    font-weight: 800;
    font-size: 23px;
    letter-spacing: -1px;
    color: #f4f6ff;
}

.brand-subtitle {
    color: #9aa3bb;
    font-size: 10px;
    letter-spacing: 1.6px;
    text-transform: uppercase;
    margin-top: -2px;
}

.hero {
    background: linear-gradient(125deg, #171d36, #24264b 55%, #48336d);
    border: 1px solid #ffffff12;
    border-radius: 28px;
    padding: 48px 38px;
    color: white;
    min-height: 320px;
    box-shadow: 0 24px 70px #0005;
}

.hero-label {
    display: inline-block;
    padding: 8px 13px;
    border: 1px solid #ffffff30;
    background: #ffffff0c;
    border-radius: 30px;
    color: #d9ddff;
    font-size: 11px;
    letter-spacing: 1px;
    margin-bottom: 22px;
}

.hero h1 {
    font-family: Manrope, sans-serif;
    font-size: clamp(35px, 4.2vw, 59px);
    line-height: 1.12;
    letter-spacing: -2.4px;
    font-weight: 800;
    color: white;
    margin: 0 0 20px;
}

.hero h1 span {
    color: #b6c0ff;
}

.hero p {
    color: #d1d6ed;
    font-size: 15px;
    line-height: 1.85;
}

.hero-art {
    min-height: 280px;
    display: flex;
    justify-content: center;
    align-items: center;
}

.orbit {
    width: 250px;
    height: 250px;
    border: 1px solid #ffffff25;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
}

.orbit:before {
    content: "";
    position: absolute;
    width: 190px;
    height: 190px;
    border: 1px dashed #ffffff35;
    border-radius: 50%;
}

.car-icon {
    font-size: 90px;
    filter: drop-shadow(0 15px 20px #0008);
    z-index: 2;
}

.floating-card {
    position: absolute;
    background: #f7f8ff;
    color: #171b30;
    box-shadow: 0 16px 40px #0005;
    border-radius: 15px;
    padding: 12px 15px;
    z-index: 3;
}

.floating-card small {
    display: block;
    font-size: 10px;
    color: #737b91;
    letter-spacing: 1px;
    margin-bottom: 5px;
}

.plate-number {
    font-family: Manrope, sans-serif;
    font-weight: 800;
    font-size: 17px;
    letter-spacing: 1.5px;
}

.plate-card {
    right: -12px;
    bottom: 12px;
}

.scan-card {
    top: 20px;
    left: -15px;
}

.section-heading {
    font-family: Manrope, sans-serif;
    font-size: 27px;
    letter-spacing: -.8px;
    font-weight: 800;
    color: var(--ink);
    margin: 0 0 8px;
}

.section-subheading {
    color: var(--muted);
    font-size: 14px;
    line-height: 1.8;
    margin-bottom: 20px;
}

.feature-card, .step-card, .metric-box, .page-banner {
    background: linear-gradient(145deg, #141929, #101421);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 17px;
    box-sizing: border-box;
}

.feature-card {
    height: 158px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    transition: transform .2s ease, border-color .2s ease;
}

.feature-card:hover {
    transform: translateY(-3px);
    border-color: #5868b8;
}

.feature-icon {
    width: 38px;
    height: 38px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 11px;
    background: #252b4c;
    color: #b8c2ff;
    font-size: 19px;
    margin-bottom: 9px;
}

.feature-card h3, .step-card h3 {
    font-family: Manrope, sans-serif;
    font-size: 14px;
    font-weight: 800;
    color: var(--ink);
    margin: 0 0 7px;
}

.feature-card p, .step-card p {
    font-size: 12px;
    line-height: 1.6;
    color: var(--muted);
    margin: 0;
}

.step-card {
    height: 140px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.step-number {
    color: #aab5ff;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1px;
    margin-bottom: 9px;
}

.page-banner {
    padding: 24px;
    margin-bottom: 22px;
}

.page-banner h1 {
    font-family: Manrope, sans-serif;
    font-size: 31px;
    letter-spacing: -1px;
    color: var(--ink);
    margin: 0 0 9px;
}

.page-banner p {
    color: var(--muted);
    line-height: 1.8;
    margin: 0;
    font-size: 14px;
}

.metric-box {
    min-height: 105px;
    padding: 18px;
}

.metric-label {
    color: var(--muted);
    font-size: 11px;
    margin-bottom: 8px;
    letter-spacing: .5px;
}

.metric-value {
    font-family: Manrope, sans-serif;
    font-size: 26px;
    font-weight: 800;
    color: var(--ink);
    overflow-wrap: anywhere;
}

.plate-result {
    background: #1a2038;
    border: 1px solid #303b66;
    border-radius: 14px;
    padding: 15px;
    margin: 8px 0;
}

.plate-result-number {
    font-family: Manrope, sans-serif;
    color: #c2caff;
    font-size: 23px;
    font-weight: 800;
    letter-spacing: 2px;
    overflow-wrap: anywhere;
}

.muted {
    color: var(--muted);
    font-size: 12px;
    line-height: 1.7;
}

div.stButton > button, div.stDownloadButton > button {
    border-radius: 12px;
    min-height: 44px;
    font-weight: 700;
    border: 1px solid #303850;
    background: #171c2c;
    color: #e8ebfa;
}

div.stButton > button[kind="primary"],
div.stDownloadButton > button[kind="primary"] {
    background: linear-gradient(120deg, #6176ff, #8c64ef);
    color: white;
    border: 0;
    box-shadow: 0 7px 18px #5267f229;
}

div.stButton > button:hover, div.stDownloadButton > button:hover {
    border-color: #7185ff;
    color: #aebaff;
}

div[data-testid="stFileUploader"] {
    background: #111522;
    border: 1px dashed #3b4562;
    border-radius: 16px;
    padding: 13px;
}

div[data-testid="stFileUploader"] section {
    background: transparent;
}

div[data-testid="stExpander"] {
    border: 1px solid var(--line);
    border-radius: 14px;
    background: #111522;
}

div[role="radiogroup"] {
    gap: 8px;
}

div[role="radiogroup"] label {
    background: #141929;
    border: 1px solid #303850;
    border-radius: 12px;
    padding: 8px 16px;
    color: #e8ebfa;
}

div[role="radiogroup"] label:has(input:checked) {
    border-color: #7185ff;
    background: #202746;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 7px;
    background: #111522;
    padding: 7px;
    border-radius: 13px;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 9px;
    color: #aab2ca;
}

.stTabs [aria-selected="true"] {
    background: #252d4a !important;
    color: #fff !important;
}

.stTextInput input, .stTextArea textarea,
.stSelectbox div[data-baseweb="select"],
.stNumberInput input {
    background: #111522 !important;
    color: #f4f6ff !important;
    border-color: #303850 !important;
}

.stCheckbox label, .stRadio label {
    color: #e2e6f5 !important;
}

[data-testid="stMetric"] {
    background: #141929;
    border: 1px solid #293149;
    border-radius: 14px;
    padding: 12px;
}

[data-testid="stDataFrame"], [data-testid="stTable"] {
    border: 1px solid var(--line);
    border-radius: 12px;
    overflow: hidden;
}

hr {
    border-color: var(--line);
}

small, .stCaption {
    color: var(--muted);
}

.footer-line {
    border-top: 1px solid var(--line);
    padding-top: 20px;
    margin-top: 35px;
    color: #78829c;
    font-size: 12px;
    text-align: center;
}

div[data-testid="stImage"] {
    margin-left: auto;
    margin-right: auto;
}

div[data-testid="stImage"] img {
    border-radius: 12px;
}

@media(max-width: 760px) {
    .block-container {
        padding: 1rem 1rem 2rem;
    }

    .hero {
        padding: 32px 24px;
        border-radius: 22px;
    }

    .hero-art {
        min-height: 220px;
        margin-top: 20px;
    }

    .orbit {
        width: 220px;
        height: 220px;
    }

    .car-icon {
        font-size: 75px;
    }

    .feature-card {
        height: auto;
        min-height: 145px;
    }

    .step-card {
        height: auto;
        min-height: 125px;
    }

    .section-heading {
        font-size: 23px;
    }
}
</style>
""", unsafe_allow_html=True)


def init_state():
    defaults = {
        "page": "Home",
        "last_result": None,
        "last_filename": None,
        "history": [],
        "processing_error": None,
        "image_upload_version": 0,
        "video_upload_version": 0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "main_navigation" not in st.session_state:
        st.session_state.main_navigation = st.session_state.page


def navigate_to(page):
    st.session_state.page = page
    st.session_state.main_navigation = page


def sync_navigation():
    st.session_state.page = st.session_state.main_navigation


@st.cache_resource(show_spinner="Loading recognition models...")
def load_pipeline():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    detector = PlateDetector(str(MODEL_PATH))
    enhancer = ImageEnhancer()
    ocr = PlateOCR()

    return RecognitionPipeline(
        detector=detector,
        enhancer=enhancer,
        ocr=ocr,
    )


def render_header():
    left, right = st.columns(
        [1.25, 2.2],
        vertical_alignment="center",
    )

    with left:
        st.markdown("""
        <div class="top-brand">
            <div class="brand-symbol">⌁</div>
            <div>
                <div class="brand-name">VISIONPLATE</div>
                <div class="brand-subtitle">Smart Plate Recognition</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.radio(
            "Main navigation",
            ["Home", "About", "Dashboard"],
            horizontal=True,
            label_visibility="collapsed",
            key="main_navigation",
            on_change=sync_navigation,
        )

    st.markdown(
        '<div style="height:1px;background:#252c40;margin-bottom:25px"></div>',
        unsafe_allow_html=True,
    )


def render_footer():
    st.markdown(
        '<div class="footer-line">VISIONPLATE · Smart Plate Recognition</div>',
        unsafe_allow_html=True,
    )


def feature_card(icon, title, description):
    st.markdown(
        f"""
        <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <h3>{title}</h3>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_home():
    left, right = st.columns(
        [1.15, .85],
        gap="large",
        vertical_alignment="center",
    )

    with left:
        st.markdown("""
        <div class="hero">
            <div class="hero-label">AI-POWERED RECOGNITION</div>
            <h1>Smart vehicle plate <span>recognition.</span></h1>
            <p>
                Detect number plates and recognize their characters
                from vehicle images and videos.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.button(
            "Explore Dashboard →",
            type="primary",
            use_container_width=True,
            key="home_explore_dashboard",
            on_click=navigate_to,
            args=("Dashboard",),
        )

    with right:
        st.markdown("""
        <div class="hero-art">
            <div class="orbit">
                <div class="car-icon">🚘</div>
                <div class="floating-card scan-card">
                    <small>Detection</small>
                    <strong>Ready to scan</strong>
                </div>
                <div class="floating-card plate-card">
                    <small>Plate preview</small>
                    <div class="plate-number">TS 09 XX 0000</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    render_footer()


def render_about():
    st.markdown("""
    <div class="page-banner">
        <h1>About VisionPlate</h1>
        <p>
            VisionPlate uses computer vision and optical character recognition
            (OCR) to detect vehicle number plates and recognize their characters
            from images and videos.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-heading">Features</div>'
        '<div class="section-subheading">'
        'Tools that help detect, enhance, and recognize vehicle number plates.'
        '</div>',
        unsafe_allow_html=True,
    )

    feature_columns = st.columns(3, gap="small")

    with feature_columns[0]:
        feature_card(
            "◎",
            "Plate detection",
            "A trained YOLO model identifies number plate regions in images "
            "and video frames.",
        )

    with feature_columns[1]:
        feature_card(
            "⌗",
            "Text recognition",
            "OCR extracts readable characters from detected number plates "
            "and provides recognition confidence.",
        )

    with feature_columns[2]:
        feature_card(
            "✦",
            "Image enhancement",
            "Optional image enhancement and night-vision processing help "
            "prepare images for recognition.",
        )

    st.markdown('<div style="height:25px"></div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-heading">How it works</div>'
        '<div class="section-subheading">'
        'Follow three simple steps to process a vehicle image or video.'
        '</div>',
        unsafe_allow_html=True,
    )

    steps = [
        (
            "01 / UPLOAD",
            "Choose a file",
            "Open the Dashboard and upload a supported vehicle image or video.",
        ),
        (
            "02 / ANALYZE",
            "Run recognition",
            "VisionPlate processes the input, detects plate regions, and "
            "attempts to recognize the characters.",
        ),
        (
            "03 / REVIEW",
            "View results",
            "Review detected plates, confidence values, and processed images "
            "or video. Download the available results and report.",
        ),
    ]

    step_columns = st.columns(3, gap="small")

    for column, step in zip(step_columns, steps):
        with column:
            st.markdown(
                f"""
                <div class="step-card">
                    <div class="step-number">{step[0]}</div>
                    <h3>{step[1]}</h3>
                    <p>{step[2]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div style="height:25px"></div>', unsafe_allow_html=True)
    render_footer()


def as_display_image(image, source_bgr=False):
    if image is None:
        return None

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    arr = np.asarray(image)

    if arr.ndim == 2:
        return arr

    if arr.ndim == 3 and arr.shape[2] == 4:
        if source_bgr:
            return cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
        return arr

    if arr.ndim == 3 and arr.shape[2] == 3 and source_bgr:
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    return arr


def to_download_bytes(image):
    if image is None:
        return None

    if isinstance(image, bytes):
        return image

    if isinstance(image, Image.Image):
        return image_to_bytes(image)

    arr = np.asarray(image)

    if arr.ndim == 2:
        success, encoded = cv2.imencode(".png", arr)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        success, encoded = cv2.imencode(
            ".png",
            cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA),
        )
    elif arr.ndim == 3 and arr.shape[2] == 3:
        success, encoded = cv2.imencode(".png", arr)
    else:
        return None

    return encoded.tobytes() if success else None


def safe_number(value, default=0.0):
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except (ValueError, TypeError):
        return default


def process_image(
    uploaded_file,
    pipeline,
    confidence=0.25,
    enhance=False,
    night_vision=False,
):
    if uploaded_file is None:
        raise ValueError("Please upload an image.")

    uploaded_file.seek(0)
    image = load_image(uploaded_file)

    if image is None:
        raise ValueError("The uploaded image could not be loaded.")

    started = time.perf_counter()

    result = pipeline.run(
        image,
        confidence=confidence,
        enhance=enhance,
        night_vision=night_vision,
    )

    result["processing_ms"] = round(
        (time.perf_counter() - started) * 1000,
        2,
    )
    result["source_name"] = uploaded_file.name
    result["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    result["mode"] = "image"

    return result


def frame_to_bgr(image, fallback):
    if image is None:
        return fallback.copy()

    if isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    arr = np.asarray(image)

    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    if arr.ndim == 3 and arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

    if arr.ndim == 3 and arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    return fallback.copy()


def process_video(
    uploaded_file,
    pipeline,
    confidence=0.25,
    enhance=False,
    night_vision=False,
):
    if uploaded_file is None:
        raise ValueError("Please upload a video.")

    suffix = Path(
        getattr(uploaded_file, "name", "")
    ).suffix.lower()

    allowed_extensions = {
        ".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"
    }

    if suffix not in allowed_extensions:
        suffix = ".mp4"

    video_bytes = uploaded_file.getvalue()

    if not video_bytes:
        raise ValueError("The uploaded video is empty.")

    started = time.perf_counter()
    progress = st.progress(0, text="Preparing video...")

    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / f"input{suffix}"
        output_path = Path(temp_dir) / "annotated_output.mp4"

        input_path.write_bytes(video_bytes)

        cap = cv2.VideoCapture(str(input_path))

        if not cap.isOpened():
            cap.release()
            progress.empty()
            raise ValueError(
                "Unable to open this video. Try exporting it "
                "as MP4 and uploading it again."
            )

        fps = safe_number(cap.get(cv2.CAP_PROP_FPS), 25.0)

        if fps <= 0:
            fps = 25.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width < 1 or height < 1:
            cap.release()
            progress.empty()
            raise ValueError("The video has invalid frame dimensions.")

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        if not writer.isOpened():
            cap.release()
            writer.release()
            progress.empty()
            raise RuntimeError(
                "Unable to initialize the MP4 video writer. "
                "Check your OpenCV video codec support."
            )

        plates_by_text = OrderedDict()
        frame_count = 0
        detections_count = 0

        try:
            while True:
                success, frame_bgr = cap.read()

                if not success:
                    break

                frame_rgb = cv2.cvtColor(
                    frame_bgr,
                    cv2.COLOR_BGR2RGB,
                )

                frame_result = pipeline.run(
                    frame_rgb,
                    confidence=confidence,
                    enhance=enhance,
                    night_vision=night_vision,
                )

                detections = frame_result.get("detections", []) or []
                plates = frame_result.get("plates", []) or []

                detections_count += len(detections)

                for plate in plates:
                    plate_text = str(
                        plate.get("text") or ""
                    ).strip()

                    if not plate_text:
                        continue

                    score = safe_number(
                        plate.get("ocr_confidence")
                    )

                    previous = plates_by_text.get(plate_text)

                    record = {
                        "text": plate_text,
                        "ocr_confidence": score,
                        "detection_confidence": safe_number(
                            plate.get("detection_confidence")
                        ),
                        "valid": plate.get("valid"),
                        "validation_message": plate.get(
                            "validation_message", ""
                        ),
                        "bbox": plate.get("bbox"),
                        "ocr_engine": plate.get("ocr_engine"),
                    }

                    if (
                        previous is None
                        or score > previous["ocr_confidence"]
                    ):
                        plates_by_text[plate_text] = record

                annotated = frame_result.get("annotated")
                output_bgr = frame_to_bgr(annotated, frame_bgr)

                if output_bgr.shape[:2] != (height, width):
                    output_bgr = cv2.resize(
                        output_bgr,
                        (width, height),
                    )

                if output_bgr.ndim != 3 or output_bgr.shape[2] != 3:
                    raise RuntimeError(
                        "The pipeline returned an unsupported video frame."
                    )

                writer.write(output_bgr)
                frame_count += 1

                if total_frames > 0:
                    fraction = min(
                        frame_count / total_frames,
                        1.0,
                    )
                    progress.progress(
                        fraction,
                        text=(
                            f"Processing frame {frame_count} "
                            f"of {total_frames}..."
                        ),
                    )
                elif frame_count % 10 == 0:
                    progress.progress(
                        min(frame_count / (frame_count + 50), 0.95),
                        text=f"Processed {frame_count} frames...",
                    )

        finally:
            cap.release()
            writer.release()

        if frame_count == 0:
            progress.empty()
            raise ValueError(
                "No readable frames were found in this video."
            )

        if (
            not output_path.is_file()
            or output_path.stat().st_size == 0
        ):
            progress.empty()
            raise RuntimeError(
                "Video processing finished, but no output video "
                "was generated."
            )

        processed_video = output_path.read_bytes()

    progress.progress(1.0, text="Video processing complete.")
    progress.empty()

    return {
        "mode": "video",
        "source_name": getattr(
            uploaded_file,
            "name",
            "uploaded_video",
        ),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "processing_ms": round(
            (time.perf_counter() - started) * 1000,
            2,
        ),
        "video_bytes": processed_video,
        "frame_count": frame_count,
        "detections_count": detections_count,
        "plates": list(plates_by_text.values()),
    }


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_plate_results(plates):
    if not plates:
        st.info("No plate text was recognized.")
        return

    for index, plate in enumerate(plates, start=1):
        plate_text = str(
            plate.get("text") or "Text not recognized"
        )
        ocr_confidence = safe_number(
            plate.get("ocr_confidence")
        )

        with st.container(border=True):
            left, right = st.columns(
                [1.5, 1],
                vertical_alignment="center",
            )

            with left:
                st.markdown(
                    f"""
                    <div class="plate-result">
                        <div class="muted">PLATE {index}</div>
                        <div class="plate-result-number">{plate_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if plate.get("validation_message"):
                    st.caption(
                        str(plate["validation_message"])
                    )

            with right:
                st.metric(
                    "OCR confidence",
                    f"{ocr_confidence:.1%}",
                )

                if plate.get("valid") is True:
                    st.success("Format matched")
                elif plate.get("valid") is False:
                    st.warning("Format not matched")

            crop = plate.get("crop")

            if crop is not None:
                with st.expander(f"View plate crop {index}"):
                    st.image(
                        as_display_image(crop),
                        caption=f"Plate {index}",
                        width=280,
                    )


def build_report(result):
    plates = []

    for plate in result.get("plates", []):
        plates.append({
            "text": str(plate.get("text", "")),
            "detection_confidence": safe_number(
                plate.get("detection_confidence")
            ),
            "ocr_confidence": safe_number(
                plate.get("ocr_confidence")
            ),
            "valid": plate.get("valid"),
            "validation_message": str(
                plate.get("validation_message", "")
            ),
            "bbox": plate.get("bbox"),
            "ocr_engine": plate.get("ocr_engine"),
        })

    report = {
        "application": "VisionPlate",
        "mode": result.get("mode", "image"),
        "source_file": result.get("source_name"),
        "timestamp": result.get("timestamp"),
        "processing_time_ms": result.get("processing_ms"),
        "detected_plate_count": len(plates),
        "plates": plates,
    }

    if result.get("mode") == "video":
        report["processed_frames"] = result.get("frame_count")
        report["total_detections"] = result.get(
            "detections_count"
        )
    else:
        report["enhancement_applied"] = result.get(
            "enhancement_applied"
        )
        report["night_vision_applied"] = result.get(
            "night_vision_applied"
        )

    return json.dumps(
        report,
        indent=2,
        default=str,
    ).encode("utf-8")


def render_image_results(result):
    plates = result.get("plates", [])
    detections = result.get("detections", [])

    st.markdown(
        '<div class="section-heading">Recognition results</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        metric_card("PLATES RECOGNIZED", len(plates))

    with c2:
        metric_card("DETECTIONS", len(detections))

    with c3:
        metric_card(
            "PROCESSING TIME",
            f'{safe_number(result.get("processing_ms")):.0f} ms',
        )

    st.markdown(
        '<div style="height:20px"></div>',
        unsafe_allow_html=True,
    )

    images = [
        (result.get("annotated"), "Annotated image"),
        (result.get("original"), "Original image"),
        (result.get("enhanced"), "Enhanced image"),
    ]

    tabs = st.tabs([item[1] for item in images])

    for tab, (image, caption) in zip(tabs, images):
        with tab:
            if image is not None:
                _, image_col, _ = st.columns(
                    [0.5, 2, 0.5],
                    gap="small",
                )

                with image_col:
                    st.image(
                        as_display_image(image),
                        caption=caption,
                        use_container_width=True,
                    )
            else:
                st.info(f"{caption} unavailable.")

    st.markdown(
        '<div style="height:20px"></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-heading">Recognized plates</div>',
        unsafe_allow_html=True,
    )

    render_plate_results(plates)

    annotated_bytes = to_download_bytes(
        result.get("annotated")
    )

    col1, col2 = st.columns(2)

    with col1:
        if annotated_bytes:
            st.download_button(
                "Download annotated image",
                data=annotated_bytes,
                file_name="visionplate_annotated.png",
                mime="image/png",
                use_container_width=True,
                key="download_annotated_image",
            )

    with col2:
        st.download_button(
            "Download recognition report",
            data=build_report(result),
            file_name="visionplate_report.json",
            mime="application/json",
            use_container_width=True,
            key="download_image_report",
        )


def render_video_results(result):
    st.markdown(
        '<div class="section-heading">Video results</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        metric_card(
            "FRAMES PROCESSED",
            result.get("frame_count", 0),
        )

    with c2:
        metric_card(
            "DETECTIONS",
            result.get("detections_count", 0),
        )

    with c3:
        metric_card(
            "PROCESSING TIME",
            f'{safe_number(result.get("processing_ms")):.0f} ms',
        )

    _, video_col, _ = st.columns(
        [0.5, 2, 0.5],
        gap="small",
    )

    with video_col:
        st.video(result["video_bytes"])

    st.download_button(
        "Download processed video",
        data=result["video_bytes"],
        file_name="visionplate_processed.mp4",
        mime="video/mp4",
        use_container_width=True,
        key="download_processed_video",
    )

    st.markdown(
        '<div style="height:20px"></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-heading">Recognized plates</div>',
        unsafe_allow_html=True,
    )

    render_plate_results(result.get("plates", []))

    st.download_button(
        "Download recognition report",
        data=build_report(result),
        file_name="visionplate_video_report.json",
        mime="application/json",
        use_container_width=True,
        key="download_video_report",
    )


def render_results(result):
    if not result:
        return

    if result.get("mode") == "video":
        render_video_results(result)
    else:
        render_image_results(result)


def reset_current_upload(upload_type):
    if upload_type == "Image":
        st.session_state.image_upload_version += 1
    else:
        st.session_state.video_upload_version += 1

    st.session_state.last_result = None
    st.session_state.last_filename = None
    st.session_state.processing_error = None

    st.rerun()


def render_dashboard():
    st.markdown("""
    <div class="page-banner">
        <h1>Recognition Dashboard</h1>
        <p>Upload an image or video to recognize number plates.</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        pipeline = load_pipeline()
    except Exception as exc:
        st.error("Could not load the recognition models.")
        st.exception(exc)
        return

    upload_col, settings_col = st.columns(
        [1.35, .85],
        gap="large",
    )

    with upload_col:
        st.markdown("### Upload file")

        upload_type = st.radio(
            "Choose input type",
            ["Image", "Video"],
            horizontal=True,
            key="dashboard_upload_type",
        )

        if upload_type == "Image":
            upload_key = (
                f"dashboard_image_upload_"
                f"{st.session_state.image_upload_version}"
            )

            uploaded_file = st.file_uploader(
                "Choose an image",
                type=["jpg", "jpeg", "png", "webp", "bmp"],
                help="Supported formats: JPG, PNG, WEBP, BMP.",
                key=upload_key,
            )
        else:
            upload_key = (
                f"dashboard_video_upload_"
                f"{st.session_state.video_upload_version}"
            )

            uploaded_file = st.file_uploader(
                "Choose a video",
                type=["mp4", "avi", "mov", "mkv", "m4v", "webm"],
                help="Supported formats depend on your OpenCV installation.",
                key=upload_key,
            )

        if uploaded_file is not None and upload_type == "Image":
            try:
                uploaded_file.seek(0)
                preview = Image.open(uploaded_file).convert("RGB")

                _, preview_col, _ = st.columns(
                    [0.5, 2, 0.5],
                    gap="small",
                )

                with preview_col:
                    st.image(
                        preview,
                        caption=uploaded_file.name,
                        use_container_width=True,
                    )

            except Exception:
                st.error("Could not preview the uploaded image.")

        elif uploaded_file is not None and upload_type == "Video":
            _, preview_col, _ = st.columns(
                [0.5, 2, 0.5],
                gap="small",
            )

            with preview_col:
                st.video(uploaded_file.getvalue())

    with settings_col:
        st.markdown("### Recognition settings")

        enhance = st.checkbox(
            "Image enhancement",
            value=False,
            key="dashboard_enhance",
        )

        night_vision = st.checkbox(
            "Night-vision enhancement",
            value=False,
            key="dashboard_night_vision",
        )

        button_col1, button_col2 = st.columns(2)

        with button_col1:
            analyze_clicked = st.button(
                "Analyze Image" if upload_type == "Image"
                else "Analyze Video",
                type="primary",
                use_container_width=True,
                disabled=uploaded_file is None,
                key="dashboard_analyze",
            )

        with button_col2:
            reset_clicked = st.button(
                "Reset Image" if upload_type == "Image"
                else "Reset Video",
                use_container_width=True,
                key="dashboard_reset",
            )

    if reset_clicked:
        reset_current_upload(upload_type)

    if analyze_clicked and uploaded_file is not None:
        try:
            with st.spinner("Analyzing..."):
                if upload_type == "Image":
                    result = process_image(
                        uploaded_file=uploaded_file,
                        pipeline=pipeline,
                        confidence=DEFAULT_CONFIDENCE,
                        enhance=enhance,
                        night_vision=night_vision,
                    )
                else:
                    result = process_video(
                        uploaded_file=uploaded_file,
                        pipeline=pipeline,
                        confidence=DEFAULT_CONFIDENCE,
                        enhance=enhance,
                        night_vision=night_vision,
                    )

            st.session_state.last_result = result
            st.session_state.last_filename = uploaded_file.name
            st.session_state.processing_error = None

            plate_count = len(result.get("plates", []))

            if result.get("mode") == "video":
                detection_count = result.get("detections_count", 0)
            else:
                detection_count = len(result.get("detections", []))

            st.session_state.history.append({
                "File": uploaded_file.name,
                "Type": upload_type,
                "Plates": plate_count,
                "Detections": detection_count,
                "Time (ms)": result.get("processing_ms", 0),
                "Timestamp": result.get("timestamp"),
            })

            st.rerun()

        except Exception as exc:
            st.session_state.processing_error = (
                f"{type(exc).__name__}: {exc}"
            )
            st.error("File processing failed.")
            st.exception(exc)

    if st.session_state.processing_error:
        st.error(st.session_state.processing_error)

    st.markdown(
        '<div style="height:28px"></div>',
        unsafe_allow_html=True,
    )

    if st.session_state.last_result:
        render_results(st.session_state.last_result)
    else:
        st.caption("Your recognition results will appear here.")

    if st.session_state.history:
        st.markdown(
            '<div style="height:32px"></div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-heading">Recent activity</div>',
            unsafe_allow_html=True,
        )

        st.dataframe(
            st.session_state.history[::-1],
            use_container_width=True,
            hide_index=True,
        )

        if st.button("Clear activity", key="clear_session_activity"):
            st.session_state.history = []
            st.rerun()

    render_footer()


def main():
    init_state()
    render_header()

    page = st.session_state.page

    if page == "Home":
        render_home()
    elif page == "About":
        render_about()
    elif page == "Dashboard":
        render_dashboard()
    else:
        navigate_to("Home")
        st.rerun()


if __name__ == "__main__":
    main()
