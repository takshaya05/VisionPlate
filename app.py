import warnings

warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*quantize_per_tensor.*deprecated.*")

import streamlit as st
from PIL import Image
import numpy as np

from modules.detector import detect_plates
from modules.ocr import read_plate
from modules.processor import process_image


st.set_page_config(
    page_title="VisionPlate | Smart Plate Recognition",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed"
)


if "results" not in st.session_state:
    st.session_state.results = None

if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html {
        scroll-behavior: smooth;
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 5%, rgba(56,189,248,.09), transparent 24%),
            radial-gradient(circle at 92% 12%, rgba(37,99,235,.08), transparent 25%),
            #06101d;
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1rem;
        padding-bottom: 0;
    }

    [data-testid="stHeader"] {
        background: rgba(6,16,29,.92);
    }

    [data-testid="stSidebar"] {
        display: none;
    }

    .brand {
        font-size: 24px;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -.5px;
    }

    .brand span {
        color: #38bdf8;
        text-shadow: 0 0 18px rgba(56,189,248,.35);
    }

    .tagline {
        color: #7f91a8;
        font-size: 11px;
        margin-top: -4px;
    }

    .nav-link {
        display: block;
        text-align: center;
        padding: 9px 12px;
        color: #cbd5e1 !important;
        text-decoration: none !important;
        font-weight: 600;
        border-radius: 10px;
        transition: .2s ease;
    }

    .nav-link:hover {
        color: #38bdf8 !important;
        background: rgba(56,189,248,.07);
    }

    .section-label {
        color: #38bdf8;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    .hero-title {
        font-size: clamp(40px, 5vw, 60px);
        line-height: 1.02;
        font-weight: 800;
        letter-spacing: -2.5px;
        color: #f8fafc;
        margin: 10px 0 18px;
    }

    .hero-title span {
        color: #38bdf8;
        text-shadow: 0 0 28px rgba(56,189,248,.30);
    }

    .hero-text {
        color: #94a3b8;
        line-height: 1.7;
        max-width: 620px;
        font-size: 15px;
    }

    .logo-box {
        min-height: 220px;
        max-width: 300px;
        margin: auto;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 20px;
        background:
            radial-gradient(circle, rgba(56,189,248,.14), transparent 48%),
            linear-gradient(145deg, rgba(14,35,57,.92), rgba(5,16,29,.90));
        border: 1px solid rgba(125,211,252,.16);
        box-shadow:
            0 18px 45px rgba(0,0,0,.22),
            0 0 35px rgba(56,189,248,.06),
            inset 0 1px 0 rgba(255,255,255,.05);
    }

    .logo-icon {
        font-size: 58px;
        filter:
            drop-shadow(0 0 10px rgba(56,189,248,.65))
            drop-shadow(0 0 24px rgba(56,189,248,.22));
    }

    .feature-card {
        min-height: 175px;
        padding: 22px;
        border-radius: 18px;
        background: linear-gradient(
            145deg,
            rgba(15,35,57,.78),
            rgba(7,20,34,.82)
        );
        border: 1px solid rgba(148,163,184,.10);
        box-shadow:
            0 12px 30px rgba(0,0,0,.12),
            inset 0 1px 0 rgba(255,255,255,.04);
        transition: .25s ease;
    }

    .feature-card:hover {
        border-color: rgba(56,189,248,.30);
        box-shadow:
            0 15px 35px rgba(0,0,0,.18),
            0 0 25px rgba(56,189,248,.07);
        transform: translateY(-3px);
    }

    .feature-icon {
        font-size: 27px;
        margin-bottom: 12px;
    }

    .feature-title {
        color: #f8fafc;
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .feature-text {
        color: #8fa1b5;
        font-size: 13px;
        line-height: 1.65;
    }

    .plate-result {
        padding: 16px;
        border-radius: 15px;
        background: rgba(8,24,40,.75);
        border: 1px solid rgba(56,189,248,.13);
        text-align: center;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.04);
    }

    .plate-text {
        color: #f8fafc;
        font-size: 28px;
        font-weight: 800;
        letter-spacing: 3px;
        text-shadow: 0 0 18px rgba(56,189,248,.18);
    }

    .footer-line {
        margin-top: 70px;
        padding: 30px 0;
        border-top: 1px solid rgba(148,163,184,.10);
        color: #718198;
        font-size: 12px;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(
            145deg,
            rgba(15,35,57,.72),
            rgba(7,20,34,.80)
        );
        border: 1px solid rgba(148,163,184,.10);
        border-radius: 15px;
        padding: 13px;
        box-shadow:
            0 8px 25px rgba(0,0,0,.12),
            inset 0 1px 0 rgba(255,255,255,.04);
    }

    [data-testid="stFileUploader"] {
        border-radius: 15px;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        margin-bottom: 24px;
    }

    .stButton > button {
        border-radius: 10px;
        min-height: 43px;
        font-weight: 600;
        border: 1px solid rgba(148,163,184,.15);
        transition: .2s ease;
    }

    .stButton > button:hover {
        border-color: rgba(56,189,248,.35);
        box-shadow: 0 0 20px rgba(56,189,248,.08);
    }

    @media (max-width: 768px) {
        .hero-title {
            font-size: 42px;
        }

        .logo-box {
            min-height: 180px;
            max-width: 250px;
        }

        .logo-icon {
            font-size: 50px;
        }

        .feature-card {
            margin-bottom: 18px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


header_left, header_right = st.columns([3, 1])

with header_left:
    st.markdown(
        """
        <div class="brand">
            🚘 Vision<span>Plate</span>
        </div>
        <div class="tagline">
            Smart Plate Recognition
        </div>
        """,
        unsafe_allow_html=True
    )

with header_right:
    st.write("")


nav1, nav2, nav3 = st.columns(3)

with nav1:
    st.markdown(
        '<a href="#home" class="nav-link">⌂ Home</a>',
        unsafe_allow_html=True
    )

with nav2:
    st.markdown(
        '<a href="#about" class="nav-link">◎ About</a>',
        unsafe_allow_html=True
    )

with nav3:
    st.markdown(
        '<a href="#dashboard" class="nav-link">⌕ Dashboard</a>',
        unsafe_allow_html=True
    )


st.markdown('<div id="home"></div>', unsafe_allow_html=True)

st.write("")
st.write("")

home_left, home_right = st.columns(
    [1.45, 1],
    gap="large"
)

with home_left:

    with st.container(border=True):

        st.markdown(
            '<div class="section-label">AI-POWERED ANPR PLATFORM</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="hero-title">
                Smart <span>Plate</span><br>
                Recognition
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="hero-text">
                VisionPlate automatically detects vehicle number plates
                and recognizes their characters from images using
                YOLO, EasyOCR and computer vision.
            </div>
            """,
            unsafe_allow_html=True
        )


with home_right:

    st.markdown(
        """
        <div class="logo-box">
            <div class="logo-icon">🚘</div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.write("")
st.write("")
st.write("")


st.markdown('<div id="about"></div>', unsafe_allow_html=True)

st.markdown(
    '<div class="section-label">ABOUT VISIONPLATE</div>',
    unsafe_allow_html=True
)

st.title("Intelligent Number Plate Recognition")

st.write(
    "A computer vision application that combines object detection "
    "and optical character recognition to automate vehicle plate "
    "identification."
)

st.write("")
st.write("")


features = [
    (
        "🎯",
        "Plate Detection",
        "YOLO identifies number plate regions and generates accurate bounding boxes."
    ),
    (
        "🔤",
        "Plate Recognition",
        "EasyOCR extracts characters from the detected number plate."
    ),
    (
        "📊",
        "Confidence Scores",
        "Detection and OCR confidence values help evaluate recognition quality."
    ),
    (
        "🔎",
        "Multiple Plate Detection",
        "Multiple number plates can be processed from a single image."
    ),
    (
        "🖥️",
        "Interactive Dashboard",
        "Upload images, run recognition and inspect results through Streamlit."
    ),
    (
        "⚡",
        "Technology",
        "Built using Python, YOLO, EasyOCR, OpenCV, NumPy and Streamlit."
    )
]


for start in range(0, len(features), 3):

    cols = st.columns(3, gap="large")

    for col, feature in zip(
        cols,
        features[start:start + 3]
    ):

        with col:

            icon, title, description = feature

            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="feature-icon">{icon}</div>
                    <div class="feature-title">{title}</div>
                    <div class="feature-text">{description}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.write("")


st.write("")
st.write("")


st.markdown('<div id="dashboard"></div>', unsafe_allow_html=True)

st.markdown(
    '<div class="section-label">ANPR DASHBOARD</div>',
    unsafe_allow_html=True
)

st.title("Detect & Recognize")

st.write(
    "Upload a vehicle image, run the AI pipeline and inspect "
    "detected plates with recognition confidence."
)

st.write("")
st.write("")


with st.container(border=True):

    uploaded_file = st.file_uploader(
        "📤 Upload Vehicle Image",
        type=["jpg", "jpeg", "png"],
        key=f"vehicle_uploader_{st.session_state.uploader_key}",
        help="Supported formats: JPG, JPEG and PNG"
    )


st.write("")


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    image_array = np.array(image)

    if st.session_state.uploaded_name != uploaded_file.name:

        st.session_state.results = None
        st.session_state.uploaded_name = uploaded_file.name

    preview, controls = st.columns(
        [1.45, 1],
        gap="large"
    )

    with preview:

        with st.container(border=True):

            st.subheader("Image Preview")

            st.image(
                image,
                caption=uploaded_file.name,
                width="stretch"
            )

    with controls:

        with st.container(border=True):

            st.subheader("Recognition")

            st.write(
                "Run the detection and character recognition process."
            )

            st.write("")

            if st.button(
                "🔍 Detect & Recognize",
                type="primary",
                width="stretch"
            ):

                with st.status(
                    "Processing image...",
                    expanded=False
                ):

                    st.session_state.results = process_image(
                        image_array,
                        detect_plates,
                        read_plate
                    )

                st.rerun()

            if st.session_state.results is not None:

                st.write("")

                if st.button(
                    "🗑️ Clear Results",
                    width="stretch"
                ):

                    st.session_state.results = None
                    st.session_state.uploaded_name = None
                    st.session_state.uploader_key += 1
                    st.rerun()

else:

    st.info(
        "Upload a vehicle image above to start VisionPlate."
    )


results = st.session_state.results


if results is not None:

    st.write("")
    st.write("")

    st.subheader("Recognition Results")

    st.write("")

    if not results:

        st.warning(
            "No license plate was detected in this image."
        )

    else:

        recognized_count = sum(
            1
            for result in results
            if result.get("plate_text")
        )

        average_yolo = sum(
            float(
                result.get(
                    "detection_confidence",
                    0
                )
            )
            for result in results
        ) / len(results)

        average_ocr = sum(
            float(
                result.get(
                    "ocr_confidence",
                    0
                )
            )
            for result in results
        ) / len(results)


        m1, m2, m3, m4 = st.columns(
            4,
            gap="medium"
        )

        with m1:
            st.metric(
                "Plates Detected",
                len(results)
            )

        with m2:
            st.metric(
                "Recognized",
                recognized_count
            )

        with m3:
            st.metric(
                "Avg YOLO",
                f"{average_yolo:.1%}"
            )

        with m4:
            st.metric(
                "Avg OCR",
                f"{average_ocr:.1%}"
            )


        st.write("")
        st.write("")


        for index, result in enumerate(
            results,
            start=1
        ):

            with st.container(border=True):

                st.markdown(
                    f"### 🚘 Detection {index}"
                )

                st.write("")

                result_image, result_data = st.columns(
                    [1.35, 1],
                    gap="large"
                )

                with result_image:

                    st.image(
                        result["crop"],
                        caption="Detected Number Plate",
                        width="stretch"
                    )

                with result_data:

                    plate_text = (
                        result.get("plate_text")
                        or "Not recognized"
                    )

                    st.caption("RECOGNIZED PLATE")

                    st.markdown(
                        f"""
                        <div class="plate-result">
                            <div class="plate-text">
                                {plate_text}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.write("")

                    yolo_confidence = float(
                        result.get(
                            "detection_confidence",
                            0
                        )
                    )

                    ocr_confidence = float(
                        result.get(
                            "ocr_confidence",
                            0
                        )
                    )

                    st.write(
                        f"YOLO Confidence: **{yolo_confidence:.2%}**"
                    )

                    st.progress(
                        min(max(yolo_confidence, 0), 1)
                    )

                    st.write(
                        f"OCR Confidence: **{ocr_confidence:.2%}**"
                    )

                    st.progress(
                        min(max(ocr_confidence, 0), 1)
                    )

                    st.write("")

                    if result.get("plate_text"):

                        st.success(
                            "Plate characters recognized successfully."
                        )

                    else:

                        st.warning(
                            "Plate detected, but characters could not be recognized."
                        )

            st.write("")


st.write("")
st.write("")


st.markdown(
    '<div class="footer-line"></div>',
    unsafe_allow_html=True
)

footer_left, footer_center, footer_right = st.columns(
    [1.2, 2, 1.2],
    gap="large"
)

with footer_left:

    st.markdown(
        """
        **🚘 VisionPlate**

        Smart Plate Recognition
        """
    )

with footer_center:

    st.caption(
        "© 2026 VisionPlate | All rights reserved."
    )

with footer_right:

    st.markdown(
        """
        **Contact Us**

        +91 XXXXX XXXXX  
        visionplate@mail.com
        """
    )