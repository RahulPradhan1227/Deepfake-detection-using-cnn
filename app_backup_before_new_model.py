import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from facenet_pytorch import MTCNN
import numpy as np


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Deepfake Image Detection",
    page_icon="🔍",
    layout="centered"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, #dbeafe 0%, transparent 35%),
            radial-gradient(circle at bottom right, #e0e7ff 0%, transparent 35%),
            linear-gradient(135deg, #f8fafc, #eef2ff);
        color: #0f172a;
    }
    .block-container {
        max-width: 950px;
        padding-top: 2.5rem;
        padding-bottom: 2rem;
    }
    .main-title {
        text-align: center;
        font-size: 46px;
        font-weight: 800;
        letter-spacing: -1px;
        color: #0f172a;
        margin-bottom: 8px;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        font-weight: 450;
        color: #475569;
        margin-bottom: 30px;
    }
    .info-card {
        background: rgba(255, 255, 255, 0.88);
        padding: 24px;
        border-radius: 18px;
        margin-bottom: 25px;
        border: 1px solid #dbeafe;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
        backdrop-filter: blur(10px);
    }
    .section-title {
        font-size: 23px;
        font-weight: 750;
        color: #1e3a8a;
        margin-top: 12px;
        margin-bottom: 14px;
    }
    .tag {
        display: inline-block;
        padding: 7px 14px;
        margin: 4px;
        border-radius: 999px;
        background: #dbeafe;
        color: #1e3a8a;
        border: 1px solid #bfdbfe;
        font-size: 13px;
        font-weight: 700;
        box-shadow: 0 2px 6px rgba(30, 64, 175, 0.08);
    }
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.88);
        border: 2px dashed #93c5fd;
        border-radius: 18px;
        padding: 12px;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
    }
    [data-testid="stFileUploader"] section {
        border: none;
    }
    [data-testid="stMetric"] {
        background: #ffffff;
        padding: 15px;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 5px 15px rgba(15, 23, 42, 0.06);
    }
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        font-weight: 700;
        border: 1px solid #2563eb;
        padding: 10px 18px;
    }
    [data-testid="stAlert"] {
        border-radius: 14px;
    }
    [data-testid="stProgressBar"] {
        height: 10px;
        border-radius: 999px;
        overflow: hidden;
    }
    [data-testid="stImage"] img {
        border-radius: 16px;
        border: 1px solid #dbeafe;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
    }
    h1, h2, h3, h4 {
        color: #0f172a;
    }
    p {
        color: #334155;
    }
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 13px;
        font-weight: 500;
        margin-top: 35px;
        padding: 20px;
        border-top: 1px solid #dbeafe;
    }
    @media (max-width: 700px) {
        .main-title { font-size: 34px; }
        .subtitle { font-size: 16px; }
        .section-title { font-size: 20px; }
        .info-card { padding: 18px; }
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PROJECT SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "deepfake_efficientnet_b0_final.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Threshold selected in the existing validation-threshold file.
CONFIDENCE_THRESHOLD = 0.008

# These figures are taken from results/final_test_evaluation.json
# for efficientnet_b0_tuned_final.pth (204 test images).
VALIDATION_ACCURACY = 99.07
TEST_ACCURACY = 99.02
ROC_AUC = 99.92


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.30),
        nn.Linear(model.classifier[1].in_features, 1)
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    return model


model = load_model()


# ============================================================
# LOAD MTCNN
# ============================================================

@st.cache_resource
def load_mtcnn():
    return MTCNN(
        image_size=224,
        margin=20,
        min_face_size=40,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        keep_all=True,
        post_process=False,
        device=DEVICE
    )


mtcnn = load_mtcnn()


# ============================================================
# IMAGE TRANSFORM
# ============================================================

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# FACE DETECTION
# ============================================================


def detect_largest_face(image):
    """Detect and extract the largest valid face using MTCNN."""

    try:
        boxes, probabilities = mtcnn.detect(image)

        if boxes is None:
            return None, None

        # Normalize the detection output to a numeric array.
        try:
            boxes = np.asarray(boxes, dtype=np.float64)
        except (TypeError, ValueError):
            return None, None

        if boxes.ndim != 2 or boxes.shape[1] != 4:
            return None, None

        valid_faces = []

        for index, box in enumerate(boxes):
            if not np.isfinite(box).all():
                continue

            x1, y1, x2, y2 = box
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

            if area > 0:
                valid_faces.append((area, index, box))

        if not valid_faces:
            return None, None

        # Choose the largest valid face.
        _, face_index, largest_box = max(
            valid_faces,
            key=lambda item: item[0]
        )

        # Extract only the selected face.
        selected_boxes = np.asarray(
            [largest_box],
            dtype=np.float64
        )

        faces = mtcnn.extract(
            image,
            selected_boxes,
            save_path=None
        )

        if faces is None:
            return None, None

        if torch.is_tensor(faces):
            if faces.ndim == 4:
                if faces.shape[0] == 0:
                    return None, None
                face = faces[0]
            elif faces.ndim == 3:
                face = faces
            else:
                return None, None
        elif isinstance(faces, (list, tuple)):
            if len(faces) == 0 or faces[0] is None:
                return None, None
            face = faces[0]
        else:
            return None, None

        if not torch.is_tensor(face):
            return None, None

        face = face.detach().cpu()

        if face.ndim != 3:
            return None, None

        # Convert channel-first RGB tensor to an image.
        if face.shape[0] in (1, 3):
            face = face.permute(1, 2, 0)

        face = face.numpy()

        if not np.issubdtype(face.dtype, np.number):
            return None, None

        if not np.isfinite(face).all():
            return None, None

        if face.max() <= 1.0:
            face = face * 255.0

        face = np.clip(face, 0, 255).astype(np.uint8)
        face_image = Image.fromarray(face).convert("RGB")

        face_confidence = None

        if probabilities is not None:
            try:
                probs = np.asarray(
                    probabilities,
                    dtype=np.float64
                ).reshape(-1)

                if face_index < len(probs):
                    probability = probs[face_index]

                    if np.isfinite(probability):
                        face_confidence = float(probability)
            except (TypeError, ValueError):
                pass

        return face_image, face_confidence

    except Exception as exc:
        raise RuntimeError(
            f"MTCNN face detection/extraction failed: {exc}"
        ) from exc
        
#=============================================================
# PREDICTION
# ============================================================

def predict_image(image):
    image_tensor = image_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image_tensor).reshape(-1)
        fake_probability = torch.sigmoid(output)[0].item()

    real_probability = 1.0 - fake_probability

    if fake_probability >= CONFIDENCE_THRESHOLD:
        prediction = "DEEPFAKE"
        confidence = fake_probability
    else:
        prediction = "REAL"
        confidence = real_probability

    return prediction, confidence, real_probability, fake_probability


# ============================================================
# HEADER AND DEVICE STATUS
# ============================================================

st.markdown(
    '<div class="main-title">🔍 Deepfake Image Detection</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">AI-powered face analysis using MTCNN and EfficientNet-B0</div>',
    unsafe_allow_html=True
)

if torch.cuda.is_available():
    st.success(f"CUDA enabled — {torch.cuda.get_device_name(0)}")
else:
    st.warning("CUDA not available — using CPU")


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown('<div class="info-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🤖 Model Information</div>', unsafe_allow_html=True)
st.markdown(
    '<span class="tag">PyTorch</span>'
    '<span class="tag">MTCNN</span>'
    '<span class="tag">EfficientNet-B0</span>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)
with col1:
    st.write("**Input Size:** 224 × 224")
     #st.write(f"**Threshold:** {CONFIDENCE_THRESHOLD:.2f}")
    st.write(f"**Fake Detection Threshold:** {CONFIDENCE_THRESHOLD:.3f}")
with col2:
    st.write(f"**Validation Accuracy:** {VALIDATION_ACCURACY:.2f}%")
    st.write(f"**Test Accuracy:** {TEST_ACCURACY:.2f}%")
    st.write(f"**ROC-AUC:** {ROC_AUC:.2f}%")

st.caption(
    "Metrics are from the latest saved evaluation report. "
    "Verify class mapping, preprocessing, and test-set separation "
    "before interpreting these results as real-world performance."
)
st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# IMAGE UPLOAD AND ANALYSIS
# ============================================================

st.markdown('<div class="section-title">📤 Upload an Image</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Choose a face image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Upload an image containing a clear, visible face."
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("Could not open this image. Please upload a valid JPG, PNG, or WEBP file.")
        st.stop()

    st.markdown('<div class="section-title">🖼️ Image Analysis</div>', unsafe_allow_html=True)

    with st.spinner("Detecting face using MTCNN..."):
        try:
            face_image, face_confidence = detect_largest_face(image)
        except Exception as exc:
            st.error(f"Face detection failed: {exc}")
            st.stop()

    if face_image is None:
        st.error("No face detected. Please upload an image containing a clear, visible face.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Detected Face")
            st.image(face_image, use_container_width=True)

        if face_confidence is not None:
            st.info(f"MTCNN face-detection confidence: {face_confidence * 100:.2f}%")

        with st.spinner("Analyzing image with EfficientNet-B0..."):
            try:
                prediction, confidence, real_probability, fake_probability = predict_image(face_image)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.stop()

        st.markdown('<div class="section-title">🧠 Prediction Result</div>', unsafe_allow_html=True)

        if prediction == "REAL":
            st.success("Prediction: REAL")
            st.write(
                f"Fake score: {fake_probability * 100:.2f}%. "
                f"This is below the detection threshold of "
                f"{CONFIDENCE_THRESHOLD * 100:.2f}%."
            )
        else:
            st.error("Prediction: DEEPFAKE")
            st.write(
                f"Fake score: {fake_probability * 100:.2f}%. "
                f"This meets or exceeds the detection threshold of "
                f"{CONFIDENCE_THRESHOLD * 100:.2f}%."
    )
        with col1:
            st.metric("REAL", f"{real_probability * 100:.2f}%")
            st.progress(float(np.clip(real_probability, 0.0, 1.0)))

        with col2:
            st.metric("DEEPFAKE", f"{fake_probability * 100:.2f}%")
            st.progress(float(np.clip(fake_probability, 0.0, 1.0)))

        st.warning(
            "These are model scores, not guaranteed probabilities of authenticity. "
            "Confirm the training label mapping before relying on predictions."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Deepfake Image Detection | MTCNN + EfficientNet-B0 + PyTorch
    </div>
    """,
    unsafe_allow_html=True
)
