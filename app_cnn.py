"""
Audiogram Severity Classifier - Deep Learning Models Only
CNN and InceptionResNetV2 transfer learning models
"""
import json
import base64
import os
import numpy as np
from pathlib import Path
from typing import Tuple
import urllib.request

import streamlit as st
from PIL import Image

# Paths
BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = Path(os.getenv("MODEL_DIR", str(BASE_DIR)))

def _model_path(relative_path: str) -> Path:
    """Always use MODEL_DIR for downloaded models."""
    return ARTIFACT_DIR / relative_path

def _ensure_model_file_exists(model_path: Path, github_url: str) -> Path:
    """Download model from GitHub if it doesn't exist or is an LFS pointer."""
    if model_path.exists() and model_path.stat().st_size > 1000:
        return model_path
    
    try:
        st.info(f"Downloading {model_path.name}... This may take 30-60 seconds.")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(github_url, str(model_path))
        return model_path
    except Exception as e:
        st.error(f"Failed to download {model_path.name}: {str(e)[:100]}")
        return None

# Model paths
MODEL_PATH = _model_path("audiogram_severity_model1.2.keras")
INCEPTIONRESNETV2_MODEL_PATH = _model_path("audiogram_severity_inceptionresnetv2.keras")
CLASS_MAP_PATH = _model_path("class_indices.json")
BACKGROUND_IMAGE_PATH = BASE_DIR / "background.jpg"
IMG_SIZE: Tuple[int, int] = (299, 299)
FREQS = [500, 1000, 2000, 3000, 4000, 6000, 8000]
DB_MIN = -10
DB_MAX = 110

# GitHub LFS CDN URLs
GITHUB_BASE = "https://media.githubusercontent.com/media/thinamawijesekara09/Audiogram-app/main"
CNN_MODEL_URL = f"{GITHUB_BASE}/audiogram_severity_model1.2.keras"
INCEPTION_MODEL_URL = f"{GITHUB_BASE}/audiogram_severity_inceptionresnetv2.keras"

st.set_page_config(page_title="Audiogram Classifier (Deep Learning)", page_icon="🎧", layout="centered")

# Background image
def get_background_image():
    if BACKGROUND_IMAGE_PATH.exists():
        with open(BACKGROUND_IMAGE_PATH, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return None

bg_image = get_background_image()

# Styling
if bg_image:
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        .stApp {{
            background-image: url("data:image/jpeg;base64,{bg_image}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            font-family: 'Inter', sans-serif;
        }}
        
        .stApp > header {{
            background-color: transparent;
            visibility: hidden;
            height: 0;
        }}
        
        h1 {{
            color: #1a1a1a !important;
            font-weight: 700 !important;
            text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.9);
            padding: 20px;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            margin-bottom: 10px;
        }}
        
        .stMarkdown p, .stMarkdown {{
            color: #2d2d2d !important;
            font-weight: 500;
            background-color: rgba(255, 255, 255, 0.95);
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        h2, h3 {{
            color: #1a1a1a !important;
            font-weight: 600 !important;
            padding: 10px;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }}
        
        .stFileUploader {{
            background-color: rgba(255, 255, 255, 0.95) !important;
            padding: 20px;
            border-radius: 15px;
            border: 2px dashed #667eea;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        }}
        
        .stFileUploader label {{
            color: #1a1a1a !important;
            font-weight: 600 !important;
            font-size: 1.1em;
        }}
        
        .stImage {{
            background-color: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        }}
        
        div[data-testid="stTable"] {{
            background-color: rgba(255, 255, 255, 0.95) !important;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        }}
        
        div[data-testid="stTable"] table {{
            color: #1a1a1a !important;
        }}
        
        div[data-testid="stTable"] th {{
            background-color: #667eea !important;
            color: white !important;
            font-weight: 600;
            padding: 12px;
        }}
        
        div[data-testid="stTable"] td {{
            color: #2d2d2d !important;
            padding: 10px;
            font-weight: 500;
        }}
        
        .stAlert {{
            background-color: rgba(255, 255, 255, 0.95) !important;
            color: #1a1a1a !important;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        </style>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        .stApp {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            font-family: 'Inter', sans-serif;
        }}
        .stApp > header {{
            background-color: transparent;
            visibility: hidden;
            height: 0;
        }}
        h1, h2, h3 {{
            color: #1a1a1a !important;
            font-weight: 600 !important;
        }}
        .stMarkdown, .stImage, .stFileUploader, div[data-testid="stTable"] {{
            background-color: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
            color: #2d2d2d !important;
        }}
        </style>
        """, unsafe_allow_html=True)

st.title("🎧 Audiogram Severity Classifier (Deep Learning)")

# Load class map
@st.cache_data(show_spinner=True)
def load_class_map():
    if not CLASS_MAP_PATH.exists():
        st.error(f"Class map file not found at {CLASS_MAP_PATH}")
        st.stop()
    with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    return idx_to_class

# CNN Model Loader
def load_cnn_classifier():
    """Lazy load CNN model."""
    try:
        from tensorflow.keras.models import load_model
        with st.spinner("Loading CNN model..."):
            model_path = _ensure_model_file_exists(MODEL_PATH, CNN_MODEL_URL)
            if model_path is None or not model_path.exists():
                st.error("CNN model unavailable")
                return None
            return load_model(model_path, compile=False, safe_mode=False)
    except Exception as e:
        st.error(f"CNN model error: {str(e)[:100]}")
        return None

# InceptionResNetV2 Model Loader
def load_inceptionresnetv2_classifier():
    """Lazy load InceptionResNetV2 model."""
    try:
        from tensorflow.keras.models import load_model
        with st.spinner("Loading InceptionResNetV2 model..."):
            model_path = _ensure_model_file_exists(INCEPTIONRESNETV2_MODEL_PATH, INCEPTION_MODEL_URL)
            if model_path is None or not model_path.exists():
                st.error("InceptionResNetV2 model unavailable")
                return None
            return load_model(model_path, compile=False, safe_mode=False)
    except Exception as e:
        st.error(f"InceptionResNetV2 model error: {str(e)[:100]}")
        return None

# Preprocessing
def preprocess_image(image: Image.Image) -> np.ndarray:
    from tensorflow.keras.utils import img_to_array
    from tensorflow.keras.applications.inception_v3 import preprocess_input
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr

def preprocess_image_inceptionresnetv2(image: Image.Image) -> np.ndarray:
    from tensorflow.keras.utils import img_to_array
    from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_inceptionresnetv2
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_inceptionresnetv2(arr)
    return arr

# Prediction functions
def predict_cnn(model, idx_to_class, image: Image.Image):
    """CNN prediction."""
    arr = preprocess_image(image)
    probs = model.predict(arr, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = idx_to_class[pred_idx]
    confidence = float(probs[pred_idx])
    return pred_label, confidence, probs

def predict_inceptionresnetv2(model, idx_to_class, image: Image.Image):
    """InceptionResNetV2 prediction."""
    arr = preprocess_image_inceptionresnetv2(image)
    probs = model.predict(arr, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = idx_to_class[pred_idx]
    confidence = float(probs[pred_idx])
    return pred_label, confidence, probs

def speech_banana_difficulties(severity_label):
    """Convert severity to speech impact."""
    severity_label = str(severity_label).strip().lower()
    mapping = {
        "normal": {
            "hard_to_hear": [],
            "notes": "Normal audiogram - speech should be perceived clearly.",
        },
        "mild": {
            "hard_to_hear": ["s", "f", "th", "sh", "t", "k"],
            "notes": "Mild hearing loss - high frequency consonants may be difficult to hear.",
        },
        "moderate": {
            "hard_to_hear": ["s", "f", "th", "sh", "t", "k", "p", "ch"],
            "notes": "Moderate hearing loss - many consonants may be unclear.",
        },
        "moderately severe": {
            "hard_to_hear": ["Most consonants", "Soft speech", "Distant speech"],
            "notes": "Moderately severe loss - amplification strongly recommended.",
        },
        "severe": {
            "hard_to_hear": ["Most speech sounds", "Soft vowels", "Conversation without amplification"],
            "notes": "Severe loss - powerful amplification or visual cues necessary.",
        },
        "profound": {
            "hard_to_hear": ["Nearly all speech sounds"],
            "notes": "Profound loss - cochlear implant or strong amplification with visual support needed.",
        },
    }
    for key in mapping:
        if key in severity_label:
            return mapping[key]
    return {"hard_to_hear": ["unknown"], "notes": "Severity label not recognized."}

# Load data
idx_to_class = load_class_map()

# UI
st.sidebar.markdown("### Model Selection")
model_choice = st.sidebar.radio(
    "Choose a model:",
    options=["CNN (Deep Learning)", "InceptionResNetV2"],
    help="Select deep learning model"
)

uploaded = st.file_uploader("Upload audiogram (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded audiogram", width=600)

    with st.spinner("Running inference..."):
        if model_choice == "CNN (Deep Learning)":
            cnn_model = load_cnn_classifier()
            if cnn_model is None:
                st.error("CNN model unavailable.")
                st.stop()
            label, conf, probs = predict_cnn(cnn_model, idx_to_class, image)
            model_info = "CNN (InceptionV3 Transfer Learning)"
        else:  # InceptionResNetV2
            inception_model = load_inceptionresnetv2_classifier()
            if inception_model is None:
                st.error("InceptionResNetV2 model unavailable.")
                st.stop()
            label, conf, probs = predict_inceptionresnetv2(inception_model, idx_to_class, image)
            model_info = "InceptionResNetV2 (Transfer Learning)"

    st.subheader("📊 Prediction Results")
    st.markdown(f"**Model Used:** {model_info}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Severity Class", label)
    with col2:
        st.metric("Confidence", f"{conf*100:.2f}%")

    difficulty_info = speech_banana_difficulties(label)
    st.subheader("🗣️ Speech Impact Analysis")
    
    if difficulty_info["hard_to_hear"]:
        hard_to_hear = ", ".join(difficulty_info["hard_to_hear"])
        st.markdown(f"**🔊 Affected Sounds:** {hard_to_hear}")
    else:
        st.markdown("**🔊 Affected Sounds:** None noted")
    
    st.markdown(f"**📝 Clinical Notes:** {difficulty_info['notes']}")

    st.subheader("📈 Detailed Classification Probabilities")
    prob_rows = [
        {"Class": idx_to_class[i], "Probability": f"{float(p)*100:.2f}%"} 
        for i, p in enumerate(probs)
    ]
    prob_rows = sorted(prob_rows, key=lambda r: float(r["Probability"].rstrip('%')), reverse=True)
    st.table(prob_rows)

    footer_lookup = {
        "CNN (Deep Learning)": f"Model: {MODEL_PATH.name}",
        "InceptionResNetV2": f"Model: {INCEPTIONRESNETV2_MODEL_PATH.name}",
    }
    st.caption(footer_lookup.get(model_choice, ""))
else:
    st.info("Upload an audiogram image to get a prediction using deep learning models.")
