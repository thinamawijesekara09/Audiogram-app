import json
import base64
import os
import numpy as np
from pathlib import Path
from typing import Tuple
import joblib
import cv2
from sklearn.cluster import KMeans
import urllib.request

import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Paths
BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = Path(os.getenv("MODEL_DIR", str(BASE_DIR)))

def _artifact_path(relative_path: str) -> Path:
    preferred = ARTIFACT_DIR / relative_path
    fallback = BASE_DIR / relative_path
    return preferred if preferred.exists() else fallback

def _ensure_model_file_exists(model_path: Path, github_url: str) -> Path:
    """Download model from GitHub if it doesn't exist or is an LFS pointer."""
    if model_path.exists():
        # Check if it's an LFS pointer file
        if model_path.stat().st_size < 200:
            try:
                content = model_path.read_bytes()
                if b"version https://git-lfs" in content:
                    st.info(f"Downloading {model_path.name} from GitHub... This may take a moment.")
                    urllib.request.urlretrieve(github_url, str(model_path))
            except Exception:
                pass
        return model_path
    
    # Download if doesn't exist
    st.info(f"Downloading {model_path.name} from GitHub... This may take a moment.")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(github_url, str(model_path))
    return model_path

MODEL_PATH = _artifact_path("audiogram_severity_model1.2.keras")
INCEPTIONRESNETV2_MODEL_PATH = _artifact_path("audiogram_severity_inceptionresnetv2.keras")
CLASS_MAP_PATH = _artifact_path("class_indices.json")
BACKGROUND_IMAGE_PATH = BASE_DIR / "background.jpg"
ML_MODEL_DIR = _artifact_path("ml_threshold_model")
RF_MODEL_PATH = ML_MODEL_DIR / "rf_model.joblib"
SVM_MODEL_PATH = ML_MODEL_DIR / "svm_model.joblib"
LABEL_ENCODER_PATH = ML_MODEL_DIR / "label_encoder.joblib"
ML_META_PATH = ML_MODEL_DIR / "meta.json"
IMG_SIZE: Tuple[int, int] = (299, 299)
FREQS = [500, 1000, 2000, 3000, 4000, 6000, 8000]
DB_MIN = -10
DB_MAX = 110

# GitHub LFS CDN URLs for models
GITHUB_BASE = "https://media.githubusercontent.com/media/thinamawijesekara09/Audiogram-app/main"
CNN_MODEL_URL = f"{GITHUB_BASE}/audiogram_severity_model1.2.keras"
INCEPTION_MODEL_URL = f"{GITHUB_BASE}/audiogram_severity_inceptionresnetv2.keras"

st.set_page_config(page_title="Audiogram Severity Classifier", page_icon="🎧", layout="centered")

# Load and encode background image
def get_background_image():
    if BACKGROUND_IMAGE_PATH.exists():
        with open(BACKGROUND_IMAGE_PATH, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return None

bg_image = get_background_image()

# Add background styling
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
        
        /* Main title styling */
        h1 {{
            color: #1a1a1a !important;
            font-weight: 700 !important;
            text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.9);
            padding: 20px;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            margin-bottom: 10px;
        }}
        
        /* Subtitle and text */
        .stMarkdown p, .stMarkdown {{
            color: #2d2d2d !important;
            font-weight: 500;
            background-color: rgba(255, 255, 255, 0.95);
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        /* Subheaders */
        h2, h3 {{
            color: #1a1a1a !important;
            font-weight: 600 !important;
            padding: 10px;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }}
        
        /* File uploader */
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
        
        /* Image container */
        .stImage {{
            background-color: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        }}
        
        /* Table styling */
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
        
        /* Info box */
        .stAlert {{
            background-color: rgba(255, 255, 255, 0.95) !important;
            color: #1a1a1a !important;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        /* Spinner */
        .stSpinner > div {{
            border-top-color: #667eea !important;
        }}
        
        /* Footer */
        .stMarkdown hr {{
            border-color: rgba(0, 0, 0, 0.2);
        }}
        </style>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            font-family: 'Inter', sans-serif;
        }
        
        .stApp > header {
            background-color: transparent;
            visibility: hidden;
            height: 0;
        }
        
        h1, h2, h3 {
            color: #1a1a1a !important;
            font-weight: 600 !important;
        }
        
        .stMarkdown, .stImage, .stFileUploader, div[data-testid="stTable"] {
            background-color: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
            color: #2d2d2d !important;
        }
        </style>
        """, unsafe_allow_html=True)

st.title("🎧 Audiogram Severity Classifier")

# ===========================
# Model Loaders (Cached)
# ===========================

@st.cache_resource(show_spinner=True)
def load_cnn_classifier():
    try:
        model_path = _ensure_model_file_exists(MODEL_PATH, CNN_MODEL_URL)
        return load_model(model_path, compile=False, safe_mode=False)
    except Exception as e:
        st.warning(f"CNN model could not be loaded: {e}")
        return None


@st.cache_resource(show_spinner=True)
def load_inceptionresnetv2_classifier():
    try:
        model_path = _ensure_model_file_exists(INCEPTIONRESNETV2_MODEL_PATH, INCEPTION_MODEL_URL)
        return load_model(model_path, compile=False, safe_mode=False)
    except Exception as e:
        st.warning(f"InceptionResNetV2 model could not be loaded: {e}")
        return None


@st.cache_data(show_spinner=True)
def load_class_map():
    if not CLASS_MAP_PATH.exists():
        st.error(f"Class map file not found at {CLASS_MAP_PATH}")
        st.stop()
    with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    return idx_to_class


@st.cache_data(show_spinner=True)
def load_ml_models():
    """Load RF/SVM models with feature column order and label encoder."""
    if not RF_MODEL_PATH.exists():
        return None, None, None, None, None

    rf_bundle = joblib.load(RF_MODEL_PATH)
    if isinstance(rf_bundle, dict):
        rf_model = rf_bundle.get('model')
        rf_feature_cols = rf_bundle.get('feature_cols')
    else:
        rf_model = rf_bundle
        rf_feature_cols = None

    svm_model = None
    svm_feature_cols = None
    if SVM_MODEL_PATH.exists():
        svm_bundle = joblib.load(SVM_MODEL_PATH)
        if isinstance(svm_bundle, dict):
            svm_model = svm_bundle.get('model')
            svm_feature_cols = svm_bundle.get('feature_cols')
        else:
            svm_model = svm_bundle
            svm_feature_cols = None

    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    return rf_model, rf_feature_cols, svm_model, svm_feature_cols, label_encoder


@st.cache_data(show_spinner=True)
def load_ml_metadata():
    """Load metadata about ML models"""
    if ML_META_PATH.exists():
        with open(ML_META_PATH, "r") as f:
            return json.load(f)
    return None


def preprocess_image(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr


def preprocess_image_inceptionresnetv2(image: Image.Image) -> np.ndarray:
    from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_inceptionresnetv2
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_inceptionresnetv2(arr)
    return arr


def merge_close_points(pts, dist_px=14):
    if not pts:
        return []
    pts = sorted(pts, key=lambda p: p[0])
    merged, used = [], [False]*len(pts)
    for i in range(len(pts)):
        if used[i]:
            continue
        cx, cy, area = pts[i]
        group = [(cx, cy, area)]
        used[i] = True
        for j in range(i+1, len(pts)):
            if used[j]:
                continue
            cx2, cy2, area2 = pts[j]
            if abs(cx2 - cx) <= dist_px and abs(cy2 - cy) <= dist_px:
                group.append((cx2, cy2, area2))
                used[j] = True
        total_area = sum(g[2] for g in group)
        mx = sum(g[0]*g[2] for g in group) / total_area
        my = sum(g[1]*g[2] for g in group) / total_area
        merged.append((float(mx), float(my), int(total_area)))
    return merged


def get_expected_x_by_kmeans(xs, k=7):
    xs = np.array(xs, dtype=np.float32).reshape(-1, 1)
    k = min(k, len(xs))
    km = KMeans(n_clusters=k, n_init='auto', random_state=42)
    km.fit(xs)
    centers = np.sort(km.cluster_centers_.reshape(-1))
    return centers


def extract_thresholds_from_pil(image: Image.Image, debug=False) -> np.ndarray:
    img_bgr = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([75, 25, 25])
    upper_blue = np.array([150, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    pts = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 3 <= area <= 3000:
            cx, cy = centroids[i]
            pts.append((float(cx), float(cy), int(area)))
    if len(pts) < 2:
        raise RuntimeError("Too few blue components detected.")
    pts = merge_close_points(pts, dist_px=14)
    xs = [p[0] for p in pts]
    expected_x = get_expected_x_by_kmeans(xs, k=len(FREQS))
    col_points = {i: [] for i in range(len(expected_x))}
    for (cx, cy, area) in pts:
        j = int(np.argmin(np.abs(expected_x - cx)))
        col_points[j].append((cx, cy, area))
    chosen = [None] * len(expected_x)
    for j in range(len(expected_x)):
        if col_points[j]:
            chosen[j] = max(col_points[j], key=lambda t: t[2])
    def y_to_db(y):
        return DB_MIN + (y / float(h - 1)) * (DB_MAX - DB_MIN)
    thresholds = [None] * len(FREQS)
    for j in range(min(len(FREQS), len(expected_x))):
        if chosen[j] is not None:
            thresholds[j] = float(y_to_db(chosen[j][1]))
    idxs = np.arange(len(FREQS))
    known = np.array([t is not None for t in thresholds], dtype=bool)
    if known.sum() < 3:
        raise RuntimeError("Too many missing columns in detection.")
    known_x = idxs[known]
    known_y = np.array([thresholds[i] for i in known_x], dtype=np.float32)
    for j in range(len(FREQS)):
        if thresholds[j] is None:
            thresholds[j] = float(np.interp(j, known_x, known_y))
    thresholds = [float(np.clip(t, DB_MIN, DB_MAX)) for t in thresholds]
    return np.array(thresholds, dtype=np.float32)


def thresholds_to_features(thr: np.ndarray) -> dict:
    thr = np.array(thr, dtype=np.float32)
    speech_avg = float(np.mean(thr[0:5]))
    high_avg = float(np.mean(thr[4:7]))
    low_avg = float(np.mean(thr[0:3]))
    slope = float(high_avg - low_avg)
    feats = {
        "thr_500": float(thr[0]),
        "thr_1k": float(thr[1]),
        "thr_2k": float(thr[2]),
        "thr_3k": float(thr[3]),
        "thr_4k": float(thr[4]),
        "thr_6k": float(thr[5]),
        "thr_8k": float(thr[6]),
        "avg_thr": float(np.mean(thr)),
        "max_thr": float(np.max(thr)),
        "min_thr": float(np.min(thr)),
        "speech_avg": speech_avg,
        "low_avg": low_avg,
        "high_avg": high_avg,
        "slope": slope,
        "variance": float(np.var(thr)),
    }
    return feats


def predict_cnn(model, idx_to_class, image: Image.Image):
    """CNN model prediction"""
    arr = preprocess_image(image)
    probs = model.predict(arr, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = idx_to_class[pred_idx]
    confidence = float(probs[pred_idx])
    return pred_label, confidence, probs


def predict_inceptionresnetv2(model, idx_to_class, image: Image.Image):
    """InceptionResNetV2 model prediction"""
    arr = preprocess_image_inceptionresnetv2(image)
    probs = model.predict(arr, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = idx_to_class[pred_idx]
    confidence = float(probs[pred_idx])
    return pred_label, confidence, probs


def predict_ml(model, feature_cols, label_encoder, image: Image.Image, model_name="RF"):
    """ML model prediction using robust threshold extraction and training feature order."""
    try:
        thr = extract_thresholds_from_pil(image, debug=False)
        feats = thresholds_to_features(thr)
        ordered = [feats[c] for c in (feature_cols or [
            "thr_500","thr_1k","thr_2k","thr_3k","thr_4k","thr_6k","thr_8k",
            "avg_thr","max_thr","min_thr","speech_avg","low_avg","high_avg","slope","variance"
        ])]
        X = np.array([ordered], dtype=np.float32)
        pred_index = int(model.predict(X)[0])
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return "Error", 0.0, {}

    try:
        prediction_label = label_encoder.inverse_transform([pred_index])[0]
    except Exception:
        prediction_label = str(pred_index)

    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)[0]
        confidence = float(np.max(probs))
    else:
        probs = None
        confidence = None

    classes = list(getattr(label_encoder, 'classes_', []))
    class_probs = {}
    if probs is not None and classes:
        for i, cls in enumerate(classes):
            if i < len(probs):
                class_probs[cls] = float(probs[i])
    elif prediction_label:
        class_probs[prediction_label] = float(confidence) if confidence is not None else 1.0

    return prediction_label, (confidence or 0.0), class_probs


def speech_banana_difficulties(severity_label):
    """Convert severity label to speech impact information - handles both str and numeric types"""
    # Convert to string if it's numeric (numpy int64, etc.)
    severity_label = str(severity_label).strip().lower()

    mapping = {
        "normal": {
            "hard_to_hear": [],
            "notes": "සාමාන්‍ය සංවාදයකදී කථා කරන ශබ්ද සාමාන්‍ය ලෙස පැහැදිලිව ඇසීමට හැකි විය යුතුය.",
        },
        "mild": {
            "hard_to_hear": ["ස", "ෆ", "ත්", "ශ", "ට", "ක"],
            "notes": "සාමාන්‍යයෙන් මුලින්ම ඇසීම අඩුවන්නේ උස සංඛ්‍යාත ව්‍යංජන ශබ්දවලටය.ඒ නිසා, විශේෂයෙන් ශබ්දය වැඩි පරිසරවලදී කථාව පැහැදිලි නොවිය හැක.”",
        },
        "moderate": {
            "hard_to_hear": ["ස", "ෆ", "ත්", "ශ", "ට", "ක", "ප", "ච", "ස", "ව"],
            "notes": "බොහෝ ව්‍යංජන ශබ්ද බලපෑමට ලක් වේ. ශබ්දය වැඩි පරිසරවලදී අවබෝධය ලබාගැනීම අපහසු වේ.",
        },
        "moderately severe": {
            "hard_to_hear": ["බොහෝ ව්‍යංජන ශබ්ද", "මෘදු කථාව", "දුරින් කරන කථාව"],
            "notes": "වර්ධනය (amplification) නොමැතිව කථාව අවබෝධ කරගැනීම දැඩි ලෙස අඩුවේ. ව්‍යංජන ශබ්දවලට වඩා ස්වර ශබ්ද ඇසීමට වැඩි හැකියාව තිබිය හැක.",
        },
        "severe": {
            "hard_to_hear": ["බොහෝ කථා ශබ්ද", "ස්වර ශබ්ද මෘදු විය හැක", "වර්ධනය නොමැති සංවාද"],
            "notes": "ඉතා ශබ්දවත් කථාවද අසන්නට අපහසු විය හැක; වර්ධනය/දෘශ්‍ය සංකේත මත දැඩි රැඳී සිටීම.",
        },
        "profound": {
            "hard_to_hear": ["ආසන්න වශයෙන් සියලුම කථන ශබ්ද නොඇසේ."],
            "notes": "සාමාන්‍ය සංවාද කථාව සාමාන්‍යයෙන් ඇසෙන්නේ නැත. ශක්තිමත් ශබ්ද වර්ධනය (amplification) හෝ CI සමඟ දෘශ්‍ය උපාය මාර්ග අවශ්‍ය වේ.",
        },
    }

    for key in mapping:
        if key in severity_label:
            return mapping[key]

    return {
        "hard_to_hear": ["unknown"],
        "notes": "Severity label not recognized in mapping.",
    }



# ===========================
# Load Models
# ===========================
cnn_model = load_cnn_classifier()
inceptionresnetv2_model = load_inceptionresnetv2_classifier()
idx_to_class = load_class_map()
rf_model, rf_feature_cols, svm_model, svm_feature_cols, label_encoder = load_ml_models()
ml_metadata = load_ml_metadata()

# ===========================
# Model Selection UI
# ===========================
st.sidebar.markdown("###  Model Selection")
model_options = []
if cnn_model is not None:
    model_options.append("CNN (Deep Learning)")
if inceptionresnetv2_model is not None:
    model_options.append("InceptionResNetV2")
if rf_model is not None:
    model_options.append("Random Forest")
if svm_model is not None:
    model_options.append("SVM")

if not model_options:
    st.error("No models are available. Check model artifacts in deployment.")
    st.stop()

model_choice = st.sidebar.radio(
    "Choose a model:",
    options=model_options,
    help="Select which model to use for classification"
)

uploaded = st.file_uploader("Upload audiogram (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded audiogram", width=600)

    with st.spinner("Running inference..."):
        if model_choice == "CNN (Deep Learning)":
            if cnn_model is None:
                st.error("CNN model is unavailable in this deployment.")
                st.stop()
            label, conf, probs = predict_cnn(cnn_model, idx_to_class, image)
            model_info = "CNN (InceptionV3 Transfer Learning)"
        elif model_choice == "InceptionResNetV2":
            if inceptionresnetv2_model is None:
                st.error("InceptionResNetV2 model is unavailable in this deployment.")
                st.stop()
            label, conf, probs = predict_inceptionresnetv2(inceptionresnetv2_model, idx_to_class, image)
            model_info = "InceptionResNetV2 (Transfer Learning)"
        elif model_choice == "Random Forest":
            if rf_model is None:
                st.error("Random Forest model not found!")
                st.stop()
            label, conf, probs = predict_ml(rf_model, rf_feature_cols, label_encoder, image, "RF")
            if ml_metadata and 'rf_accuracy' in ml_metadata:
                model_info = f"Random Forest (Accuracy: {ml_metadata['rf_accuracy']*100:.2f}%)"
            else:
                model_info = "Random Forest"
        else:  # SVM
            if svm_model is None:
                st.error("SVM model not found! (only RF available)")
                st.stop()
            label, conf, probs = predict_ml(svm_model, svm_feature_cols, label_encoder, image, "SVM")
            if ml_metadata and 'svm_accuracy' in ml_metadata:
                model_info = f"Support Vector Machine (Accuracy: {ml_metadata['svm_accuracy']*100:.2f}%)"
            else:
                model_info = "SVM"

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

    # Show probability table
    st.subheader("📈 Detailed Classification Probabilities")
    if isinstance(probs, dict):
        prob_rows = [
            {"Class": cls, "Probability": f"{float(p)*100:.2f}%"} for cls, p in probs.items()
        ]
    else:
        prob_rows = [
            {"Class": idx_to_class[i], "Probability": f"{float(p)*100:.2f}%"} for i, p in enumerate(probs)
        ]
    prob_rows = sorted(prob_rows, key=lambda r: float(r["Probability"].rstrip('%')), reverse=True)
    st.table(prob_rows)


    footer_lookup = {
        "CNN (Deep Learning)": f"Model: {MODEL_PATH.name} | Class map: {CLASS_MAP_PATH.name}",
        "InceptionResNetV2": f"Model: {INCEPTIONRESNETV2_MODEL_PATH.name} | Class map: {CLASS_MAP_PATH.name}",
        "Random Forest": f"Model: {RF_MODEL_PATH.name} | Label encoder: {LABEL_ENCODER_PATH.name}",
        "SVM": f"Model: {SVM_MODEL_PATH.name} | Label encoder: {LABEL_ENCODER_PATH.name}",
    }
    st.caption(footer_lookup.get(model_choice, ""))
else:
    st.info("Upload an audiogram image to get a prediction.")
