# Audiogram Severity Classifier - Deployment Guide

This project has two separate deployment configurations:

## 1. **Main Deployment (RF/SVM) - Stable & Lightweight**
- **URL**: https://web-production-0cdf9.up.railway.app
- **Models**: Random Forest, Support Vector Machine
- **Memory**: 512 MB (stable)
- **Use Case**: Production deployment for reliable predictions
- **Files**: 
  - `app.py` - Main application
  - `Dockerfile` - Container config
  - `railway.toml` - Railway deployment config

### Features:
- ✅ Fast inference
- ✅ Low memory footprint
- ✅ Stable and always accessible
- ✅ Threshold-based ML models

---

## 2. **Deep Learning Deployment (CNN/InceptionResNetV2) - High Accuracy**
- **Status**: Ready to deploy as separate service
- **Models**: CNN, InceptionResNetV2
- **Memory Required**: 1-2 GB (for TensorFlow)
- **Use Case**: High-accuracy deep learning inference
- **Files**:
  - `app_cnn.py` - CNN-only application
  - `Dockerfile.cnn` - CNN-optimized container
  - `railway-cnn.toml` - CNN Railway config

### Features:
- ✅ Deep transfer learning models
- ✅ Higher accuracy for complex cases
- ✅ Self-healing model downloads from GitHub LFS
- ✅ Separate from stable deployment

---

## Deployment Instructions for CNN Service

### Option A: Automatic (Recommended)
1. Create new Railway project
2. Connect GitHub repo
3. Set buildpack: Dockerfile
4. Set Dockerfile to: `Dockerfile.cnn`
5. Configure environment:
   - `MODEL_DIR=/opt/models`
6. Deploy

### Option B: Manual via Railway CLI
```bash
railway up --dockerfile Dockerfile.cnn
```

### Environment Variables
- `MODEL_DIR=/opt/models` - Where models are stored/downloaded
- `PORT=8501` - Streamlit port (set automatically by Railway)

---

## Model Download Behavior

Both deployments use **lazy loading**:
- Models are downloaded from GitHub LFS media CDN on first use
- First classification request takes 30-60 seconds (one-time)
- Subsequent requests are instant (cached locally)

### GitHub LFS URLs
- CNN: `https://media.githubusercontent.com/media/thinamawijesekara09/Audiogram-app/main/audiogram_severity_model1.2.keras`
- InceptionResNetV2: `https://media.githubusercontent.com/media/thinamawijesekara09/Audiogram-app/main/audiogram_severity_inceptionresnetv2.keras`

---

## Architecture

```
User Request
    ↓
    ├─→ ML/RF/SVM Route  → web-production-0cdf9.up.railway.app (stable, 512MB)
    │
    └─→ DL/CNN Route     → [new-cnn-service].railway.app (high accuracy, 1-2GB)
```

---

## Memory Requirements

| Model Type | Minimum | Recommended |
|-----------|---------|-------------|
| RF/SVM    | 512 MB  | 512 MB      |
| CNN       | 1 GB    | 2 GB        |

---

## Files Overview

### Main Deployment
- `app.py` - RF/SVM classifier with optional DL fallback
- `Dockerfile` - Lightweight container
- `requirements.txt` - Python dependencies

### CNN Deployment
- `app_cnn.py` - Streamlined CNN/InceptionResNetV2 only
- `Dockerfile.cnn` - Optimized for deep learning
- `railway-cnn.toml` - Railway configuration
- `class_indices.json` - Severity class labels

---

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run RF/SVM app
streamlit run app.py

# Run CNN app (requires TensorFlow)
streamlit run app_cnn.py
```

---

## Troubleshooting

### If CNN app crashes on Railway:
1. Check container memory (increase to 2GB)
2. Check Railway logs for OOM errors
3. Ensure GitHub LFS connectivity (test URLs above)

### If models fail to load:
1. Check `MODEL_DIR` environment variable
2. Verify GitHub LFS files are accessible
3. Check Railway container logs for download errors

---

## Next Steps

1. ✅ **Current Status**: RF/SVM app deployed and stable
2. **TODO**: Deploy CNN app to separate Railway service
3. **TODO** (Optional): Create landing page linking both services

