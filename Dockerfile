FROM python:3.11-slim

# Force cache bust for Railway - build version 3
LABEL rebuild="3"

WORKDIR /app

# Install dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files (excluding .keras files which are LFS-tracked)
COPY app.py .
COPY *.json .
COPY ml_threshold_model/ ml_threshold_model/
COPY .streamlit/ .streamlit/

# Keep model artifacts in a non-mounted path too
RUN mkdir -p /opt/models
COPY class_indices.json /opt/models/
COPY ml_threshold_model/ /opt/models/ml_threshold_model/

# Download keras models directly from GitHub LFS (bypass pointer files)
# Build timestamp: 2026-03-08T03:35:00Z (forces cache invalidation)
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN echo "Downloading keras models from GitHub LFS..." && \
    wget -O /app/audiogram_severity_model1.2.keras \
    "https://media.githubusercontent.com/media/thinamawijesekara09/Audiogram-app/main/audiogram_severity_model1.2.keras" && \
    wget -O /app/audiogram_severity_inceptionresnetv2.keras \
    "https://media.githubusercontent.com/media/thinamawijesekara09/Audiogram-app/main/audiogram_severity_inceptionresnetv2.keras" && \
    echo "Download complete. Copying to /opt/models..." && \
    cp /app/audiogram_severity_model1.2.keras /opt/models/ && \
    cp /app/audiogram_severity_inceptionresnetv2.keras /opt/models/ && \
    echo "Verifying file sizes..." && \
    ls -lh /app/audiogram_severity_model1.2.keras && \
    ls -lh /app/audiogram_severity_inceptionresnetv2.keras && \
    ls -lh /opt/models/audiogram_severity_model1.2.keras && \
    ls -lh /opt/models/audiogram_severity_inceptionresnetv2.keras

# Expose port for Streamlit
EXPOSE 8501

# Set environment variables
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV MODEL_DIR=/opt/models

# Run Streamlit
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --logger.level=error --client.showErrorDetails=false"]
