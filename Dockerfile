FROM python:3.11-slim

# Force cache bust for Railway - build version 4
LABEL rebuild="4"

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

# Install git and git-lfs, then pull real model files
RUN apt-get update && apt-get install -y git git-lfs && rm -rf /var/lib/apt/lists/*

# Copy .gitattributes and initialize git-lfs
COPY .gitattributes .
RUN git init . && git lfs install --local

# Copy the LFS pointer files and pull real files
COPY audiogram_severity_model1.2.keras .
COPY audiogram_severity_inceptionresnetv2.keras .

# Pull real files from LFS
RUN echo "Pulling real keras models from git-lfs..." && \
    git lfs pull -I "*.keras" || echo "LFS pull partial - downloading from GitHub instead..." && \
    if [ ! -s audiogram_severity_model1.2.keras ] || grep -q "version https://git-lfs" audiogram_severity_model1.2.keras; then \
      wget -O audiogram_severity_model1.2.keras "https://media.githubusercontent.com/media/thinamawijesekara09/Audiogram-app/main/audiogram_severity_model1.2.keras"; \
    fi && \
    if [ ! -s audiogram_severity_inceptionresnetv2.keras ] || grep -q "version https://git-lfs" audiogram_severity_inceptionresnetv2.keras; then \
      wget -O audiogram_severity_inceptionresnetv2.keras "https://media.githubusercontent.com/media/thinamawijesekara09/Audiogram-app/main/audiogram_severity_inceptionresnetv2.keras"; \
    fi && \
    echo "Copying models to /opt/models..." && \
    cp audiogram_severity_model1.2.keras /opt/models/ && \
    cp audiogram_severity_inceptionresnetv2.keras /opt/models/ && \
    echo "Verifying keras models..." && \
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
