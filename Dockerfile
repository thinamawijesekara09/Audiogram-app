FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY app.py .
COPY audiogram_severity_model1.2.keras .
COPY audiogram_severity_inceptionresnetv2.keras .
COPY *.json .
COPY ml_threshold_model/ ml_threshold_model/
COPY .streamlit/ .streamlit/

# Keep model artifacts in a non-mounted path too
RUN mkdir -p /opt/models
COPY audiogram_severity_model1.2.keras /opt/models/
COPY audiogram_severity_inceptionresnetv2.keras /opt/models/
COPY class_indices.json /opt/models/
COPY ml_threshold_model/ /opt/models/ml_threshold_model/

# Fail early if model artifacts are missing in image
RUN ls -lh /app/audiogram_severity_model1.2.keras /app/audiogram_severity_inceptionresnetv2.keras

# Expose port for Streamlit
EXPOSE 8501

# Set environment variables
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV MODEL_DIR=/opt/models

# Run Streamlit
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --logger.level=error --client.showErrorDetails=false"]
