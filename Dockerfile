FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY app.py .
COPY *.json .
COPY ml_threshold_model/ ml_threshold_model/
COPY .streamlit/ .streamlit/

# Keep model artifacts directory for runtime downloads
RUN mkdir -p /opt/models

# Set environment variables
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV MODEL_DIR=/opt/models

# Expose port for Streamlit
EXPOSE 8501

# Run Streamlit (models download on first use)
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --logger.level=error --client.showErrorDetails=false"]
