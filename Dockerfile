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

# If .keras files are Git LFS pointers, replace them with real binaries
RUN python - <<'PY'
from pathlib import Path
import urllib.request

repo_base = "https://media.githubusercontent.com/media/thinamawijesekara09/Audiogram-app/main"
artifacts = {
	"audiogram_severity_model1.2.keras": f"{repo_base}/audiogram_severity_model1.2.keras",
	"audiogram_severity_inceptionresnetv2.keras": f"{repo_base}/audiogram_severity_inceptionresnetv2.keras",
}

def is_lfs_pointer(path: Path) -> bool:
	if not path.exists() or path.stat().st_size == 0:
		return True
	head = path.read_bytes()[:200]
	return head.startswith(b"version https://git-lfs.github.com/spec/v1")

for name, url in artifacts.items():
	targets = [Path("/app") / name, Path("/opt/models") / name]
	for target in targets:
		if is_lfs_pointer(target):
			target.parent.mkdir(parents=True, exist_ok=True)
			print(f"Downloading real artifact for {name} -> {target}")
			urllib.request.urlretrieve(url, str(target))

for name in artifacts:
	for target in [Path("/app") / name, Path("/opt/models") / name]:
		size_mb = target.stat().st_size / (1024 * 1024)
		print(f"{target}: {size_mb:.2f} MB")
PY

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
