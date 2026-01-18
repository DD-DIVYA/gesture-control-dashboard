FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY Backend\(Ml_python\)/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY Backend\(Ml_python\)/ .

# Download YOLOv8 model if not present
RUN python -c "import urllib.request; urllib.request.urlretrieve('https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt', 'yolov8n.pt')" || echo "Model download failed, will download at runtime"

# Expose port
EXPOSE 10000

# Set environment variables
ENV PORT=10000
ENV WS_HOST=0.0.0.0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:10000/health')" || exit 1

# Run the application
CMD ["python", "ws_server.py"]