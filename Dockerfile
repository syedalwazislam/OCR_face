FROM python:3.9-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk2.0-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    tesseract-ocr \
    libzbar0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel
RUN pip install ninja

# Speed up builds
ENV CMAKE_GENERATOR=Ninja
ENV FORCE_CMAKE=1

# Install PyTorch CPU first (important for YOLO stability)
RUN pip install torch==2.1.2+cpu torchvision==0.16.2+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# Copy requirements
COPY requirements.txt .

# FIX CRITICAL ISSUE: NumPy must be pinned
RUN pip install --no-cache-dir "numpy<2"

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# IMPORTANT: Railway-safe startup command
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
