FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps required for dlib/face-recognition/opencv/tesseract/pyzbar
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

COPY requirements.txt .

# upgrade pip first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# install python deps
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}
