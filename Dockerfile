FROM python:3.9-slim
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake libopenblas-dev liblapack-dev \
    libx11-dev libgtk2.0-dev libgl1 libglib2.0-0 \
    libsm6 libxext6 libxrender1 tesseract-ocr libzbar0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel ninja

ENV CMAKE_GENERATOR=Ninja
ENV FORCE_CMAKE=1

# Step 1: Pin NumPy FIRST before anything else touches it
RUN pip install --no-cache-dir "numpy==1.26.4"

# Step 2: Install scikit-image pinned to NumPy 1.x compatible version
RUN pip install --no-cache-dir "scikit-image==0.21.0"

# Step 3: PyTorch CPU
RUN pip install --no-cache-dir torch==2.1.2+cpu torchvision==0.16.2+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# Step 4: Everything else
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Safety net — force NumPy back if anything above drifted it
RUN pip install --no-cache-dir --force-reinstall "numpy==1.26.4" "scikit-image==0.21.0"

COPY . .

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
