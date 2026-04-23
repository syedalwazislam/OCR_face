import logging
import asyncio

import numpy as np
import cv2
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.encoders import jsonable_encoder

from webtest import (
    CNICProcessor,
    detect_cnic_fields,
    process_cnic_front,
    detect_face_in_image,
    verify_face_live,
    capture_live_face,
    extract_picture_from_cnic,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="CNIC Processing API", version="1.0.0")

# ---------------------------------------------------------------------------
# Model initialisation  (once at startup)
# ---------------------------------------------------------------------------
try:
    cnic_processor = CNICProcessor("runs/detect/train3/weights/best.pt")
    model_status = "custom"
    logger.info("✅ Custom YOLO model loaded (runs/detect/train3/weights/best.pt)")
except Exception as e:
    logger.warning(f"⚠️  Custom model failed to load ({e}). Falling back to yolov8n.pt")
    try:
        cnic_processor = CNICProcessor("yolov8n.pt")
        cnic_processor.class_names = {
            0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
            4: "airplane", 5: "bus", 6: "train", 7: "truck",
            8: "boat", 9: "traffic light",
        }
        model_status = "default"
        logger.info("✅ Default YOLO model loaded (yolov8n.pt)")
    except Exception as e2:
        logger.error(f"❌ Both models failed to load: {e2}")
        cnic_processor = None
        model_status = "unavailable"


# ---------------------------------------------------------------------------
# Custom JSON encoder for numpy types
# ---------------------------------------------------------------------------
NUMPY_ENCODER = {
    np.bool_: bool,
    np.integer: int,
    np.floating: float,
    np.ndarray: lambda x: x.tolist(),
}


def _safe_encode(data: dict) -> dict:
    """Encode a response dict, safely handling numpy types."""
    return jsonable_encoder(data, custom_encoder=NUMPY_ENCODER)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_image_bytes(data: bytes, filename: str = "image") -> np.ndarray:
    """Decode raw bytes into an OpenCV BGR array."""
    if not data:
        raise HTTPException(status_code=400, detail=f"{filename} is empty")
    image_array = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=400, detail=f"Could not decode image: {filename}"
        )
    return image


async def _read_upload(upload: UploadFile) -> np.ndarray:
    """
    Read an UploadFile asynchronously (non-blocking) and return an OpenCV image.
    Running file.read() in an executor prevents blocking the async event loop.
    """
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, upload.file.read)
    return _read_image_bytes(data, upload.filename or "upload")


def _require_model() -> None:
    """Raise 503 if the model failed to load at startup."""
    if cnic_processor is None:
        raise HTTPException(
            status_code=503,
            detail="Model unavailable — check server logs for startup errors",
        )


def _extract_cnic_face(front_image: np.ndarray) -> np.ndarray:
    """Extract the CNIC portrait region from a front-side CNIC image."""
    detections = detect_cnic_fields(
        front_image, cnic_processor.model, cnic_processor.class_names
    )
    if not detections:
        raise HTTPException(
            status_code=422, detail="No CNIC fields detected in the image"
        )

    cnic_picture, _ = extract_picture_from_cnic(front_image, detections)
    if cnic_picture is None:
        raise HTTPException(
            status_code=422,
            detail="Could not extract face picture from CNIC card",
        )
    return cnic_picture


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", summary="Health check")
async def health():
    return {"status": "ok", "model_status": model_status}


@app.post("/extract-cnic", summary="Extract CNIC fields only (no face verification)")
async def extract_cnic(
    cnic_image: UploadFile = File(..., description="Front-side CNIC image"),
):
    """
    Detect CNIC fields with the trained YOLO model, extract and clean text
    fields, and return structured CNIC data.  No face verification is performed.
    """
    _require_model()
    try:
        front_image = await _read_upload(cnic_image)

        detections = detect_cnic_fields(
            front_image, cnic_processor.model, cnic_processor.class_names
        )
        if not detections:
            raise HTTPException(
                status_code=422, detail="No CNIC fields detected in the image"
            )

        extracted_data, all_data, cnic_picture = process_cnic_front(
            front_image, cnic_processor
        )
        if not extracted_data:
            raise HTTPException(
                status_code=422, detail="No data extracted from CNIC"
            )

        return _safe_encode({
            "model_status": model_status,
            "fields": extracted_data,
            "has_cnic_face": cnic_picture is not None,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unhandled error in /extract-cnic")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/verify-face",
    summary="Verify CNIC face against uploaded selfie (cloud-safe)",
)
async def verify_face_upload(
    cnic_image: UploadFile = File(..., description="Front-side CNIC image"),
    selfie_image: UploadFile = File(..., description="Selfie from client webcam/camera"),
):
    """
    Cloud-safe face verification:
    - Extracts face from CNIC image
    - Reads selfie uploaded by the client
    - Compares both faces and returns a verification result
    """
    _require_model()
    try:
        front_image = await _read_upload(cnic_image)
        live_face_image = await _read_upload(selfie_image)

        cnic_picture = _extract_cnic_face(front_image)

        detected_face = detect_face_in_image(live_face_image)
        if detected_face is None:
            raise HTTPException(
                status_code=422,
                detail="No face detected in the uploaded selfie image",
            )

        verification = verify_face_live(cnic_picture, detected_face)

        return _safe_encode({
            "model_status": model_status,
            "mode": "uploaded_selfie",
            "verification": verification,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unhandled error in /verify-face")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/verify-face-local-webcam",
    summary="Verify CNIC face against server webcam (local only)",
)
async def verify_face_local_webcam(
    cnic_image: UploadFile = File(..., description="Front-side CNIC image"),
):
    """
    Local-only face verification (not suitable for cloud deployment):
    - Extracts face from CNIC image
    - Captures a live selfie from the **server** webcam
    - Compares both faces and returns a verification result
    """
    _require_model()
    try:
        front_image = await _read_upload(cnic_image)
        cnic_picture = _extract_cnic_face(front_image)

        live_face = capture_live_face()
        if live_face is None:
            raise HTTPException(
                status_code=422,
                detail="No face captured from webcam (user cancelled or no face detected)",
            )

        verification = verify_face_live(cnic_picture, live_face)

        return _safe_encode({
            "model_status": model_status,
            "mode": "server_webcam",
            "verification": verification,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unhandled error in /verify-face-local-webcam")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
