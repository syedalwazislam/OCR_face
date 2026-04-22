from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import cv2
import uvicorn

from webtest import (
    CNICProcessor,
    detect_cnic_fields,
    process_cnic_front,
    detect_face_in_image,
    verify_face_live,
    capture_live_face,
    extract_picture_from_cnic,
)


app = FastAPI(title="CNIC Processing API", version="1.0.0")

# Initialize CNIC processor once at startup (mirrors console Mode 1 behavior)
try:
    cnic_processor = CNICProcessor("runs/detect/train3/weights/best.pt")
    model_status = "custom"
except Exception:
    # Fallback to default YOLO model
    cnic_processor = CNICProcessor("yolov8n.pt")
    cnic_processor.class_names = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        4: "airplane",
        5: "bus",
        6: "train",
        7: "truck",
        8: "boat",
        9: "traffic light",
    }
    model_status = "default"


def _read_image_file(upload: UploadFile) -> np.ndarray:
    """Read an uploaded image into an OpenCV BGR array."""
    data = upload.file.read()
    if not data:
        raise HTTPException(status_code=400, detail=f"{upload.filename} is empty")
    image_array = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=400, detail=f"Could not decode image {upload.filename}"
        )
    return image


def _extract_cnic_face(front_image: np.ndarray) -> np.ndarray:
    """Extract the CNIC portrait area from a front-side CNIC image."""
    detections = detect_cnic_fields(
        front_image, cnic_processor.model, cnic_processor.class_names
    )
    if not detections:
        raise HTTPException(status_code=422, detail="No CNIC fields detected in the image")

    cnic_picture, _ = extract_picture_from_cnic(front_image, detections)
    if cnic_picture is None:
        raise HTTPException(
            status_code=422,
            detail="Could not extract face picture from CNIC card",
        )
    return cnic_picture


@app.get("/health", summary="Health check")
async def health():
    return {"status": "ok", "model_status": model_status}


@app.post("/extract-cnic", summary="Extract CNIC fields only (no face verification)")
async def extract_cnic(
    cnic_image: UploadFile = File(..., description="Front-side CNIC image"),
):
    """
    Pure extraction API:
    - Detect CNIC fields using the trained YOLO model
    - Extract and clean text fields
    - Returns structured CNIC data, no face verification
    """
    try:
        front_image = _read_image_file(cnic_image)

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

        response = {
            "model_status": model_status,
            "fields": extracted_data,
            "has_cnic_face": cnic_picture is not None,
        }
        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/verify-face",
    summary="Verify CNIC face against uploaded selfie (Render/cloud-safe)",
)
async def verify_face_upload(
    cnic_image: UploadFile = File(
        ..., description="Front-side CNIC image (face will be auto-extracted)"
    ),
    selfie_image: UploadFile = File(
        ..., description="Captured selfie image from client webcam/mobile camera"
    ),
):
    """
    Cloud-safe face verification API:
    - Extracts face from CNIC image
    - Reads selfie image uploaded by client app/browser
    - Compares CNIC face vs selfie using face_recognition / DeepFace / OpenCV
    """
    try:
        front_image = _read_image_file(cnic_image)
        live_face = _read_image_file(selfie_image)

        cnic_picture = _extract_cnic_face(front_image)
        detected_face = detect_face_in_image(live_face)
        if detected_face is None:
            raise HTTPException(
                status_code=422,
                detail="No face detected in uploaded selfie image",
            )

        verification = verify_face_live(cnic_picture, detected_face)

        response = {
            "model_status": model_status,
            "mode": "uploaded_selfie",
            "verification": verification,
        }
        return jsonable_encoder(response, custom_encoder={
    np.bool_: bool,
    np.integer: int,
    np.floating: float,
    np.ndarray: lambda x: x.tolist(),
})


    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/verify-face-local-webcam",
    summary="Verify CNIC face against server webcam (local machine only)",
)
async def verify_face_local_webcam(
    cnic_image: UploadFile = File(
        ..., description="Front-side CNIC image (face will be auto-extracted)"
    ),
):
    """
    Local-only face verification API:
    - Extracts face from CNIC image
    - Captures live selfie from server webcam
    - Intended for local desktop testing, not cloud deployment
    """
    try:
        front_image = _read_image_file(cnic_image)
        cnic_picture = _extract_cnic_face(front_image)

        live_face = capture_live_face()
        if live_face is None:
            raise HTTPException(
                status_code=422,
                detail="No face captured from webcam (user cancelled or no face detected)",
            )

        verification = verify_face_live(cnic_picture, live_face)
        response = {
            "model_status": model_status,
            "mode": "server_webcam",
            "verification": verification,
        }
        return jsonable_encoder(response, custom_encoder={
    np.bool_: bool,
    np.integer: int,
    np.floating: float,
    np.ndarray: lambda x: x.tolist(),
})


    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

