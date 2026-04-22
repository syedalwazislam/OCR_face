from ultralytics import YOLO
import cv2
import easyocr
import csv
import os
import pandas as pd
import re
import json
from datetime import datetime
import numpy as np
import math
from deskew import determine_skew

# Check for face recognition libraries
FACE_RECOGNITION_AVAILABLE = False
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    pass

DEEPFACE_AVAILABLE = False
try:
    import deepface
    DEEPFACE_AVAILABLE = True
except ImportError:
    pass

# Check for face recognition libraries
FACE_RECOGNITION_AVAILABLE = False
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    pass

DEEPFACE_AVAILABLE = False
try:
    import deepface
    DEEPFACE_AVAILABLE = True
except ImportError:
    pass

class CNICProcessor:
    def __init__(self, model_path='runs/detect/train3/weights/best.pt'):
        # Load your custom trained YOLO model
        self.model = YOLO(model_path)
        self.reader = easyocr.Reader(['en'], gpu=False)
        
        # Your custom class names - define them directly
        self.class_names = {
            0: 'CNIC-HHMI', 1: 'bdate', 2: 'country', 3: 'edate', 
            4: 'fname', 5: 'gender', 6: 'id', 7: 'idate', 
            8: 'name', 9: 'picture'
        }

def detect_cnic_fields(image, model, class_names):
    """Detect CNIC fields using custom YOLO model"""
    results = model(image)
    detections = []
    
    for result in results:
        if result.boxes is not None:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                detections.append({
                    'class_id': cls,
                    'class_name': class_names[cls],  # Use the class_names from processor
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2]
                })
    
    return detections


def deskew_img(image: np.ndarray) -> np.ndarray:
    """
    Deskew an image using the estimated skew angle.
    Works for both grayscale and BGR images.
    """
    try:
        # Use grayscale for angle estimation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        angle = determine_skew(gray)

        old_height, old_width = image.shape[:2]
        angle_radian = math.radians(angle)

        # Compute new bounding dimensions
        new_width = abs(np.sin(angle_radian) * old_height) + abs(
            np.cos(angle_radian) * old_width
        )
        new_height = abs(np.sin(angle_radian) * old_width) + abs(
            np.cos(angle_radian) * old_height
        )

        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        rot_mat[1, 2] += (new_width - old_width) / 2
        rot_mat[0, 2] += (new_height - old_height) / 2

        deskewed = cv2.warpAffine(
            image,
            rot_mat,
            (int(round(new_width)), int(round(new_height))),
            borderValue=(0, 0, 0),
        )
        return deskewed
    except Exception:
        # If skew estimation fails, return original
        return image


def denoise_img(image: np.ndarray) -> np.ndarray:
    """
    Denoise image using bilateral filtering (preserves edges, smooths noise).
    Works for both grayscale and BGR images.
    """
    return cv2.bilateralFilter(image, 5, 55, 60)


def preprocess_image_for_ocr(roi: np.ndarray) -> np.ndarray:
    """
    Enhance image for better OCR results:
      - deskew
      - normalize contrast
      - denoise
      - adaptive threshold
      - morphological cleanup
    Returns a single-channel (grayscale/binary) image.
    """
    # 1) Deskew on original ROI (color or gray)
    deskewed = deskew_img(roi)

    # 2) Convert to grayscale
    gray = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY) if len(deskewed.shape) == 3 else deskewed

    # 3) Normalize contrast
    norm = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # 4) Denoise (bilateral filter to keep text edges sharp)
    denoised = denoise_img(norm)

    # 5) Adaptive thresholding for better text recognition
    processed = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # 6) Morphological close to fill small gaps in characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
    
    return processed

def clean_extracted_text(text, field_type):
    """Clean and extract relevant text based on field type"""
    if not text:
        return ""
    
    # IMPORTANT: Extract field-specific patterns FIRST from original text
    # before removing common phrases
    
    # Field-specific extraction (from original text)
    if field_type == 'id':
        # Extract CNIC number pattern: XXXXX-XXXXXXX-X (try original text first)
        cnic_pattern = r'\d{5}-\d{7}-\d{1}'
        match = re.search(cnic_pattern, text)
        if match:
            return match.group(0)
        # Alternative pattern with spaces
        cnic_pattern_alt = r'\d{5}\s*-\s*\d{7}\s*-\s*\d{1}'
        match = re.search(cnic_pattern_alt, text)
        if match:
            return match.group(0).replace(' ', '')
        return ""  # Return empty if no CNIC pattern found
    
    elif field_type in ['bdate', 'idate', 'edate']:
        # Extract date pattern: DD.MM.YYYY or DD-MM-YYYY or DD/MM/YYYY
        date_patterns = [
            r'\d{2}\.\d{2}\.\d{4}',  # DD.MM.YYYY
            r'\d{2}-\d{2}-\d{4}',    # DD-MM-YYYY
            r'\d{2}/\d{2}/\d{4}',    # DD/MM/YYYY
            r'\d{1,2}\.\d{1,2}\.\d{4}',  # D.M.YYYY or DD.M.YYYY
            r'\d{1,2},\d{1,2}\.\d{4}',  # Handle comma errors: 10,11.1987 -> 10.11.1987
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                date_str = match.group(0)
                # Fix comma errors in dates
                date_str = date_str.replace(',', '.')
                return date_str
        return ""  # Return empty if no date pattern found
    
    elif field_type == 'gender':
        # First try exact matches
        gender_pattern = r'\b(M|F|Male|Female|MALE|FEMALE)\b'
        match = re.search(gender_pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).upper() if len(match.group(0)) == 1 else match.group(0).title()
        
        # Handle OCR errors - look for patterns that might be "M" or "Male"
        # Common OCR errors: "Gcccler" -> "M", "Gcnder" -> "M", etc.
        # If text contains single letter M or F, extract it
        single_letter = re.search(r'\b([MF])\b', text, re.IGNORECASE)
        if single_letter:
            return single_letter.group(0).upper()
        
        # If text is short and contains M or F, try to extract
        if len(text) < 10:
            if 'M' in text.upper() and 'F' not in text.upper():
                return 'M'
            elif 'F' in text.upper():
                return 'F'
        
        return ""  # Return empty if no gender found
    
    elif field_type == 'country':
        # Extract country name (usually "Pakistan" or country codes)
        country_pattern = r'\b(Pakistan|PAKISTAN|PK|UAE|United Arab Emirates)\b'
        match = re.search(country_pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).title()
        return ""  # Return empty if no country found
    
    elif field_type in ['name', 'fname']:
        # Common phrases to remove from names
        name_phrases_to_remove = [
            "PAKISTAN", "National", "Identity", "Card", "Name", "Father", 
            "ather", "Name:", "Father Name", "Father's name", "Fathers name",
            "Gender", "Country", "Stay", "Identity Number", "Date", "Birth",
            "Issue", "Expiry", "Signature", "Holder", "ISLAMIC", "REPUBLIC",
            "OF", "PAKISTAN", "National Identity Card", "M", "F", "Male", "Female"
        ]
        
        cleaned = text
        # Remove common phrases (case-insensitive, word boundaries)
        for phrase in name_phrases_to_remove:
            # Use word boundary to avoid partial matches
            pattern = r'\b' + re.escape(phrase) + r'\b'
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        
        # Remove numbers and special characters, keep only letters and spaces
        cleaned = re.sub(r'[^a-zA-Z\s]', '', cleaned)
        # Remove extra whitespace
        cleaned = ' '.join(cleaned.split())
        # Remove single character words (likely OCR errors)
        words = [w for w in cleaned.split() if len(w) > 1]
        # Remove very short words that are likely OCR errors (less than 2 chars)
        words = [w for w in words if len(w) >= 2]
        
        # Take first 2-4 words as name (typical name structure)
        if len(words) > 4:
            words = words[:4]
        
        result = ' '.join(words).strip()
        return result if result else ""
    
    # For CNIC Header, return empty (we don't need it)
    elif field_type == 'CNIC-HHMI':
        return ""
    
    # General cleaning for other fields
    # Common phrases to remove
    common_phrases = [
        "PAKISTAN", "National Identity Card", "Name", "Father Name", 
        "Gender", "Country of Stay", "Identity Number", "Date of Issue", 
        "Date of Expiry", "Date of Birth", "Signature", "Holder",
        "ISLAMIC REPUBLIC OF PAKISTAN", "Date", "of", "Birth",
        "Issue", "Expiry", "Identity", "Number", "Country", "Stay"
    ]
    
    cleaned = text
    for phrase in common_phrases:
        cleaned = re.sub(re.escape(phrase), "", cleaned, flags=re.IGNORECASE)
    
    # Remove extra whitespace and special characters
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    cleaned = re.sub(r'^[^\w]+|[^\w]+$', '', cleaned)
    
    return cleaned

def extract_text_from_roi(image, bbox, reader, field_type=''):
    """Extract text from detected region with improved preprocessing"""
    x1, y1, x2, y2 = map(int, bbox)
    
    # Add padding (increased for better context)
    padding = 15
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.shape[1], x2 + padding)
    y2 = min(image.shape[0], y2 + padding)
    
    roi = image[y1:y2, x1:x2]
    
    if roi.size == 0:
        return ""
    
    # Try multiple preprocessing approaches
    processed_roi = preprocess_image_for_ocr(roi)
    
    # Also try original grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    
    # Perform OCR with both images and take best result
    all_texts = []
    
    for img in [processed_roi, gray_roi]:
        try:
            results = reader.readtext(img, detail=1, paragraph=False, text_threshold=0.3, width_ths=0.7, height_ths=0.7)
            for (_, text, conf) in results:
                if conf > 0.4:  # Slightly higher confidence threshold
                    all_texts.append((text.strip(), conf))
        except Exception as e:
            continue
    
    if not all_texts:
        return ""
    
    # Combine texts, prioritizing higher confidence
    all_texts.sort(key=lambda x: x[1], reverse=True)
    combined_text = ' '.join([text for text, _ in all_texts])
    
    # Clean the extracted text based on field type
    cleaned = clean_extracted_text(combined_text, field_type)
    
    return cleaned

def extract_picture_from_cnic(image, detections):
    """Extract the picture/face region from CNIC card"""
    picture_regions = []
    
    for detection in detections:
        if detection['class_name'] == 'picture':
            x1, y1, x2, y2 = map(int, detection['bbox'])
            # Add padding
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.shape[1], x2 + padding)
            y2 = min(image.shape[0], y2 + padding)
            
            picture_roi = image[y1:y2, x1:x2]
            if picture_roi.size > 0:
                picture_regions.append({
                    'image': picture_roi,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': detection['confidence']
                })
    
    # Return the picture with highest confidence
    if picture_regions:
        best_picture = max(picture_regions, key=lambda x: x['confidence'])
        return best_picture['image'], best_picture['bbox']
    return None, None

def detect_face_in_image(image):
    """Detect face in image using OpenCV Haar Cascade"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        # Return the largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])  # width * height
        x, y, w, h = largest_face
        face_roi = image[y:y+h, x:x+w]
        return face_roi, (x, y, w, h)
    return None, None

def compare_faces_face_recognition(face1, face2):
    """Compare two faces using face_recognition library"""
    if not FACE_RECOGNITION_AVAILABLE:
        return None, "face_recognition library not available"
    
    try:
        # Convert BGR to RGB (face_recognition uses RGB)
        face1_rgb = cv2.cvtColor(face1, cv2.COLOR_BGR2RGB)
        face2_rgb = cv2.cvtColor(face2, cv2.COLOR_BGR2RGB)
        
        # Get face encodings
        encoding1 = face_recognition.face_encodings(face1_rgb)
        encoding2 = face_recognition.face_encodings(face2_rgb)
        
        if len(encoding1) == 0:
            return None, "No face found in CNIC picture"
        if len(encoding2) == 0:
            return None, "No face found in selfie"
        
        # Calculate distance (lower is more similar)
        distance = face_recognition.face_distance([encoding1[0]], encoding2[0])[0]
        
        # Threshold: typically 0.6 is used, lower means more similar
        is_match = distance < 0.6
        similarity = (1 - distance) * 100  # Convert to percentage
        
        return {
            'is_match': is_match,
            'similarity': similarity,
            'distance': distance,
            'method': 'face_recognition'
        }, None
    except Exception as e:
        return None, f"Error in face_recognition: {str(e)}"

def compare_faces_deepface(face1, face2):
    """Compare two faces using DeepFace library"""
    if not DEEPFACE_AVAILABLE:
        return None, "DeepFace library not available. Install with: pip install deepface"
    
    try:
        from deepface import DeepFace
        # Save temporary images
        temp1 = 'temp_face1.jpg'
        temp2 = 'temp_face2.jpg'
        cv2.imwrite(temp1, face1)
        cv2.imwrite(temp2, face2)
        
        # Verify faces
        result = DeepFace.verify(img1_path=temp1, img2_path=temp2, 
                                model_name='VGG-Face', enforce_detection=False)
        
        # Clean up
        if os.path.exists(temp1):
            os.remove(temp1)
        if os.path.exists(temp2):
            os.remove(temp2)
        
        return {
            'is_match': result['verified'],
            'similarity': result['distance'],  # Lower is more similar
            'method': 'deepface'
        }, None
    except Exception as e:
        # Clean up on error
        for temp in ['temp_face1.jpg', 'temp_face2.jpg']:
            if os.path.exists(temp):
                os.remove(temp)
        return None, f"Error in DeepFace: {str(e)}"

def compare_faces_opencv(face1, face2):
    """Compare faces using OpenCV (histogram comparison) - fallback method"""
    try:
        # Resize faces to same size
        face1_resized = cv2.resize(face1, (128, 128))
        face2_resized = cv2.resize(face2, (128, 128))
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(face1_resized, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(face2_resized, cv2.COLOR_BGR2GRAY)
        
        # Calculate histogram
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        
        # Compare histograms
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # Normalize to percentage
        similarity = correlation * 100
        
        # Threshold: correlation > 0.7 is considered similar
        is_match = correlation > 0.7
        
        return {
            'is_match': is_match,
            'similarity': similarity,
            'correlation': correlation,
            'method': 'opencv_histogram'
        }, None
    except Exception as e:
        return None, f"Error in OpenCV comparison: {str(e)}"

def verify_face_with_selfie(cnic_picture, selfie_path):
    """
    Verify if the face in CNIC picture matches the selfie
    
    Args:
        cnic_picture: Extracted picture from CNIC card (numpy array)
        selfie_path: Path to selfie image
    
    Returns:
        verification_result: Dictionary with verification results
    """
    print("\n" + "="*60)
    print("FACE VERIFICATION")
    print("="*60)
    
    if cnic_picture is None:
        return {'error': 'No picture found in CNIC card'}
    
    if not os.path.exists(selfie_path):
        return {'error': f'Selfie image not found at {selfie_path}'}
    
    # Load selfie
    selfie_image = cv2.imread(selfie_path)
    if selfie_image is None:
        return {'error': 'Could not read selfie image'}
    
    # Extract faces from both images
    print("🔍 Extracting face from CNIC picture...")
    cnic_face, cnic_face_bbox = detect_face_in_image(cnic_picture)
    
    print("🔍 Extracting face from selfie...")
    selfie_face, selfie_face_bbox = detect_face_in_image(selfie_image)
    
    if cnic_face is None:
        return {'error': 'Could not detect face in CNIC picture'}
    
    if selfie_face is None:
        return {'error': 'Could not detect face in selfie'}
    
    # Save extracted faces for inspection
    cv2.imwrite('cnic_extracted_face.jpg', cnic_face)
    cv2.imwrite('selfie_extracted_face.jpg', selfie_face)
    print("💾 Extracted faces saved: 'cnic_extracted_face.jpg', 'selfie_extracted_face.jpg'")
    
    # Try different face comparison methods
    verification_result = {
        'cnic_face_detected': True,
        'selfie_face_detected': True,
        'methods_tried': []
    }
    
    # Method 1: face_recognition (most accurate)
    if FACE_RECOGNITION_AVAILABLE:
        print("\n📊 Comparing faces using face_recognition library...")
        result, error = compare_faces_face_recognition(cnic_face, selfie_face)
        if result:
            verification_result['face_recognition'] = result
            verification_result['methods_tried'].append('face_recognition')
            print(f"   Similarity: {result['similarity']:.2f}%")
            print(f"   Match: {'✅ YES' if result['is_match'] else '❌ NO'}")
    
    # Method 2: DeepFace
    if DEEPFACE_AVAILABLE:
        print("\n📊 Comparing faces using DeepFace...")
        result, error = compare_faces_deepface(cnic_face, selfie_face)
        if result:
            verification_result['deepface'] = result
            verification_result['methods_tried'].append('deepface')
            print(f"   Match: {'✅ YES' if result['is_match'] else '❌ NO'}")
    
    # Method 3: OpenCV (fallback)
    print("\n📊 Comparing faces using OpenCV histogram...")
    result, error = compare_faces_opencv(cnic_face, selfie_face)
    if result:
        verification_result['opencv'] = result
        verification_result['methods_tried'].append('opencv')
        print(f"   Similarity: {result['similarity']:.2f}%")
        print(f"   Match: {'✅ YES' if result['is_match'] else '❌ NO'}")
    
    # Determine final verification result
    if 'face_recognition' in verification_result:
        verification_result['final_verification'] = verification_result['face_recognition']['is_match']
        verification_result['confidence'] = verification_result['face_recognition']['similarity']
    elif 'deepface' in verification_result:
        verification_result['final_verification'] = verification_result['deepface']['is_match']
        verification_result['confidence'] = 'N/A'
    elif 'opencv' in verification_result:
        verification_result['final_verification'] = verification_result['opencv']['is_match']
        verification_result['confidence'] = verification_result['opencv']['similarity']
    else:
        verification_result['final_verification'] = False
        verification_result['error'] = 'Could not perform face comparison'
    
    return verification_result

def process_cnic_front(front_imagee, cnic_processor):
    """Process all CNIC fields from front side using YOLO detection"""
    print("🚀 Starting CNIC Front Side Processing...")
    
    # Detect all CNIC fields
    detections = detect_cnic_fields(front_imagee, cnic_processor.model, cnic_processor.class_names)
    
    if len(detections) == 0:
        print("❌ No CNIC fields detected!")
        return {}, [], None
    
    print(f"✅ Detected {len(detections)} CNIC fields")
    
    # Extract picture from CNIC
    cnic_picture, picture_bbox = extract_picture_from_cnic(front_imagee, detections)
    if cnic_picture is not None:
        cv2.imwrite('cnic_picture_extracted.jpg', cnic_picture)
        print(f"📸 CNIC picture extracted and saved to 'cnic_picture_extracted.jpg'")
    
    # Process all detected fields
    extracted_data = {}
    all_filtered_data = []
    
    # Sort detections by y-coordinate (top to bottom)
    detections.sort(key=lambda x: x['bbox'][1])
    
    for detection in detections:
        field_name = detection['class_name']
        # Skip picture field as it doesn't contain extractable text
        if field_name == 'picture':
            continue
            
        text = extract_text_from_roi(front_imagee, detection['bbox'], cnic_processor.reader, field_name)
        
        if text:
            # Map field names to display names
            display_names = {
                'name': 'Name',
                'fname': 'Father Name', 
                'id': 'ID Card Number',
                'bdate': 'Date of Birth',
                'idate': 'Date of Issue',
                'edate': 'Date of Expiry',
                'gender': 'Gender',
                'country': 'Country',
                'picture': 'Picture',
                'CNIC-HHMI': 'CNIC Header'
            }
            
            display_name = display_names.get(field_name, field_name)
            # Skip CNIC Header as it contains too much noise
            if field_name == 'CNIC-HHMI':
                print(f"⏭️  Skipping {display_name} (contains multiple fields)")
                continue
            extracted_data[display_name] = text
            all_filtered_data.append(text)
            print(f"📋 {display_name}: {text} (conf: {detection['confidence']:.2f})")
        else:
            print(f"⚠️ {field_name}: No text extracted (conf: {detection['confidence']:.2f})")
    
    return extracted_data, all_filtered_data, cnic_picture

def display_detected_fields(image, detections, window_name='Detected CNIC Fields'):
    """Display image with detected fields"""
    display_image = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection['bbox'])
        class_name = detection['class_name']
        confidence = detection['confidence']
        
        # Draw bounding box
        cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(display_image, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow(window_name, display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def validate_cnic_data(data):
    """Validate and format extracted CNIC data"""
    validated = {}
    
    # Validate CNIC number format
    if 'ID Card Number' in data:
        cnic = data['ID Card Number']
        # Ensure proper format: XXXXX-XXXXXXX-X
        cnic_clean = re.sub(r'[^\d-]', '', cnic)
        if re.match(r'\d{5}-\d{7}-\d{1}', cnic_clean):
            validated['ID Card Number'] = cnic_clean
        else:
            validated['ID Card Number'] = cnic  # Keep original if validation fails
    
    # Validate and format dates
    date_fields = {
        'Date of Birth': 'bdate',
        'Date of Issue': 'idate',
        'Date of Expiry': 'edate'
    }
    
    for field_name, _ in date_fields.items():
        if field_name in data:
            date_str = data[field_name]
            # Try to standardize date format
            date_patterns = [
                (r'(\d{2})\.(\d{2})\.(\d{4})', r'\1.\2.\3'),  # DD.MM.YYYY
                (r'(\d{2})-(\d{2})-(\d{4})', r'\1.\2.\3'),   # DD-MM-YYYY -> DD.MM.YYYY
                (r'(\d{2})/(\d{2})/(\d{4})', r'\1.\2.\3'),   # DD/MM/YYYY -> DD.MM.YYYY
            ]
            
            for pattern, replacement in date_patterns:
                match = re.search(pattern, date_str)
                if match:
                    validated[field_name] = match.group(0).replace('-', '.').replace('/', '.')
                    break
            else:
                validated[field_name] = date_str
    
    # Copy other fields as-is
    for key, value in data.items():
        if key not in validated:
            validated[key] = value
    
    return validated

def save_results(data, filename='cnic_front_data.csv'):
    """Save extracted data to CSV, JSON, and text files"""
    if not data:
        print("No data to save")
        return
    
    # Validate data first
    validated_data = validate_cnic_data(data)
    
    # Prepare CSV data
    csv_data = []
    for field_name, text in validated_data.items():
        csv_data.append({
            'Field': field_name,
            'Value': text
        })
    
    df = pd.DataFrame(csv_data)
    
    # Handle file permission errors
    try:
        df.to_csv(filename, index=False)
        print(f"💾 Data saved to {filename}")
    except PermissionError:
        # Try with a timestamped filename
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        new_filename = filename.replace('.csv', f'_{timestamp}.csv')
        df.to_csv(new_filename, index=False)
        print(f"⚠️  Original file locked. Data saved to {new_filename}")
        filename = new_filename  # Update for JSON/TXT files
    except Exception as e:
        print(f"❌ Error saving CSV: {e}")
        return
    
    # Save as JSON for better structure
    try:
        json_filename = filename.replace('.csv', '.json')
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(validated_data, f, indent=2, ensure_ascii=False)
        print(f"💾 JSON data saved to {json_filename}")
    except Exception as e:
        print(f"⚠️  Could not save JSON: {e}")
    
    # Also save as formatted text file
    try:
        txt_filename = filename.replace('.csv', '.txt')
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("  CNIC FRONT SIDE EXTRACTED DATA\n")
            f.write("=" * 50 + "\n\n")
            
            # Order fields logically
            field_order = ['Name', 'Father Name', 'ID Card Number', 'Date of Birth', 
                          'Date of Issue', 'Date of Expiry', 'Gender', 'Country']
            
            for field in field_order:
                if field in validated_data:
                    f.write(f"{field:20s}: {validated_data[field]}\n")
            
            # Add any remaining fields
            for field_name, text in validated_data.items():
                if field_name not in field_order:
                    f.write(f"{field_name:20s}: {text}\n")
            
            f.write("\n" + "=" * 50 + "\n")
        print(f"💾 Formatted text summary saved to '{txt_filename}'")
    except Exception as e:
        print(f"⚠️  Could not save text file: {e}")

def create_annotated_image(image, detections, output_path='cnic_annotated.jpg'):
    """Create and save annotated image"""
    annotated_image = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection['bbox'])
        class_name = detection['class_name']
        confidence = detection['confidence']
        
        # Draw bounding box
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label with background
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(annotated_image, (x1, y1-label_size[1]-10), (x1+label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(annotated_image, label, (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    cv2.imwrite(output_path, annotated_image)
    print(f"🖼️ Annotated image saved as '{output_path}'")

if __name__ == '__main__':
    # Use correct paths - adjust based on your folder structure
    f_img = './Output_Images/obd.jpeg'
    
    # Check if file exists
    if not os.path.exists(f_img):
        print(f"ERROR: Front image not found at {f_img}")
        f_img = '../Dataset/p_front.jpg'
        if not os.path.exists(f_img):
            print(f"Also not found at {f_img}")
            exit(1)
    
    # Read image
    front_image = cv2.imread(f_img)
    
    if front_image is None:
        print(f"ERROR: Could not read front image")
        exit(1)
    
    print("🔄 Initializing CNIC Processor...")
    
    # Initialize Custom YOLO model for CNIC fields
    try:
        cnic_processor = CNICProcessor('runs/detect/train3/weights/best.pt')
        print("✅ Custom YOLO model loaded successfully!")
    except Exception as e:
        print(f"❌ Custom model not found: {e}")
        print("⚠️ Using default YOLO model (will detect generic objects)")
        cnic_processor = CNICProcessor('yolov8n.pt')
        # For default model, use COCO class names
        cnic_processor.class_names = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 
            5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light'
        }
    
    # Detect CNIC fields using custom model
    print("\n🔍 Detecting CNIC fields...")
    front_detections = detect_cnic_fields(front_image, cnic_processor.model, cnic_processor.class_names)
    print(f"✅ Detected {len(front_detections)} CNIC fields on front side")
    
    # Display detected fields
    display_detected_fields(front_image, front_detections, 'Detected CNIC Fields')
    
    # Create and save annotated image
    create_annotated_image(front_image, front_detections, 'cnic_detected_fields.jpg')
    
    # Process and extract text from all fields
    print("\n📖 Extracting text from detected fields...")
    extracted_data, all_data, cnic_picture = process_cnic_front(front_image, cnic_processor)
    
    # STEP 4: Face Verification (if selfie provided)
    selfie_path = './Dataset/test1.jpeg'
    verification_result = None
    
    if selfie_path and cnic_picture is not None:
        if os.path.exists(selfie_path):
            print(f"\n🔍 Starting face verification with selfie: {selfie_path}")
            verification_result = verify_face_with_selfie(cnic_picture, selfie_path)
            
            # Add verification result to extracted data
            if verification_result and 'final_verification' in verification_result:
                extracted_data['Face Verification'] = '✅ MATCH' if verification_result['final_verification'] else '❌ NO MATCH'
                if 'confidence' in verification_result and verification_result['confidence'] != 'N/A':
                    extracted_data['Face Similarity'] = f"{verification_result['confidence']:.2f}%"
                
                # Print verification summary
                print("\n" + "="*60)
                print("FACE VERIFICATION SUMMARY")
                print("="*60)
                print(f"   Status: {'✅ VERIFIED' if verification_result['final_verification'] else '❌ NOT VERIFIED'}")
                if 'confidence' in verification_result and verification_result['confidence'] != 'N/A':
                    print(f"   Similarity: {verification_result['confidence']:.2f}%")
                print(f"   Methods used: {', '.join(verification_result.get('methods_tried', []))}")
                print("="*60)
            elif verification_result and 'error' in verification_result:
                print(f"\n⚠️  Face verification error: {verification_result['error']}")
        else:
            print(f"\n⚠️  Selfie image not found at: {selfie_path}")
            print("   Face verification skipped.")
    elif selfie_path and cnic_picture is None:
        print("\n⚠️  Face verification skipped: Could not extract picture from CNIC card")
    elif selfie_path is None:
        print("\n💡 Tip: To enable face verification, set 'selfie_path' variable with path to selfie image")
        print("   Example: selfie_path = './Dataset/selfie.jpg'")
    
    # Save results
    if extracted_data:
        save_results(extracted_data)
        
        # Print final summary with statistics
        print("\n" + "="*60)
        print("🎉 CNIC PROCESSING COMPLETED!")
        print("="*60)
        
        # Show extraction statistics
        total_fields = len(extracted_data)
        required_fields = ['Name', 'Father Name', 'ID Card Number', 'Date of Birth']
        extracted_required = sum(1 for field in required_fields if field in extracted_data)
        
        print(f"\n📊 Extraction Statistics:")
        print(f"   • Total fields extracted: {total_fields}")
        print(f"   • Required fields found: {extracted_required}/{len(required_fields)}")
        
        print(f"\n📋 Extracted Data:")
        print("-" * 60)
        
        # Display in logical order
        field_order = ['Name', 'Father Name', 'ID Card Number', 'Date of Birth', 
                      'Date of Issue', 'Date of Expiry', 'Gender', 'Country']
        
        for field in field_order:
            if field in extracted_data:
                print(f"   ✓ {field:20s}: {extracted_data[field]}")
        
        # Show any additional fields
        for field, value in extracted_data.items():
            if field not in field_order:
                print(f"   • {field:20s}: {value}")
        
        if extracted_required < len(required_fields):
            missing = [f for f in required_fields if f not in extracted_data]
            print(f"\n⚠️  Missing required fields: {', '.join(missing)}")
            print("   Consider checking image quality or model accuracy.")
        
        # Display face verification results
        if verification_result:
            print("\n" + "="*60)
            print("FACE VERIFICATION RESULTS")
            print("="*60)
            
            if 'error' in verification_result:
                print(f"❌ Error: {verification_result['error']}")
            else:
                final_match = verification_result.get('final_verification', False)
                confidence = verification_result.get('confidence', 'N/A')
                
                if isinstance(confidence, (int, float)):
                    print(f"🎯 Verification: {'✅ MATCH' if final_match else '❌ NO MATCH'}")
                    print(f"📊 Confidence: {confidence:.2f}%")
                else:
                    print(f"🎯 Verification: {'✅ MATCH' if final_match else '❌ NO MATCH'}")
                
                print(f"🔧 Methods used: {', '.join(verification_result.get('methods_tried', []))}")
                
                # Add verification result to extracted data
                extracted_data['Face_Verification'] = 'MATCH' if final_match else 'NO MATCH'
                if isinstance(confidence, (int, float)):
                    extracted_data['Face_Verification_Confidence'] = f"{confidence:.2f}%"
        
        print("\n" + "="*60)
    else:
        print("❌ No data extracted from CNIC")
        print("   Possible reasons:")
        print("   • Image quality too low")
        print("   • Fields not detected properly")
        print("   • OCR failed to extract text")
    
    print("\n✅ Front side processing completed!")