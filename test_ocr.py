import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import os

class DocumentScanner:
    def __init__(self):
        pass
    
    def enhance_to_scan_quality(self, image_path, output_path):
        """
        Convert any image to high-quality scanned document look
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image {image_path}")
            return None
        
        # Step 1: Resize for consistency (optional)
        img = self._resize_image(img, max_dimension=2000)
        
        # Step 2: Automatic document detection and cropping
        processed = self._auto_document_crop(img)
        
        # Step 3: Apply scanning effects
        scanned = self._apply_scan_effects(processed)
        
        # Step 4: Save result
        cv2.imwrite(output_path, scanned)
        print(f"Saved scanned version to: {output_path}")
        return scanned
    
    def _resize_image(self, image, max_dimension=2000):
        """Resize image while maintaining aspect ratio"""
        height, width = image.shape[:2]
        
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(image, (new_width, new_height), 
                            interpolation=cv2.INTER_AREA)
        return image
    
    def _auto_document_crop(self, image):
        """Automatically detect and crop document boundaries"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter for edge-preserving smoothing
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Edge detection
        edges = cv2.Canny(filtered, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        # Try to find document-like contour
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            if len(approx) == 4:  # Likely a document
                # Get the four corners
                points = approx.reshape(4, 2)
                
                # Order points: top-left, top-right, bottom-right, bottom-left
                rect = self._order_points(points)
                
                # Apply perspective transform
                warped = self._four_point_transform(image, rect)
                return warped
        
        # If no document contour found, return original
        return image
    
    def _order_points(self, pts):
        """Order points for perspective transform"""
        rect = np.zeros((4, 2), dtype="float32")
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        
        return rect
    
    def _four_point_transform(self, image, pts):
        """Apply perspective transform"""
        (tl, tr, br, bl) = pts
        
        # Calculate width and height
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        # Destination points
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        
        # Compute transform matrix and apply
        M = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        
        return warped
    
    def _apply_scan_effects(self, image):
        """Apply scanning effects to make it look like a scanned document"""
        # Convert to grayscale for document processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Step 1: Noise reduction
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Step 2: Contrast enhancement using CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Step 3: Sharpening
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Step 4: Binarization (for crisp black/white)
        # Adaptive thresholding for documents
        binary = cv2.adaptiveThreshold(sharpened, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Step 5: Add slight color tint (optional - for warmer scan look)
        # Convert back to color with slight yellow/brown tint
        color_scan = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        # Add slight warmth to mimic scanned paper
        color_scan[:, :, 0] = color_scan[:, :, 0] * 0.9  # Reduce blue
        color_scan[:, :, 2] = color_scan[:, :, 2] * 0.9  # Reduce red
        color_scan = color_scan.astype(np.uint8)
        
        # Step 6: Add subtle noise/grain (optional - for realism)
        noise = np.random.normal(0, 1.5, color_scan.shape).astype(np.uint8)
        color_scan = cv2.add(color_scan, noise)
        
        # Step 7: Add slight vignette effect (optional)
        color_scan = self._add_vignette(color_scan, scale=0.8)
        
        return color_scan
    
    def _add_vignette(self, image, scale=0.9):
        """Add subtle vignette effect"""
        rows, cols = image.shape[:2]
        
        # Create vignette mask
        kernel_x = cv2.getGaussianKernel(cols, cols/3)
        kernel_y = cv2.getGaussianKernel(rows, rows/3)
        kernel = kernel_y * kernel_x.T
        
        # Normalize kernel
        mask = 255 * kernel / np.linalg.norm(kernel)
        mask = mask * scale + (1 - scale) * 255
        mask = mask.astype(np.uint8)
        
        # Apply vignette to each channel
        result = image.copy()
        for i in range(3):
            result[:, :, i] = cv2.addWeighted(image[:, :, i], 
                                            scale, 
                                            mask, 
                                            1-scale, 
                                            0)
        
        return result

# Alternative: Simple one-function solution
def simple_scan_effect(image_path, output_path):
    """
    Simple function to convert any image to scanned quality
    """
    # Read image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Noise reduction
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # Apply adaptive thresholding for crisp black/white
    scanned = cv2.adaptiveThreshold(enhanced, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Convert back to color with warm tint
    color_result = cv2.cvtColor(scanned, cv2.COLOR_GRAY2BGR)
    color_result[:, :, 0] = color_result[:, :, 0] * 0.95  # Reduce blue
    color_result[:, :, 2] = color_result[:, :, 2] * 0.95  # Reduce red
    
    # Save
    cv2.imwrite(output_path, color_result)
    return color_result

def batch_process_folder(input_folder, output_folder):
    """
    Process all images in a folder
    """
    scanner = DocumentScanner()
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Supported image formats
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    processed_count = 0
    for filename in os.listdir(input_folder):
        filepath = os.path.join(input_folder, filename)
        
        # Check if it's an image file
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            output_path = os.path.join(output_folder, 
                                      f"scanned_{filename}")
            
            try:
                scanner.enhance_to_scan_quality(filepath, output_path)
                processed_count += 1
                print(f"✓ Processed: {filename}")
            except Exception as e:
                print(f"✗ Error processing {filename}: {str(e)}")
    
    print(f"\nProcessed {processed_count} images successfully!")

# Usage Examples
if __name__ == "__main__":
    # Method 1: Process single image
    scanner = DocumentScanner()
    scanner.enhance_to_scan_quality("./Dataset/obd.jpeg", "CNIC_scanned.jpg")
    
    # Method 2: Simple version
    simple_scan_effect("./Dataset/obd.jpeg", "CNIC_simple_scan.jpg")
    