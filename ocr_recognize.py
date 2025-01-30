import cv2
import pytesseract
from PIL import Image
import numpy as np
import logging

# ================== OCR Configuration ==================

# If Tesseract is not in your system's PATH, specify its location
# For example, on Windows:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Uncomment and set the path if necessary
# pytesseract.pytesseract.tesseract_cmd = r'YOUR_TESSERACT_PATH'

# ================== OCR Functions ==================

def recognize_number(frame, roi_coords, expected_number=None):
    """
    Recognizes a number within a specified region of a frame.

    Args:
        frame (numpy.ndarray): The webcam frame.
        roi_coords (dict): Dictionary containing 'x', 'y', 'w', 'h' of the ROI.
        expected_number (str or int, optional): The expected number for validation.

    Returns:
        str or None: The recognized number if successful and matches expected_number (if provided), else None.
    """
    x, y, w, h = roi_coords['x'], roi_coords['y'], roi_coords['w'], roi_coords['h']
    roi = frame[y:y+h, x:x+w]

    # Preprocessing
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply thresholding to get a binary image
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)
    # Optionally, apply dilation to enhance features
    kernel = np.ones((2,2), np.uint8)
    processed = cv2.dilate(thresh, kernel, iterations=1)

    # OCR
    pil_img = Image.fromarray(processed)
    config = '--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789'
    recognized_text = pytesseract.image_to_string(pil_img, config=config)

    # Clean the recognized text
    recognized_number = ''.join(filter(str.isdigit, recognized_text))

    # Logging the recognized number
    if recognized_number:
        logging.info(f"Recognized Number: {recognized_number}")
    else:
        logging.warning("No number recognized in the specified ROI.")

    # Validation (optional)
    if expected_number is not None:
        if str(recognized_number) != str(expected_number):
            logging.warning(f"Recognized number '{recognized_number}' does not match expected '{expected_number}'.")
            return None

    return recognized_number if recognized_number else None


