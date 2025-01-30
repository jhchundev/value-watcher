import cv2
import pytesseract
from PIL import Image
import numpy as np
import logging

from config import load_config

# ================== OCR Configuration ==================

config = load_config()
WEBCAMS = config.get("webcams", [])
SLACK_WEBHOOK_URL = config.get("slack_webhook_url", "")
CHECK_INTERVAL = config.get("check_interval", 300)
FAIL_THRESHOLD_PERCENT = config.get("fail_threshold_percent", 50)
LOG_FILE = config.get("log_file", 'local_webcam_monitor.log')
EXPORT_FILE = config.get("export_file", 'active_webcams.csv')
SPECIFIC_ACTIVE_NUMBER = config.get("specific_active_number", 2)
LOG_ROTATION_MAX_BYTES = config.get("log_rotation_max_bytes", 5 * 1024 * 1024)  # 5 MB
LOG_ROTATION_BACKUP_COUNT = config.get("log_rotation_backup_count", 5)

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


def main():
    # Configure logging
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filemode='a',
    )
    # Load webcam configuration
    if not WEBCAMS:
        logging.error("No webcams configured. Exiting...")
        return

    # Initialize webcam capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Error: Could not open webcam.")
        return

    # Set webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    # Main loop
    recognize_number(cap, {'x': 100, 'y': 100, 'w': 200, 'h': 100}, expected_number=150)


if __name__ == "__main__":
    main()