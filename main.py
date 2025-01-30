import cv2
import requests
import time
import logging
from datetime import datetime
import sys
import os
from contextlib import contextmanager
import csv
import json
from logging.handlers import RotatingFileHandler
import shutil
import threading
import pytesseract
from PIL import Image
import numpy as np

# ================== Configuration Loader ==================

def load_config(config_file='config.json'):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

config = load_config()

WEBCAMS = config.get("webcams", [])
SLACK_WEBHOOK_URL = config.get("slack_webhook_url", "")
CHECK_INTERVAL = config.get("check_interval", 300)
FAIL_THRESHOLD_PERCENT = config.get("fail_threshold_percent", 50)
LOG_FILE = config.get("log_file", 'local_webcam_monitor.log')
EXPORT_FILE = config.get("export_file", 'active_webcams.csv')
SPECIFIC_ACTIVE_NUMBER = config.get("specific_active_number", 2)
LOG_ROTATION_MAX_BYTES = config.get("log_rotation_max_bytes", 5*1024*1024)  # 5 MB
LOG_ROTATION_BACKUP_COUNT = config.get("log_rotation_backup_count", 5)

# ================== Logging Setup with Rotation ==================

# Define a rotating file handler
rotating_handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=LOG_ROTATION_MAX_BYTES,
    backupCount=LOG_ROTATION_BACKUP_COUNT
)
rotating_handler.setLevel(logging.INFO)
rotating_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Define a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        rotating_handler,
        console_handler
    ]
)

# ================== Context Manager to Suppress stderr ==================

@contextmanager
def suppress_stderr():
    """
    A context manager to suppress stderr temporarily.
    Useful for suppressing OpenCV error messages.
    """
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

# ================== OCR Configuration ==================

# If Tesseract is not in your system's PATH, specify its location
# For example, on Windows:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Uncomment and set the path if necessary
# pytesseract.pytesseract.tesseract_cmd = r'YOUR_TESSERACT_PATH'

# ================== Functions ==================

def recognize_number(frame, roi_coords, expected_number=None):
    """
    Recognizes a number within a specified region of a frame.

    Args:
        frame (numpy.ndarray): The webcam frame.
        roi_coords (tuple): Coordinates of the ROI (x, y, w, h).
        expected_number (str or int, optional): The expected number for validation.

    Returns:
        str: The recognized number.
    """
    x, y, w, h = roi_coords
    roi = frame[y:y+h, x:x+w]

    # Preprocessing
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply thresholding to get a binary image
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)
    # Optionally, apply dilation or erosion to enhance features
    kernel = np.ones((2,2), np.uint8)
    processed = cv2.dilate(thresh, kernel, iterations=1)

    # OCR
    pil_img = Image.fromarray(processed)
    config = '--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789'
    recognized_text = pytesseract.image_to_string(pil_img, config=config)

    # Clean the recognized text
    recognized_number = ''.join(filter(str.isdigit, recognized_text))

    # Validation (optional)
    if expected_number is not None:
        if str(recognized_number) != str(expected_number):
            logging.warning(f"Recognized number {recognized_number} does not match expected {expected_number}.")
            return None

    return recognized_number


def check_webcam_status(cam, timeout=5, retries=3):
    """
    Checks if a specific webcam is accessible and functional with retries.
    Additionally, recognizes a number within a defined ROI.

    Args:
        cam (dict): Dictionary containing webcam 'name', 'index', 'roi', and 'expected_number'.
        timeout (int): Time in seconds to wait for the webcam.
        retries (int): Number of retry attempts.

    Returns:
        bool: True if webcam is accessible and the recognized number matches the expected number, False otherwise.
    """
    for attempt in range(retries):
        with suppress_stderr():
            cap = cv2.VideoCapture(cam['index'])

        if not cap.isOpened():
            logging.error(
                f"{cam['name']} (Index {cam['index']}) could not be opened. Attempt {attempt + 1} of {retries}.")
            time.sleep(1)
            continue

        # **Set the desired resolution (1000x800)**
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        start_time = time.time()
        recognized_number = None

        while True:
            ret, frame = cap.read()
            if ret:
                # Rotate the frame 90 degrees to the left (counter-clockwise) if necessary
                rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                recognized_number = recognize_number(
                    rotated_frame,
                    (cam['roi']['x'], cam['roi']['y'], cam['roi']['w'], cam['roi']['h']),
                    cam.get('expected_number')
                )
                if recognized_number is not None:
                    logging.info(f"{cam['name']} (Index {cam['index']}): Recognized Number: {recognized_number}")
                    cap.release()
                    return True
                else:
                    logging.warning(f"{cam['name']} (Index {cam['index']}): Number mismatch or recognition failed.")
                    break
            elif time.time() - start_time > timeout:
                logging.error(f"{cam['name']} (Index {cam['index']}) is unresponsive.")
                cap.release()
                break
            time.sleep(0.5)

        cap.release()
    return False

def send_slack_notification(message):
    """
    Sends a notification message to Slack.

    Args:
        message (str): The message to send to Slack.
    """
    payload = {
        "text": message
    }
    try:
        response = requests.post(SLACK_WEBHOOK_URL, json=payload)
        if response.status_code != 200:
            logging.error(f'Failed to send Slack notification. Status code: {response.status_code}, Response: {response.text}')
        else:
            logging.info('Slack notification sent successfully.')
    except requests.RequestException as e:
        logging.error(f'Error sending Slack notification: {e}')

def export_log_info(active_cams, export_file='active_webcams.csv'):
    """
    Exports the list of active webcams to a CSV file.

    Args:
        active_cams (list): List of active webcam names.
        export_file (str): Path to the export CSV file.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        with open(export_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            # Write header if file is empty
            if os.stat(export_file).st_size == 0:
                writer.writerow(['Timestamp', 'Active Webcams'])
            writer.writerow([timestamp, '; '.join(active_cams)])
        logging.info(f'Exported active webcams to {export_file} at {timestamp}.')
    except Exception as e:
        logging.error(f'Failed to export active webcams: {e}')

def periodic_export(export_interval=3600):
    """
    Periodically exports the log file to a specified location.

    Args:
        export_interval (int): Time between exports in seconds.
    """
    while True:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        destination = f'exported_logs/webcam_monitor_{timestamp}.log'
        try:
            os.makedirs('exported_logs', exist_ok=True)
            shutil.copy(LOG_FILE, destination)
            logging.info(f'Exported log file to {destination}.')
        except Exception as e:
            logging.error(f'Failed to export log file: {e}')
        time.sleep(export_interval)

def monitor_webcams():
    """
    Main function to monitor webcams and send notifications based on failure thresholds.
    Includes OCR-based number recognition within defined ROIs.
    """
    logging.info('Starting webcam monitoring service...')
    WEBCAM_STATUS = {cam['name']: True for cam in WEBCAMS}  # Initial status

    while True:
        total_cams = len(WEBCAMS)
        failures = 0
        failed_cams = []
        active_cams = []

        if total_cams == 0:
            logging.warning('No webcams configured for monitoring.')
        else:
            logging.info('Checking status of each webcam...')
            for cam in WEBCAMS:
                is_up = check_webcam_status(cam)
                if not is_up:
                    failures += 1
                    failed_cams.append(cam['name'])
                    if WEBCAM_STATUS[cam['name']]:
                        # Webcam just failed
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        message = (
                            f":warning: *Webcam Alert*\n"
                            f"At {timestamp}, {cam['name']} is not responding or recognized number is incorrect."
                        )
                        send_slack_notification(message)
                        WEBCAM_STATUS[cam['name']] = False
                else:
                    active_cams.append(cam['name'])
                    if not WEBCAM_STATUS[cam['name']]:
                        # Webcam just recovered
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        message = (
                            f":white_check_mark: *Webcam Recovery*\n"
                            f"At {timestamp}, {cam['name']} has started responding again with correct number."
                        )
                        send_slack_notification(message)
                        WEBCAM_STATUS[cam['name']] = True
                time.sleep(1)  # Brief pause between checks

            failure_percentage = (failures / total_cams) * 100
            logging.info(f'Webcam Check: {failures}/{total_cams} failures ({failure_percentage:.2f}%)')
            logging.info(f'Active Webcams: {", ".join(active_cams)}')

            # Specific Condition: When exactly SPECIFIC_ACTIVE_NUMBER webcams are active
            if len(active_cams) == SPECIFIC_ACTIVE_NUMBER:
                export_log_info(active_cams, EXPORT_FILE)

            if failure_percentage >= FAIL_THRESHOLD_PERCENT:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                failed_cam_names = ', '.join(failed_cams)
                message = (
                    f":warning: *Webcam Alert*\n"
                    f"At {timestamp}, {failures} out of {total_cams} webcams are not responding "
                    f"({failure_percentage:.2f}%).\n"
                    f"Failed webcams: {failed_cam_names}"
                )
                send_slack_notification(message)
            else:
                logging.info('Failure threshold not met. No aggregate notification sent.')

        logging.info(f'Waiting for {CHECK_INTERVAL} seconds before next check...\n')
        time.sleep(CHECK_INTERVAL)

# ================== Start Periodic Export in Separate Thread ==================

export_thread = threading.Thread(target=periodic_export, args=(3600,), daemon=True)  # Export every hour
export_thread.start()

# ================== Entry Point ==================

if __name__ == '__main__':
    try:
        monitor_webcams()
    except KeyboardInterrupt:
        logging.info('Webcam monitoring stopped manually.')
    except Exception as e:
        logging.exception(f'An unexpected error occurred: {e}')
