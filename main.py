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
from typing import Tuple, Optional, List

from config import load_config

# ================== Configuration Loader ==================

config = load_config()

WEBCAMS = config.get("webcams", [])
SLACK_WEBHOOK_URL = config.get("slack_webhook_url", "")
CHECK_INTERVAL = config.get("check_interval", 300)
FAIL_THRESHOLD_PERCENT = config.get("fail_threshold_percent", 50)
LOG_FILE = config.get("log_file", 'local_webcam_monitor.log')
EXPORT_FILE = config.get("export_file", 'active_webcams.csv')
SPECIFIC_ACTIVE_NUMBER = config.get("specific_active_number", 2)
LOG_ROTATION_MAX_BYTES = config.get("log_rotation_max_bytes", 5 * 1024 * 1024)
LOG_ROTATION_BACKUP_COUNT = config.get("log_rotation_backup_count", 5)

# ================== Logging Setup with Rotation ==================

rotating_handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=LOG_ROTATION_MAX_BYTES,
    backupCount=LOG_ROTATION_BACKUP_COUNT
)
rotating_handler.setLevel(logging.INFO)
rotating_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

logging.basicConfig(
    level=logging.INFO,
    handlers=[rotating_handler, console_handler]
)

# ================== Context Manager to Suppress stderr ==================

@contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

# ================== OCR Configuration ==================
# If needed, specify the full path to your Tesseract executable:
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Example on Linux

# ================== OCR Utility Functions ==================

def preprocess_image_for_ocr(frame: np.ndarray, roi_coords: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Preprocesses the image for better OCR accuracy by applying color segmentation,
    morphological operations, and resizing.
    """
    x, y, w, h = roi_coords
    roi = frame[y:y + h, x:x + w]

    # Convert to HSV for easier color segmentation
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define a mask range for "yellow" text. Adjust as necessary.
    YELLOW_LOWER = np.array([20, 100, 100])
    YELLOW_UPPER = np.array([35, 255, 255])

    mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Convert masked area back to grayscale
    masked_roi = cv2.bitwise_and(roi, roi, mask=mask)
    gray = cv2.cvtColor(masked_roi, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Additional morphological steps to remove small specks
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Resize for better OCR accuracy
    processed = cv2.resize(morph, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)

    return processed

def extract_text_from_image(image: np.ndarray) -> Tuple[str, List[float]]:
    """
    Extracts text and confidence scores from a preprocessed image using Tesseract OCR.
    """
    # Tesseract config: psm=8 (single word/line), whitelisting digits only
    custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789'
    data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)

    recognized_string = ""
    confidences = []

    for i, text_val in enumerate(data['text']):
        if text_val.strip():
            # Filter out any non-digit characters, ensuring only integers are captured
            filtered_text = "".join([ch for ch in text_val if ch.isdigit()])
            if filtered_text:
                recognized_string += filtered_text
                conf = data['conf'][i]
                if conf != -1:  # Tesseract can give -1 for "no confidence"
                    confidences.append(float(conf))

    return recognized_string, confidences

# ================== Number Recognition Classes ==================

class NumberRecognizer:
    def __init__(self, confidence_threshold: int = 80):
        self.confidence_threshold = confidence_threshold

    def recognize_number(
        self,
        frame: np.ndarray,
        roi_coords: Tuple[int, int, int, int],
        expected_threshold: Optional[int] = None
    ) -> Tuple[Optional[int], bool]:
        """
        Recognizes an integer number from the specified region of interest in the frame.

        :param frame: The full webcam frame as a NumPy array (BGR).
        :param roi_coords: Tuple (x, y, w, h) specifying the region-of-interest.
        :param expected_threshold: An integer indicating if recognized value > threshold => Slack alert.

        :return: (recognized_value, is_over_threshold)
                 recognized_value is an integer or None if failed.
                 is_over_threshold is a boolean indicating if the recognized value > expected_threshold.
        """
        processed_image = preprocess_image_for_ocr(frame, roi_coords)
        recognized_string, confidences = extract_text_from_image(processed_image)

        if not recognized_string:
            logging.warning("No valid digits recognized in the ROI.")
            return None, False

        if not confidences:
            logging.warning("Unable to get any confidence values from Tesseract.")
            return None, False

        avg_confidence = np.mean(confidences)
        if avg_confidence < self.confidence_threshold:
            logging.warning(
                f"Low OCR confidence ({avg_confidence:.2f}). "
                f"Raw recognized string: '{recognized_string}'"
            )
            return None, False

        # Parse recognized_string as integer
        try:
            recognized_value = int(recognized_string)
        except ValueError:
            logging.warning(f"Could not parse recognized text '{recognized_string}' as integer.")
            return None, False

        # Check if recognized value is over the threshold
        is_over_threshold = False
        if expected_threshold is not None and recognized_value > expected_threshold:
            is_over_threshold = True

        return recognized_value, is_over_threshold

class AggregatedNumberRecognizer:
    def __init__(self, attempts: int = 5, confidence_threshold: int = 80):
        self.attempts = attempts
        self.confidence_threshold = confidence_threshold
        self.recognizer = NumberRecognizer(confidence_threshold=self.confidence_threshold)

    def recognize_number_aggregated(
        self,
        frame: np.ndarray,
        roi_coords: Tuple[int, int, int, int],
        expected_threshold: Optional[int] = None,
        delay_between_attempts: float = 0.1
    ) -> Tuple[Optional[int], bool]:
        """
        Attempts to recognize a number multiple times and aggregates the results.

        :param frame: The full webcam frame as a NumPy array (BGR).
        :param roi_coords: Tuple (x, y, w, h) specifying the region-of-interest.
        :param expected_threshold: An integer indicating if recognized value > threshold => Slack alert.
        :param delay_between_attempts: Delay between retry attempts in seconds.

        :return: (best_recognized_value, is_any_over_threshold)
        """
        recognized_vals = []
        threshold_hits = 0

        for attempt in range(self.attempts):
            val, over_flag = self.recognizer.recognize_number(
                frame,
                roi_coords,
                expected_threshold=expected_threshold
            )
            if val is not None:
                recognized_vals.append(val)
                if over_flag:
                    threshold_hits += 1
            time.sleep(delay_between_attempts)  # Small delay between attempts

        if not recognized_vals:
            return None, False

        # Determine the most common recognized value
        most_common_val = max(set(recognized_vals), key=recognized_vals.count)

        # Optionally, select the value closest to the most common value
        best_val = min(recognized_vals, key=lambda x: abs(x - most_common_val))

        is_any_over_threshold = (threshold_hits > 0)
        return best_val, is_any_over_threshold

# ================== Slack Notification ==================

def send_slack_notification(message: str) -> None:
    payload = {"text": message}
    try:
        response = requests.post(SLACK_WEBHOOK_URL, json=payload)
        if response.status_code != 200:
            logging.error(
                f'Failed to send Slack notification. '
                f'Status code: {response.status_code}, Response: {response.text}'
            )
        else:
            logging.info('Slack notification sent successfully.')
    except requests.RequestException as e:
        logging.error(f'Error sending Slack notification: {e}')

# ================== Export Info to CSV ==================

def export_log_info(active_cams: List[str], export_file: str = EXPORT_FILE) -> None:
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        with open(export_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if os.stat(export_file).st_size == 0:
                writer.writerow(['Timestamp', 'Active Webcams'])
            writer.writerow([timestamp, '; '.join(active_cams)])
        logging.info(f'Exported active webcams to {export_file} at {timestamp}.')
    except Exception as e:
        logging.error(f'Failed to export active webcams: {e}')

# ================== Periodic Export of Logs ==================

def periodic_export(export_interval: int = 3600, log_file: str = LOG_FILE) -> None:
    while True:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        destination = f'exported_logs/webcam_monitor_{timestamp}.log'
        try:
            os.makedirs('exported_logs', exist_ok=True)
            shutil.copy(log_file, destination)
            logging.info(f'Exported log file to {destination}.')
        except Exception as e:
            logging.error(f'Failed to export log file: {e}')
        time.sleep(export_interval)

# ================== Webcam Checker ==================

class WebcamChecker:
    def __init__(self, webcam: dict, retries: int = 10, timeout: int = 5):
        self.name = webcam['name']
        self.index = webcam['index']
        self.roi = (webcam['roi']['x'], webcam['roi']['y'],
                    webcam['roi']['w'], webcam['roi']['h'])
        self.threshold_value = webcam.get('threshold_value', 50)
        self.retries = retries
        self.timeout = timeout
        self.recognizer = AggregatedNumberRecognizer(attempts=self.retries)

    def check_status(self) -> bool:
        """
        Attempts to open the webcam, read frames, and perform OCR to recognize numbers.
        Returns True if successful, False otherwise.
        """
        for attempt in range(1, self.retries + 1):
            with suppress_stderr():
                cap = cv2.VideoCapture(self.index)

            if not cap.isOpened():
                logging.error(
                    f"{self.name} (Index {self.index}) could not be opened. "
                    f"Attempt {attempt} of {self.retries}."
                )
                time.sleep(1)
                continue

            # Optional: set desired resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            start_time = time.time()

            recognized_value = None
            is_over_threshold = False

            while True:
                ret, frame = cap.read()
                if ret:
                    # Rotate frame if necessary
                    rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                    # Attempt to recognize the number
                    recognized_value, is_over_threshold = self.recognizer.recognize_number_aggregated(
                        rotated_frame,
                        self.roi,
                        expected_threshold=self.threshold_value
                    )

                    if recognized_value is not None:
                        logging.info(
                            f"{self.name} (Index {self.index}): "
                            f"Recognized Number: {recognized_value}"
                        )

                        # If threshold is crossed, send Slack
                        if is_over_threshold:
                            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            message = (
                                f":exclamation: *Threshold Alert*\n"
                                f"At {timestamp}, {self.name} recognized value "
                                f"({recognized_value}) is above threshold."
                            )
                            send_slack_notification(message)

                        cap.release()
                        return True
                    else:
                        # Recognition failed; try next attempt
                        logging.warning(
                            f"{self.name} (Index {self.index}): Number recognition failed on attempt {attempt}."
                        )
                        break
                elif time.time() - start_time > self.timeout:
                    logging.error(f"{self.name} (Index {self.index}) is unresponsive.")
                    break

                time.sleep(0.5)

            cap.release()

        return False

# ================== Main Monitoring Loop ==================

def monitor_webcams():
    logging.info('Starting webcam monitoring service...')
    webcam_checkers = {cam['name']: WebcamChecker(cam) for cam in WEBCAMS}
    webcam_status = {cam['name']: True for cam in WEBCAMS}  # Keep track of last known status

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
                checker = webcam_checkers[cam['name']]
                is_up = checker.check_status()
                if not is_up:
                    failures += 1
                    failed_cams.append(cam['name'])
                    # If previously it was up, notify Slack of failure
                    if webcam_status[cam['name']]:
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        message = (
                            f":warning: *Webcam Alert*\n"
                            f"At {timestamp}, {cam['name']} is not responding or "
                            f"no valid number recognized."
                        )
                        send_slack_notification(message)
                        webcam_status[cam['name']] = False
                else:
                    active_cams.append(cam['name'])
                    # If previously it was down, notify Slack of recovery
                    if not webcam_status[cam['name']]:
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        message = (
                            f":white_check_mark: *Webcam Recovery*\n"
                            f"At {timestamp}, {cam['name']} is responding again with a valid number."
                        )
                        send_slack_notification(message)
                        webcam_status[cam['name']] = True
                time.sleep(1)

            # Logging status
            failure_percentage = (failures / total_cams) * 100 if total_cams else 0
            logging.info(f'Webcam Check: {failures}/{total_cams} failures ({failure_percentage:.2f}%)')
            logging.info(f'Active Webcams: {", ".join(active_cams)}')

            # Export info if exactly SPECIFIC_ACTIVE_NUMBER cameras are active
            if len(active_cams) == SPECIFIC_ACTIVE_NUMBER:
                export_log_info(active_cams, EXPORT_FILE)

            # If failure threshold met, send an aggregate alert
            if failure_percentage >= FAIL_THRESHOLD_PERCENT:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                failed_cam_names = ', '.join(failed_cams)
                message = (
                    f":warning: *Webcam Alert*\n"
                    f"At {timestamp}, {failures} out of {total_cams} webcams "
                    f"are failing ({failure_percentage:.2f}%).\n"
                    f"Failed webcams: {failed_cam_names}"
                )
                send_slack_notification(message)
            else:
                logging.info('Failure threshold not met. No aggregate notification sent.')

        logging.info(f'Waiting for {CHECK_INTERVAL} seconds before next check...\n')
        time.sleep(CHECK_INTERVAL)

# ================== Start Periodic Export in Separate Thread ==================

def start_periodic_export():
    export_thread = threading.Thread(target=periodic_export, args=(3600,), daemon=True)  # Export logs every hour
    export_thread.start()

# ================== Entry Point ==================

if __name__ == '__main__':
    start_periodic_export()
    try:
        monitor_webcams()
    except KeyboardInterrupt:
        logging.info('Webcam monitoring stopped manually.')
    except Exception as e:
        logging.exception(f'An unexpected error occurred: {e}')
