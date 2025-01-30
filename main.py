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
LOG_ROTATION_MAX_BYTES = config.get("log_rotation_max_bytes", 5*1024*1024)
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

# ================== Main Number Recognition Function ==================

def recognize_number(frame, roi_coords, expected_threshold=None, confidence_threshold=80):
    """
    Recognizes a (yellow) number with optional decimal points from a black background.

    :param frame:           The full webcam frame as a NumPy array (BGR).
    :param roi_coords:      Tuple (x, y, w, h) specifying the region-of-interest.
    :param expected_threshold: A float indicating if recognized value > threshold => Slack alert.
    :param confidence_threshold: Minimum average confidence (0-100) required from Tesseract.

    :return: (recognized_value, is_over_threshold)
             recognized_value is a float or None if failed.
             is_over_threshold is a boolean indicating if the recognized value > expected_threshold.
    """
    x, y, w, h = roi_coords
    roi = frame[y:y+h, x:x+w]

    # Convert to HSV for easier color segmentation
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define a mask range for "yellow" text. Adjust as necessary.
    # Hue range for yellow is typically around 20-35, but fine-tune
    YELLOW_LOWER = np.array([20, 100, 100])
    YELLOW_UPPER = np.array([35, 255, 255])

    mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # We have a mask isolating the yellow text on black background.
    # Convert masked area back to grayscale
    masked_roi = cv2.bitwise_and(roi, roi, mask=mask)
    gray = cv2.cvtColor(masked_roi, cv2.COLOR_BGR2GRAY)

    # In many black background + colored text scenarios, Otsu's threshold often works well:
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Additional morphological steps to remove small specks
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Resize for better OCR accuracy
    # Increase factor if text is small
    processed = cv2.resize(morph, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)

    # Tesseract config: psm=8 (single word/line), whitelisting digits and the decimal point
    custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789'
    data = pytesseract.image_to_data(processed, config=custom_config, output_type=pytesseract.Output.DICT)

    recognized_string = ""
    confidences = []

    for i, text_val in enumerate(data['text']):
        if text_val.strip():
            # Filter out clearly invalid tokens (like leftover letters or weird symbols)
            # Since we used whitelist, mostly digits and '.' come through, but let's be safe
            valid_chars = [ch for ch in text_val if ch.isdigit() or ch == '.']
            filtered_text = "".join(valid_chars)
            if filtered_text:
                recognized_string += filtered_text
                conf = data['conf'][i]
                if conf != -1:  # Tesseract can give -1 for "no confidence"
                    confidences.append(float(conf))

    if not recognized_string:
        logging.warning("No valid digits recognized in the ROI.")
        return None, False

    if not confidences:  # Means Tesseract returned -1 or no bounding box
        logging.warning("Unable to get any confidence values from Tesseract.")
        return None, False

    avg_confidence = np.mean(confidences)
    if avg_confidence < confidence_threshold:
        logging.warning(
            f"Low OCR confidence ({avg_confidence:.2f}). "
            f"Raw recognized string: '{recognized_string}'"
        )
        return None, False

    # Now parse recognized_string as float
    try:
        recognized_value = float(recognized_string)
    except ValueError:
        logging.warning(f"Could not parse recognized text '{recognized_string}' as float.")
        return None, False

    # Check if recognized value is over the threshold
    is_over_threshold = False
    if expected_threshold is not None and recognized_value > expected_threshold:
        is_over_threshold = True

    return recognized_value, is_over_threshold

# ================== Aggregated Recognition ==================

def recognize_number_aggregated(frame, roi_coords, expected_threshold=None, attempts=3):
    """
    Tries multiple times to recognize a number from the same frame region.
    If successful in any attempt, returns the most common recognized float.
    Also returns True if it crosses the threshold in any attempt.

    :param frame:            The frame.
    :param roi_coords:       (x, y, w, h).
    :param expected_threshold: If recognized value > threshold => Slack.
    :param attempts:         Number of attempts to do OCR.

    :return: (final_value, is_over_threshold)
    """
    recognized_vals = []
    threshold_hits = 0

    for _ in range(attempts):
        val, over_flag = recognize_number(frame, roi_coords, expected_threshold)
        if val is not None:
            recognized_vals.append(val)
            if over_flag:
                threshold_hits += 1
        time.sleep(0.1)  # small delay

    if not recognized_vals:
        return None, False

    # Get the most common recognized float (simple mode approach)
    # Because floats can be slightly different, we can round them or
    # just pick the last recognized. If you want "true" majority voting,
    # consider rounding to 1-2 decimals for grouping.
    # Example: round to 1 decimal
    recognized_rounded = [round(x, 1) for x in recognized_vals]
    most_common_rounded = max(set(recognized_rounded), key=recognized_rounded.count)

    # If you want the *closest float* from recognized_vals to that mode, pick it:
    best_val = None
    min_diff = float('inf')
    for rv in recognized_vals:
        if abs(round(rv, 1) - most_common_rounded) < min_diff:
            min_diff = abs(round(rv, 1) - most_common_rounded)
            best_val = rv

    # If any attempt indicated it's over the threshold => report True
    is_any_over_threshold = (threshold_hits > 0)
    return best_val, is_any_over_threshold

# ================== Slack Notification ==================

def send_slack_notification(message):
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

def export_log_info(active_cams, export_file='active_webcams.csv'):
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

def periodic_export(export_interval=3600):
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

# ================== Webcam Check ==================

def check_webcam_status(cam, timeout=5, retries=3):
    """
    Attempt to open a webcam multiple times, read frames, and run OCR to see if
    a valid number is recognized. If recognized, also check if it is above the threshold.
    """
    for attempt in range(retries):
        with suppress_stderr():
            cap = cv2.VideoCapture(cam['index'])

        if not cap.isOpened():
            logging.error(
                f"{cam['name']} (Index {cam['index']}) could not be opened. "
                f"Attempt {attempt + 1} of {retries}."
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
                # If your camera orientation is rotated, rotate if needed:
                # e.g., rotate 90 deg CCW
                rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                # Attempt to recognize the number multiple times
                recognized_value, is_over_threshold = recognize_number_aggregated(
                    rotated_frame,
                    (cam['roi']['x'], cam['roi']['y'], cam['roi']['w'], cam['roi']['h']),
                    expected_threshold=cam.get('threshold_value', 50),
                    attempts=3
                )

                if recognized_value is not None:
                    logging.info(
                        f"{cam['name']} (Index {cam['index']}): "
                        f"Recognized Number: {recognized_value}"
                    )

                    # If threshold is crossed, send Slack
                    if is_over_threshold:
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        message = (
                            f":exclamation: *Threshold Alert*\n"
                            f"At {timestamp}, {cam['name']} recognized value "
                            f"({recognized_value}) is above threshold."
                        )
                        send_slack_notification(message)

                    cap.release()
                    return True
                else:
                    # We recognized *nothing*, break to try next attempt.
                    logging.warning(
                        f"{cam['name']} (Index {cam['index']}): Number recognition failed this round."
                    )
                    break
            elif time.time() - start_time > timeout:
                logging.error(f"{cam['name']} (Index {cam['index']}) is unresponsive.")
                cap.release()
                break

            time.sleep(0.5)

        cap.release()
    return False

# ================== Main Monitoring Loop ==================

def monitor_webcams():
    logging.info('Starting webcam monitoring service...')
    WEBCAM_STATUS = {cam['name']: True for cam in WEBCAMS}  # Keep track of last known status

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
                    # If previously it was up, now we notify Slack that it failed
                    if WEBCAM_STATUS[cam['name']]:
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        message = (
                            f":warning: *Webcam Alert*\n"
                            f"At {timestamp}, {cam['name']} is not responding or "
                            f"no valid number recognized."
                        )
                        send_slack_notification(message)
                        WEBCAM_STATUS[cam['name']] = False
                else:
                    active_cams.append(cam['name'])
                    if not WEBCAM_STATUS[cam['name']]:
                        # Recovery
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        message = (
                            f":white_check_mark: *Webcam Recovery*\n"
                            f"At {timestamp}, {cam['name']} is responding again with valid number."
                        )
                        send_slack_notification(message)
                        WEBCAM_STATUS[cam['name']] = True
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
export_thread = threading.Thread(target=periodic_export, args=(3600,), daemon=True)  # Export logs every hour
export_thread.start()

# ================== Entry Point ==================
if __name__ == '__main__':
    try:
        monitor_webcams()
    except KeyboardInterrupt:
        logging.info('Webcam monitoring stopped manually.')
    except Exception as e:
        logging.exception(f'An unexpected error occurred: {e}')
