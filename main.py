import cv2
import requests
import time
import logging
from datetime import datetime

# ================== Configuration ==================

# Range of device indices to scan for webcams
# Adjust START_INDEX and END_INDEX based on your system
START_INDEX = 0
END_INDEX = 10  # Scan device indices from 0 to 9

# Slack webhook URL
SLACK_WEBHOOK_URL = 'https://hooks.slack.com/services/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'

# Monitoring parameters
CHECK_INTERVAL = 10  # Time between checks in seconds
FAIL_THRESHOLD_PERCENT = 150  # Percentage to trigger Slack notification

# Logging configuration
LOG_FILE = 'local_webcam_monitor.log'

# ================== Logging Setup ==================

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ================== Functions ==================

def detect_webcams(start=0, end=10):
    """
    Detects connected webcams by attempting to open device indices.

    Args:
        start (int): Starting device index.
        end (int): Ending device index (exclusive).

    Returns:
        list: List of accessible device indices.
    """
    available_cams = []
    logging.info('Detecting connected webcams...')
    for index in range(start, end):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            logging.info(f'Webcam found at device index {index}.')
            available_cams.append(index)
            cap.release()
        else:
            logging.debug(f'No webcam found at device index {index}.')
    logging.info(f'Total webcams detected: {len(available_cams)}')
    return available_cams

def check_webcam_status(index, timeout=5):
    """
    Checks if a specific webcam is accessible and functional.

    Args:
        index (int): Device index of the webcam.
        timeout (int): Time in seconds to wait for the webcam.

    Returns:
        bool: True if webcam is accessible, False otherwise.
    """
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    start_time = time.time()
    if not cap.isOpened():
        logging.error(f'Webcam at device index {index} could not be opened.')
        return False
    while True:
        ret, frame = cap.read()
        if ret:
            logging.debug(f'Webcam at index {index} is functioning.')
            cap.release()
            return True
        elif time.time() - start_time > timeout:
            logging.error(f'Webcam at device index {index} is unresponsive.')
            cap.release()
            return False
        time.sleep(0.5)

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

def monitor_webcams():
    """
    Main function to monitor webcams and send notifications based on failure thresholds.
    """
    logging.info('Starting webcam monitoring service...')
    while True:
        available_cams = detect_webcams(START_INDEX, END_INDEX)
        total_cams = len(available_cams)
        failures = 0
        failed_cams = []

        logging.info('Checking status of each webcam...')
        for cam_index in available_cams:
            if not check_webcam_status(cam_index):
                failures += 1
                failed_cams.append(cam_index)
            time.sleep(1)  # Brief pause between checks

        if total_cams == 0:
            logging.warning('No webcams detected. Skipping this monitoring cycle.')
        else:
            failure_percentage = (failures / total_cams) * 100
            logging.info(f'Webcam Check: {failures}/{total_cams} failures ({failure_percentage:.2f}%)')

            if failure_percentage >= FAIL_THRESHOLD_PERCENT:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                failed_cam_indices = ', '.join(map(str, failed_cams))
                message = (
                    f":warning: *Webcam Alert*\n"
                    f"At {timestamp}, {failures} out of {total_cams} webcams are not responding "
                    f"({failure_percentage:.2f}%).\n"
                    f"Failed webcam device indices: {failed_cam_indices}"
                )
                send_slack_notification(message)
            else:
                logging.info('Failure threshold not met. No notification sent.')

        logging.info(f'Waiting for {CHECK_INTERVAL} seconds before next check...\n')
        time.sleep(CHECK_INTERVAL)

# ================== Entry Point ==================

if __name__ == '__main__':
    try:
        monitor_webcams()
    except KeyboardInterrupt:
        logging.info('Webcam monitoring stopped manually.')
    except Exception as e:
        logging.exception(f'An unexpected error occurred: {e}')
