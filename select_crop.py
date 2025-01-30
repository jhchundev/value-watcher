import cv2
import json
import os

def load_config(config_file='config.json'):
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    else:
        return {}

def save_config(config, config_file='config.json'):
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)

def select_roi(cam):
    cap = cv2.VideoCapture(cam['index'])
    if not cap.isOpened():
        print(f"Error: Could not open webcam {cam['name']} (Index {cam['index']}).")
        return None

    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read frame from webcam {cam['name']} (Index {cam['index']}).")
        cap.release()
        return None

    # Rotate the frame 90 degrees to the left (counter-clockwise)
    rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Optionally resize the rotated frame for easier selection
    frame_display = cv2.resize(rotated_frame, (1000, 800))
    cv2.imshow(f"Select ROI for {cam['name']}", frame_display)

    print(f"Select ROI for {cam['name']} and press ENTER or SPACE. Press 'c' to cancel.")
    roi = cv2.selectROI(f"Select ROI for {cam['name']}", frame_display, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    cap.release()

    if roi[2] == 0 or roi[3] == 0:
        print(f"No ROI selected for {cam['name']}. Skipping.")
        return None

    # Adjust ROI coordinates if the frame was resized
    scale_x = rotated_frame.shape[1] / 800
    scale_y = rotated_frame.shape[0] / 600
    x, y, w, h = roi
    x = int(x * scale_x)
    y = int(y * scale_y)
    w = int(w * scale_x)
    h = int(h * scale_y)

    print(f"Selected ROI for {cam['name']}: x={x}, y={y}, w={w}, h={h}")
    return {'x': x, 'y': y, 'w': w, 'h': h}

def main():
    config_file = 'config.json'
    config = load_config(config_file)

    if 'webcams' not in config or not config['webcams']:
        print("No webcams configured in config.json.")
        return

    for cam in config['webcams']:
        print(f"\nConfiguring ROI for webcam: {cam['name']} (Index {cam['index']})")
        roi = select_roi(cam)
        if roi:
            cam['roi'] = roi
            print(f"ROI for {cam['name']} set to: {roi}")
        else:
            print(f"ROI for {cam['name']} not set.")

    save_config(config, config_file)
    print("\nAll ROIs have been configured and saved to config.json.")

if __name__ == "__main__":
    main()
