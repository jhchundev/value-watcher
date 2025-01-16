import cv2

def capture_sample_frame(cam_index=0):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"Cannot open webcam at index {cam_index}")
        return

    print("Press 's' to save the sample frame or 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow('Sample Frame - Press s to save', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            cv2.imwrite('sample_frame.jpg', frame)
            print("Sample frame saved as 'sample_frame.jpg'")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_sample_frame()
