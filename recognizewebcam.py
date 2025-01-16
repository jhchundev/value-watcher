import cv2

def list_webcams(start=0, end=10):
    print("Scanning for connected webcams...")
    for index in range(start, end):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Webcam found at device index {index}.")
            ret, frame = cap.read()
            if ret:
                cv2.imshow(f'Webcam {index}', frame)
                print(f"Press any key to close Webcam {index} view.")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            cap.release()
        else:
            print(f"No webcam found at device index {index}.")

if __name__ == "__main__":
    list_webcams()
