import cv2
import sys

def test_camera(index=0):
    print(f"[*] Opening camera at index {index}...")
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print(f"[!] Unable to open camera at index {index}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Failed to grab frame.")
            break

        cv2.putText(frame, f"Intel FHD Camera [Index {index}] - Press Q to exit", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Intel FHD Machine Vision Camera Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    test_camera(idx)
