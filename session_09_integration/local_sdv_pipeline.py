import threading
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

state = {
    "realsense_frame": None,
    "fhd_frame": None,
    "running": True,
    "fps_rs": 0.0,
    "fps_fhd": 0.0
}

def auto_detect_fhd_camera():
    for i in range(12):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                return i
    return 0

def realsense_thread(model_path):
    # Removed specific device flag to allow OpenVINO to auto-optimize threads
    model = YOLO(model_path, task="detect")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    align = rs.align(rs.stream.color)
    pipeline.start(config)

    frame_count = 0
    t0 = time.time()

    try:
        while state["running"]:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                continue

            aligned = align.process(frames)
            c_frame = aligned.get_color_frame()
            d_frame = aligned.get_depth_frame()
            if not c_frame or not d_frame:
                continue

            img = np.asanyarray(c_frame.get_data())
            results = model.predict(source=img, imgsz=640, verbose=False)
            
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                dist = d_frame.get_distance(cx, cy)
                cls_name = model.names[int(box.cls[0])]
                
                c = (0, 0, 255) if 0.0 < dist < 1.2 else (0, 255, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
                cv2.putText(img, f"{cls_name} {dist:.2f}m", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)

            frame_count += 1
            if time.time() - t0 >= 1.0:
                state["fps_rs"] = frame_count / (time.time() - t0)
                frame_count = 0
                t0 = time.time()

            state["realsense_frame"] = img
            time.sleep(0.01)  # Micro-sleep prevents thread locking
    finally:
        pipeline.stop()

def lane_fhd_thread(cam_index):
    cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_count = 0
    t0 = time.time()

    while state["running"] and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # OPTIMIZATION: Massive blur to destroy noise/shirt patterns
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        # OPTIMIZATION: Stricter edge detection
        edges = cv2.Canny(blur, 100, 200)
        
        mask = np.zeros_like(edges)
        poly = np.array([[(int(w*0.1), h), (int(w*0.45), int(h*0.6)), (int(w*0.55), int(h*0.6)), (int(w*0.9), h)]], np.int32)
        cv2.fillPoly(mask, poly, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # OPTIMIZATION: Stricter line generation math
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=60, minLineLength=50, maxLineGap=40)
        overlay = np.zeros_like(frame)
        
        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l.flatten()[:4]
                cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 4)
        
        blended = cv2.addWeighted(frame, 0.8, overlay, 1.0, 0)
        
        frame_count += 1
        if time.time() - t0 >= 1.0:
            state["fps_fhd"] = frame_count / (time.time() - t0)
            frame_count = 0
            t0 = time.time()

        state["fhd_frame"] = blended
        time.sleep(0.01)  # Micro-sleep prevents thread locking
    cap.release()

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/best_openvino_model/"
    dynamic_fhd_idx = auto_detect_fhd_camera()

    t1 = threading.Thread(target=realsense_thread, args=(model_path,))
    t2 = threading.Thread(target=lane_fhd_thread, args=(dynamic_fhd_idx,))
    t1.start()
    t2.start()

    try:
        while True:
            f_rs = state["realsense_frame"]
            f_lane = state["fhd_frame"]

            if f_rs is not None and f_lane is not None:
                r1 = cv2.resize(f_rs, (640, 480))
                r2 = cv2.resize(f_lane, (640, 480))

                cv2.putText(r1, f"RealSense Detection (FPS: {state['fps_rs']:.1f})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(r2, f"FHD Lane Detection (FPS: {state['fps_fhd']:.1f})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                combined = np.hstack((r1, r2))
                cv2.imshow("Intel UP Square 6000 - Local SDV Dashboard", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        state["running"] = False
        t1.join()
        t2.join()
        cv2.destroyAllWindows()
