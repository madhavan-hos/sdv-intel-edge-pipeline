import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

class ObstacleDetector:
    def __init__(self, model_path="models/best_openvino_model/"):
        self.model = YOLO(model_path, task="detect")
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.align = rs.align(rs.stream.color)
        self.pipeline.start(self.config)

    def process_next_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            return None

        color_img = np.asanyarray(color_frame.get_data())
        
        results = self.model.predict(source=color_img, imgsz=640, device="intel:cpu", verbose=False)
        boxes = results[0].boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = self.model.names[cls_id]

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            distance = depth_frame.get_distance(cx, cy)

            color = (0, 255, 0)
            status = "CLEAR"
            if 0.0 < distance < 1.0:
                color = (0, 0, 255)
                status = "COLLISION WARNING"
            elif 1.0 <= distance < 2.5:
                color = (0, 255, 255)
                status = "CAUTION"

            cv2.rectangle(color_img, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {conf:.2f} | {distance:.2f}m [{status}]"
            cv2.putText(color_img, label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(color_img, (cx, cy), 4, color, -1)

        return color_img

    def stop(self):
        self.pipeline.stop()

if __name__ == "__main__":
    import sys
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "models/best_openvino_model/"
    det = ObstacleDetector(model_dir)
    try:
        while True:
            frame = det.process_next_frame()
            if frame is not None:
                cv2.imshow("SDV Smart Obstacle Distance Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        det.stop()
        cv2.destroyAllWindows()
