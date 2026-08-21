import pyrealsense2 as rs
import numpy as np
import cv2

class RealSenseSensorFusion:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        self.align = rs.align(rs.stream.color)
        self.pipeline.start(self.config)
        self.colorizer = rs.colorizer()

    def get_aligned_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            return None, None, None
            
        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_frame, color_frame

    def get_distance_at_point(self, depth_frame, x, y):
        return depth_frame.get_distance(int(x), int(y))

    def stop(self):
        self.pipeline.stop()

if __name__ == "__main__":
    fusion = RealSenseSensorFusion()
    try:
        while True:
            color_img, depth_frame, _ = fusion.get_aligned_frames()
            if color_img is None:
                continue
                
            h, w = color_img.shape[:2]
            cx, cy = w // 2, h // 2
            dist = fusion.get_distance_at_point(depth_frame, cx, cy)
            
            cv2.circle(color_img, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(color_img, f"Center Distance: {dist:.2f} m", (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
            cv2.imshow("RealSense Aligned RGB-Depth Fusion", color_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        fusion.stop()
        cv2.destroyAllWindows()
