import pyrealsense2 as rs
import numpy as np
import cv2
import time

def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    print("[*] Starting RealSense D435i RGB + Depth pipeline...")
    pipeline.start(config)

    colorizer = rs.colorizer()
    frame_count = 0
    counter_start = time.perf_counter()
    capture_fps = 0.0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())

            frame_count += 1
            now = time.perf_counter()
            elapsed = now - counter_start
            if elapsed >= 1.0:
                capture_fps = frame_count / elapsed
                frame_count = 0
                counter_start = now

            cv2.putText(
                color_image,
                f"RealSense SDK Capture: {capture_fps:.1f} FPS",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            combined = np.hstack((color_image, depth_image))
            cv2.imshow("RealSense D435i (Color | Depth Colormap)", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
