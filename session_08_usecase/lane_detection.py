import cv2
import numpy as np

class LaneDetector:
    def __init__(self):
        pass

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        lower_white = np.array([0, 200, 0], dtype=np.uint8)
        upper_white = np.array([255, 255, 255], dtype=np.uint8)
        white_mask = cv2.inRange(hls, lower_white, upper_white)
        
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        combined_edges = cv2.bitwise_or(edges, white_mask)

        mask = np.zeros_like(combined_edges)
        roi_pts = np.array([[
            (int(w * 0.05), h),
            (int(w * 0.42), int(h * 0.58)),
            (int(w * 0.58), int(h * 0.58)),
            (int(w * 0.95), h)
        ]], dtype=np.int32)
        cv2.fillPoly(mask, roi_pts, 255)
        roi_edges = cv2.bitwise_and(combined_edges, mask)

        lines = cv2.HoughLinesP(roi_edges, 1, np.pi/180, threshold=30, minLineLength=40, maxLineGap=120)
        lane_overlay = np.zeros_like(frame)

        left_lines, right_lines = [], []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if slope < -0.3:
                    left_lines.append((x1, y1, x2, y2))
                elif slope > 0.3:
                    right_lines.append((x1, y1, x2, y2))

        for x1, y1, x2, y2 in left_lines + right_lines:
            cv2.line(lane_overlay, (x1, y1), (x2, y2), (0, 255, 0), 4)

        result = cv2.addWeighted(frame, 0.85, lane_overlay, 1.0, 0)
        cv2.putText(result, "Intel FHD Camera: Lane Tracking Active", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return result

if __name__ == "__main__":
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    detector = LaneDetector()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        output = detector.process_frame(frame)
        cv2.imshow("Lane Detection", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
