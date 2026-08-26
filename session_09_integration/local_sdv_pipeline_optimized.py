"""Responsive version of the local SDV integration pipeline.

The original local_sdv_pipeline.py is intentionally left unchanged.  This
version keeps camera acquisition independent from OpenVINO/YOLO inference and
always processes the newest RealSense frame instead of building a stale queue.
"""

import argparse
import os
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO


PIPELINE_VERSION = "2026.08.26-lane-v6"


state = {
    "running": True,
    "realsense_frame": None,
    "realsense_packet": None,
    "realsense_sequence": 0,
    "detections": [],
    "fhd_frame": None,
    "fps_rs_capture": 0.0,
    "fps_rs_inference": 0.0,
    "fps_fhd": 0.0,
    "inference_ms": 0.0,
}
state_lock = threading.Lock()
new_realsense_frame = threading.Condition(state_lock)


def set_thread_affinity(role):
    """Reserve CPU time for camera capture on Linux edge hardware."""
    if not hasattr(os, "sched_setaffinity"):
        return

    available = sorted(os.sched_getaffinity(0))
    if len(available) < 4:
        return

    if role == "realsense":
        selected = {available[0]}
    elif role == "fhd":
        selected = {available[1]}
    elif role == "inference":
        selected = set(available[2:])
    else:
        return

    try:
        # pid=0 means the calling Linux thread. OpenVINO workers created after
        # this call inherit the inference thread's restricted CPU set.
        os.sched_setaffinity(0, selected)
        print(f"[*] {role} CPU affinity: {sorted(selected)}")
    except OSError as exc:
        print(f"[!] Unable to set {role} CPU affinity: {exc}")


def linux_fhd_candidates():
    """Return V4L indices while explicitly excluding RealSense video nodes."""
    candidates = []
    video_root = Path("/sys/class/video4linux")
    if not video_root.exists():
        return candidates

    for device in video_root.glob("video*"):
        try:
            # str.removeprefix() was added in Python 3.9. The UP Square
            # workshop image uses Python 3.8, so parse the known prefix using
            # slicing instead.
            index = int(device.name[len("video"):])
            device_name = (device / "name").read_text().strip()
        except (OSError, ValueError):
            continue

        if "realsense" in device_name.lower():
            continue

        # Prefer the workshop FHD camera, then generic USB cameras.
        lowered = device_name.lower()
        priority = 0 if "fhd" in lowered else 1 if "usb" in lowered else 2
        candidates.append((priority, index, device_name))

    return sorted(candidates)


def auto_detect_fhd_camera():
    candidates = linux_fhd_candidates()
    indices = [index for _, index, _ in candidates]
    if not indices:
        indices = list(range(12))

    for i in indices:
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                matched = next((name for _, idx, name in candidates if idx == i), "unknown")
                print(f"[*] Auto-detected FHD candidate /dev/video{i}: {matched}")
                return i
    return 0


def realsense_capture_thread(target_fps):
    set_thread_affinity("realsense")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, target_fps)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, target_fps)
    align = rs.align(rs.stream.color)
    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    frame_count = 0
    counter_start = time.perf_counter()

    try:
        while state["running"]:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                continue

            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # Copy the SDK buffers because librealsense reuses them after the
            # next capture.  The inference worker can then safely run later.
            color_image = np.asanyarray(color_frame.get_data()).copy()
            depth_image = np.asanyarray(depth_frame.get_data()).copy()

            frame_count += 1
            now = time.perf_counter()
            elapsed = now - counter_start

            with new_realsense_frame:
                state["realsense_sequence"] += 1
                state["realsense_frame"] = color_image
                state["realsense_packet"] = (
                    state["realsense_sequence"],
                    color_image,
                    depth_image,
                    depth_scale,
                )
                if elapsed >= 1.0:
                    state["fps_rs_capture"] = frame_count / elapsed
                    frame_count = 0
                    counter_start = now
                new_realsense_frame.notify()
    finally:
        pipeline.stop()


def robust_depth_distance(depth_image, depth_scale, x1, y1, x2, y2):
    """Return the median valid depth near a detection's center."""
    height, width = depth_image.shape[:2]
    cx = min(max((x1 + x2) // 2, 0), width - 1)
    cy = min(max((y1 + y2) // 2, 0), height - 1)
    radius = max(2, min(abs(x2 - x1), abs(y2 - y1)) // 10)
    sample = depth_image[
        max(0, cy - radius):min(height, cy + radius + 1),
        max(0, cx - radius):min(width, cx + radius + 1),
    ]
    valid = sample[sample > 0]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid)) * depth_scale


def realsense_inference_thread(model_path, confidence, iou):
    set_thread_affinity("inference")
    model = YOLO(model_path, task="detect")
    print("[*] Inference location: LOCAL UP Square OpenVINO CPU")
    print(f"[*] Model classes: {model.names}")
    print(f"[*] Detection thresholds: confidence={confidence:.2f}, IoU={iou:.2f}")
    last_sequence = -1
    frame_count = 0
    counter_start = time.perf_counter()

    while state["running"]:
        with new_realsense_frame:
            new_realsense_frame.wait_for(
                lambda: not state["running"]
                or (
                    state["realsense_packet"] is not None
                    and state["realsense_packet"][0] != last_sequence
                ),
                timeout=1.0,
            )
            if not state["running"]:
                break
            sequence, color_image, depth_image, depth_scale = state["realsense_packet"]
            last_sequence = sequence

        inference_start = time.perf_counter()
        results = model.predict(
            source=color_image,
            imgsz=640,
            conf=confidence,
            iou=iou,
            verbose=False,
        )
        inference_ms = (time.perf_counter() - inference_start) * 1000.0

        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            distance = robust_depth_distance(
                depth_image, depth_scale, x1, y1, x2, y2
            )
            class_name = model.names[int(box.cls[0])]
            confidence_score = float(box.conf[0])

            if 0.0 < distance < 1.2:
                color = (0, 0, 255)
                status = "COLLISION WARNING"
            elif 1.2 <= distance < 2.5:
                color = (0, 255, 255)
                status = "CAUTION"
            else:
                color = (0, 255, 0)
                status = "CLEAR"

            detections.append(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                    class_name,
                    confidence_score,
                    distance,
                    status,
                    color,
                )
            )

        frame_count += 1
        now = time.perf_counter()
        elapsed = now - counter_start
        with state_lock:
            state["detections"] = detections
            state["inference_ms"] = inference_ms
            if elapsed >= 1.0:
                state["fps_rs_inference"] = frame_count / elapsed
                frame_count = 0
                counter_start = now


def fit_lane_boundary(segments, y_top, y_bottom, frame_width):
    """Fit one stable boundary through all accepted Hough segments."""
    if not segments:
        return None

    points = np.array(
        [(x1, y1) for x1, y1, _, _ in segments]
        + [(x2, y2) for _, _, x2, y2 in segments],
        dtype=np.float32,
    )
    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    if abs(vy) < 1e-6:
        return None

    x_top = int(x0 + (y_top - y0) * vx / vy)
    x_bottom = int(x0 + (y_bottom - y0) * vx / vy)
    x_top = min(max(x_top, 0), frame_width - 1)
    x_bottom = min(max(x_bottom, 0), frame_width - 1)
    return (x_bottom, y_bottom, x_top, y_top)


def smooth_lane(previous, current, alpha=0.35):
    if current is None:
        return previous
    if previous is None:
        return current
    return tuple(
        int((1.0 - alpha) * old + alpha * new)
        for old, new in zip(previous, current)
    )


def detect_lane_boundaries(frame):
    """Detect left/right lane boundaries while rejecting horizontal edges."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # A small blur suppresses sensor noise without erasing thin pen/tape lanes.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 45, 135)

    # Preserve common workshop markings (blue pen/tape and yellow tape). Road
    # lane markings of other colors are still detected by their strong edges.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(
        hsv,
        np.array([90, 35, 35], dtype=np.uint8),
        np.array([140, 255, 255], dtype=np.uint8),
    )
    yellow = cv2.inRange(
        hsv,
        np.array([15, 60, 60], dtype=np.uint8),
        np.array([40, 255, 255], dtype=np.uint8),
    )
    colored_markings = cv2.bitwise_or(blue, yellow)
    colored_markings = cv2.morphologyEx(
        colored_markings,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
    )
    candidates = cv2.bitwise_or(edges, colored_markings)

    y_top = int(h * 0.35)
    y_bottom = h - 1
    roi = np.zeros_like(candidates)
    polygon = np.array(
        [[
            (int(w * 0.02), y_bottom),
            (int(w * 0.30), y_top),
            (int(w * 0.70), y_top),
            (int(w * 0.98), y_bottom),
        ]],
        dtype=np.int32,
    )
    cv2.fillPoly(roi, polygon, 255)
    candidates = cv2.bitwise_and(candidates, roi)

    lines = cv2.HoughLinesP(
        candidates,
        1,
        np.pi / 180,
        threshold=22,
        minLineLength=25,
        maxLineGap=90,
    )

    left_segments = []
    right_segments = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = map(int, line.flatten()[:4])
            dx = x2 - x1
            dy = y2 - y1
            midpoint_x = (x1 + x2) / 2.0

            # Paper/table edges are almost horizontal; real perspective lane
            # boundaries have a meaningful vertical component.
            if abs(dy) < 0.35 * max(abs(dx), 1):
                continue

            if abs(dx) <= 2:
                if midpoint_x < w * 0.5:
                    left_segments.append((x1, y1, x2, y2))
                else:
                    right_segments.append((x1, y1, x2, y2))
                continue

            slope = dy / dx
            if slope < -0.35 and midpoint_x < w * 0.68:
                left_segments.append((x1, y1, x2, y2))
            elif slope > 0.35 and midpoint_x > w * 0.32:
                right_segments.append((x1, y1, x2, y2))

    return (
        fit_lane_boundary(left_segments, y_top, y_bottom, w),
        fit_lane_boundary(right_segments, y_top, y_bottom, w),
    )


def lane_fhd_thread(cam_index):
    set_thread_affinity("fhd")
    cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_count = 0
    counter_start = time.perf_counter()
    previous_left = None
    previous_right = None
    left_misses = 0
    right_misses = 0

    while state["running"] and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        detected_left, detected_right = detect_lane_boundaries(frame)
        if detected_left is None:
            left_misses += 1
            if left_misses > 5:
                previous_left = None
        else:
            left_misses = 0
            previous_left = smooth_lane(previous_left, detected_left)

        if detected_right is None:
            right_misses += 1
            if right_misses > 5:
                previous_right = None
        else:
            right_misses = 0
            previous_right = smooth_lane(previous_right, detected_right)

        overlay = np.zeros_like(frame)
        if previous_left is not None:
            cv2.line(
                overlay,
                previous_left[:2],
                previous_left[2:],
                (0, 255, 0),
                5,
            )
        if previous_right is not None:
            cv2.line(
                overlay,
                previous_right[:2],
                previous_right[2:],
                (0, 255, 0),
                5,
            )

        blended = cv2.addWeighted(frame, 0.8, overlay, 1.0, 0)
        if previous_left is not None and previous_right is not None:
            lane_status = "LANE BOUNDARIES DETECTED"
            status_color = (0, 255, 0)
        elif previous_left is not None or previous_right is not None:
            lane_status = "ONE LANE BOUNDARY DETECTED"
            status_color = (0, 255, 255)
        else:
            lane_status = "SEARCHING FOR LANE BOUNDARIES"
            status_color = (0, 165, 255)
        cv2.putText(
            blended,
            lane_status,
            (10, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            status_color,
            2,
        )
        frame_count += 1
        now = time.perf_counter()
        elapsed = now - counter_start

        with state_lock:
            state["fhd_frame"] = blended
            if elapsed >= 1.0:
                state["fps_fhd"] = frame_count / elapsed
                frame_count = 0
                counter_start = now

    cap.release()


def draw_detections(frame, detections):
    for (
        x1,
        y1,
        x2,
        y2,
        class_name,
        confidence,
        distance,
        status,
        color,
    ) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{class_name} {confidence:.2f} | {distance:.2f}m [{status}]",
            (x1, max(y1 - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Responsive local SDV dashboard")
    parser.add_argument(
        "model_path",
        nargs="?",
        default="models/best_openvino_model/",
        help="OpenVINO model directory",
    )
    parser.add_argument(
        "--fhd-index",
        type=int,
        default=None,
        help="Use a known FHD camera index instead of automatic detection",
    )
    parser.add_argument(
        "--rs-fps",
        type=int,
        choices=(15, 30),
        default=15,
        help="Requested RealSense color/depth FPS (default: 15)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum object confidence from 0.0 to 1.0 (default: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="Non-maximum-suppression IoU threshold (default: 0.45)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        raise SystemExit("--confidence must be between 0.0 and 1.0")
    if not 0.0 <= args.iou <= 1.0:
        raise SystemExit("--iou must be between 0.0 and 1.0")
    # OpenCV otherwise creates its own worker pool in addition to OpenVINO's
    # pool, which can starve USB camera acquisition on a four-core Atom CPU.
    cv2.setNumThreads(1)
    fhd_index = args.fhd_index
    if fhd_index is None:
        fhd_index = auto_detect_fhd_camera()
    print(f"[*] SDV optimized pipeline version: {PIPELINE_VERSION}")
    print(f"[*] Using FHD camera index: {fhd_index}")
    print(f"[*] Requesting RealSense RGB + depth at 640x480 @ {args.rs_fps} FPS")

    threads = [
        threading.Thread(
            target=realsense_capture_thread,
            args=(args.rs_fps,),
            daemon=True,
        ),
        threading.Thread(
            target=realsense_inference_thread,
            args=(args.model_path, args.confidence, args.iou),
            daemon=True,
        ),
        threading.Thread(target=lane_fhd_thread, args=(fhd_index,), daemon=True),
    ]
    for thread in threads:
        thread.start()

    display_frames = 0
    display_start = time.perf_counter()
    display_fps = 0.0
    last_displayed_sequence = -1

    try:
        while state["running"]:
            with state_lock:
                real_frame = (
                    None
                    if state["realsense_frame"] is None
                    else state["realsense_frame"].copy()
                )
                lane_frame = (
                    None if state["fhd_frame"] is None else state["fhd_frame"].copy()
                )
                detections = list(state["detections"])
                capture_fps = state["fps_rs_capture"]
                inference_fps = state["fps_rs_inference"]
                inference_ms = state["inference_ms"]
                fhd_fps = state["fps_fhd"]
                realsense_sequence = state["realsense_sequence"]

            if real_frame is not None and lane_frame is not None:
                draw_detections(real_frame, detections)
                real_frame = cv2.resize(real_frame, (640, 480))
                lane_frame = cv2.resize(lane_frame, (640, 480))

                if realsense_sequence != last_displayed_sequence:
                    display_frames += 1
                    last_displayed_sequence = realsense_sequence
                now = time.perf_counter()
                elapsed = now - display_start
                if elapsed >= 1.0:
                    display_fps = display_frames / elapsed
                    display_frames = 0
                    display_start = now

                cv2.putText(
                    real_frame,
                    f"RealSense Capture: {capture_fps:.1f} FPS | AI: {inference_fps:.2f} FPS",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    real_frame,
                    f"AI latency: {inference_ms:.0f} ms | RS display: {display_fps:.1f} FPS",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    lane_frame,
                    f"FHD Lane Detection (FPS: {fhd_fps:.1f})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

                combined = np.hstack((real_frame, lane_frame))
                cv2.imshow("Intel UP Square 6000 - Optimized Local SDV Dashboard", combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        with new_realsense_frame:
            state["running"] = False
            new_realsense_frame.notify_all()
        for thread in threads:
            thread.join(timeout=2.0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
