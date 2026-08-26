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


PIPELINE_VERSION = "2026.08.26-fps-v4"


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


def realsense_inference_thread(model_path):
    set_thread_affinity("inference")
    model = YOLO(model_path, task="detect")
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
        results = model.predict(source=color_image, imgsz=640, verbose=False)
        inference_ms = (time.perf_counter() - inference_start) * 1000.0

        height, width = depth_image.shape[:2]
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = min(max((x1 + x2) // 2, 0), width - 1)
            cy = min(max((y1 + y2) // 2, 0), height - 1)
            distance = float(depth_image[cy, cx]) * depth_scale
            class_name = model.names[int(box.cls[0])]
            color = (0, 0, 255) if 0.0 < distance < 1.2 else (0, 255, 0)
            detections.append((x1, y1, x2, y2, class_name, distance, color))

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


def lane_fhd_thread(cam_index):
    set_thread_affinity("fhd")
    cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_count = 0
    counter_start = time.perf_counter()

    while state["running"] and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        edges = cv2.Canny(blur, 100, 200)

        mask = np.zeros_like(edges)
        poly = np.array(
            [[
                (int(w * 0.1), h),
                (int(w * 0.45), int(h * 0.6)),
                (int(w * 0.55), int(h * 0.6)),
                (int(w * 0.9), h),
            ]],
            np.int32,
        )
        cv2.fillPoly(mask, poly, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(
            masked_edges,
            1,
            np.pi / 180,
            threshold=60,
            minLineLength=50,
            maxLineGap=40,
        )
        overlay = np.zeros_like(frame)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.flatten()[:4]
                cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 4)

        blended = cv2.addWeighted(frame, 0.8, overlay, 1.0, 0)
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
    for x1, y1, x2, y2, class_name, distance, color in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{class_name} {distance:.2f}m",
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
    return parser.parse_args()


def main():
    args = parse_args()
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
            args=(args.model_path,),
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
