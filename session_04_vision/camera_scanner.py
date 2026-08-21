import cv2
import subprocess
import re

def list_v4l2_devices():
    print("[*] Scanning system for V4L2 USB Video Devices...")
    try:
        output = subprocess.check_output(["v4l2-ctl", "--list-devices"], text=True)
        print(output)
    except Exception:
        print("[!] v4l2-ctl utility not found. Scanning via OpenCV indices...")

def scan_opencv_cameras(max_tested=10):
    available_cams = []
    for index in range(max_tested):
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                available_cams.append((index, w, h))
            cap.release()
    return available_cams

def scan_realsense_devices():
    print("[*] Scanning for Intel RealSense Devices...")
    try:
        import pyrealsense2 as rs
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            print(" [-] No Intel RealSense devices detected.")
            return []
        rs_list = []
        for dev in devices:
            name = dev.get_info(rs.camera_info.name)
            serial = dev.get_info(rs.camera_info.serial_number)
            print(f" [+] Found RealSense: {name} (S/N: {serial})")
            rs_list.append((name, serial))
        return rs_list
    except ImportError:
        print(" [!] pyrealsense2 library not available.")
        return []

if __name__ == "__main__":
    list_v4l2_devices()
    cams = scan_opencv_cameras()
    print(f"[*] Available OpenCV Camera Indices: {cams}")
    scan_realsense_devices()
