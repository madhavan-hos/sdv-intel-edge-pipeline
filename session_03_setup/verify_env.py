import sys
import cv2
import numpy as np
import openvino as ov

print("=" * 60)
print("SDV Workshop - Environment Verification")
print("=" * 60)
print(f"Python Version:    {sys.version.split()[0]}")
print(f"OpenCV Version:   {cv2.__version__}")
print(f"NumPy Version:    {np.__version__}")
print(f"OpenVINO Version: {ov.__version__}")

try:
    import pyrealsense2 as rs
    try:
        from importlib.metadata import version
        rs_version = version("pyrealsense2")
    except Exception:
        # Some distro/librealsense builds intentionally do not expose
        # pyrealsense2.__version__. Successful import is what matters here.
        rs_version = getattr(rs, "__version__", "INSTALLED (version unavailable)")
    print(f"pyrealsense2:     {rs_version}")
except ImportError:
    print("pyrealsense2:     NOT INSTALLED (Required for RealSense D435i)")

core = ov.Core()
devices = core.available_devices
print(f"OpenVINO Available Devices: {devices}")
for d in devices:
    dev_name = core.get_property(d, "FULL_DEVICE_NAME")
    print(f"  - {d}: {dev_name}")
print("=" * 60)
