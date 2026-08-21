import argparse
from ultralytics import YOLO

def export_model(weights_path, half_precision=True):
    print(f"[*] Loading PyTorch weights from: {weights_path}")
    model = YOLO(weights_path)
    
    precision = "FP16" if half_precision else "FP32"
    print(f"[*] Exporting to OpenVINO format ({precision})...")
    
    exported_path = model.export(
        format="openvino",
        half=half_precision,
        imgsz=640
    )
    print(f"[+] OpenVINO model exported successfully to: {exported_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="runs/detect/train/weights/best.pt")
    parser.add_argument("--fp32", action="store_true", help="Export as FP32 (default is FP16)")
    args = parser.parse_args()

    export_model(args.weights, half_precision=not args.fp32)
