from ultralytics import YOLO

def train():
    model = YOLO("yolo26n.pt")
    print("[*] Starting YOLO training on HPC Server...")
    results = model.train(
        data="idd.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        project="runs/detect",
        name="train"
    )
    print("[+] Training completed successfully!")

if __name__ == "__main__":
    train()
