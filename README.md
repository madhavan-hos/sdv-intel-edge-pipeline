# Intel Unnati HPC & Edge AI Vision Stack: SDV Workshop

Welcome to the **Build Your First SDV Vehicles** workshop repository! This project demonstrates a complete Software-Defined Vehicle (SDV) perception pipeline, migrating a trained YOLO model from an HPC server to an Intel edge compute device using OpenVINO.

## 🛠 Hardware Configuration
* **Edge Compute:** Intel UP Squared 6000 (Intel Atom x6425RE / LPDDR4 8GB)
* **Depth & Object Sensor:** Intel RealSense D435i (Color + Active IR Depth)
* **Lane Sensor:** Intel Full HD Machine Vision USB 2.0 Camera
* **Cloud/HPC Server:** Intel Xeon Node (Used for training and model export)

> **⚠️ Hardware Note:** The Intel RealSense D435i **must** be connected to a blue **USB 3.2 port** on the UP Squared 6000. If connected to a USB 2.0 port, the high-bandwidth dual-stream (RGB + Depth) will time out and crash the pipeline.

---

## 🚀 Step 1: Project Initialization

If you have just cloned this repository or need to generate the file structure:

```bash
# Generate the full session-by-session directory structure
python3 build_project.py

# Navigate into the generated project folder
cd sdv_workshop_project
```

---

## 📦 Step 2: Environment Setup (Edge Device)

Edge devices have limited storage. To avoid downloading massive, unnecessary GPU libraries (which will fill your disk), we will create an isolated virtual environment and strictly install the CPU versions of the AI frameworks.

```bash
# 1. Create and activate a Python virtual environment
python3 -m venv ai_env
source ai_env/bin/activate

# 2. Clear old pip caches to free up disk space
rm -rf ~/.cache/pip

# 3. Install lightweight, CPU-only PyTorch
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)

# 4. Install the remaining workshop dependencies
pip install -r session_03_setup/requirements.txt

# 5. Verify the installation
python session_03_setup/verify_env.py
```

---

## 📥 Step 3: Model Transfer (HPC to Edge)

You must securely copy the OpenVINO FP16 optimized model from the HPC server down to your local UP Squared 6000 edge device.

```bash
# Ensure the models directory exists
mkdir -p models/best_openvino_model/

# Secure Copy (SCP) the model files from the HPC server
# Replace <username> and <HPC_IP> with your assigned credentials
scp -r <username>@<HPC_IP>:/mnt/ai-cluster/projects/road_object/runs/detect/train/weights/best_openvino_model/* models/best_openvino_model/
```

Verify the files (`best.xml`, `best.bin`, `metadata.yaml`) are present:
```bash
ls models/best_openvino_model/
```

---

## 📷 Step 4: Camera Diagnostics & Discovery (Session 4)

Before running the full SDV pipeline, verify both cameras are transmitting properly.

**1. Scan for Connected Cameras:**
```bash
python session_04_vision/camera_scanner.py
```

**2. Test RealSense Depth Stream:**
```bash
python session_04_vision/test_realsense.py
```
*(If this throws a `Frame didn't arrive within 5000` error, ensure the camera is plugged into a USB 3.0 port).*

**3. Test FHD Machine Vision Stream:**
```bash
python session_04_vision/test_fhd_camera.py <INDEX>
```
*(Replace `<INDEX>` with the ID found by the camera scanner, e.g., 4 or 6).*

---

## 🧠 Step 5: Isolated Use Case Testing (Session 8)

Test the individual AI modules before fusing them.

**Lane Detection (FHD Camera):**
```bash
python session_08_usecase/lane_detection.py
```

**Obstacle Distance Estimation (RealSense + OpenVINO):**
```bash
python session_08_usecase/obstacle_detection.py models/best_openvino_model/
```

---

## 🏎️ Step 6: Full SDV System Integration (Session 9)

Run the fully optimized, multi-threaded SDV dashboard. This script fuses both camera streams, applies Gaussian blurring and optimized Canny edge detection for stable lane tracking, and utilizes OpenVINO for real-time obstacle distance estimation. 

*Note: The script features dynamic auto-detection for the FHD camera, so no manual index is required.*

```bash
# Ensure you are in the project root and the environment is active
source ai_env/bin/activate

# Launch the unified SDV Perception Pipeline
python session_09_integration/local_sdv_pipeline.py models/best_openvino_model/
```

### Dashboard Distance Logic:
* **Green (Clear):** Distance > 2.5 meters.
* **Yellow (Caution):** Distance between 1.2m and 2.5m.
* **Red (Collision Warning):** Distance < 1.2m.
