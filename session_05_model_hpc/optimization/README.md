# OpenVINO Precision Optimization Lab

This optional Session 5 lab exports the same trained YOLO checkpoint as three
separate OpenVINO models. Keeping the variants separate makes the accuracy,
size, and latency trade-off visible and prevents one export from silently
overwriting another.

## Precision overview

| Variant | Export | Calibration | Workshop role |
|---|---|---|---|
| FP32 | Full 32-bit floating point | No | Accuracy/reference baseline |
| FP16 | 16-bit floating point | No | Smaller model; useful for supported GPU/VPU hardware |
| INT8 | 8-bit integer post-training quantization | Yes | Usually the best CPU/edge performance candidate |

Quantization affects model storage and compute precision. It does not change
the RealSense/FHD camera capture rate. INT8 can reduce accuracy, so benchmark
latency and validate mAP before selecting it for the final demonstration.

## 1. Prepare the HPC environment

Use the same Conda environment that trained the model:

```bash
conda activate ai
cd /mnt/ai-cluster/projects/road_object
python -c 'import ultralytics, openvino; print(ultralytics.__version__, openvino.__version__)'
```

The workshop HPC image uses Ultralytics 8.4.35. That release accepts
`half=True` and `int8=True`; newer releases prefer `quantize=16` and
`quantize=8`. The provided script detects which argument style is available.

## 2. Export all three variants

Run the script from the cloned workshop repository on the HPC server:

```bash
python session_05_model_hpc/optimization/export_openvino_precisions.py \
  --weights /mnt/ai-cluster/projects/road_object/runs/detect/train/weights/best.pt \
  --data /mnt/ai-cluster/projects/road_object/idd.yaml \
  --precision all \
  --fraction 1.0 \
  --imgsz 640 \
  --output-dir /mnt/ai-cluster/projects/road_object/optimized_models
```

INT8 calibration uses representative images referenced by `idd.yaml`. Use the
same camera domain and preprocessing expected in deployment. A smaller
`--fraction`, such as `0.25`, speeds up a workshop trial but may reduce the
quality of calibration.

The command creates:

```text
optimized_models/
├── best_fp32_openvino_model/
├── best_fp16_openvino_model/
└── best_int8_openvino_model/
```

Each directory also contains `workshop_export_manifest.json` recording the
requested precision, source checkpoint, calibration input, and tool version.

To export one variant only:

```bash
python session_05_model_hpc/optimization/export_openvino_precisions.py \
  --weights /mnt/ai-cluster/projects/road_object/runs/detect/train/weights/best.pt \
  --precision fp16 \
  --output-dir /mnt/ai-cluster/projects/road_object/optimized_models
```

For INT8, always include `--data`:

```bash
python session_05_model_hpc/optimization/export_openvino_precisions.py \
  --weights /mnt/ai-cluster/projects/road_object/runs/detect/train/weights/best.pt \
  --precision int8 \
  --data /mnt/ai-cluster/projects/road_object/idd.yaml \
  --fraction 1.0 \
  --output-dir /mnt/ai-cluster/projects/road_object/optimized_models
```

## 3. Validate accuracy on the HPC server

Validate every directory against the same validation dataset:

```bash
python -c 'from ultralytics import YOLO; YOLO("/mnt/ai-cluster/projects/road_object/optimized_models/best_fp32_openvino_model").val(data="/mnt/ai-cluster/projects/road_object/idd.yaml", imgsz=640, device="intel:cpu")'

python -c 'from ultralytics import YOLO; YOLO("/mnt/ai-cluster/projects/road_object/optimized_models/best_fp16_openvino_model").val(data="/mnt/ai-cluster/projects/road_object/idd.yaml", imgsz=640, device="intel:cpu")'

python -c 'from ultralytics import YOLO; YOLO("/mnt/ai-cluster/projects/road_object/optimized_models/best_int8_openvino_model").val(data="/mnt/ai-cluster/projects/road_object/idd.yaml", imgsz=640, device="intel:cpu")'
```

Record mAP50-95, mAP50, precision, and recall. FP32 is the reference; select
INT8 only if its accuracy loss is acceptable for the workshop use case.

## 4. Copy a selected variant to the UP Square

Run on the UP Square from the workshop repository root:

```bash
mkdir -p models/best_int8_openvino_model
scp -r startup2@10.10.9.19:/mnt/ai-cluster/projects/road_object/optimized_models/best_int8_openvino_model/* \
  models/best_int8_openvino_model/
```

Then run the dashboard with that directory:

```bash
python session_09_integration/local_sdv_pipeline_optimized.py \
  models/best_int8_openvino_model/ \
  --fhd-index 0 \
  --rs-fps 15 \
  --confidence 0.25
```

## 5. Benchmark on the deployment device

The UP Square result is the decision metric. Find the XML name inside each
copied model directory and run the existing benchmark:

```bash
find models/best_int8_openvino_model -maxdepth 1 -name '*.xml'
python session_07_acceleration/benchmark_devices.py \
  models/best_int8_openvino_model/best_int8.xml
```

Use the actual XML filename printed by `find`. Also compare the dashboard's AI
latency/FPS because preprocessing and postprocessing are not included in a raw
OpenVINO graph benchmark.

## Notes about the workshop dataset

The supplied `idd.yaml` currently points both `train` and `val` to broad image
directories. For meaningful validation, point them to distinct train and
validation splits. Calibration images may overlap training data, but the final
accuracy comparison must use a held-out validation/test split.
