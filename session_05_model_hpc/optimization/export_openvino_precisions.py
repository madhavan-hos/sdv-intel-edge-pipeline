"""Export isolated FP32, FP16, and calibrated INT8 OpenVINO variants."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import ultralytics
from ultralytics import YOLO

try:
    from ultralytics.cfg import DEFAULT_CFG_DICT
except ImportError:
    DEFAULT_CFG_DICT = {}


PRECISION_VALUES = {
    "fp32": 32,
    "fp16": 16,
    "int8": 8,
}


def export_arguments(precision, data, fraction, imgsz):
    """Build arguments for both legacy and current Ultralytics releases."""
    args = {
        "format": "openvino",
        "imgsz": imgsz,
        "batch": 1,
        "dynamic": False,
    }

    # Ultralytics 8.4.38+ standardizes precision as quantize=32/16/8.
    # Workshop HPC images such as 8.4.35 use the legacy half/int8 flags.
    if "quantize" in DEFAULT_CFG_DICT:
        args["quantize"] = PRECISION_VALUES[precision]
    elif precision == "fp16":
        args["half"] = True
    elif precision == "int8":
        args["int8"] = True

    if precision == "int8":
        if data is None:
            raise ValueError("INT8 export requires --data for calibration")
        args["data"] = str(data)
        args["fraction"] = fraction

    return args


def export_variant(weights, precision, output_root, data, fraction, imgsz, overwrite):
    destination = output_root / (weights.stem + "_" + precision + "_openvino_model")
    if destination.exists() and not overwrite:
        raise FileExistsError(
            "{} already exists; use --overwrite to replace it".format(destination)
        )
    if destination.exists():
        shutil.rmtree(str(destination))

    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ov_export_", dir=str(output_root)) as temp_dir:
        temp_weights = Path(temp_dir) / (weights.stem + "_" + precision + weights.suffix)
        shutil.copy2(str(weights), str(temp_weights))

        model = YOLO(str(temp_weights))
        args = export_arguments(precision, data, fraction, imgsz)
        print("[*] Exporting {} with arguments: {}".format(precision.upper(), args))
        exported_path = Path(model.export(**args))
        shutil.copytree(str(exported_path), str(destination))

    manifest = {
        "source_weights": str(weights.resolve()),
        "precision": precision.upper(),
        "imgsz": imgsz,
        "batch": 1,
        "dynamic": False,
        "calibration_data": str(data.resolve()) if data is not None else None,
        "calibration_fraction": fraction if precision == "int8" else None,
        "ultralytics_version": ultralytics.__version__,
    }
    (destination / "workshop_export_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("[+] Saved {} model to {}".format(precision.upper(), destination))
    return destination


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export workshop YOLO weights to OpenVINO precision variants"
    )
    parser.add_argument("--weights", required=True, type=Path, help="Source .pt weights")
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "int8", "all"),
        default="all",
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="Dataset YAML used for representative INT8 calibration images",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of calibration data to use (default: 1.0)",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("optimized_models"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.weights.is_file():
        raise SystemExit("Weights file not found: {}".format(args.weights))
    if args.weights.suffix.lower() != ".pt":
        raise SystemExit("--weights must point to the original PyTorch .pt model")
    if not 0.0 < args.fraction <= 1.0:
        raise SystemExit("--fraction must be greater than 0 and at most 1")
    if args.precision in ("int8", "all"):
        if args.data is None or not args.data.is_file():
            raise SystemExit("INT8 export requires an existing --data YAML file")

    precisions = ("fp32", "fp16", "int8") if args.precision == "all" else (args.precision,)
    print("[*] Ultralytics version: {}".format(ultralytics.__version__))
    for precision in precisions:
        export_variant(
            weights=args.weights,
            precision=precision,
            output_root=args.output_dir,
            data=args.data,
            fraction=args.fraction,
            imgsz=args.imgsz,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
