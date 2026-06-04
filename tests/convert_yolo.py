"""
Generic YOLO → Hailo ONNX conversion script.

Usage:
    PYTHONNOUSERSITE=1 python3.10 tests/convert_yolo.py \\
        --model-name yolov8s \\
        --onnx-path /path/to/yolov8s.onnx \\
        <output_dir>

Writes to <output_dir>:
  - <model-name>.hailo8.onnx
  - logs.json
  - result.json  ({"success": true/false, "error": "...", "output": "..."})
"""

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "conversion-toolchain"))

from conversion_toolchain.logger import Logs
from conversion_toolchain.utils import convert_to_hailo_onnx


# ── Calibration images (reuse the yolov8n COCO val2017 zip) ─────────────────
_CALIB_ZIP = Path("/home/ayoub/tmp/yolov8n.zip")

_YOLO_OPTIONS = {
    "start_node_names": [],
    "end_node_names":   [],   # auto-detected by find_hailo_end_nodes
    "height":   640,
    "width":    640,
    "channels": 3,
    # Hailo always expects NHWC calibration data; the SDK handles the model's NCHW internally.
    "nchw":     False,
    "means":    [0.0, 0.0, 0.0],
    "stds":     [255.0, 255.0, 255.0],
}


def _redirect_sdk_build_dir():
    """Redirect SDK compilation artefacts to a user-writable temp dir."""
    import tempfile
    from hailo_sdk_common.paths_manager.paths import SDKPaths
    sdk = SDKPaths()
    if not sdk._is_release:
        sdk._is_release = True
        sdk._build_dir = tempfile.mkdtemp(prefix="hailo_build_")


def _extract_calib_images(tmp_dir: Path) -> list:
    """Extract calibration JPEGs from the shared calibration zip."""
    calib_dir = tmp_dir / "calib"
    calib_dir.mkdir()
    src = zipfile.ZipFile(_CALIB_ZIP)
    images = [n for n in src.namelist()
              if n.startswith("calibration_dataset/") and "." in n.split("/")[-1]]
    paths = []
    for i, name in enumerate(images[:100]):
        dest = calib_dir / f"img{i:03d}.jpg"
        dest.write_bytes(src.read(name))
        paths.append(str(dest))
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--onnx-path",  required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_file = out_dir / "result.json"

    try:
        _redirect_sdk_build_dir()

        tmp = out_dir / "work"
        tmp.mkdir(exist_ok=True)

        print(f"[calib] Extracting calibration images ...")
        calib_images = _extract_calib_images(tmp)
        print(f"[calib] {len(calib_images)} images ready")

        output_onnx = str(out_dir / f"{args.model_name}.hailo8.onnx")
        logs = Logs()

        # options copy (convert_to_hailo_onnx pops keys from it)
        options = dict(_YOLO_OPTIONS)

        print(f"[conv] Converting {args.onnx_path} ...")
        convert_to_hailo_onnx(args.onnx_path, output_onnx, options, calib_images, logs)

        logs.save_as_json(str(out_dir / "logs.json"))
        print(f"[conv] Done → {output_onnx}")

        result_file.write_text(json.dumps({"success": True, "output": output_onnx}))
        sys.exit(0)

    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        print(f"[error] {e}", file=sys.stderr)
        result_file.write_text(json.dumps({"success": False, "error": str(e), "trace": trace}))
        sys.exit(1)


if __name__ == "__main__":
    main()
