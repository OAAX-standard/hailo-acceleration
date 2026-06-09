"""
Standalone conversion script: YOLOv8n ONNX → Hailo ONNX.

Run with the system Python that has the Hailo SDK:
    PYTHONNOUSERSITE=1 python3.10 tests/convert_yolov8n.py <output_dir>

Writes to <output_dir>:
  - yolov8n.hailo8.onnx
  - logs.json
  - result.json  (exit summary: {"success": true/false, "error": "..."})
"""

import json
import os
import sys
import zipfile
from pathlib import Path


# Insert the conversion toolchain package into the path.
sys.path.insert(0, str(Path(__file__).parents[1] / "conversion-toolchain"))

from conversion_toolchain.logger import Logs
from conversion_toolchain.utils import convert_to_hailo_onnx, unzip_file

# The Hailo SDK detects itself as a dev install (dist-packages, not site-packages)
# and writes compilation artefacts to /usr/local/lib/python3.10/sdk_client/build/
# which is owned by another user.  Redirect to a user-writable temp dir.
def _redirect_sdk_build_dir():
    import tempfile
    from hailo_sdk_common.paths_manager.paths import SDKPaths
    sdk = SDKPaths()
    if not sdk._is_release:
        tmp = tempfile.mkdtemp(prefix="hailo_build_")
        sdk._is_release = True
        sdk._build_dir = tmp

_redirect_sdk_build_dir()

YOLOV8N_ONNX = Path("/home/ayoub/nvidia-acceleration/yolov8n.onnx")
YOLOV8N_ZIP  = Path("/home/ayoub/tmp/yolov8n.zip")

OPTIONS = {
    # Stop at the 6 raw conv outputs from the detection head (3 scales × bbox + class).
    # Everything after these nodes (Reshape, DFL, decode) is unsupported by Hailo and
    # must be done on the host.  This is the standard Hailo YOLOv8 export format.
    "start_node_names": [],
    "end_node_names": [
        "/model.22/cv2.0/cv2.0.2/Conv",
        "/model.22/cv2.1/cv2.1.2/Conv",
        "/model.22/cv2.2/cv2.2.2/Conv",
        "/model.22/cv3.0/cv3.0.2/Conv",
        "/model.22/cv3.1/cv3.1.2/Conv",
        "/model.22/cv3.2/cv3.2.2/Conv",
    ],
    "height":   640,
    "width":    640,
    "channels": 3,
    # Hailo always expects NHWC calibration data regardless of the model's NCHW input.
    # The SDK handles the transposition internally during translation.
    "nchw":     False,
    "means":    [0.0, 0.0, 0.0],
    "stds":     [255.0, 255.0, 255.0],
}


def build_input_zip(dest: Path) -> Path:
    zip_path = dest / "yolov8n_input.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(YOLOV8N_ONNX, "yolov8n.onnx")
        zf.writestr("options.json", json.dumps(OPTIONS))
        src = zipfile.ZipFile(YOLOV8N_ZIP)
        images = [n for n in src.namelist()
                  if n.startswith("calibration_dataset/") and "." in n.split("/")[-1]]
        for i, name in enumerate(images[:100]):
            zf.writestr(f"calib/img{i:03d}.jpg", src.read(name))
        print(f"[pack] {len(images[:100])} calibration images packed")
    return zip_path


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <output_dir>", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    result_file = out_dir / "result.json"

    try:
        # Package the model
        work = out_dir / "work"
        work.mkdir(exist_ok=True)
        print("[pack] Building input zip ...")
        input_zip = build_input_zip(work)

        # Unpack via toolchain helper
        extract_dir = work / "extracted"
        print("[unzip] Extracting ...")
        onnx_path, input_js, calib_images = unzip_file(str(input_zip), str(extract_dir))
        print(f"[unzip] ONNX: {Path(onnx_path).name}, calib: {len(calib_images)} images")

        # Run conversion
        output_onnx = str(out_dir / "yolov8n.hailo8.onnx")
        logs = Logs()
        print("[conv] Starting Hailo conversion (translate → optimize → compile) ...")
        convert_to_hailo_onnx(onnx_path, output_onnx, input_js, calib_images, logs)

        # Save logs
        logs.save_as_json(str(out_dir / "logs.json"))
        print(f"[conv] Done. Output: {output_onnx}")

        result_file.write_text(json.dumps({"success": True, "output": output_onnx}))
        sys.exit(0)

    except Exception as e:
        import traceback
        msg = traceback.format_exc()
        print(f"[error] {e}", file=sys.stderr)
        result_file.write_text(json.dumps({"success": False, "error": str(e), "trace": msg}))
        sys.exit(1)


if __name__ == "__main__":
    main()
