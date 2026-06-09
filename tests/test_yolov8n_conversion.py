"""
Integration test: YOLOv8n ONNX → Hailo ONNX conversion.

Packages yolov8n.onnx with 100 COCO val2017 calibration images, runs the full
Hailo conversion pipeline (translate → optimize → compile), and verifies the
output.

Run with:
    pytest tests/test_yolov8n_conversion.py -v -s
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import onnx
import pytest

_REPO_ROOT      = Path(__file__).parents[1]
_CONVERT_SCRIPT = _REPO_ROOT / "tests" / "convert_yolov8n.py"
_YOLOV8N_ONNX   = Path("/home/ayoub/nvidia-acceleration/yolov8n.onnx")
_YOLOV8N_ZIP    = Path("/home/ayoub/tmp/yolov8n.zip")

# Stop at the 6 raw conv outputs (3 scales × bbox + class).
# The DFL decode head is unsupported in Hailo and is run on the host instead.
_END_NODES = [
    "/model.22/cv2.0/cv2.0.2/Conv", "/model.22/cv2.1/cv2.1.2/Conv", "/model.22/cv2.2/cv2.2.2/Conv",
    "/model.22/cv3.0/cv3.0.2/Conv", "/model.22/cv3.1/cv3.1.2/Conv", "/model.22/cv3.2/cv3.2.2/Conv",
]

# Python interpreter that has the Hailo SDK installed.
_SDK_PYTHON = "python3.10"
_SDK_ENV    = {"PYTHONNOUSERSITE": "1"}


def _sdk_available() -> bool:
    r = subprocess.run(
        [_SDK_PYTHON, "-c", "import hailo_sdk_client"],
        capture_output=True,
        env={**__import__("os").environ, **_SDK_ENV},
    )
    return r.returncode == 0


# ── Session fixture: run conversion once, share across all tests ─────────────

@pytest.fixture(scope="session")
def conversion_output(tmp_path_factory):
    """Invoke convert_yolov8n.py and return the output directory Path."""
    if not _sdk_available():
        pytest.skip(
            f"Hailo SDK not available via '{_SDK_PYTHON} PYTHONNOUSERSITE=1'. "
            "Check that hailo_dataflow_compiler is installed."
        )
    if not _YOLOV8N_ONNX.exists():
        pytest.skip(f"YOLOv8n ONNX not found: {_YOLOV8N_ONNX}")
    if not _YOLOV8N_ZIP.exists():
        pytest.skip(f"Calibration zip not found: {_YOLOV8N_ZIP}")

    out_dir = tmp_path_factory.mktemp("yolov8n_out")

    print(f"\n  Output dir: {out_dir}")
    print( "  Running conversion — this takes several minutes ...")

    import os
    # hailo_platform needs libhailort.so.4.19.0 — prepend our bundled copy.
    _hailort_419 = _REPO_ROOT / "runtime-library/deps/hailort/X86_64_4.19.0"
    existing_ldpath = os.environ.get("LD_LIBRARY_PATH", "")
    ld_path = f"{_hailort_419}:{existing_ldpath}" if _hailort_419.exists() else existing_ldpath
    env = {**os.environ, **_SDK_ENV, "LD_LIBRARY_PATH": ld_path}

    result = subprocess.run(
        [_SDK_PYTHON, str(_CONVERT_SCRIPT), str(out_dir)],
        env=env,
        timeout=600,
    )

    result_file = out_dir / "result.json"
    if not result_file.exists():
        pytest.fail(f"convert_yolov8n.py did not write result.json (exit {result.returncode})")

    meta = json.loads(result_file.read_text())
    if not meta.get("success"):
        pytest.fail(
            f"Conversion failed:\n{meta.get('trace', meta.get('error', 'unknown'))}"
        )

    return out_dir


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestYolov8nConversion:

    def test_output_onnx_exists(self, conversion_output):
        assert (conversion_output / "yolov8n.hailo8.onnx").exists()

    def test_logs_json_written(self, conversion_output):
        assert (conversion_output / "logs.json").exists()

    def test_logs_record_all_pipeline_stages(self, conversion_output):
        text = (conversion_output / "logs.json").read_text().lower()
        assert "translat" in text, "Translation stage missing from logs"
        assert "optim"    in text, "Optimization stage missing from logs"
        assert "compil"   in text, "Compilation stage missing from logs"

    def test_output_loads_with_onnx_library(self, conversion_output):
        model = onnx.load(str(conversion_output / "yolov8n.hailo8.onnx"))
        assert model.graph is not None

    def test_output_opset_compatible_with_hailo_ort(self, conversion_output):
        """Output must use opset ≤ 16 to be loadable by the Hailo ORT build."""
        model = onnx.load(str(conversion_output / "yolov8n.hailo8.onnx"))
        for opset in model.opset_import:
            if opset.domain == "":
                assert opset.version <= 16, \
                    f"Opset {opset.version} exceeds Hailo ORT max (16)"

    def test_output_contains_hailo_op_node(self, conversion_output):
        model = onnx.load(str(conversion_output / "yolov8n.hailo8.onnx"))
        op_types = {n.op_type for n in model.graph.node}
        assert "HailoOp" in op_types, \
            f"No HailoOp node. Op types present: {op_types}"

    def test_output_has_graph_inputs_and_outputs(self, conversion_output):
        model = onnx.load(str(conversion_output / "yolov8n.hailo8.onnx"))
        assert len(model.graph.input)  >= 1, "Output model has no inputs"
        assert len(model.graph.output) >= 1, "Output model has no outputs"

    def test_output_passes_onnx_checker(self, conversion_output):
        model = onnx.load(str(conversion_output / "yolov8n.hailo8.onnx"))
        onnx.checker.check_model(model)

    def test_output_is_smaller_than_input(self, conversion_output):
        """Hailo ONNX should be smaller than the original (weights compiled into HEF)."""
        hailo_size = (conversion_output / "yolov8n.hailo8.onnx").stat().st_size
        orig_size  = _YOLOV8N_ONNX.stat().st_size
        print(f"\n  Original: {orig_size / 1e6:.1f} MB → Hailo: {hailo_size / 1e6:.1f} MB")
        assert hailo_size < orig_size, \
            f"Hailo ONNX ({hailo_size / 1e6:.1f} MB) is not smaller than original ({orig_size / 1e6:.1f} MB)"

    def test_model_runs_on_hailo_chip(self, conversion_output):
        """Load and run one inference on the Hailo chip using the runtime."""
        model_path = str(conversion_output / "yolov8n.hailo8.onnx")
        yolo_bin   = _REPO_ROOT / "tests/runtime/build/yolo_test"
        if not yolo_bin.exists():
            pytest.skip(f"yolo_test binary not found at {yolo_bin} — build tests first")

        result = subprocess.run(
            [
                str(yolo_bin), model_path,
                "--input-name", "input0",
                "--imgsz", "640",
                "--warmup", "2",
                "--runs",   "10",
                "--in-flight", "2",
                "--no-validate",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, \
            f"yolo_test failed (exit {result.returncode}):\n{result.stderr}"

        # Extract and print the results section
        lines = result.stdout.splitlines()
        results_section = "\n".join(
            l for l in lines if any(k in l for k in ("Results", "latency", "Throughput", "Load time"))
        )
        print(f"\n{results_section}")

        assert "Throughput" in result.stdout, "No throughput in output"
        # Sanity check: throughput > 1 img/s (chip is actually running)
        for line in lines:
            if "Throughput" in line:
                fps = float(line.split(":")[-1].strip().split()[0])
                assert fps > 1.0, f"Suspiciously low throughput: {fps} img/s"
                print(f"  → {fps:.1f} img/s on Hailo-8")
