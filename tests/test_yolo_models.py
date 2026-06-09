"""
Parametrized integration test: convert multiple YOLO models to Hailo ONNX
and benchmark each on the Hailo chip.

Models that fail at any step are reported as failed and the suite continues.

Run:
    pytest tests/test_yolo_models.py -v -s
"""

import json
import os
import subprocess
from pathlib import Path

import onnx
import pytest

_REPO_ROOT      = Path(__file__).parents[1]
_CONVERT_SCRIPT = _REPO_ROOT / "tests/convert_yolo.py"
_YOLO_BIN       = _REPO_ROOT / "tests/runtime/build/yolo_test"
_SDK_PYTHON     = "python3.10"
_SDK_ENV        = {"PYTHONNOUSERSITE": "1"}
_HAILORT_419    = _REPO_ROOT / "runtime-library/deps/hailort/X86_64_4.19.0"

MODELS = [
    {
        "name":       "yolov8s",
        "onnx":       str(_REPO_ROOT / "yolov8s.onnx"),
        "input_name": "images",
    },
    {
        "name":       "yolo11n",
        "onnx":       "/home/ayoub/model_onnx_export/ultralytics/exported/yolo11n.onnx",
        "input_name": "images",
    },
    {
        "name":       "yolo26n",
        "onnx":       "/home/ayoub/model_onnx_export/ultralytics/exported/yolo26n.onnx",
        "input_name": "images",
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sdk_env() -> dict:
    ld = str(_HAILORT_419) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    return {**os.environ, **_SDK_ENV, "LD_LIBRARY_PATH": ld}


def _convert(model: dict, out_dir: Path) -> dict:
    """Run convert_yolo.py and return the result.json dict."""
    result = subprocess.run(
        [_SDK_PYTHON, str(_CONVERT_SCRIPT),
         "--model-name", model["name"],
         "--onnx-path",  model["onnx"],
         str(out_dir)],
        env=_sdk_env(),
        timeout=700,
    )
    result_file = out_dir / "result.json"
    if not result_file.exists():
        return {"success": False, "error": f"no result.json (exit {result.returncode})"}
    return json.loads(result_file.read_text())


def _benchmark(model_path: str, input_name: str) -> dict:
    """Run yolo_test and return parsed metrics."""
    result = subprocess.run(
        [str(_YOLO_BIN), model_path,
         "--input-name", input_name,
         "--imgsz", "640",
         "--warmup", "5", "--runs", "50",
         "--in-flight", "1",
         "--no-validate"],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        return {"error": result.stderr[:500]}

    metrics = {}
    for line in result.stdout.splitlines():
        for key in ("Load time", "Avg latency", "Min latency", "p95 latency", "Throughput"):
            if key in line:
                try:
                    metrics[key] = float(line.split(":")[-1].strip().split()[0])
                except ValueError:
                    pass
    return metrics


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session", params=MODELS, ids=[m["name"] for m in MODELS])
def converted_model(request, tmp_path_factory):
    """Convert one model to Hailo ONNX. Yields (model_cfg, hailo_onnx_path | None)."""
    model = request.param
    onnx_path = Path(model["onnx"])

    if not onnx_path.exists():
        pytest.skip(f"Source ONNX not found: {onnx_path}")

    out_dir = tmp_path_factory.mktemp(f"hailo_{model['name']}")
    print(f"\n[{model['name']}] Converting {onnx_path.name} ...")

    result = _convert(model, out_dir)

    if not result.get("success"):
        # Don't fail the fixture — let tests mark themselves as failed
        print(f"[{model['name']}] Conversion FAILED: {result.get('error', '')[:200]}")
        return model, None

    hailo_path = Path(result["output"])
    print(f"[{model['name']}] → {hailo_path.name}  ({hailo_path.stat().st_size / 1e6:.1f} MB)")
    return model, hailo_path


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestYoloModels:

    def _require_conversion(self, converted_model):
        """Skip test if conversion failed."""
        model, hailo_path = converted_model
        if hailo_path is None:
            pytest.fail(f"[{model['name']}] conversion failed — see fixture output")
        return model, hailo_path

    def test_conversion_succeeded(self, converted_model):
        model, hailo_path = converted_model
        if hailo_path is None:
            pytest.fail(f"[{model['name']}] conversion failed")
        assert hailo_path.exists()

    def test_output_contains_hailo_op(self, converted_model):
        model, hailo_path = self._require_conversion(converted_model)
        m = onnx.load(str(hailo_path))
        op_types = {n.op_type for n in m.graph.node}
        assert "HailoOp" in op_types, f"[{model['name']}] No HailoOp node"

    def test_output_opset_compatible(self, converted_model):
        model, hailo_path = self._require_conversion(converted_model)
        m = onnx.load(str(hailo_path))
        for opset in m.opset_import:
            if opset.domain == "":
                assert opset.version <= 16, \
                    f"[{model['name']}] Opset {opset.version} > 16"

    def test_runs_on_hailo_chip(self, converted_model):
        model, hailo_path = self._require_conversion(converted_model)
        if not _YOLO_BIN.exists():
            pytest.skip("yolo_test binary not built")

        print(f"\n[{model['name']}] Benchmarking on Hailo chip ...")
        metrics = _benchmark(str(hailo_path), model["input_name"])

        if "error" in metrics:
            pytest.fail(f"[{model['name']}] yolo_test failed: {metrics['error']}")

        fps = metrics.get("Throughput", 0)
        avg_ms = metrics.get("Avg latency", 0)
        p95_ms = metrics.get("p95 latency", 0)

        print(f"  Throughput:  {fps:.1f} img/s")
        print(f"  Avg latency: {avg_ms:.1f} ms")
        print(f"  p95 latency: {p95_ms:.1f} ms")

        assert fps > 1.0, f"[{model['name']}] Suspiciously low throughput: {fps} img/s"
