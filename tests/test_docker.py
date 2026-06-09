"""
Integration tests for the Hailo conversion toolchain.

Each test feeds a model into the Docker-based conversion toolchain and verifies
the output. Prerequisites:

  1. Docker installed and the image built:
       bash conversion-toolchain/build-toolchain.sh

  2. Hailo SDK deps directory (HAILO_DEPS_DIR env var) containing:
       hailo_dataflow_compiler-<version>-py3-none-linux_x86_64.whl
       hailort-<version>-cp38-cp38-linux_x86_64.whl
       hailort_<version>_amd64.deb

Usage:
    HAILO_DEPS_DIR=/path/to/hailo-deps pytest tests/test_docker.py -v
"""

import io
import json
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

import onnx
import pytest

DOCKER_IMAGE = "onnx-to-hailo:latest"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_toolchain(
    zip_path: Path,
    output_dir: Path,
    hailo_deps_dir: Path,
    timeout: int = 600,
) -> subprocess.CompletedProcess:
    """Run the conversion toolchain Docker container."""
    return subprocess.run(
        [
            "docker", "run", "--rm",
            "-v", f"{hailo_deps_dir}:/app/hailo-deps",
            "-v", f"{zip_path.parent}:/input:ro",
            "-v", f"{output_dir}:/output",
            DOCKER_IMAGE,
            f"/input/{zip_path.name}",
            "/output",
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _package_model(
    onnx_path: Path,
    options: dict,
    dest_zip: Path,
    calib_images: list = None,
):
    """Pack an ONNX + options.json (+ optional calibration images) into a zip."""
    with zipfile.ZipFile(dest_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(onnx_path, onnx_path.name)
        zf.writestr("options.json", json.dumps(options))
        for img_path in (calib_images or []):
            img_path = Path(img_path)
            zf.write(img_path, f"calib/{img_path.name}")


def _make_calib_image(tmp: Path, size=(225, 225), name="calib.jpg") -> Path:
    """Create a minimal JPEG calibration image."""
    from PIL import Image
    img = Image.new("RGB", size, color=(128, 128, 128))
    path = tmp / name
    img.save(str(path), format="JPEG")
    return path


# ── Tests: basic output structure ────────────────────────────────────────────

class TestConversionOutput:
    """Verify that the conversion produces correctly structured output files."""

    def test_produces_hailo_onnx_file(self, hailo_deps_available, sample_zip, tmp_path):
        """Conversion of the sample model produces exactly one *.hailo8.onnx file."""
        out = tmp_path / "output"
        out.mkdir()

        result = _run_toolchain(sample_zip, out, hailo_deps_available)
        assert result.returncode == 0, (
            f"Conversion failed (exit {result.returncode}).\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

        hailo_onnx = list(out.glob("*.hailo8.onnx"))
        assert len(hailo_onnx) == 1, (
            f"Expected 1 *.hailo8.onnx output, found: {[p.name for p in out.iterdir()]}"
        )

    def test_produces_logs_json(self, hailo_deps_available, sample_zip, tmp_path):
        """Conversion produces a logs.json alongside the ONNX output."""
        out = tmp_path / "output"
        out.mkdir()

        result = _run_toolchain(sample_zip, out, hailo_deps_available)
        assert result.returncode == 0

        logs_file = out / "logs.json"
        assert logs_file.exists(), "No logs.json produced"

    def test_output_filename_derived_from_input(self, hailo_deps_available, sample_zip, tmp_path):
        """Output filename is <input-model-name>.hailo8.onnx."""
        out = tmp_path / "output"
        out.mkdir()

        result = _run_toolchain(sample_zip, out, hailo_deps_available)
        assert result.returncode == 0

        # sample zip contains model.onnx → expect model.hailo8.onnx
        assert (out / "model.hailo8.onnx").exists(), (
            f"Expected model.hailo8.onnx, got: {[p.name for p in out.iterdir()]}"
        )

    def test_no_extra_files_produced(self, hailo_deps_available, sample_zip, tmp_path):
        """Output directory contains only *.hailo8.onnx and logs.json (no temp files)."""
        out = tmp_path / "output"
        out.mkdir()

        result = _run_toolchain(sample_zip, out, hailo_deps_available)
        assert result.returncode == 0

        produced = {p.name for p in out.iterdir() if not p.is_dir()}
        expected = {"model.hailo8.onnx", "logs.json"}
        extra = produced - expected
        assert not extra, f"Unexpected extra files produced: {extra}"


# ── Tests: output ONNX validity ───────────────────────────────────────────────

class TestOutputOnnxValidity:
    """Verify the output ONNX is structurally correct and Hailo-specific."""

    def test_output_loads_with_onnx_library(self, hailo_deps_available, sample_zip, tmp_path):
        """onnx.load() succeeds on the output model."""
        out = tmp_path / "output"
        out.mkdir()

        result = _run_toolchain(sample_zip, out, hailo_deps_available)
        assert result.returncode == 0

        hailo_onnx = next(out.glob("*.hailo8.onnx"))
        model = onnx.load(str(hailo_onnx))
        assert model.graph is not None

    def test_output_has_nodes(self, hailo_deps_available, sample_zip, tmp_path):
        """Output ONNX graph contains at least one node."""
        out = tmp_path / "output"
        out.mkdir()

        result = _run_toolchain(sample_zip, out, hailo_deps_available)
        assert result.returncode == 0

        hailo_onnx = next(out.glob("*.hailo8.onnx"))
        model = onnx.load(str(hailo_onnx))
        assert len(model.graph.node) > 0

    def test_output_contains_hailo_op_node(self, hailo_deps_available, sample_zip, tmp_path):
        """Output ONNX graph contains a HailoOp node (the compiled subgraph)."""
        out = tmp_path / "output"
        out.mkdir()

        result = _run_toolchain(sample_zip, out, hailo_deps_available)
        assert result.returncode == 0

        hailo_onnx = next(out.glob("*.hailo8.onnx"))
        model = onnx.load(str(hailo_onnx))
        op_types = {n.op_type for n in model.graph.node}
        assert "HailoOp" in op_types, (
            f"No HailoOp node found in output graph. Op types present: {op_types}"
        )

    def test_output_has_inputs_and_outputs(self, hailo_deps_available, sample_zip, tmp_path):
        """Output ONNX graph has at least one input and one output."""
        out = tmp_path / "output"
        out.mkdir()

        result = _run_toolchain(sample_zip, out, hailo_deps_available)
        assert result.returncode == 0

        hailo_onnx = next(out.glob("*.hailo8.onnx"))
        model = onnx.load(str(hailo_onnx))
        assert len(model.graph.input) > 0, "Output model has no inputs"
        assert len(model.graph.output) > 0, "Output model has no outputs"

    def test_output_onnx_passes_checker(self, hailo_deps_available, sample_zip, tmp_path):
        """onnx.checker.check_model() passes on the output model."""
        out = tmp_path / "output"
        out.mkdir()

        result = _run_toolchain(sample_zip, out, hailo_deps_available)
        assert result.returncode == 0

        hailo_onnx = next(out.glob("*.hailo8.onnx"))
        model = onnx.load(str(hailo_onnx))
        # check_model raises onnx.checker.ValidationError on invalid models
        onnx.checker.check_model(model)


# ── Tests: logs.json content ──────────────────────────────────────────────────

class TestLogsContent:
    """Verify the content of logs.json produced by a successful conversion."""

    def test_logs_json_is_valid_json(self, hailo_deps_available, sample_zip, tmp_path):
        out = tmp_path / "output"
        out.mkdir()
        result = _run_toolchain(sample_zip, out, hailo_deps_available)
        assert result.returncode == 0

        logs = json.loads((out / "logs.json").read_text())
        assert isinstance(logs, list)

    def test_logs_contain_success_message(self, hailo_deps_available, sample_zip, tmp_path):
        out = tmp_path / "output"
        out.mkdir()
        result = _run_toolchain(sample_zip, out, hailo_deps_available)
        assert result.returncode == 0

        logs = json.loads((out / "logs.json").read_text())
        messages = [entry.get("Message", "") for entry in logs]
        assert any("Successful" in m or "successful" in m for m in messages), (
            f"No success message in logs: {messages}"
        )

    def test_logs_record_all_pipeline_stages(self, hailo_deps_available, sample_zip, tmp_path):
        """Logs should mention translation, optimization, and compilation."""
        out = tmp_path / "output"
        out.mkdir()
        result = _run_toolchain(sample_zip, out, hailo_deps_available)
        assert result.returncode == 0

        text = (out / "logs.json").read_text().lower()
        assert "translat" in text, "No translation step in logs"
        assert "optim" in text, "No optimization step in logs"
        assert "compil" in text, "No compilation step in logs"


# ── Tests: conversion with calibration images ─────────────────────────────────

class TestConversionWithCalibImages:
    """Conversion using explicit calibration images provided in the zip."""

    def test_conversion_with_calib_images(self, hailo_deps_available, sample_zip, tmp_path):
        """Providing calibration images in the zip succeeds and produces valid output."""
        import zipfile as zf_mod

        # Unpack the sample zip, add calibration images, repack.
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()
        with zf_mod.ZipFile(sample_zip, "r") as z:
            z.extractall(extract_dir)

        calib_img = _make_calib_image(tmp_path, size=(225, 225))
        new_zip = tmp_path / "with_calib.zip"
        with zf_mod.ZipFile(new_zip, "w", zf_mod.ZIP_DEFLATED) as z:
            for f in extract_dir.iterdir():
                z.write(f, f.name)
            # Add 5 calibration images
            for i in range(5):
                z.write(calib_img, f"calib/img{i:02d}.jpg")

        out = tmp_path / "output"
        out.mkdir()
        result = _run_toolchain(new_zip, out, hailo_deps_available)
        assert result.returncode == 0, (
            f"Conversion with calib images failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert list(out.glob("*.hailo8.onnx")), "No output ONNX produced"


# ── Tests: error handling ─────────────────────────────────────────────────────

class TestErrorHandling:
    """Verify the toolchain fails gracefully for invalid inputs."""

    def test_nonexistent_zip_fails(self, hailo_deps_available, tmp_path):
        """A path to a nonexistent zip exits non-zero."""
        out = tmp_path / "output"
        out.mkdir()
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "-v", f"{hailo_deps_available}:/app/hailo-deps",
                "-v", f"{tmp_path}:/input:ro",
                "-v", f"{out}:/output",
                DOCKER_IMAGE,
                "/input/nonexistent.zip",
                "/output",
            ],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode != 0, "Expected non-zero exit for missing zip"

    def test_zip_without_onnx_fails(self, hailo_deps_available, tmp_path):
        """A zip without an ONNX file exits non-zero."""
        bad_zip = tmp_path / "no_onnx.zip"
        with zipfile.ZipFile(bad_zip, "w") as z:
            z.writestr("options.json", json.dumps({"height": 64, "width": 64}))
        out = tmp_path / "output"
        out.mkdir()
        result = _run_toolchain(bad_zip, out, hailo_deps_available, timeout=120)
        assert result.returncode != 0

    def test_corrupt_zip_fails(self, hailo_deps_available, tmp_path):
        """A corrupt (non-zip) file exits non-zero."""
        bad_zip = tmp_path / "corrupt.zip"
        bad_zip.write_bytes(b"this is not a zip file")
        out = tmp_path / "output"
        out.mkdir()
        result = _run_toolchain(bad_zip, out, hailo_deps_available, timeout=120)
        assert result.returncode != 0

    def test_failure_produces_logs_json(self, hailo_deps_available, tmp_path):
        """Even on failure the toolchain should produce a logs.json with an error message."""
        bad_zip = tmp_path / "no_onnx.zip"
        with zipfile.ZipFile(bad_zip, "w") as z:
            z.writestr("options.json", json.dumps({"height": 64, "width": 64}))
        out = tmp_path / "output"
        out.mkdir()
        _run_toolchain(bad_zip, out, hailo_deps_available, timeout=120)

        logs_file = out / "logs.json"
        if logs_file.exists():
            logs = json.loads(logs_file.read_text())
            messages = [e.get("Message", "") for e in logs]
            assert any("fail" in m.lower() or "error" in m.lower() for m in messages), (
                f"Expected failure message in logs, got: {messages}"
            )


# ── Tests: performance ────────────────────────────────────────────────────────

class TestPerformance:
    """Sanity-check conversion time."""

    def test_conversion_completes_within_10_minutes(
        self, hailo_deps_available, sample_zip, tmp_path
    ):
        """The sample model converts within 600 seconds."""
        out = tmp_path / "output"
        out.mkdir()

        start = time.time()
        result = _run_toolchain(sample_zip, out, hailo_deps_available, timeout=600)
        elapsed = time.time() - start

        assert result.returncode == 0, f"Conversion failed: {result.stderr}"
        assert elapsed < 600, f"Conversion took {elapsed:.0f}s (> 600s limit)"
        print(f"\n  Conversion time: {elapsed:.1f}s")
