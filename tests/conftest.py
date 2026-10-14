"""
Shared fixtures for the Hailo conversion toolchain test suite.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

# Path to the sample model.zip shipped in the repo.
SAMPLE_ZIP = Path(__file__).parents[1] / "conversion-toolchain" / "artifacts" / "model.zip"
SAMPLE_ONNX = Path(__file__).parents[1] / "conversion-toolchain" / "artifacts" / "model.onnx"

DOCKER_IMAGE = "onnx-to-hailo:latest"

# Optional path to Hailo SDK deps (needed for full Docker conversion tests).
# Set HAILO_DEPS_DIR=/path/to/hailo-deps before running to enable those tests.
HAILO_DEPS_DIR = os.environ.get("HAILO_DEPS_DIR", "")


def _docker_image_available() -> bool:
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
        if r.returncode != 0:
            return False
        r = subprocess.run(
            ["docker", "images", "-q", DOCKER_IMAGE],
            capture_output=True, text=True, timeout=5,
        )
        return bool(r.stdout.strip())
    except Exception:
        return False


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


@pytest.fixture(scope="session")
def sample_zip():
    if not SAMPLE_ZIP.exists():
        pytest.skip(f"Sample zip not found: {SAMPLE_ZIP}")
    return SAMPLE_ZIP


@pytest.fixture(scope="session")
def sample_onnx():
    if not SAMPLE_ONNX.exists():
        pytest.skip(f"Sample ONNX not found: {SAMPLE_ONNX}")
    return SAMPLE_ONNX


@pytest.fixture(scope="session")
def docker_available():
    if not _docker_image_available():
        pytest.skip(
            f"Docker image '{DOCKER_IMAGE}' not available. "
            "Build it with: bash conversion-toolchain/build-toolchain.sh"
        )


@pytest.fixture(scope="session")
def hailo_deps_available(docker_available):
    """Requires both Docker image and Hailo SDK deps directory."""
    if not HAILO_DEPS_DIR or not Path(HAILO_DEPS_DIR).is_dir():
        pytest.skip(
            "Hailo SDK deps directory not set or not found. "
            "Set HAILO_DEPS_DIR=/path/to/hailo-deps to enable full conversion tests."
        )
    return Path(HAILO_DEPS_DIR)
