# Tests

## Runtime tests (C++)

Test the OAAX v2 runtime API against the Hailo backend.

### Build

```bash
bash runtime-library/build-runtime.sh X86_64
bash tests/runtime/build-tests.sh
```

### Run

**Without a model** (API guards, lifecycle — no hardware required):
```bash
cd tests/runtime/build
./simple_test
./lifecycle_test
```

**With a Hailo ONNX model** (requires Hailo device):
```bash
./simple_test    <model.hailo8.onnx>
./lifecycle_test <model.hailo8.onnx> [--input-name NAME] [--imgsz N]
./multi_model_test <model.hailo8.onnx> [model2.hailo8.onnx]
./yolo_test      <model.hailo8.onnx> [--runs N] [--warmup N] [--in-flight N]
```

---

## Conversion toolchain tests (Python)

End-to-end integration tests: a model is fed into the Docker-based Hailo
conversion toolchain and the output is verified.

### Prerequisites

1. **Docker image built:**
   ```bash
   bash conversion-toolchain/build-toolchain.sh
   ```

2. **Hailo SDK deps** in a local directory (`HAILO_DEPS_DIR`):
   - `hailo_dataflow_compiler-<version>-py3-none-linux_x86_64.whl`
   - `hailort-<version>-cp38-cp38-linux_x86_64.whl`
   - `hailort_<version>_amd64.deb`

3. **Python dependencies:**
   ```bash
   pip install pytest onnx Pillow
   ```

### Run

```bash
HAILO_DEPS_DIR=/path/to/hailo-deps pytest tests/test_docker.py -v
```

### What is tested

| Class | What it covers |
|---|---|
| `TestConversionOutput` | Output files exist and are named correctly |
| `TestOutputOnnxValidity` | Output ONNX loads, has nodes, contains a HailoOp, passes `onnx.checker` |
| `TestLogsContent` | `logs.json` is valid JSON, contains success + all pipeline stages |
| `TestConversionWithCalibImages` | Conversion succeeds when calibration images are included in the zip |
| `TestErrorHandling` | Bad/corrupt/incomplete inputs exit non-zero gracefully |
| `TestPerformance` | Conversion completes within 10 minutes |

All tests use the sample model in `conversion-toolchain/artifacts/model.zip` as input,
or build a modified zip on the fly.
