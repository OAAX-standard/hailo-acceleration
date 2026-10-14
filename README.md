# Hailo-8 Acceleration

[![CI](https://github.com/OAAX-standard/hailo-acceleration/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/OAAX-standard/hailo-acceleration/actions/workflows/ci.yml)

OAAX v2 runtime and conversion toolchain for Hailo-8 AI accelerator chips. Provides a C shared library and a Docker-based conversion toolchain that together form a complete inference pipeline on Hailo hardware.

> For a general introduction to the OAAX standard, see the [OAAX repository](https://github.com/oaax-standard/OAAX).

## How the two components fit together

```
ONNX model
    │
    ▼
[Conversion Toolchain]  ── Docker image; translates, quantizes, and compiles the model
    │                       for Hailo-8 hardware, outputs a *.hailo8.onnx file
    ▼
Hailo ONNX model
    │
    ▼
[Runtime Library]  ── C shared library; loads the Hailo ONNX and runs async inference
    │                  via the OAAX v2 C API (oaax_runtime.h)
    ▼
Inference results
```

## Repository structure

```
conversion-toolchain/   Docker-based model compiler (ONNX → Hailo ONNX)
runtime-library/        C shared library (OAAX v2 API)
  include/
    oaax_runtime.h      Public API header
  src/
    runtime_core.c      API implementation (Hailo ORT execution provider)
    runtime_utils.c     ORT helpers, type mapping, free_tensors
  deps/
    onnxruntime/        Pre-built Hailo ORT static libraries per HailoRT version
                        (+ the ORT source trees they were built from, used only by
                        cross-compile-*.sh, not by CMake)
    hailort/            HailoRT sources and pre-built shared libraries
    utils/              Queue and logger
tests/
  runtime/              C++ test binaries (simple_test, lifecycle_test, queue_test, ...)
  test_docker.py        Docker conversion integration tests
  test_yolov8n_conversion.py  YOLOv8n end-to-end convert + run test
  test_yolo_models.py   Parametrized multi-model conversion suite
scripts/
  setup-env.sh          Install cross-compilation toolchains
.github/workflows/      CI (lint, toolchain build, runtime build + tests, S3 upload)
VERSION, CHANGELOG.md   Release version and history
ThirdPartyNotices.txt   Licenses of bundled third-party code
```

## Building the runtime library

```bash
# Install cross-compilation toolchains (once, requires root)
sudo bash scripts/setup-env.sh

# Build for X86_64 (default) or AARCH64
bash runtime-library/build-runtime.sh X86_64
bash runtime-library/build-runtime.sh AARCH64
```

Output: `runtime-library/artifacts/runtime-library-<PLATFORM>-<HAILORT_VERSION>.tar.gz`

The X86_64 build targets Haswell or newer CPUs (`-march=haswell`, AVX2).

Each archive contains:
- `libRuntimeLibrary.so` — the OAAX runtime
- `libonnxruntime_providers_hailo.so` — Hailo execution provider for ORT
- `libonnxruntime_providers_shared.so` — ORT shared provider
- `libhailort.so.*` — HailoRT library
- `oaax_runtime.h` — public API header

Supported HailoRT versions: `4.17.0`, `4.18.0`, `4.19.0`, `4.20.0` (default `4.20.0`).
Override with a second argument:
```bash
bash runtime-library/build-runtime.sh X86_64 4.19.0
```

## Building the conversion toolchain

The toolchain requires the following Hailo SDK packages mounted at `/app/hailo-deps`
inside the container at runtime (not at build time):

- `hailo_dataflow_compiler-<version>-py3-none-linux_x86_64.whl`
- `hailort-<version>-cp38-cp38-linux_x86_64.whl`
- `hailort_<version>_amd64.deb`

```bash
bash conversion-toolchain/build-toolchain.sh
```

Output: `conversion-toolchain/artifacts/oaax-hailo-toolchain.tar`

### Running the conversion

Pack your model into a zip file containing:

| File | Description |
|---|---|
| `model.onnx` | The ONNX model to convert |
| `options.json` | Conversion config (see below) |
| `calib/*.jpg` | Calibration images (optional — built-in images used if absent) |

**`options.json` format:**
```json
{
  "start_node_names": [],
  "end_node_names":   [],
  "height":   640,
  "width":    640,
  "channels": 3,
  "nchw":     false,
  "means":    [0.0, 0.0, 0.0],
  "stds":     [255.0, 255.0, 255.0]
}
```

> **`end_node_names`** can be left empty (`[]`). The toolchain auto-detects the
> appropriate cut points for YOLO-family models (cv2/cv3 Conv outputs before the
> DFL decode head, which Hailo SDK does not support).

> **`nchw`** controls calibration image layout. Always set to `false` — Hailo
> expects NHWC calibration data regardless of the model's actual input format;
> the SDK handles the transposition internally.

```bash
docker run \
  -v /path/to/hailo-deps:/app/hailo-deps \
  -v /path/to/input:/app/input:ro \
  -v /path/to/output:/app/output \
  onnx-to-hailo:latest /app/input/model.zip /app/output
```

Output: `<model-name>.hailo8.onnx` + `logs.json`

## Runtime API (v2)

The public interface is `runtime-library/include/oaax_runtime.h`.

### Call sequence

```c
// 1. Initialize (once)
const char *keys[]   = {"log_level", "log_file"};
const char *values[] = {"2",         "runtime.log"};
Config cfg = {2, keys, values};
runtime_init(cfg);

// 2. Load one or more models
ModelConfig mc = {.file_path = "model.hailo8.onnx"};
runtime_load_models(1, &mc);

// 3. Async inference loop
runtime_enqueue_input(0, input_tensors);          // non-blocking
runtime_retrieve_output(&model_id, &out, 5000);   // 5 s timeout

// 4. Cleanup
runtime_cleanup();   // idempotent
```

### Key types

| Type | Description |
|---|---|
| `RuntimeStatus` | 19-value enum — check every return value |
| `Tensors` | `{id, num_tensors, TensorDescriptor*}` — owns its data |
| `TensorDescriptor` | `{name, data_type, rank, shape, data_size, data}` |
| `Config` | Key-value string pairs passed to `runtime_init` |
| `ModelConfig` | `{file_path, model_data, model_size, config}` — file or in-memory |

### `runtime_init` config keys

| Key | Values | Default |
|---|---|---|
| `"log_level"` | `0`=DEBUG … `3`=ERROR | `1` (INFO) |
| `"log_file"` | file path | `"runtime.log"` |
| `"n_threads"` | 1–16 total ORT threads, split evenly across all model replicas | `4` |
| `"perf_mode"` | `"eco"` (40% of cores) or `"power"` (90%); used only if `n_threads` is unset | unset |
| `"n_replicas"` | ≥ 1 ORT sessions (worker threads) per model | `1` |

`ModelConfig.config` is currently unused; all tuning goes through `runtime_init`.

### Queues

Each model has an input queue and the runtime has one output queue, both holding
100 items. When a queue is full the **oldest** item is dropped (and freed) to
make room, so `runtime_enqueue_input` still returns success. Keep
`runtime_retrieve_output` draining results. Ownership of `input_tensors` passes
to the runtime only on success.

On any non-`RUNTIME_STATUS_SUCCESS` return, call `runtime_get_error()` for a
human-readable description.

## Model compatibility

Tested on Hailo SDK 3.25.0 / HailoRT 4.20.0:

| Model | Conversion | Throughput (640×640, batch=1) |
|---|---|---|
| YOLOv8n | ✅ | ~67 img/s |
| YOLOv8s | ✅ | ~57 img/s |
| YOLO11n | ❌ | Backbone attention layers unsupported in SDK 3.25.0 |
| YOLO26n | ❌ | Backbone attention layers unsupported in SDK 3.25.0 |

> No models are shipped in this repository. The YOLO models above come from
> Ultralytics and are licensed under AGPL-3.0; export them yourself and check
> that license before redistributing them or their Hailo conversions.

## Testing

### C++ runtime tests (no hardware required for API tests)

```bash
# Build the runtime first, then build the tests
bash tests/runtime/build-tests.sh

cd tests/runtime/build

# API-only (no model needed)
./simple_test
./lifecycle_test
./queue_test

# Full inference (requires Hailo device + a converted model)
./lifecycle_test   model.hailo8.onnx --input-name input0 --imgsz 640
./yolo_test        model.hailo8.onnx --input-name input0 --runs 50
./multi_model_test model.hailo8.onnx
```

### Python integration tests

```bash
pip install pytest onnx Pillow

# Docker conversion error-handling (Docker image must be built)
pytest tests/test_docker.py::TestErrorHandling -v

# Full conversion + chip inference (requires Hailo SDK deps + hardware)
HAILO_DEPS_DIR=/path/to/hailo-deps pytest tests/test_docker.py -v
pytest tests/test_yolov8n_conversion.py -v -s   # ~6 min
pytest tests/test_yolo_models.py -v -s           # ~10 min per model
```

The YOLO tests look for `yolov8n.onnx`, `yolov8s.onnx`, `yolo11n.onnx` and
`yolo26n.onnx` in the repo root. Override with the `YOLOV8N_ONNX`,
`YOLO11N_ONNX` and `YOLO26N_ONNX` env vars; `YOLOV8N_ZIP` sets the calibration
zip location (default: system temp dir).

See [tests/README.md](tests/README.md) for full details.

## Pre-built artifacts

Pre-built runtimes (one per HailoRT version) are published to S3 on every CI run (`latest` only from `main`; `<version>` is the content of `VERSION` on `main`, the branch name on PRs):

```
s3://oaax/runtimes/<version>/HAILO/{x86_64,aarch64}/Ubuntu/library-<hailort_version>.tar.gz
s3://oaax/runtimes/latest/HAILO/{x86_64,aarch64}/Ubuntu/library-<hailort_version>.tar.gz

s3://oaax/conversion-toolchain/<version>/HAILO/oaax-hailo-toolchain.tar
s3://oaax/conversion-toolchain/latest/HAILO/oaax-hailo-toolchain.tar
```

Community contributions and usage examples are available in the
[OAAX contributions](https://github.com/oaax-standard/contributions) and
[examples](https://github.com/oaax-standard/examples) repositories.
