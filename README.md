# Hailo-8 Acceleration

OAAX runtime and conversion toolchain for Hailo-8 AI accelerator chips. Provides a shared C library and a Docker-based conversion toolchain that together form a complete inference pipeline.

## How the two components fit together

```
ONNX model
    │
    ▼
[Conversion Toolchain]  — Docker image; wraps the model in a HailoOp subgraph
    │
    ▼
Hailo ONNX model
    │
    ▼
[Runtime Library]  — C shared library; loads the model and runs async inference via oaax_runtime.h
    │
    ▼
Inference results
```

## Repository structure

- [Conversion toolchain](conversion-toolchain): Docker-based model optimizer (ONNX → Hailo ONNX).
- [Runtime library](runtime-library): C shared library implementing the OAAX v2 API (`oaax_runtime.h`).

## Building the runtime library

```bash
bash runtime-library/build-runtime.sh
```

Output: `runtime-library/build/libRuntimeLibrary.so`

The build defaults to `PLATFORM=X86_64` and `HAILORT_VERSION=4.20.0`. Edit `build-runtime.sh` or pass CMake flags directly to override.

## Building the conversion toolchain

The toolchain requires the following Hailo dependencies mounted at `/app/hailo-deps` inside the container:

- `hailo_dataflow_compiler-<version>-py3-none-linux_x86_64.whl`
- `hailort-<version>-cp38-cp38-linux_x86_64.whl`
- `hailort_<version>_amd64.deb`

```bash
bash conversion-toolchain/build-toolchain.sh
```

### Running the toolchain

The input must be a zip file containing: an ONNX model, a calibration image folder, and a `options.json` config:

```json
{
  "start_node_names": ["Conv1"],
  "end_node_names": ["Softmax1"],
  "height": 416, "width": 416, "channels": 3,
  "nchw": true,
  "means": [0.485, 0.456, 0.406],
  "stds":  [0.229, 0.224, 0.225]
}
```

```bash
docker run -v /path/to/hailo-deps:/app/hailo-deps \
           -v /path/to/input:/app/input \
           -v /path/to/output:/app/output \
           onnx-to-hailo:latest /app/input/input.zip /app/output
```

## Runtime API (v2)

The public interface is `runtime-library/include/oaax_runtime.h`. The expected call sequence is:

```c
runtime_init(config);
runtime_load_models(n, model_configs);

// async inference loop:
runtime_enqueue_input(model_id, input_tensors);   // non-blocking
runtime_retrieve_output(&model_id, &output, ms);  // timeout in ms

runtime_cleanup();
```

Key types: `Config`, `ModelConfig`, `Tensors`, `TensorDescriptor`, `RuntimeStatus`.

Config keys accepted by `runtime_init`: `"log_level"` (0–3), `"log_file"` (path).  
Config keys accepted per model in `ModelConfig.config`: `"n_threads"` (default 4).

On any non-`RUNTIME_STATUS_SUCCESS` return, call `runtime_get_error()` for a description.

## Pre-built artifacts

Pre-built runtimes and usage examples are available in the [OAAX contributions](https://github.com/oaax-standard/contributions) and [examples](https://github.com/oaax-standard/examples) repositories.
