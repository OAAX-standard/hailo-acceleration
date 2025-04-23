# OAAX Hailo runtime library

This Hailo implementation of an OAAX runtime is a wrapper around the [ONNX Runtime](https://github.com/hailo-ai/onnxruntime) with the Hailo execution provider.

## Pre-requisites

Before you start, make sure you have set up your environment using the following script:

```bash
sudo bash scripts/setup-env.sh
```

This will install the required dependencies and set up the environment for building the OAAX runtime.
The script will also set up the cross-compilation toolchain for the target architecture (X86_64 and AARCH64).

## Getting started

The OAAX runtime is leveraging the ONNX Runtime library to load and run the model. ORT requires
the [CPU INFOrmation library](https://github.com/pytorch/cpuinfo), and the [RE2 library](https://github.com/google/re2), and the [HailoRT library](https://github.com/hailo-ai/hailort).

All of these dependencies are included in the `deps` directory, and are already cross-compiled for the target architectures using the compiler toolchain set up using the shell script above.
However, you can recompile them separately by running the shell scripts inside each directory.

To build the OAAX runtime, run the following command:

```bash
bash build-runtimes.sh
```

This will create an `artifacts/` directory containing the compiled shared library: `libRuntimeLibrary.so` along with other HailoRT-related libraries.

## Running Inference using the OAAX runtime

### Requirements

The target machine must have Ubuntu 20.04 or later installed (or equivalent Debian version), and the HailoRT driver must also be installed.

### Usage

To run the inference process, you need to have the compiled model file, the shared library built in the previous step, along with a simple C/C++ code that loads the shared library and the model and runs it. Ensure that you have all necessary HailoRT-related libraries available and reachable to the AI application by setting the `LD_LIBRARY_PATH` environment variable to include the path to the HailoRT libraries.

You can find diverse examples and applications of using the OAAX runtime in the
[examples](https://github.com/oaax-standard/examples) repository.
