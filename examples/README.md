# Full example

## Overview

This example demonstrates how to use the OAAX toolchain to compile and run a face detection model on Hailo-8. The example includes the following steps:
1. Pre-requisites
2. Model compilation using the OAAX toolchain
3. Model inference using the OAAX runtime

## Pre-requisites

Before you start, make sure you have (prefereably) two machines, a host and a target machine. The host machine should be x86_64 based, and the target machine should be running Ubuntu 20+ and have a Hailo chip attached and a compatible driver installed (for example, v4.20.0).

On the host machine, you need to have Docker installed to be able to use the toolchain. The toolchain can either be built from source or downloaded from the [contributions repository](https://github.com/oaax-standard/contributions?tab=readme-ov-file#overview).
Moreover, you need to download some Hailo DFC dependencies, which are available in the Hailo Developer Zone, and them available on the host machine in one folder (for consistency with the example, the folder should be named `./hailo-deps`). The dependencies are:
- `hailort_4.20.0_amd64.deb`
- `hailort-4.20.0-cp38-cp38-linux_x86_64.whl`
- `hailo_dataflow_compiler-3.30.0-py3-none-linux_x86_64.whl`

On the target machine, you need to have the HailoRT driver installed, and have downloaded the precompiled the runtime library from the [contributions repository](https://github.com/oaax-standard/contributions?tab=readme-ov-file#overview).


## Model Compilation using the OAAX toolchain

1. Assuming you have downloaded the precompiled toolchain in the **present working directory**, you can run the following command to load toolchain:

```bash
docker load -i ./oaax-hailo-toolchain.tar
```
This will load the toolchain into Docker. You can then run the toolchain using the following command:

2. The next step is to prepare the toolchain model archive, it needs to contain the ONNX model, configuration JSON, and calibration images. The files are all available in the [resources](resources) folder. You can create the archive using this command:

```bash
cd resources
zip -r input.zip model.onnx config.json val2017/
```

3. Run the toolchain:

```bash
docker run -it --rm -v $(pwd):/workspace -w /workspace oaax-hailo-toolchain:latest
```

## Model Inference using the OAAX runtime