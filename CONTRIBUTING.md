# Contributing to OAAX Hailo Acceleration

Thank you for your interest in contributing! Before you start, please read the
[OAAX general contributing guide](https://github.com/OAAX-standard/OAAX/blob/main/CONTRIBUTING.md)
for project-wide conventions (bug reports, commit messages and DCO sign-off, code style).

## How to contribute

1. **Open an issue** to discuss the bug or feature before writing code.
2. **Fork the repository** and create a branch from `main`.
3. **Make your changes**, keeping each PR focused on one thing.
4. **Run the tests** locally (see below) to confirm nothing is broken.
5. **Sign off every commit** (`git commit -s`) and open a pull request.

## Development environment

CI builds on Ubuntu 22.04 (x86_64). Install the cross-compilation toolchains once per machine (requires root):

```bash
sudo bash scripts/setup-env.sh
```

Building the conversion toolchain needs Docker, and converting models needs the Hailo SDK packages listed in the
[README](README.md#building-the-conversion-toolchain).

## Running the tests

```bash
bash runtime-library/build-runtime.sh X86_64          # build the runtime first
bash tests/runtime/build-tests.sh                     # then the C++ tests
cd tests/runtime/build && ./simple_test && ./lifecycle_test && ./queue_test   # no hardware needed

pip install pytest onnx Pillow                        # Python tests (Docker image must be built first)
pytest tests/test_docker.py::TestErrorHandling -v
```

Without `HAILO_DEPS_DIR` set, the Docker error-handling tests skip, so a green run only proves the build. The full
conversion and inference tests need the Hailo SDK packages and a Hailo device; see the [README](README.md#testing).
