# Contributing to OAAX Hailo Acceleration

Thank you for your interest in contributing! Before you start, please read the
[OAAX general contributing guide](https://github.com/OAAX-standard/OAAX/blob/main/CONTRIBUTING.md)
for project-wide conventions (bug reports, commit messages and DCO sign-off).

## How to contribute

1. **Open an issue** for a bug or feature, before writing code for anything beyond a small fix.
2. **Fork the repository** and create a branch from `main`.
3. **Make your changes**, keeping each PR focused on one thing.
4. **Run the tests** locally (see below) to confirm nothing is broken.
5. **Sign off every commit** (`git commit -s`) and open a pull request.

## Development environment

CI builds on Ubuntu 22.04 (x86_64). Install the build dependencies once per machine (requires root):

```bash
sudo bash scripts/setup-env.sh
```

`setup-env.sh` is required (the runtime build expects its compilers in `/opt`). It installs system packages and downloads two
compiler toolchains (`x86_64-unknown-linux-gnu-gcc-9.5.0`, `gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu`) and CMake 3.31.7 into `/opt`,
replacing the `cmake` links in `/usr/local/bin`: use a VM or container if you do not want that. The runtime build also needs `patchelf`
(`sudo apt-get install patchelf`), which the script does not install.

Building the conversion toolchain needs Docker, and converting models needs the Hailo SDK packages listed in the
[README](README.md#building-the-conversion-toolchain).

## Running the tests

```bash
bash runtime-library/build-runtime.sh X86_64          # build the runtime first
bash tests/runtime/build-tests.sh                     # then the C++ tests
(cd tests/runtime/build && ./simple_test && ./lifecycle_test && ./queue_test)   # no hardware needed

pip install pytest onnx Pillow                        # Python tests (Docker image must be built first)
pytest tests/test_docker.py::TestErrorHandling -v
```

CI also checks C formatting with clang-format 19 against the `.clang-format` in the repository root; the exact command
is in `.github/workflows/ci.yml`.

Without `HAILO_DEPS_DIR` set, the Docker error-handling tests skip, so a green run only proves the build. The full
conversion and inference tests need the Hailo SDK packages and a Hailo device; see the [README](README.md#testing).

Jobs that upload artifacts need repository secrets and can fail on a PR from a fork; that is not caused by your change.
