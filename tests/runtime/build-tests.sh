#!/bin/bash
# Build the C++ runtime tests against the prebuilt runtime library.
# Run runtime-library/build-runtime.sh first so the library is in place.
set -e

cd "$(dirname "$0")"

RUNTIME_LIB_DIR="${RUNTIME_LIB_DIR:-$(pwd)/../../runtime-library/build}"
BUILD_DIR="$(pwd)/build"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. -DRUNTIME_LIB_DIR="$RUNTIME_LIB_DIR" -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)"

# Symlink shared libs into the build dir so $ORIGIN RPATH resolves at runtime.
# This includes libRuntimeLibrary.so, the Hailo provider libs, and libhailort.
for lib in "$RUNTIME_LIB_DIR"/*.so*; do
    [ -e "$lib" ] || continue
    ln -sf "$lib" .
done

echo ""
echo "Tests built in: $BUILD_DIR"
echo ""
echo "Run without a model (API-only tests):"
echo "  ./simple_test"
echo "  ./lifecycle_test"
echo ""
echo "Run with a Hailo ONNX model:"
echo "  ./simple_test <model.onnx>"
echo "  ./lifecycle_test <model.onnx> [--input-name NAME] [--imgsz N]"
echo "  ./multi_model_test <model.onnx> [model2.onnx] [--input-name NAME] [--imgsz N]"
echo "  ./yolo_test <model.onnx> [--runs N] [--warmup N] [--in-flight N] [--imgsz N]"
