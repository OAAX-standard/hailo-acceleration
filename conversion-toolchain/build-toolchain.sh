set -e

cd "$(dirname "$0")" || exit 1

# Build the toolchain as a Docker image
docker build -t onnx-to-hailo:latest .

# Save the Docker image as a tarball
mkdir -p artifacts
docker save onnx-to-hailo:latest -o ./artifacts/oaax-hailo-toolchain.tar

# You can load and run the toolchain with:
#   docker load -i ./artifacts/oaax-hailo-toolchain.tar
#   docker run -v /path/to/hailo-deps:/app/hailo-deps \
#              -v /path/to/input:/app/input \
#              -v /path/to/output:/app/output \
#              onnx-to-hailo:latest /app/input/model.zip /app/output
