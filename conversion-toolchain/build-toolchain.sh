set -e

cd "$(dirname "$0")" || exit 1

rm -rf build 2&> /dev/null || true
mkdir build

# Build the toolchain as a Docker image
docker build -t onnx-to-hailo:latest .

# Save the Docker image as a tarball
docker save onnx-to-hailo:latest -o ./build/onnx-to-hailo-latest.tar

# You can run the conversion toolchain using the following command:
#docker load  -i ./build/onnx-to-hailo-latest.tar
#docker run -v ./hailo-deps:/app/hailo-deps -v ./artifacts:/app2 onnx-to-hailo:latest /app2/model.zip /app2

