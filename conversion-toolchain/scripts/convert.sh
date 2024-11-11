set -e

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <onnx-path> <output-directory>" >&2
  exit 1
fi
onnx_path=$(realpath "$1")
output_directory=$(realpath "$2")

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR" || exit 1

yes | dpkg -i ../hailo-deps/hailort_4.*.0_amd64.deb
pip install ../hailo-deps/hailo_dataflow_compiler-3.*.0-py3-none-linux_x86_64.whl
pip install ../hailo-deps/hailort-4.*.0-cp38-cp38-linux_x86_64.whl

conversion_toolchain --zip-path "$onnx_path" --output-dir "$output_directory"
