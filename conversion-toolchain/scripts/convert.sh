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
# The HailoRT wheel is tagged per Python version (cp310 for the image's Python 3.10).
py_tag="cp$(python -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"
pip install ../hailo-deps/hailort-4.*.0-"$py_tag"-"$py_tag"-linux_x86_64.whl

conversion_toolchain --zip-path "$onnx_path" --output-dir "$output_directory"
