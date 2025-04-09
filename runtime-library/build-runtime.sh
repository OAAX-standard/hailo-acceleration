set -e

cd "$(dirname "$0")" || exit 1

rm -rf build 2&> /dev/null || true
mkdir build

cd build

PLATFORM=X86_64 # or AAARCH64
HAILORT_VERSION=4.20.0
cmake .. -DPLATFORM=$PLATFORM -DHAILORT_VERSION=$HAILORT_VERSION
make -j