set -e

cd "$(dirname "$0")"

BUILD_DIR="$(pwd)/build"
ARTIFACTS_DIR="$(pwd)/artifacts"
mkdir -p $BUILD_DIR
rm -rf $ARTIFACTS_DIR
mkdir -p $ARTIFACTS_DIR

# read platform from command line argument
if [ "$#" -ne 1 ]; then
  PLATFORMS=("X86_64" "AARCH64")
  echo "No platform specified. Building for both X86_64 and AARCH64."
  echo "You can specify a platform by running: ./build-runtimes.sh <X86_64|AARCH64>"
  sleep 1
else
  if [ "$1" != "X86_64" ] && [ "$1" != "AARCH64" ]; then
    echo "Invalid platform specified. Use X86_64 or AARCH64."
    exit 1
  fi
  echo "Platform specified: $1"
  PLATFORMS=("$1")
fi

cd ${BUILD_DIR}

HAILORT_VERSIONS=("4.17.0" "4.18.0" "4.19.0" "4.20.0")

for hailort_version in "${HAILORT_VERSIONS[@]}"; do
  for platform in "${PLATFORMS[@]}"; do
    echo ">>>> Building for platform: $platform && HailoRT version: $hailort_version"
    rm -rf *
    cmake .. -DPLATFORM=$platform -DHAILORT_VERSION=$hailort_version
    make -j
    echo "Build complete. The following shared libraries were created:"
    ls ./*.so*
    echo "Copying shared libraries to artifacts directory..."
    mkdir -p ${ARTIFACTS_DIR}/$platform-${hailort_version}/
    cp ./*.so* ${ARTIFACTS_DIR}/$platform-${hailort_version}/
    tar czf ${ARTIFACTS_DIR}/runtime-library-${platform}-${hailort_version}.tar.gz -C ${ARTIFACTS_DIR}/$platform-${hailort_version} ./*.so*
    echo "Shared libraries for $platform have been copied to ${ARTIFACTS_DIR}/$platform-${hailort_version}/"
  done
done
