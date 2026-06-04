set -e

cd "$(dirname "$0")" || exit 1

PLATFORM="${1:-X86_64}"
HAILORT_VERSION="${2:-4.20.0}"

rm -rf build 2>/dev/null || true
mkdir build
cd build

cmake .. -DPLATFORM="$PLATFORM" -DHAILORT_VERSION="$HAILORT_VERSION"
make -j"$(nproc)"

# Package all runtime libraries + public header into a distributable archive.
cd ..
mkdir -p artifacts

STAGING=$(mktemp -d)
cp build/libRuntimeLibrary.so                   "$STAGING/"
cp build/libonnxruntime_providers_hailo.so      "$STAGING/"
cp build/libonnxruntime_providers_shared.so     "$STAGING/"
cp build/libhailort.so.*                        "$STAGING/" 2>/dev/null || true
cp include/oaax_runtime.h                       "$STAGING/"

tar czf "artifacts/runtime-library-${PLATFORM}.tar.gz" -C "$STAGING" .
rm -rf "$STAGING"

echo "Artifact: runtime-library/artifacts/runtime-library-${PLATFORM}.tar.gz"
