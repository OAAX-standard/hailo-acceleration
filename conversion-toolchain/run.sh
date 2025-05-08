set -e

cd "$(dirname "$0")" || exit 1

hailort_pkg="/home/robin/Ayoub-Development/conversion_with_ayoub/hailort_4.20.0_amd64.deb"
hailort_wheel="/home/robin/Ayoub-Development/conversion_with_ayoub/hailort-4.20.0-cp38-cp38-linux_x86_64.whl"
hailort_dfc="/home/robin/Ayoub-Development/conversion_with_ayoub/hailo_dataflow_compiler-3.30.0-py3-none-linux_x86_64.whl"

# create hailo-deps2 directory if it doesn't exist
mkdir -p ./tmp-hailo-deps2

# copy the files to hailo-deps2 directory
cp "$hailort_pkg" ./tmp-hailo-deps2/
cp "$hailort_wheel" ./tmp-hailo-deps2/
cp "$hailort_dfc" ./tmp-hailo-deps2/

# Load docker image
# docker load -i ./artifacts/*.tar

cd ./tmp-artifacts
sudo rm -rf ./model.zip output/
zip -r model.zip ./*

cd ..

# Run the docker container
docker run -v ./tmp-hailo-deps2/:/app/hailo-deps \
    -v ./tmp-artifacts:/app/input \
    -v ./tmp-artifacts/output:/app/output \
    oaax-hailo-toolchain:1.0.0 /app/input/model.zip /app/output
