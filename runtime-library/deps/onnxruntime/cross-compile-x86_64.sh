#!/bin/bash

set -e

BASE_DIR="$(cd "$(dirname "$0")"; pwd)";

declare -a hailort_versionnames=("4.17.0" "4.18.0" "4.19.0")

## now loop through the above array
for i in "${!hailort_versionnames[@]}"
do

  cd $BASE_DIR
  ort_foldername="onnxruntime-${hailort_versionnames[$i]}"
  cd $ort_foldername

  rm -rf build

  export HailoRT_DIR=$BASE_DIR/../hailort/X86_64_${hailort_versionnames[$i]}
  export HailoRT_SOURCE_DIR=$BASE_DIR/../hailort/hailort-${hailort_versionnames[$i]}
  export PROTO_BIN_DIR=$BASE_DIR/__protoc-3.18.0__/bin/protoc
  export CPATH="$BASE_DIR/../nlohmann/single_include"

  rm -rf $BASE_DIR/$ort_foldername/cmakes
  mkdir $BASE_DIR/$ort_foldername/cmakes
  rm -rf $BASE_DIR/../lib
  mkdir  $BASE_DIR/../lib
  rm -rf $BASE_DIR/../include

  find $HailoRT_SOURCE_DIR -name "*.cmake" -exec cp {} $BASE_DIR/$ort_foldername/cmakes \;
  cp -rf $HailoRT_DIR/include $BASE_DIR/..
  cp -rf $HailoRT_DIR/* $BASE_DIR/../lib

  export PATH=/snap/bin:/opt/x86_64-unknown-linux-gnu-gcc-9.5.0/bin:/usr/bin:/usr/local/bin
  export CPATH="$BASE_DIR/../nlohmann/single_include"
  export CC=/opt/x86_64-unknown-linux-gnu-gcc-9.5.0/bin/x86_64-unknown-linux-gnu-gcc
  export CXX=/opt/x86_64-unknown-linux-gnu-gcc-9.5.0/bin/x86_64-unknown-linux-gnu-g++

bash ./build.sh --config Release --parallel --x86 --skip_tests --use_hailo --cmake_extra_defines \
  CMAKE_POSITION_INDEPENDENT_CODE=ON \
  onnxruntime_CROSS_COMPILING=ON \
  CMAKE_LIBRARY_PATH="$BASE_DIR/../lib" \
  CMAKE_PREFIX_PATH="$BASE_DIR/$ort_foldername/cmakes" \
  onnxruntime_BUILD_UNIT_TESTS=OFF \
  onnxruntime_ENABLE_CPUINFO=OFF

  #build nsync
  cd build/Linux/Release/external/nsync
  make
  cd ../../../../..

  cd ..

  rm -rf X86_64-${hailort_versionnames[$i]} || true
  mkdir X86_64-${hailort_versionnames[$i]}  || true

  cp -rf ./$ort_foldername/include ./X86_64-${hailort_versionnames[$i]}  || true
  RELEASE="./$ort_foldername/build/Linux/Release"  || true
  cp $RELEASE/*.a ./X86_64-${hailort_versionnames[$i]} 2>>/dev/null  || true
  cp $RELEASE/*.so ./X86_64-${hailort_versionnames[$i]} 2>>/dev/null  || true

  shopt -s globstar  || true
  cp -rf $RELEASE/**/*.a ./X86_64-${hailort_versionnames[$i]} 2>>/dev/null  || true

  rm -rf $BASE_DIR/../lib
  rm -rf $BASE_DIR/../include
  rm -rf $BASE_DIR/$ort_foldername/cmakes

  find $BASE_DIR -name "*.so*" -exec chmod 644 {} \;

  printf "\nDone :) X86_64-${hailort_versionnames[$i]} \n"
done


