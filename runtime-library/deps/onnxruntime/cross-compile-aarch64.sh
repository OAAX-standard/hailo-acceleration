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

  export HailoRT_DIR=$BASE_DIR/../hailort/AARCH64_${hailort_versionnames[$i]}
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


  # export CROSS_NAME=gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu
  export CROSS_NAME=gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu
  export CROSS_ROOT=/opt/${CROSS_NAME}
  export COMPILER_PREFIX=aarch64-none-linux-gnu-
  export CPPFLAGS="-I${CROSS_ROOT}/include"
  export CFLAGS="-I${CROSS_ROOT}/include"
  export AR=${CROSS_ROOT}/bin/${COMPILER_PREFIX}ar
  export AS=${CROSS_ROOT}/bin/${COMPILER_PREFIX}as
  export LD=${CROSS_ROOT}/bin/${COMPILER_PREFIX}ld
  export RANLIB=${CROSS_ROOT}/bin/${COMPILER_PREFIX}ranlib
  export CC=${CROSS_ROOT}/bin/${COMPILER_PREFIX}gcc
  export CXX=${CROSS_ROOT}/bin/${COMPILER_PREFIX}g++
  export NM=${CROSS_ROOT}/bin/${COMPILER_PREFIX}nm

bash ./build.sh --config Release --parallel --arm64 --skip_tests --use_hailo --cmake_extra_defines \
  CMAKE_POSITION_INDEPENDENT_CODE=ON \
    onnxruntime_BUILD_UNIT_TESTS=OFF \
    CMAKE_BUILD_TYPE=Release \
    CPUINFO_TARGET_PROCESSOR=arm64 \
    CMAKE_SYSTEM_PROCESSOR=aarch64 \
    CPUINFO_BUILD_BENCHMARKS=OFF \
    CPUINFO_RUNTIME_TYPE=static \
    CMAKE_SYSTEM_NAME=Linux \
    CMAKE_C_COMPILER=${CC} \
    CMAKE_CXX_COMPILER=${CXX} \
    CMAKE_LIBRARY_PATH="$BASE_DIR/../lib" \
    CMAKE_PREFIX_PATH="$BASE_DIR/$ort_foldername/cmakes" \
    CMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
    CMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
    CMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY \
    CMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ONLY \
    onnxruntime_CROSS_AARCH64COMPILING=ON \
    onnxruntime_CROSS_COMPILING=ON \
    onnxruntime_ENABLE_CPUINFO=ON \
    ONNX_CUSTOM_PROTOC_EXECUTABLE=${PROTO_BIN_DIR}

  #build nsync
  cd build/Linux/Release/external/nsync
  make
  cd ../../../../..

  cd ..

  rm -rf AARCH64-${hailort_versionnames[$i]} || true
  mkdir AARCH64-${hailort_versionnames[$i]}  || true

  cp -rf ./$ort_foldername/include ./AARCH64-${hailort_versionnames[$i]}  || true
  RELEASE="./$ort_foldername/build/Linux/Release"  || true
  cp $RELEASE/*.a ./AARCH64-${hailort_versionnames[$i]} 2>>/dev/null  || true
  cp $RELEASE/*.so ./AARCH64-${hailort_versionnames[$i]} 2>>/dev/null  || true

  shopt -s globstar  || true
  cp -rf $RELEASE/**/*.a ./AARCH64-${hailort_versionnames[$i]} 2>>/dev/null  || true

  rm -rf $BASE_DIR/../lib
  rm -rf $BASE_DIR/../include
  rm -rf $BASE_DIR/$ort_foldername/cmakes

  find $BASE_DIR -name "*.so*" -exec chmod 644 {} \;

  printf "\nDone :) AARCH64-${hailort_versionnames[$i]} \n"
done

