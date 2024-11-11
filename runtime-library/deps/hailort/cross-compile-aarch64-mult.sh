#!/bin/bash

set -e

BASE_DIR="$(cd "$(dirname "$0")"; pwd)";
cd $BASE_DIR
declare -a hailort_versionnames=("4.17.0" "4.18.0" "4.19.0")

# create source and destination folders
declare -a hailort_source_foldernames=("${hailort_versionnames[@]/#/hailort-}")
declare -a hailort_destination_foldernames=("${hailort_versionnames[@]/#/AARCH64_}")

## now loop through the above array
for i in "${!hailort_source_foldernames[@]}"
do
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

  cd ${hailort_source_foldernames[$i]}
  rm -rf build

  cmake -H. -Bbuild  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=${CC} \
  -DCMAKE_CXX_COMPILER=${CXX} \
  -DCMAKE_LINKER=${LD} \
  -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
  -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
  -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY \
  -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ONLY


  cmake --build build --config release --target libhailort -j8

  cd build

  make -j

  cd ../..

  rm -rf ${hailort_destination_foldernames[$i]} || true
  mkdir ${hailort_destination_foldernames[$i]}

  # read version from hailort_foldername
  version=$(echo ${hailort_source_foldernames[$i]} | cut -d'-' -f2)
  cp -rf ./${hailort_source_foldernames[$i]}/hailort/libhailort/include ./${hailort_destination_foldernames[$i]}
  cp ./${hailort_source_foldernames[$i]}/build/hailort/libhailort/src/libhailort.so.4.* ./${hailort_destination_foldernames[$i]}
  cp ./${hailort_source_foldernames[$i]}/build/hailort/libhailort/libhef_proto.a ./${hailort_destination_foldernames[$i]}
  cp ./${hailort_source_foldernames[$i]}/build/hailort/libhailort/libscheduler_mon_proto.a ./${hailort_destination_foldernames[$i]}

  find $BASE_DIR -name "*.so*" -exec chmod 644 {} \;
  find $BASE_DIR -name "protoc-3.21.12.0" -exec chmod 644 {} \;


done
