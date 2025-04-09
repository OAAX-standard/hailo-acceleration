#!/bin/bash

set -e

BASE_DIR="$(cd "$(dirname "$0")"; pwd)";
cd $BASE_DIR

declare -a hailort_versionnames=("4.17.0" "4.18.0" "4.19.0" "4.20.0")

# create source and destination folders
declare -a hailort_source_foldernames=("${hailort_versionnames[@]/#/hailort-}")
declare -a hailort_destination_foldernames=("${hailort_versionnames[@]/#/X86_64_}")

## now loop through the above array
for i in "${!hailort_source_foldernames[@]}"
do

  rm -rf ${hailort_destination_foldernames[$i]} || true
  mkdir ${hailort_destination_foldernames[$i]}

  export CC=/opt/x86_64-unknown-linux-gnu-gcc-9.5.0/bin/x86_64-unknown-linux-gnu-gcc
  export CXX=/opt/x86_64-unknown-linux-gnu-gcc-9.5.0/bin/x86_64-unknown-linux-gnu-g++

  export CROSS_ROOT="/opt/x86_64-unknown-linux-gnu-gcc-9.5.0"
  export COMPILER_PREFIX="x86_64-unknown-linux-gnu-"
  export PATH=$CROSS_ROOT/bin:/snap/bin:/bin:/usr/bin:/snap/bin:/usr/local/bin:opt/x86_64-unknown-linux-gnu-gcc-9.5.0/bin:/usr/bin
  export SYSROOT="/opt/x86_64-unknown-linux-gnu-gcc-9.5.0/x86_64-unknown-linux-gnu/sysroot"

  export CPPFLAGS="-I${CROSS_ROOT}/include "
  export CFLAGS="-I${CROSS_ROOT}/include "
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
  -DCMAKE_C_COMPILER=${CROSS_ROOT}/bin/${COMPILER_PREFIX}gcc \
  -DCMAKE_CXX_COMPILER=${CROSS_ROOT}/bin/${COMPILER_PREFIX}g++ \
  -DCMAKE_LINKER=${CROSS_ROOT}/bin/${COMPILER_PREFIX}ld \
  -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
  -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
  -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY \
  -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ONLY \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5


  cmake --build build --config release --target libhailort -j8

  cd build

  make -j

  cd ../..

  # read version from hailort_foldernames
  version=$(echo "${hailort_source_foldernames[$i]}" | cut -d'-' -f2)
  echo $version > ${hailort_destination_foldernames[$i]}/HAILORT_VERSION
  cp -rf ./${hailort_source_foldernames[$i]}/hailort/libhailort/include ./${hailort_destination_foldernames[$i]}
  cp ./${hailort_source_foldernames[$i]}/build/hailort/libhailort/src/libhailort.so.4.* ./${hailort_destination_foldernames[$i]}
  cp ./${hailort_source_foldernames[$i]}/build/hailort/libhailort/libhef_proto.a ./${hailort_destination_foldernames[$i]}
  cp ./${hailort_source_foldernames[$i]}/build/hailort/libhailort/libscheduler_mon_proto.a ./${hailort_destination_foldernames[$i]}

  find $BASE_DIR -name "*.so*" -exec chmod 644 {} \;
  find $BASE_DIR -name "protoc-3.21.12.0" -exec chmod 644 {} \;

done

