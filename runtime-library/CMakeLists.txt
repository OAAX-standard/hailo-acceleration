cmake_minimum_required(VERSION 3.10.2)

project(RuntimeLibrary)

add_definitions(-DONNXRUNTIME_API_VERSION=11)


# where is the nx-c source code located?
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(DEPS_DIR ${CMAKE_CURRENT_LIST_DIR}/deps)

# add string option to specify the target platform
set(PLATFORM "NONE" CACHE STRING "The target platform")

# add string option to specify the target platform
set(HAILORT_VERSION "NONE" CACHE STRING "The HailoRT version")

message("HAILORT_VERSION: ${HAILORT_VERSION}")

######################### customize when cross-compiling ###############################################################

# set COMPILER_PREFIX, for example, "" for default compiler, arm-linux- , or aarch64-linux- etc for cross compilers
# set generic build target
set(ADDITONAL_LIBS "")

if (PLATFORM STREQUAL "X86_64")
  set(GENERIC_BUILD_TARGET X86_64)
  set(CROSS_ROOT "/opt/x86_64-unknown-linux-gnu-gcc-9.5.0")
  set(COMPILER_PREFIX "x86_64-unknown-linux-gnu-")
  set(SYSROOT "/opt/x86_64-unknown-linux-gnu-gcc-9.5.0/x86_64-unknown-linux-gnu/sysroot")
  set(CMAKE_BUILD_FLAGS " -O3 -fno-math-errno -fopenmp -std=c99 -march=haswell")
elseif (PLATFORM STREQUAL "AARCH64")
  set(GENERIC_BUILD_TARGET AARCH64)
  # set(CROSS_ROOT "/opt/gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu")
  set(CROSS_ROOT "/opt/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu")
  set(COMPILER_PREFIX "aarch64-none-linux-gnu-")
else()
  message(FATAL_ERROR "Please specify the target platform: X86_64 or AARCH64")
endif()

set(ADDITONAL_LIBS "clog")
if (HAILORT_VERSION STREQUAL "4.17.0")
  set(ADDITONAL_LIBS "")
endif()

set(CMAKE_C_COMPILER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}gcc)
set(CMAKE_CXX_COMPILER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}g++)
set(CMAKE_LINKER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}ld)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

#--build: the machine you are building on
#--host: the machine you are building for
#--target: the machine that GCC will produce binary for
set(BUILD X86_64-linux-gnu)
set(HOST ${GENERIC_BUILD_TARGET}-linux)
set(TARGET ${GENERIC_BUILD_TARGET})

# set compiler flags
# see https://gist.github.com/robinvanemden/72aa87b1a656d5cd4a06ea0a9b77e4ac
# also: NEON on aarch64 is automatically implied (https://stackoverflow.com/a/29891469)

######################### webassembly wamr settings ####################################################################

# set make system
set(CMAKE_SYSTEM_NAME Linux)

######################## webassembly wamr library build ################################################################

# Reset default linker flags
set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS " ")
set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS " ")

# compiling and linking flags
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wformat -Wformat-security -fdata-sections -ffunction-sections -Wl,--gc-sections")

####################### toolchain arguments ############################################################################

INCLUDE(CMakeForceCompiler)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_BUILD_FLAGS " ${CMAKE_BUILD_FLAGS} -std=c99 -fno-common -Wall -Wextra")
set(CMAKE_BUILD_FLAGS " ${CMAKE_BUILD_FLAGS} -Wno-missing-field-initializers -fno-exceptions")
set(CMAKE_BUILD_FLAGS " ${CMAKE_BUILD_FLAGS} -Wno-attribute-warning -fdata-sections -ffunction-sections -Wl,--gc-sections ")

set(CMAKE_C_FLAGS " ${CMAKE_C_FLAGS} ${CMAKE_BUILD_FLAGS}  -D_GNU_SOURCE -I${CROSS_ROOT}/include")
set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} ${CMAKE_BUILD_FLAGS}  -D_GNU_SOURCE -I${CROSS_ROOT}/include")
set(CMAKE_CPP_FLAGS " ${CMAKE_CPP_FLAGS} ${CMAKE_BUILD_FLAGS}  -D_GNU_SOURCE -I${CROSS_ROOT}/include")

message(STATUS "SDKTARGETSYSROOT=${SDKTARGETSYSROOT}")
message(STATUS "CROSS_ROOT=${CROSS_ROOT}")

set(CMAKE_AR ${CROSS_ROOT}/bin/${COMPILER_PREFIX}ar)
set(CMAKE_AS ${CROSS_ROOT}/bin/${COMPILER_PREFIX}as)
set(CMAKE_LD ${CROSS_ROOT}/bin/${COMPILER_PREFIX}ld)
set(CMAKE_RANLIB ${CROSS_ROOT}/bin/${COMPILER_PREFIX}ranlib)
set(CMAKE_C_COMPILER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}gcc)
set(CMAKE_CXX_COMPILER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}g++)
set(CMAKE_ASM_COMPILER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}gcc)
set(CMAKE_NM ${CROSS_ROOT}/bin/${COMPILER_PREFIX}NM)

# this is the file system root of the target
set(CMAKE_FIND_ROOT_PATH ${SYSROOT})

# search for programs in the build host directories
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# for libraries and headers in the target directories
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

######################## generic build settings ########################################################################

# set some basic c, c++ and linker flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_BUILD_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_BUILD_FLAGS}")
set(LINKER_FLAGS "${LINKER_FLAGS} ${CMAKE_BUILD_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_BUILD_FLAGS}")
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_BUILD_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_BUILD_FLAGS}")

# add all source files
file(GLOB_RECURSE SRC
        ${SRC_DIR}/runtime_utils.c
        ${SRC_DIR}/runtime_core.c
        ${DEPS_DIR}/utils/queue.c
        ${DEPS_DIR}/utils/logger.c
        )
######################## main bin build settings #######################################################################

# binary target
add_library(RuntimeLibrary SHARED ${SRC})

# target include / link dirs
target_include_directories(RuntimeLibrary PUBLIC ${INCLUDE_DIR})
target_include_directories(RuntimeLibrary PUBLIC ${DEPS_DIR}/utils)

set(RE2_DIR "${DEPS_DIR}/re2")
set(ONNX_RUNTIME_DIR "${DEPS_DIR}/onnxruntime")

target_link_libraries(RuntimeLibrary PUBLIC pthread atomic dl stdc++ m)

target_include_directories(RuntimeLibrary PUBLIC ${RE2_DIR}/include)
target_link_directories(RuntimeLibrary PUBLIC "${RE2_DIR}/${GENERIC_BUILD_TARGET}")
target_link_libraries(RuntimeLibrary PUBLIC re2)

set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_BUILD_FLAGS} ")
target_link_libraries(RuntimeLibrary PRIVATE ${LLVM_AVAILABLE_LIBS} -ldl pthread ${CGRLIB} -lgomp -lstdc++ -lm -ldl)

target_include_directories(RuntimeLibrary PUBLIC ${CPUINFO_DIR}/include)
target_link_directories(RuntimeLibrary PUBLIC "${CPUINFO_DIR}/${GENERIC_BUILD_TARGET}")
target_link_libraries(RuntimeLibrary PUBLIC cpuinfo)

target_link_libraries(RuntimeLibrary PUBLIC -Wl,-rpath,./)
target_link_libraries(RuntimeLibrary PUBLIC -Wl,--disable-new-dtags)

target_include_directories(RuntimeLibrary PUBLIC ${ONNX_RUNTIME_DIR}/${GENERIC_BUILD_TARGET}-${HAILORT_VERSION}/include)
target_include_directories(RuntimeLibrary PUBLIC ${ONNX_RUNTIME_DIR}/${GENERIC_BUILD_TARGET}-${HAILORT_VERSION}/include/onnxruntime/core/session)
target_link_directories(RuntimeLibrary PUBLIC "${ONNX_RUNTIME_DIR}/${GENERIC_BUILD_TARGET}-${HAILORT_VERSION}")
target_link_libraries(
        RuntimeLibrary
        PUBLIC
        -Wl,--start-group

        onnxruntime_session
        onnxruntime_optimizer
        onnxruntime_providers
        onnxruntime_util
        onnxruntime_framework
        onnxruntime_graph
        onnxruntime_mlas
        onnxruntime_common
        onnxruntime_flatbuffers
        absl_base
        absl_log_severity
        absl_malloc_internal
        absl_raw_logging_internal
        absl_spinlock_wait
        absl_throw_delegate
        absl_hashtablez_sampler
        absl_raw_hash_set
        absl_hash
        absl_city
        absl_low_level_hash
        absl_int128
        absl_exponential_biased
        absl_cord
        absl_cord_internal
        absl_cordz_functions
        absl_cordz_handle
        absl_cordz_info
        absl_strings
        absl_strings_internal
        absl_graphcycles_internal
        absl_synchronization
        absl_civil_time
        absl_time
        absl_time_zone
        absl_bad_optional_access
        absl_bad_variant_access
        flatbuffers
        onnx
        onnx_proto
        protobuf-lite
        cpuinfo
        nsync_cpp

        ${ADDITONAL_LIBS}

        -Wl,--end-group
)

add_custom_command(TARGET RuntimeLibrary POST_BUILD
        COMMAND cp ${ONNX_RUNTIME_DIR}/${GENERIC_BUILD_TARGET}-${HAILORT_VERSION}/libonnxruntime_providers_hailo.so ${CMAKE_CURRENT_BINARY_DIR}/
        COMMAND cp ${ONNX_RUNTIME_DIR}/${GENERIC_BUILD_TARGET}-${HAILORT_VERSION}/libonnxruntime_providers_shared.so ${CMAKE_CURRENT_BINARY_DIR}/
        COMMAND cp ${ONNX_RUNTIME_DIR}/../hailort/${GENERIC_BUILD_TARGET}_${HAILORT_VERSION}/libhailort.so.* ${CMAKE_CURRENT_BINARY_DIR}/
)
