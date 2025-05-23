cmake_minimum_required(VERSION 3.0.0)

option(HAILO_BUILD_PYBIND "Build Python binding" OFF)
option(HAILO_BUILD_EMULATOR "Build hailort for emulator" OFF)
option(HAILO_BUILD_UT "Build Unit Tests" OFF)
option(HAILO_BUILD_DMABUF_TESTS "Build DMA buffer tests. Relevant only if HAILO_BUILD_UT is ON" OFF)
option(HAILO_BUILD_HW_DEBUG_TOOL "Build hw debug tool" OFF)
option(HAILO_BUILD_GSTREAMER "Compile gstreamer plugins" OFF)
option(HAILO_BUILD_EXAMPLES "Build examples" OFF)
option(HAILO_OFFLINE_COMPILATION "Don't download external dependencies" OFF)
option(HAILO_BUILD_SERVICE "Build hailort service" OFF)
option(HAILO_BUILD_PROFILER "Build hailort profiler" ON)
option(HAILO_COMPILE_WARNING_AS_ERROR "Add compilation flag for treating compilation warnings as errors" OFF)
option(HAILO_SUPPORT_PACKAGING "Create HailoRT package (internal)" OFF)

if (HAILO_COMPILE_WARNING_AS_ERROR)
    if(WIN32)
        set(HAILORT_COMPILE_OPTIONS ${HAILORT_COMPILE_OPTIONS} /WX)
    elseif(UNIX)
        set(HAILORT_COMPILE_OPTIONS ${HAILORT_COMPILE_OPTIONS} -Werror)
    else()
        message(FATAL_ERROR "Unexpeced host, stopping build")
    endif()
endif()

# Flag for emulator (FPGA/Veloce)
if(HAILO_BUILD_EMULATOR)
    message(WARNING "HailoRT is building with Emulator flag on")
    set(HAILORT_COMPILE_OPTIONS ${HAILORT_COMPILE_OPTIONS} -DHAILO_EMULATOR)
endif()

# Set firmware version
add_definitions( -DFIRMWARE_VERSION_MAJOR=4 )
add_definitions( -DFIRMWARE_VERSION_MINOR=17 )
add_definitions( -DFIRMWARE_VERSION_REVISION=0 )
if(HAILO_BUILD_SERVICE)
    add_definitions( -DHAILO_SUPPORT_MULTI_PROCESS )
endif()

# TODO: temporary hack to support offline builds. Remove HAILO_OFFLINE_COMPILATION and use FETCHCONTENT_FULLY_DISCONNECTED
if(HAILO_OFFLINE_COMPILATION)
    set(FETCHCONTENT_FULLY_DISCONNECTED ON CACHE INTERNAL "")
    set(HAILO_OFFLINE_COMPILATION OFF CACHE INTERNAL "")
endif()

# TODO: move protobuf and grpc to inner cmake files
set(HAILO_EXTERNAL_DIR ${CMAKE_CURRENT_LIST_DIR}/external)
set(HAILO_EXTERNALS_CMAKE_SCRIPTS ${CMAKE_CURRENT_LIST_DIR}/cmake/external/)
include(${HAILO_EXTERNALS_CMAKE_SCRIPTS}/protobuf.cmake)
if(HAILO_BUILD_SERVICE)
    include(${HAILO_EXTERNALS_CMAKE_SCRIPTS}/grpc.cmake)
endif()

set(HAILORT_INC_DIR ${PROJECT_SOURCE_DIR}/hailort/libhailort/include)
set(HAILORT_SRC_DIR ${PROJECT_SOURCE_DIR}/hailort/libhailort/src)
set(HAILORT_COMMON_DIR ${PROJECT_SOURCE_DIR}/hailort/)
set(COMMON_INC_DIR ${PROJECT_SOURCE_DIR}/common/include)
set(DRIVER_INC_DIR ${PROJECT_SOURCE_DIR}/hailort/drivers/common)
set(RPC_DIR ${PROJECT_SOURCE_DIR}/hailort/rpc)

if(CMAKE_SYSTEM_NAME STREQUAL QNX)
    include(${HAILO_EXTERNALS_CMAKE_SCRIPTS}/pevents.cmake)
endif()

if(HAILO_BUILD_SERVICE)
    add_subdirectory(rpc)
endif()

add_subdirectory(common)
add_subdirectory(libhailort)
add_subdirectory(hailortcli)
if(HAILO_BUILD_HW_DEBUG_TOOL)
    add_subdirectory(tools/hw_debug)
endif()

if(HAILO_BUILD_SERVICE)
    add_subdirectory(hailort_service)
endif()

if(HAILO_WIN_DRIVER)
    add_subdirectory(drivers/win)
endif()

if(HAILO_SUPPORT_PACKAGING)
    add_subdirectory(packaging)
endif()

# Compile PCIe Driver if QNX
if(CMAKE_SYSTEM_NAME STREQUAL QNX)
    add_subdirectory(drivers/qnx)
endif()
