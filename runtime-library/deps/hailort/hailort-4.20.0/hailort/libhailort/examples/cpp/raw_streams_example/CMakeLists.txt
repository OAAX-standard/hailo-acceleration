cmake_minimum_required(VERSION 3.5.0)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

find_package(HailoRT 4.20.0 EXACT REQUIRED)

add_executable(cpp_raw_streams_example raw_streams_example.cpp)
target_link_libraries(cpp_raw_streams_example PRIVATE HailoRT::libhailort Threads::Threads)

if(WIN32)
    target_compile_options(cpp_raw_streams_example PRIVATE
        /DWIN32_LEAN_AND_MEAN
        /DNOMINMAX                  # NOMINMAX is required in order to play nice with std::min/std::max (otherwise Windows.h defines it's own)
        /wd4201 /wd4251
    )
endif()

set_target_properties(cpp_raw_streams_example PROPERTIES CXX_STANDARD 14)