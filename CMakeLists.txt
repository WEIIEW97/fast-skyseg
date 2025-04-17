cmake_minimum_required(VERSION 3.10)
project(ncnn_mbv3_qint8)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(ncnn_DIR "/home/william/Codes/ncnn-20241226-ubuntu-2204/lib/cmake/ncnn" CACHE PATH "NCNN config directory")
set(ncnn_include_dir "/home/william/Codes/ncnn-20241226-ubuntu-2204/include" CACHE PATH "NCNN root directory")
find_package(ncnn REQUIRED)
find_package(OpenCV REQUIRED)

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

file(GLOB_RECURSE ncnn_sources ${CMAKE_CURRENT_SOURCE_DIR}/ncnn/*.cc)
file(GLOB_RECURSE ncnn_headers ${CMAKE_CURRENT_SOURCE_DIR}/ncnn/*.h)

add_executable(${PROJECT_NAME}
    ${ncnn_sources}
)

target_link_libraries(
    ${PROJECT_NAME}
    ncnn
    ${OpenCV_LIBS}
    ${OpenMP_CXX_LIBRARIES}
)

target_include_directories(${PROJECT_NAME}
    PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ncnn_include_dir}
)
