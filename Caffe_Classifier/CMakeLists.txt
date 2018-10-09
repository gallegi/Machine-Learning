
cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 11)
project(classify_bien_bao)
find_package(Caffe REQUIRED)
add_executable(classify_bien_bao sign_detection.cpp)
include_directories(${Caffe_INCLUDE_DIRS})
target_link_libraries(classify_bien_bao ${Caffe_LIBRARIES} )
