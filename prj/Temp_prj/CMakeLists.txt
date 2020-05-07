cmake_minimum_required(VERSION 3.0)

project(Temp_prj)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Temp_prj main.cpp)
target_link_libraries(Temp_prj ${OpenCV_LIBS})