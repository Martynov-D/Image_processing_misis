cmake_minimum_required(VERSION 3.0)

project(Lab1)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Lab1 lab1.cpp)
target_link_libraries(Lab1 ${OpenCV_LIBS})