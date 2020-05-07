cmake_minimum_required(VERSION 3.0)

project(Lab3)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Lab3 lab3.cpp)
target_link_libraries(Lab3 ${OpenCV_LIBS})