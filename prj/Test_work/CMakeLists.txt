cmake_minimum_required(VERSION 3.0)

project(Test_work)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Test_work main.cpp)
target_link_libraries(Test_work ${OpenCV_LIBS})