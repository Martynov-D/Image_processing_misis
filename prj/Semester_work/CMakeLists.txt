cmake_minimum_required(VERSION 3.0)

project(Semester_work)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Semester_work main.cpp)
target_link_libraries(Semester_work ${OpenCV_LIBS})