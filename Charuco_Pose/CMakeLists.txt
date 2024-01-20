# Minimum required CMake version
cmake_minimum_required(VERSION 3.12)

# Project name
project(Charuco_Pose)
set(CMAKE_BUILD_TYPE Debug)
# Find OpenCV package
find_package(OpenCV REQUIRED)

# Add the source file
add_executable(Charuco_Pose src/detect_Charuco_pose.cpp)
include_directories(include)

# Link OpenCV libraries
target_link_libraries(Charuco_Pose ${OpenCV_LIBS})
