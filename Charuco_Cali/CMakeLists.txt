# Minimum required CMake version
cmake_minimum_required(VERSION 3.12)

# Project name
project(Charuco_Cali)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Add the source file
add_executable(camera_calibration src/camera_calibration.cpp)

# Link against OpenCV
target_link_libraries(camera_calibration ${OpenCV_LIBS})
