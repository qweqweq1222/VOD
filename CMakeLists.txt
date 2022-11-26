cmake_minimum_required(VERSION 3.21)

project(start LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)

add_library(lb lib.h optim.cpp src.cpp)
target_link_libraries(lb 
  Ceres::ceres
  Eigen3::Eigen3 
  opencv::calib3d
  opencv::core
  opencv::flann
  opencv::highgui
  opencv::imgcodecs
  opencv::imgproc
)

add_executable(start main.cpp)
target_link_libraries(start lb)

