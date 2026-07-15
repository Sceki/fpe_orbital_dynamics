cmake_minimum_required(VERSION 3.18...3.30)
project(fpe_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(pybind11 CONFIG REQUIRED)
find_package(Threads REQUIRED)

# Eigen: prefer a system install, otherwise fetch a pinned release.
find_package(Eigen3 3.4 CONFIG QUIET)
if(NOT TARGET Eigen3::Eigen)
  message(STATUS "Eigen3 not found on system - fetching 3.4.0")
  include(FetchContent)
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
  set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
  )
  FetchContent_MakeAvailable(eigen)
endif()

pybind11_add_module(_core cpp/src/bindings.cpp)
target_include_directories(_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)
target_link_libraries(_core PRIVATE Eigen3::Eigen Threads::Threads)

if(MSVC)
  target_compile_options(_core PRIVATE /O2)
else()
  target_compile_options(_core PRIVATE -O3)
endif()

install(TARGETS _core DESTINATION fpe)

