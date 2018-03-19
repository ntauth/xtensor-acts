cmake_minimum_required(VERSION 3.7)
project(xtensor_pground)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

if (NOT(${XTENSOR_INCLUDE_DIR}))
    set(XTENSOR_INCLUDE_DIR ~/miniconda2/include)
endif ()

# Include files
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${XTENSOR_INCLUDE_DIR})

# Source files
add_executable(xtensor_pground src/main.cpp)
