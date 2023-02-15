cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(sum_arrays CXX C CUDA)

include_directories(../include)
add_executable(reduceInteger reduceInteger.cu)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")