cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(sum_arrays CXX C CUDA)

include_directories(../include)


add_executable(sum_arrays sum_arrays.cu)

