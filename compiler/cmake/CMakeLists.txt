# samples
# ref:https://github.com/robertmaynard/code-samples/blob/master/posts/cmake/CMakeLists.txt


# Cmake最低版本号要求
cmake_minimum_required(VERSION 2.8)

# 项目信息
project(cmake_and_cuda LANGUAGES CXX CUDA)


# 指定生成目标
add_executable(test1 test1.cu)



