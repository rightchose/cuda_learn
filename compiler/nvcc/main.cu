#include<stdio.h>
#include<iostream>
#include "foo.cuh"

/* 
    nvcc -o main.out main.cu foo.cu
 */

int main()
{
    std::cout << "Hello NVCC" << std::endl;
    useCUDA();
    return 0;
}