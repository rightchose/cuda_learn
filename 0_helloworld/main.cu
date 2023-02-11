#include<stdio.h>


/* 
    nvcc main.cu
 */
__global__ void hello_world(void)
{
    printf("GPU: Hello world!\n");
}

int main(int argc, char **argv)
{
    printf("CPU: Hello world!\n");
    hello_world<<<1, 10>>>();
    cudaDeviceReset(); // 如果没有这行，可能从gpu中输出hello world
    return 0;
}