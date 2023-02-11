#include <stdio.h>

/*
    了解CUDA编程中的一些隐含变量，涉及threadIdx、blockIdx、blockDim、gridDim

    CUDA编程的模型：one kernel -> one grid -> multi blocks
    gridDim: 描述一个grid里面的Blocks的组织形式
    blockDim：描述一个block里面的Threads的组织形式

    对于GPU上的一个线程要确定其“位置”(全局索引)，需要知道其在block中的位置以及该block在grid中的位置
    CUDA中每个线程隐含了blockIdx以及threadIdx，通过这两个变量可以确定线程的全局索引

 */

/* 
    打印每个线程的threadIdx, blockIdx,
 */
__global__ void printIdx(void)
{
    // printf("gridDim: (%d,%d,%d), blockDim: (%d,%d,%d)\n",
    //        gridDim.x, gridDim.y, gridDim.z,
    //        blockDim.x, blockDim.y, blockDim.z);
    printf("blockIdx: (%d,%d,%d), threadIdx: (%d,%d,%d)\n",
           blockIdx.x, blockIdx.y, blockIdx.z,
           threadIdx.x, threadIdx.y, threadIdx.z);

    int index = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    
    index = index * blockDim.x * blockDim.y * blockDim.z + threadIdx.x + threadIdx.y *blockDim.x + threadIdx.z *blockDim.x *blockDim.y;
    printf("thread id: %d\n", index);
}

/*
    这里截取一段有关blockIdx和threadIdx的打印结果
    Ouput:
        blockIdx: (2,2,0), threadIdx: (0,0,0)
        blockIdx: (2,2,0), threadIdx: (0,1,0)
        blockIdx: (2,2,0), threadIdx: (0,0,1)
        blockIdx: (2,2,0), threadIdx: (0,1,1)
        blockIdx: (2,2,0), threadIdx: (0,0,2)
        blockIdx: (2,2,0), threadIdx: (0,1,2)
        blockIdx: (0,0,2), threadIdx: (0,0,0)
        blockIdx: (0,0,2), threadIdx: (0,1,0)
        blockIdx: (0,0,2), threadIdx: (0,0,1)
        blockIdx: (0,0,2), threadIdx: (0,1,1)
        blockIdx: (0,0,2), threadIdx: (0,0,2)
        blockIdx: (0,0,2), threadIdx: (0,1,2)
    分析：
        1、打印时，以Block为单位进行打印，Blocks间的打印结果无必然顺序。
        2、Block内的线程，threadIdx是一个dim3类型，打印时按照xyz的顺序依次增加。
            例如:    
                    (x,y,z) -> (x,y,z) -> (x,y,z)
                    (0,0,0) -> (0,1,0) -> (0,0,1)
                    (0,0,1) -> (0,1,1) -> (0,0,2)
            假设，block数目为1，也就是grid内只含有一个block。
            那么线程标识为： threadIdx.x + threadIdx.y * threadDim.x + threadIdx.z * threadDim.x * threadIdm.z
 */

int main(int argc, char **argv)
{
    dim3 grid(3, 4, 5); // 定义grid的大小
    dim3 block(1, 2, 3); // 定义block的大小
    printIdx<<<grid, block>>>();
    cudaDeviceReset();
    return 0;
}