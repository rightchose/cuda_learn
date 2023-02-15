#include <cuda_runtime.h>
#include <stdio.h>
#include "freshman.h"


/* 
    计算模型：
        对于size长度的int数组g_idata，将其划分成1024大小的X干个block，申请一个长度为X的g_odata数组存储结果
        g_odata存储每个block的中的和
        这块理解起来比较难，这里举个有关下面代码的简例子，（反正就是这么神奇
        设定block大小为8，这样tid介于0~7，我们假设该block对于的区段，依次存放0、1、2、...7。
        blockDim.x大小为16，stride依次为1、2、4。
        计算时如果tid %（2*1，2*2，2*4） == 0 就在tid位置上加上对应的值
            tid: 0      1   2       3   4       5       6       7
    stride:   1  +1     x   +3      x   +5      x       +7      x   （第一次for
              2  +2     x   x       x   x       x       x       x     第二次
              4  +4     x   x       x   x       x       x       x   第三次
        如果不限制同步，那么最好每个位置的值会变成
                7、1、5、3、9、5、13、7（一种可能）
            但在for循环中，使用了__syncthreads语句，强制该block中所有能到达该语句的程序
            对于tid = 0~7的8个线程，每次for循环都会等待其他的结果，
            可以看到，依据stride共有四次循环，最后依次循环除了tid=0，有加法运算，其余无，
            因此在最后一次时，会将其他的tid运算的结果加到tid=0上，当然，这一过程中在多次for循环中都有发生
            分析下，直接加法应该就是7次运算周期，
            如果GPU的话，多个计算单元，可以完成多个线程的for循环，上面实际只需要三个周期。
            当然累加的串行计算实现对比不太合适，如果将CPU所有核心用上，肯定更快，
            但数据规模上去后，显然还是GPU的多核心更有优势，多个SM，每个SM有多个单元，并行处理能力很强。      
 */
__global__ void warmup(int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    // boundary check
    if (tid >= n)
        return;
    // convert global data pointer to the
    int *idata = g_idata + blockIdx.x * blockDim.x;
    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if ((tid % (2 * stride)) == 0)
        {
            idata[tid] += idata[tid + stride];
        }
        // synchronize within block
        // __syncthreads will wait for all warps in a block to reach that point in your code.
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    // boundary check
    if (tid >= n)
        return;
    // convert global data pointer to the
    int *idata = g_idata + blockIdx.x * blockDim.x;
    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if ((tid % (2 * stride)) == 0)
        {
            idata[tid] += idata[tid + stride];
        }
        // synchronize within block
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

/*

    这段同样难理解，还是举上一个例子
    tid 0-7
    tid stride/index   stride/index    stride/index
    0       1/0(+1)        2/0(+2)         4/0(+4)
    1       1/2(+3)        2/4(+6)         4/x                  
    2       1/4(+5)        2/x             4/x
    3       1/6(+7)        2/x             4/x
    4       1/x            2/x             4/x
    5       1/x            2/x             4/x
    6       1/x            2/x             4/x
    7       1/x            2/x             4/x
 */

__global__ void reduceNeighboredLess(int *g_idata, int *g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    // convert global data pointer to the local point of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;
    if (idx > n)
        return;
    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        // convert tid into local array index
        int index = 2 * stride * tid;
        if (index < blockDim.x)
        {
            idata[index] += idata[index + stride];
        }
        __syncthreads();
    }
    // write result for this block to global men
    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceInterleaved(int *g_idata, int *g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    // convert global data pointer to the local point of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;
    if (idx >= n)
        return;
    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {

        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    // write result for this block to global men
    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

int main(int argc, char **argv)
{
    initDevice(0);
    bool bResult = false;

    int size = 1 << 24;
    printf(" with array size %d\n", size);

    // execution configuration
    int blocksize = 1024;
    if(argc > 1)
    {
        blocksize = atoi(argv[1]);
    }
    dim3 block(blocksize, 1);
    dim3 grid((size - 1) / block.x + 1, 1);
    printf("grid %d block %d\n", grid.x, block.x);

    // allocate host memory
    size_t bytes = size * sizeof(int);
    int *idata_host = (int *)malloc(bytes);
    int *odata_host = (int *)malloc(grid.x * sizeof(int));
    int *tmp = (int *)malloc(bytes);

    // initialize the array
    initialData_int(idata_host, size);

    memcpy(tmp, idata_host, bytes);
    double iStart, iElaps;
    int gpu_sum = 0;

    // device memory
    int *idata_dev = NULL;
    int *odata_dev = NULL;
    CHECK(cudaMalloc((void **)&idata_dev, bytes));
    CHECK(cudaMalloc((void **)&odata_dev, grid.x * sizeof(int)));

    // cpu reduction
    int cpu_sum = 0;
    iStart = cpuSecond();
    // cpu_sum = recursiveReduce(tmp, size);
    for (int i = 0; i < size; i++)
        cpu_sum += tmp[i];
    printf("cpu sum:%d \n", cpu_sum);
    iElaps = cpuSecond() - iStart;
    printf("cpu reduce                 elapsed %lf ms cpu_sum: %d\n", iElaps, cpu_sum);

    // warmup

    CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    warmup<<<grid, block>>>(idata_dev, odata_dev, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += odata_host[i];
    printf("gpu warmup                 elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
           iElaps, gpu_sum, grid.x, block.x);

    // kernel 1:reduceNeighbored

    CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    reduceNeighbored<<<grid, block>>>(idata_dev, odata_dev, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += odata_host[i];
    printf("gpu reduceNeighbored       elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
           iElaps, gpu_sum, grid.x, block.x);

    // kernel 2:reduceNeighboredLess

    CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    reduceNeighboredLess<<<grid, block>>>(idata_dev, odata_dev, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += odata_host[i];
    printf("gpu reduceNeighboredLess   elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
           iElaps, gpu_sum, grid.x, block.x);

    // kernel 3:reduceInterleaved
    CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    reduceInterleaved<<<grid, block>>>(idata_dev, odata_dev, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += odata_host[i];
    printf("gpu reduceInterleaved      elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
           iElaps, gpu_sum, grid.x, block.x);
    // free host memory

    free(idata_host);
    free(odata_host);
    CHECK(cudaFree(idata_dev));
    CHECK(cudaFree(odata_dev));

    // reset device
    cudaDeviceReset();

    // check the results
    if (gpu_sum == cpu_sum)
    {
        printf("Test success!\n");
    }
    return EXIT_SUCCESS;
}