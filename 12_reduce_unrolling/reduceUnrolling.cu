#include <cuda_runtime.h>
#include <stdio.h>
#include "freshman.h"


__global__ void warmup(int *g_idata, int *g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    if(tid >= n)
        return;
    int *idata = g_idata + blockIdx.x * blockDim.x;

    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if((tid % (2 * stride)) == 0)
        {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    if(tid == 0)
        g_odata[blockIdx.x] = idata[0];
}


__global__ void reduceUnroll2(int *g_idata, int *g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + 2 * blockIdx.x * blockDim.x;
    if(tid >= n)
        return;
    int *idata = g_idata + blockIdx.x * blockDim.x * 2;
    if(idx + blockDim.x < n)
    {
        g_idata[idx] += g_idata[idx + blockDim.x];
    }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if(tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    if(tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnroll4(int *g_idata, int *g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x * 4;

    if(tid >= n)
        return;
    int *idata = g_idata + blockIdx.x * blockDim.x * 4;
    // 关于这里的判断条件还是很奇怪
    if(idx + blockDim.x < n)
    {
        g_idata[idx] += g_idata[idx + blockDim.x];
        g_idata[idx] += g_idata[idx + blockDim.x * 2];
        g_idata[idx] += g_idata[idx + blockDim.x * 3];
    }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if(tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    if(tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnroll8(int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x * 8 + threadIdx.x;
    // boundary check
    if (tid >= n)
        return;
    // convert global data pointer to the
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;
    if (idx + blockDim.x < n)
    {
        g_idata[idx] += g_idata[idx + blockDim.x];
        g_idata[idx] += g_idata[idx + blockDim.x * 2];
        g_idata[idx] += g_idata[idx + blockDim.x * 3];
        g_idata[idx] += g_idata[idx + blockDim.x * 4];
        g_idata[idx] += g_idata[idx + blockDim.x * 5];
        g_idata[idx] += g_idata[idx + blockDim.x * 6];
        g_idata[idx] += g_idata[idx + blockDim.x * 7];
    }
    __syncthreads();
    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
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

__global__ void reduceUnrollWarp8(int *g_idata, int *g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x * 8;
    if(tid >= n)
        return;
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;
    // unrolling8
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int a5 = g_idata[idx + 4 * blockDim.x];
        int a6 = g_idata[idx + 5 * blockDim.x];
        int a7 = g_idata[idx + 6 * blockDim.x];
        int a8 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    }
    __syncthreads();
    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }
        // synchronize within block
        __syncthreads();
    }

    // write result for this block to global mem
    // 不理解
    if (tid < 32)
    {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceCompleteUnrollWarp8(int *g_idata, int *g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x * 8 + threadIdx.x;
    // boundary check
    if (tid >= n)
        return;
    // convert global data pointer to the
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int a5 = g_idata[idx + 4 * blockDim.x];
        int a6 = g_idata[idx + 5 * blockDim.x];
        int a7 = g_idata[idx + 6 * blockDim.x];
        int a8 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    }

    __syncthreads();
    // in-place reduction in global memory
    if (blockDim.x >= 1024 && tid < 512)
        idata[tid] += idata[tid + 512];
    __syncthreads();
    if (blockDim.x >= 512 && tid < 256)
        idata[tid] += idata[tid + 256];
    __syncthreads();
    if (blockDim.x >= 256 && tid < 128)
        idata[tid] += idata[tid + 128];
    __syncthreads();
    if (blockDim.x >= 128 && tid < 64)
        idata[tid] += idata[tid + 64];
    __syncthreads();

    // write result for this block to global mem
    if (tid < 32)
    {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

int main(int argc, char **argv)
{
    initDevice(0);

    bool bResult = false;

    int size = 1 << 24;

    int blocksize = 1024;

    dim3 block(blocksize);
    dim3 grid((size - 1) / block.x + 1, 1);
    printf("grid %d, block %d\n", grid.x, block.x);

    size_t bytes = size * sizeof(int);
    int *idata_host = (int *)malloc(bytes);
    int *odata_host = (int *)malloc(grid.x * sizeof(int));
    int *tmp = (int *)malloc(bytes);

    initialData_int(idata_host, size);
    memcpy(tmp, idata_host, bytes);

    double iStart, iElaps;
    int gpu_sum = 0;

    // device memory
    int *idata_dev = NULL;
    int *odata_dev = NULL;
    CHECK(cudaMalloc((void **)&idata_dev, bytes));
    CHECK(cudaMalloc((void **)&odata_dev, bytes));

    // cpu reduction
    int cpu_sum = 0;
    iStart = cpuSecond();
    for (int i = 0; i < size; ++i)
        cpu_sum += idata_host[i];
    printf("cpu sum: %d\n", cpu_sum);
    iElaps = cpuSecond() - iStart;
    printf("cpu reduce elaps %lf ms cpu_sum: %d\n", iElaps, cpu_sum);

    // kernel 1:warmup
    CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    warmup<<<grid.x / 2, block>>>(idata_dev, odata_dev, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; ++i)
        gpu_sum += odata_host[i];
    printf("gpu sum: %d\n", gpu_sum);
    printf("gpu warmup elapsed %lf ms \n", iElaps);

    // kernel 2: reduceUnrolling2
    CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    reduceUnroll2<<<grid.x / 2, block>>>(idata_dev, odata_dev, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 2; i++)
        gpu_sum += odata_host[i];
    printf("reduceUnrolling2 elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
           iElaps, gpu_sum, grid.x / 2, block.x);

    // kernel 3: reduceUnrolling4
    CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    reduceUnroll4<<<grid.x / 4, block>>>(idata_dev, odata_dev, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 4; i++)
        gpu_sum += odata_host[i];
    printf("reduceUnrolling4 elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
           iElaps, gpu_sum, grid.x / 4, block.x);

    // kernel 4:reduceUnrolling8
    CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    reduceUnroll8<<<grid.x / 8, block>>>(idata_dev, odata_dev, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 8; i++)
        gpu_sum += odata_host[i];
    printf("reduceUnrolling8 elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
           iElaps, gpu_sum, grid.x / 8, block.x);

    // kernel 5: reuduceUnrollWarp8
    CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    reduceUnrollWarp8<<<grid.x / 8, block>>>(idata_dev, odata_dev, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 8; i++)
        gpu_sum += odata_host[i];
    printf("reduceUnrollingWarp8 elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
           iElaps, gpu_sum, grid.x / 8, block.x);

    // kernel 3:reduceCompleteUnrollWarp8
    CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    reduceCompleteUnrollWarp8<<<grid.x / 8, block>>>(idata_dev, odata_dev, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 8; i++)
        gpu_sum += odata_host[i];
    printf("reduceCompleteUnrollWarp8   elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
           iElaps, gpu_sum, grid.x / 8, block.x);
}