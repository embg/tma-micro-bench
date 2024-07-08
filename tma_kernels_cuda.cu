#include "tma_kernels_cuda.h"
#include <stdio.h>

__global__ void grid_constant_kernel(float* dst, const float* src, int M, int N){
    printf("Hello World from GPU!\n");
}

void launch_grid_constant_kernel(float* dst, float* src, int M, int N)
{
    dim3 threadsPerBlock(256, 1);
    dim3 numBlocks(100, 1, 1);
    grid_constant_kernel<<<numBlocks, threadsPerBlock>>>(dst, src, M, N);
}
