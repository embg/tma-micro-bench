#include "tma_kernels_cuda.h"
#include <cudaTypedefs.h>
#include <stdio.h>
#include <cuda.h>
#include <assert.h>

#define CUDA_CHECK(call) \
    do { \
        CUresult status = call; \
        if (status != CUDA_SUCCESS) { \
            const char* err; \
            cuGetErrorString(status, &err); \
            fprintf(stderr, "CUDA error at line %d in file %s: %s\n", __LINE__, __FILE__, err); \
            exit(1); \
        } \
    } while (0)

namespace {
size_t cdiv(size_t a, size_t b) {
    return (a + (b - 1)) / b;
}
}

__global__ void grid_constant_kernel(
    float* dst, const __grid_constant__ CUtensorMap dst_desc,
    const float* src,  const __grid_constant__ CUtensorMap src_desc,
    size_t M, size_t N)
{
    printf("Hello World from GPU!\n");
}

void launch_grid_constant_kernel(float* dst, float* src, size_t M, size_t N)
{
    CUtensorMap dst_desc{};
    CUtensorMap src_desc{};
    
    // Note: fastest-moving dimension always comes first here
    constexpr uint32_t rank = 2;
    const uint64_t size[rank] = {N, M}; // elements
    const uint64_t stride[rank - 1] = {N * sizeof(float)}; // bytes
    const uint32_t box_size[rank] = {BLOCK_N, BLOCK_M}; // elements
    const uint32_t elem_stride[rank] = {1, 1}; // elements

    CUDA_CHECK(cuTensorMapEncodeTiled(
        &src_desc,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        rank,
        src,
        size,
        stride,
        box_size,
        elem_stride,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));
    
    CUDA_CHECK(cuTensorMapEncodeTiled(
        &dst_desc,                // CUtensorMap *tensorMap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        rank,
        dst,
        size,
        stride,
        box_size,
        elem_stride,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));

    dim3 threadsPerBlock(32, 1, 1);
    dim3 numBlocks(cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N), 1, 1);
    grid_constant_kernel<<<numBlocks, threadsPerBlock>>>(dst, dst_desc, src, src_desc, M, N);
}
