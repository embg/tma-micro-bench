#include "tma_kernels_cuda.h"
#include <cudaTypedefs.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda/barrier>
#include <cuda/ptx>
#include <assert.h>

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

#define CUDA_CHECK(call) \
    do { \
        CUresult status = call; \
        if (status != CUDA_SUCCESS) { \
            const char* err; \
            cuGetErrorName(status, &err); \
            fprintf(stderr, "CUDA error at line %d in file %s: %s\n", __LINE__, __FILE__, err); \
            exit(1); \
        } \
    } while (0)

#define cdiv(a, b) (((a) + ((b) - 1)) / (b))

__global__ void grid_constant_kernel(
    float* tensor,
    const __grid_constant__ CUtensorMap desc,
    size_t M, size_t N)
{
  __shared__ alignas(128) float tma_buf[BLOCK_M][BLOCK_N];

  // Calculate coordinates for load / store
  const size_t grid_n = cdiv(N, BLOCK_N);
  const size_t pid_m = blockIdx.x / grid_n;
  const size_t pid_n = blockIdx.x % grid_n;
  const size_t offs_m = pid_m * BLOCK_M;
  const size_t offs_n = pid_n * BLOCK_N;

  // Initialize shared memory barrier with the number of threads participating in the barrier.
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;

  if (threadIdx.x == 0) {
    // Initialize barrier. All `blockDim.x` threads in block participate.
    init(&bar, blockDim.x);
    // Make initialized barrier visible in async proxy.
    cde::fence_proxy_async_shared_cta();    
  }
  // Syncthreads so initialized barrier is visible to all threads.
  __syncthreads();

  barrier::arrival_token token;
  if (threadIdx.x == 0) {
    // Initiate bulk tensor copy.
    cde::cp_async_bulk_tensor_2d_global_to_shared(tma_buf, &desc, offs_n, offs_m, bar);
    // Arrive on the barrier and tell how many bytes are expected to come in.
    token = cuda::device::barrier_arrive_tx(bar, 1, BLOCK_M * BLOCK_N * sizeof(float));
  } else {
    // Other threads just arrive.
    token = bar.arrive();
  }
  // Wait for the data to have arrived.
  bar.wait(std::move(token));

  // Increment all values in the tensor
  static_assert(BLOCK_M == 64);
  static_assert(BLOCK_N == 64);
  const size_t laneIdx = threadIdx.x & 31;
  for (int row = 0; row < BLOCK_M; row++) {
    const size_t rowStart = row * BLOCK_N;
    tma_buf[row][laneIdx] += 1;
    tma_buf[row][laneIdx + 32] += 1;
  }
  
  // Wait for shared memory writes to be visible to TMA engine.
  cde::fence_proxy_async_shared_cta();
  __syncthreads();
  // After syncthreads, writes by all threads are visible to TMA engine.

  // Initiate TMA transfer to copy shared memory to global memory
  if (threadIdx.x == 0) {
    cde::cp_async_bulk_tensor_2d_shared_to_global(&desc, offs_n, offs_m, (uint8_t*)tma_buf);
    // Wait for TMA transfer to have finished reading shared memory.
    // Create a "bulk async-group" out of the previous bulk copy operation.
    cde::cp_async_bulk_commit_group();
    // Wait for the group to have completed reading from shared memory.
    cde::cp_async_bulk_wait_group_read<0>();
  }

  // Destroy barrier. This invalidates the memory region of the barrier. If
  // further computations were to take place in the kernel, this allows the
  // memory location of the shared memory barrier to be reused.
  if (threadIdx.x == 0) {
    (&bar)->~barrier();
  }
}

TmaDesc::TmaDesc(float* gmem_ptr, size_t M, size_t N) {
  // Note: fastest-moving dimension always comes first here
  constexpr uint32_t rank = 2;
  const uint64_t size[rank] = {N, M}; // elements
  const uint64_t stride[rank - 1] = {N * sizeof(float)}; // bytes
  const uint32_t box_size[rank] = {BLOCK_N, BLOCK_M}; // elements
  const uint32_t elem_stride[rank] = {1, 1}; // elements

  CUDA_CHECK(cuTensorMapEncodeTiled(
      &desc_,
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
      rank,
      (void*)gmem_ptr,
      size,
      stride,
      box_size,
      elem_stride,
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  ));
}

void launch_grid_constant_kernel(float* tensor, size_t M, size_t N)
{
    TmaDesc desc(tensor, M, N);
    dim3 threadsPerBlock(32, 1, 1);
    dim3 numBlocks(cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N), 1, 1);
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, grid_constant_kernel);
    const size_t sharedSizeBytes = attr.sharedSizeBytes;
    cudaFuncSetAttribute(grid_constant_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024 - sharedSizeBytes);
    grid_constant_kernel<<<numBlocks, threadsPerBlock, 227 * 1024 - sharedSizeBytes>>>(tensor, *desc.get(), M, N);
}

// launch_ondevice_kernel
