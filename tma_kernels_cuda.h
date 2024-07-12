#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda.h>

// Make sure to keep this in sync with tma_triton_kernel.py
constexpr size_t BLOCK_M = 64;
constexpr size_t BLOCK_N = 64;

void launch_grid_constant_kernel(float* tensor, size_t M, size_t N);
void launch_fence_kernel(uint8_t* desc, size_t M, size_t N);
void launch_ondevice_kernel(uint8_t* desc, size_t M, size_t N);

class TmaDesc {
  public:
    TmaDesc(float* gmem_ptr, size_t M, size_t N);

    TmaDesc(const TmaDesc& other) = delete;
    TmaDesc& operator=(const TmaDesc& other) = delete;
    TmaDesc(TmaDesc&& other) = delete;
    TmaDesc& operator=(TmaDesc&& other) = delete;
    ~TmaDesc() = default;
    
    CUtensorMap* get() {
        return &desc_;
    }

  private:
    CUtensorMap desc_;
};
