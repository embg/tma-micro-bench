#pragma once

#include <cstddef>
#include <cstdint>

// Make sure to keep this in sync with tma_triton_kernel.py
constexpr size_t BLOCK_M = 64;
constexpr size_t BLOCK_N = 64;

void launch_grid_constant_kernel(float* dst, float* src, size_t M, size_t N);

// @pre desc_cpu_ptr must point to a buffer of at least 128 bytes
void init_tma_descriptor(uint8_t* desc_cpu_ptr, float* tensor_gmem_ptr, size_t M, size_t N);
