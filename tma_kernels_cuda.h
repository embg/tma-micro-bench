#pragma once

#include <cstddef>

// Make sure to keep this in sync with tma_triton_kernel.py
constexpr size_t BLOCK_M = 64;
constexpr size_t BLOCK_N = 64;

void launch_grid_constant_kernel(float* dst, float* src, size_t M, size_t N);
