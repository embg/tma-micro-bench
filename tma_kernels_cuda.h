#pragma once

#include <cstddef>

constexpr size_t BLOCK_M = 256;
constexpr size_t BLOCK_N = 256;

void launch_grid_constant_kernel(float* dst, float* src, size_t M, size_t N);
