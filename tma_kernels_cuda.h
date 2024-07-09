#pragma once

#include <cstddef>

constexpr size_t BLOCK_M = 64;
constexpr size_t BLOCK_N = 64;

void launch_grid_constant_kernel(float* dst, float* src, size_t M, size_t N);
