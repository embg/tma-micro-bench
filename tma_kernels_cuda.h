#pragma once

#define BLOCK_M 256
#define BLOCK_N 256

void launch_grid_constant_kernel(float* dst, float* src, int M, int N);
