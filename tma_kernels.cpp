#include "tma_kernels_cuda.h"
#include "ATen/ATen.h"
#include "torch/extension.h"

namespace {
void validate(at::Tensor dst, at::Tensor src) {
  if (dst.size(0) != src.size(0)) {
    throw std::runtime_error("dst.size(0) != src.size(0)");
  }
  if (dst.size(1) != src.size(1)) {
    throw std::runtime_error("dst.size(1) != src.size(1)");
  }
  if (dst.scalar_type() != at::kFloat) {
    throw std::runtime_error("dst.scalar_type() != at::kFloat");
  }
  if (src.scalar_type() != at::kFloat) {
    throw std::runtime_error("src.scalar_type() != at::kFloat");
  }
}
}

void copy_2d_tma_grid_const(at::Tensor dst, at::Tensor src) {
  validate(dst, src);
  launch_grid_constant_kernel(
      dst.data_ptr<float>(),
      src.data_ptr<float>(),
      src.size(0),
      src.size(1)
  );
}

TORCH_LIBRARY(tma_kernels, m) {
  m.def("copy_2d_tma_grid_const", copy_2d_tma_grid_const);
}
