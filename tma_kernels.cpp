#include "tma_kernels.h"
#include "ATen/ATen.h"
#include "torch/extension.h"

namespace {
void validateSizes(at::Tensor dst, at::Tensor src) {
  if (dst.size(0) != src.size(0)) {
    throw std::runtime_error("dst.size(0) != src.size(0)");
  }
  if (dst.size(1) != src.size(1)) {
    throw std::runtime_error("dst.size(1) != src.size(1)");
  }
}
}

void copy_2d_tma_grid_const(at::Tensor dst, at::Tensor src) {
  validateSizes(dst, src);
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
