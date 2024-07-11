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

void fill_tma_desc_for_tensor(at::Tensor cpu_desc, at::Tensor device_tensor) {
  if (!device_tensor.device().is_cuda()) {
    throw std::runtime_error("device_tensor must be on cuda");
  }
  if (!cpu_desc.device().is_cpu()) {
    throw std::runtime_error("cpu_desc must be on cpu");
  }
  TmaDesc desc(device_tensor.data_ptr<float>(), device_tensor.size(0), device_tensor.size(1));
  CUtensorMap desc_raw = desc.get();
  memcpy(cpu_desc.data_ptr<char>(), &desc_raw, sizeof(CUtensorMap));
}

void experiment_teardown(at::Tensor dst, at::Tensor src) {
  
}

TORCH_LIBRARY(tma_kernels, m) {
  m.def("copy_2d_tma_grid_const", copy_2d_tma_grid_const);
  m.def("experiment_setup", experiment_setup);
  m.def("fill_tma_desc_for_tensor", fill_tma_desc_for_tensor);
}
