#include "tma_kernels_cuda.h"
#include "ATen/ATen.h"
#include "torch/extension.h"

void add1_tma_grid_const(at::Tensor tensor) {
  if (tensor.scalar_type() != at::kFloat) {
    throw std::runtime_error("tensor.scalar_type() != at::kFloat");
  }
  launch_grid_constant_kernel(
      tensor.data_ptr<float>(),
      tensor.size(0),
      tensor.size(1)
  );
}

void add1_tma_byref_excl_memcpy(at::Tensor desc, at::Tensor tensor) {
  if (tensor.scalar_type() != at::kFloat) {
    throw std::runtime_error("tensor.scalar_type() != at::kFloat");
  }
  launch_fence_kernel(
      desc.data_ptr<uint8_t>(),
      tensor.size(0),
      tensor.size(1)
  );
}

void add1_tma_ondevice(at::Tensor desc, at::Tensor tensor) {
  if (tensor.scalar_type() != at::kFloat) {
    throw std::runtime_error("tensor.scalar_type() != at::kFloat");
  }
  launch_ondevice_kernel(
      desc.data_ptr<uint8_t>(),
      tensor.size(0),
      tensor.size(1)
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
  memcpy(cpu_desc.data_ptr(), desc.get(), sizeof(CUtensorMap));
}

TORCH_LIBRARY(tma_kernels, m) {
  m.def("add1_tma_grid_const", add1_tma_grid_const);
  m.def("fill_tma_desc_for_tensor", fill_tma_desc_for_tensor);
  m.def("add1_tma_byref_excl_memcpy", add1_tma_byref_excl_memcpy);
  m.def("add1_tma_ondevice", add1_tma_ondevice);
}
