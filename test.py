import torch

torch.ops.load_library("tma_kernels.so")

if __name__ == "__main__":
    m = 9 * 128
    n = 12 * 256
    k = 4096

    a = torch.randn(m, n, device="cuda")
    b = torch.randn(m, n, device="cuda")

    print(dir(torch.ops.tma_kernels))
    torch.ops.tma_kernels.copy_2d_tma_grid_const(b, a)
