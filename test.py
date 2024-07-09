import torch

torch.ops.load_library("tma_kernels.so")

if __name__ == "__main__":
    m = 9 * 128
    n = 12 * 256

    src = torch.randn(m, n, device="cuda")
    dst = torch.randn(m, n, device="cuda")
    
    assert not torch.allclose(src, dst)
    torch.ops.tma_kernels.copy_2d_tma_grid_const(dst, src)
    torch.cuda.synchronize()
    assert torch.allclose(src, dst)
