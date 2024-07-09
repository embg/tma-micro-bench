import torch

torch.ops.load_library("tma_kernels.so")

if __name__ == "__main__":
    shapes = [
        (256 * 9, 128 * 12),
        (128 * 7, 256 * 13),
        (4096, 4096),
        (64, 64),
        (4096, 64),
        (64, 4096)
    ]
    for m,n in shapes:
        src = torch.randn(m, n, device="cuda")
        dst = torch.randn(m, n, device="cuda")
        val = torch.clone(src)
        src != dst
        torch.ops.tma_kernels.copy_2d_tma_grid_const(dst, src)
        torch.cuda.synchronize()
        src == dst
        dst == val
