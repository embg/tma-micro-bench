import torch
import tma_triton_kernel
torch.ops.load_library("tma_kernels.so")

def test(tma_copy_func, m, n):
    src = torch.randn(m, n, device="cuda")
    dst = torch.randn(m, n, device="cuda")
    val = torch.clone(src)
    assert not torch.equal(src, dst)
    tma_copy_func(dst, src)
    torch.cuda.synchronize()
    assert torch.equal(val, dst)
    assert torch.equal(val, src)

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
        test(torch.ops.tma_kernels.copy_2d_tma_grid_const, m, n)
        test(tma_triton_kernel.copy_2d_tma_triton, m, n)
