import torch
torch.ops.load_library("tma_kernels.so")

def test(add1_kernel, m, n):
    tensor = torch.randn(m, n, device="cuda")
    val = torch.clone(tensor) + 1
    torch.cuda.synchronize()
    add1_kernel(tensor)
    torch.cuda.synchronize()
    assert torch.allclose(val, tensor)

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
        print("shape:",m,n)
        test(torch.ops.tma_kernels.add1_tma_grid_const, m, n)
