import torch
torch.ops.load_library("tma_kernels.so")

gpu_desc = torch.empty(128, device="cuda", dtype=torch.uint8)

def test(m, n):
    tensor = torch.randn(m, n, device="cuda")
    cpu_desc = torch.empty(128, device="cpu", dtype=torch.uint8)
    torch.ops.tma_kernels.fill_tma_desc_for_tensor(cpu_desc, tensor)
    gpu_desc.copy_(cpu_desc)
    val = torch.clone(tensor) + 1
    torch.cuda.synchronize()
    #torch.ops.tma_kernels.add1_tma_grid_const(tensor, True)
    torch.cuda.synchronize()
    torch.ops.tma_kernels.add1_tma_byref_excl_memcpy(gpu_desc, tensor)
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
        test(m, n)
