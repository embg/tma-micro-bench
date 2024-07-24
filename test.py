import torch
torch.ops.load_library("tma_kernels.so")

gpu_desc = torch.empty(128, device="cuda", dtype=torch.uint8)

def run_grid_const(desc, tensor):
    torch.ops.tma_kernels.add1_tma_grid_const(tensor)
    
def run_byref_memcpy(desc, tensor):
    torch.ops.tma_kernels.add1_tma_byref_excl_memcpy(desc, tensor)
    
def run_ondevice(desc, tensor):
    torch.ops.tma_kernels.add1_tma_ondevice(desc, tensor)
    
def run_ondevice_cpfence(desc, tensor):
    torch.ops.tma_kernels.add1_tma_ondevice_cpfence(desc, tensor)

def test(func, m, n):
    tensor = torch.randn(m, n, device="cuda")
    cpu_desc = torch.empty(128, device="cpu", dtype=torch.uint8)
    torch.ops.tma_kernels.fill_tma_desc_for_tensor(cpu_desc, tensor)
    gpu_desc.copy_(cpu_desc)
    val = torch.clone(tensor) + 1
    func(gpu_desc, tensor)
    assert torch.allclose(val, tensor)

if __name__ == "__main__":
    shapes = [
        (256 * 9, 128 * 12),
        (128 * 7, 256 * 13),
        (64, 64),
        (4096, 64),
        (64, 4096)
    ]
    for m,n in shapes:
        # These tests all pass
        test(run_grid_const, m, n)
        test(run_byref_memcpy, m, n)
        test(run_ondevice, m, n)
        
        # Currently broken
        test(run_ondevice_cpfence, m, n)
