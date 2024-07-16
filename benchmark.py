import torch
import triton
torch.ops.load_library("tma_kernels.so")

configs = [
    triton.testing.Benchmark(
        x_names=["M"],
        x_vals=[i for i in range(64, 132 * 10 * 64, 33 * 64)],
        line_arg="provider",
        # line_vals=["copy_2d_tma_grid_const", "byref_incl_memcpy", "byref_excl_memcpy", "ondevice"],
        # line_names=["grid_constant", "byref_incl_memcpy", "byref_excl_memcpy", "ondevice"],
        line_vals=["copy_2d_tma_grid_const", "byref_incl_memcpy", "slow_memcpy", "ondevice"],
        line_names=["grid_constant", "async_memcpy_and_fence", "sync_memcpy_and_fence", "tensormap.replace"],
        ylabel="latency (us)",
        plot_name="TMA 2D add1 kernel latency: tensor shape (M,64)",
        args={},
        y_log=False
    )
]

cpu_desc_nopin = torch.empty(128, device="cpu", dtype=torch.uint8, pin_memory=False)
cpu_desc_pin = torch.empty(128, device="cpu", dtype=torch.uint8, pin_memory=True)   
gpu_desc = torch.empty(128, device="cuda", dtype=torch.uint8, pin_memory=False)   

@triton.testing.perf_report(configs)
def benchmark(M, provider):
    A = torch.zeros((M, 64), device="cuda", dtype=torch.float32)
    rep = 100 # ms
    quantiles = [0.5, 0.2, 0.8]
    if provider == "copy_2d_tma_grid_const":
        def gc():
            torch.ops.tma_kernels.add1_tma_grid_const(A)
        ms = triton.testing.do_bench(
            lambda: gc(),
            warmup = 25, rep = rep, return_mode = "min"
        )
    elif provider == "byref_incl_memcpy":
        def byref():
            torch.ops.tma_kernels.fill_tma_desc_for_tensor(cpu_desc_pin, A)
            gpu_desc.copy_(cpu_desc_pin, non_blocking=True)
            torch.ops.tma_kernels.add1_tma_byref_excl_memcpy(gpu_desc, A)
        ms = triton.testing.do_bench(byref, warmup = 25, rep = rep, return_mode = "min")
    elif provider == "slow_memcpy":
        def byref2():
            torch.ops.tma_kernels.fill_tma_desc_for_tensor(cpu_desc_nopin, A)
            gpu_desc.copy_(cpu_desc_nopin, non_blocking=False)
            torch.ops.tma_kernels.add1_tma_byref_excl_memcpy(gpu_desc, A)
        ms = triton.testing.do_bench(byref2, warmup = 25, rep = rep, return_mode = "min")
    elif provider == "ondevice":
        torch.ops.tma_kernels.fill_tma_desc_for_tensor(cpu_desc_nopin, A)
        gpu_desc.copy_(cpu_desc_nopin, non_blocking=False)
        ms = triton.testing.do_bench(
            lambda: torch.ops.tma_kernels.add1_tma_ondevice(gpu_desc, A),
            warmup = 25, rep = rep, return_mode = "min"
        )
    else:
        return

    def us(ms):
        return ms * 1000
    
    return us(ms)


def main():
    benchmark.run(show_plots=True, print_data=True, save_path=".")


if __name__ == "__main__":
    main()
