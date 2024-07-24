import torch
import triton
torch.ops.load_library("tma_kernels.so")

configs = [
    triton.testing.Benchmark(
        x_names=["M"],
        x_vals=[i for i in range(64, 132 * 6 * 64, 33 * 64)],
        line_arg="provider",
        # line_vals=["copy_2d_tma_grid_const", "byref_incl_memcpy", "byref_excl_memcpy", "ondevice"],
        # line_names=["grid_constant", "byref_incl_memcpy", "byref_excl_memcpy", "ondevice"],
        # line_vals=["copy_2d_tma_grid_const", "byref_incl_memcpy", "slow_memcpy", "ondevice"],
        # line_names=["grid_constant", "async_memcpy_and_fence", "sync_memcpy_and_fence", "tensormap.replace"],
        line_vals=["copy_2d_tma_grid_const", "ondevice"],#"slow_memcpy", "fast_memcpy"],
        line_names=["grid_constant", "ondevice"],#"sync_memcpy_and_fence", "async_memcpy_and_fence"],
        styles=[("tab:blue", "solid"), ("tab:red", "dashed")],#("tab:green", "solid"), ("tab:orange", "solid")],
        ylabel="latency (us)",
        plot_name="TMA 2D add1 kernel latency: tensor shape (M,64)",
        args={},
        y_log=False
    )
] 

@triton.testing.perf_report(configs)
def benchmark(M, provider):
    A = torch.zeros((M, 64), device="cuda", dtype=torch.float32)
    rep = 100 # ms
    quantiles = [0.5, 0.2, 0.8]
    cpu_desc = torch.empty(128, device="cpu", dtype=torch.uint8, pin_memory=True)
    torch.ops.tma_kernels.fill_tma_desc_for_tensor(cpu_desc, A)
    gpu_desc = cpu_desc.cuda(non_blocking=True)
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        if provider == "copy_2d_tma_grid_const":
            ms = triton.testing.do_bench_cudagraph(
                lambda: torch.ops.tma_kernels.add1_tma_grid_const(A),
                return_mode="min"
            )
        elif provider == "ondevice":
            ms = triton.testing.do_bench_cudagraph(
                lambda: torch.ops.tma_kernels.add1_tma_ondevice(gpu_desc, A),
                return_mode="min"
            )
        elif provider == "ondevice_cpfence":
            pass # currently broken
        else:
            return

    def us(ms):
        return ms * 1000
    
    return us(ms)


def main():
    benchmark.run(show_plots=True, print_data=True, save_path=".")


if __name__ == "__main__":
    main()
