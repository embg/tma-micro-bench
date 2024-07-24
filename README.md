# tma-micro-bench

```
sudo dnf install cuda-toolkit-12-4
export CUDA_HOME=/usr/local/cuda-12.4/
export TORCH_CUDA_ARCH_LIST="9.0a"
python setup.py develop
python test.py
python benchmark.py
```

![graph](graph.png)
