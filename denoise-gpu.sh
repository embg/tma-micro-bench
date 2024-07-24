#!/bin/bash
# Script by @bertmaher

# There's a whole presentation about stable benchmarking here:
# https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9956-best-practices-when-benchmarking-cuda-applications_V2.pdf

# Lock GPU clocks
sudo nvidia-smi -i 5 -pm 1 >&/dev/null                # persistent mode
sudo nvidia-smi --power-limit=700 -i 5 >& /dev/null   # lock to 700 W
sudo nvidia-smi -lgc 1980 -i 5 >& /dev/null           # lock to 1980 MHz.

# TODO: On my devgpu, device 6 is apparently attached to NUMA node 3.  How did
# I discover this?
#
# `nvidia-smi -i 6 -pm 1` prints the PCI bus ID (00000000:C6:00.0)
#
# You can also get this from `nvidia-smi -x -q` and looking for minor_number
# and pci_bus_id
#
# Then, `cat /sys/bus/pci/devices/0000:c6:00.0/numa_node` prints 3
# is it always the case that device N is on numa node N/2? :shrug:
#
# Maybe automate this process or figure out if it always holds?
#
# ... Or you can just `nvidia-smi topo -mp` and it will just print out exactly
# what you want, like this:

#       GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7    mlx5_0  mlx5_1  mlx5_2  mlx5_3  CPU Affinity    NUMA Affinity
# GPU0   X      PXB     SYS     SYS     SYS     SYS     SYS     SYS     NODE    SYS     SYS     SYS     0-23,96-119     0
# GPU6  SYS     SYS     SYS     SYS     SYS     SYS      X      PXB     SYS     SYS     SYS     NODE    72-95,168-191   3

export CUDA_VISIBLE_DEVICES=5
numactl -m 1 -c 1 "$@"

# Unlock GPU clock
sudo nvidia-smi -rgc -i 5 >& /dev/null
