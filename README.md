# CUDA FlashAttention

This project implements the forward pass of causal attention in CUDA.

It contains two versions:

- A naive implementation that creates the full attention matrix in global memory.
- A tiled FlashAttention-style implementation that uses shared memory tiling.

It follows the algorithm outlined in the FlashAttention-2 paper.

## Design

I have intentionally implemented the FlashAttention kernel in a very basic way. 
The goal was to implement every operation myself to understand the whole kernel, so I did not use external libraries such as cuBLAS for matrix multiplication or warp shuffles for reductions.

## Correctness

`compare.cu` runs both implementations with the same inputs and compares their outputs.

Current results on a 3090 Ti (SM 86):

```text
Max absolute error: 5.96046e-08
Mismatches: 0 / 65536
```

Build and run the comparison:

```bash
nvcc compare.cu naive_kernels.cu flash_kernels.cu -o compare
./compare
```

## Profiling results

DRAM traffic was measured on a 3090 Ti using NVIDIA Nsight Compute:

```bash
sudo ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum ./naive
sudo ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum ./flash
```

| Implementation | Reads | Writes | Total |
| --- | ---: | ---: | ---: |
| Naive | 9.21 MB | 4.57 MB | 13.78 MB |
| FlashAttention | 0.80 MB | 0.055 MB | 0.855 MB |

The naive totals include all three kernels. The FlashAttention implementation uses approximately 16.1x less DRAM traffic.
