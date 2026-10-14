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

## Next steps

The next step is to profile both implementations and compare:

- Kernel execution time
- Global memory traffic
- Memory bandwidth
- Shared-memory usage

Performance results will be added after profiling.
