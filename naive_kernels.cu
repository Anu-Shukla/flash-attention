#include "attention.cuh"

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cmath>

__global__ void attention_score(const float* q, const float* k, float* s) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= N || row >= N) return;

    if (col <= row) {
        float value = 0.0f;
        for (int i = 0; i < d; i++) {
            value += q[row * d + i] * k[col * d + i];
        }
        s[row * N + col] = value * scale;
    } else {
        s[row * N + col] = -INFINITY;
    }
}

__global__ void softmax_s(float* s) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float local_max = -INFINITY;
    float local_sum = 0.0f;

    using BlockReduce = cub::BlockReduce<float, SOFTMAX_DIM>;
    __shared__ typename BlockReduce::TempStorage storage;
    __shared__ float row_max;
    __shared__ float row_sum;

    for (int col = tid; col < N; col += blockDim.x) {
        local_max = fmaxf(local_max, s[row * N + col]);
    }

    float max_result = BlockReduce(storage).Reduce(local_max, cub::Max());
    if (tid == 0) row_max = max_result;
    __syncthreads();

    for (int col = tid; col < N; col += blockDim.x) {
        local_sum += expf(s[row * N + col] - row_max);
    }

    float sum_result = BlockReduce(storage).Sum(local_sum);
    if (tid == 0) row_sum = sum_result;
    __syncthreads();

    for (int col = tid; col < N; col += blockDim.x) {
        s[row * N + col] = expf(s[row * N + col] - row_max) / row_sum;
    }
}

__global__ void attention_output(const float* p, const float* v, float* o) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= d || row >= N) return;

    float value = 0.0f;
    for (int i = 0; i < N; i++) {
        value += p[row * N + i] * v[i * d + col];
    }
    o[row * d + col] = value;
}

void launch_naive(const float* q,
                  const float* k,
                  const float* v,
                  float* score_workspace,
                  float* o) {
    dim3 matrix_block(16, 16);
    dim3 score_grid((N + 15) / 16, (N + 15) / 16);
    dim3 output_grid((d + 15) / 16, (N + 15) / 16);

    attention_score<<<score_grid, matrix_block>>>(q, k, score_workspace);
    softmax_s<<<N, SOFTMAX_DIM>>>(score_workspace);
    attention_output<<<output_grid, matrix_block>>>(score_workspace, v, o);
}
