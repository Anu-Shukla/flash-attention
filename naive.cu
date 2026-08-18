#include <cub/thread/thread_operators.cuh>
#include <driver_types.h>
#include <stdlib.h>
#include <cmath>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

constexpr int N = 1024;
constexpr int d = 64;
constexpr float scale = 1.0f/8.0f;
constexpr int SOFTMAX_DIM = 256;

__global__ void attention_score(const float* d_q, const float* d_k, float* d_s) {
    int col_s = blockIdx.x * blockDim.x + threadIdx.x;
    int row_s = blockIdx.y * blockDim.y + threadIdx.y;

    if (col_s >= N || row_s >= N) {
        return;
    }

    float accumulator = 0.0f;
    if (col_s <= row_s) {
        for (int i = 0; i < d; i++) {
            accumulator += d_q[row_s * d + i] * d_k[col_s * d+ i];
        }
        d_s[row_s * N + col_s] = accumulator * scale;
    } else {
        d_s[row_s * N + col_s] = -INFINITY;
    }
}

__global__ void softmax_s(float* d_s) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    float local_max = -INFINITY; float local_sum = 0;
    __shared__ float shared_row_max; __shared__ float shared_row_sum;
    using BlockReduce = cub::BlockReduce<float, SOFTMAX_DIM>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    //make each thread find its local sum and max
    for (int col = tid; col < N; col += SOFTMAX_DIM) {
        local_max = max(local_max, d_s[row * N + col]);
    }
    float row_max = BlockReduce(temp_storage).Reduce(local_max, cub::Max());

    if (tid == 0) {
        shared_row_max = row_max;
    }
    __syncthreads();

    for (int col = tid; col < N; col += SOFTMAX_DIM) {
        local_sum += expf(d_s[row * N + col] - shared_row_max);
    }
    float row_sum = BlockReduce(temp_storage).Sum(local_sum);

    if (tid == 0) {
        shared_row_sum = row_sum;
    }
    __syncthreads(); 

    for(int col = tid; col < N; col += SOFTMAX_DIM) {
        d_s[row * N + col] = expf(d_s[row * N + col] - shared_row_max) / shared_row_sum;
    }
}

__global__ void attention_output(const float* d_p, const float* d_v, float* d_o) {
    int col_o = blockIdx.x * blockDim.x + threadIdx.x;
    int row_o = blockIdx.y * blockDim.y + threadIdx.y;

    if (col_o >= d || row_o >= N) {
        return;
    }

    float accumulator = 0.0f;
    for (int i = 0; i < N; i++) {
        accumulator += d_p[row_o * N + i] * d_v[i * d + col_o];
    }
    d_o[row_o * d + col_o] = accumulator;
}

void fill_random(float* data, int elems) {
    for (int i = 0; i < elems; i++) {
        data[i] = ((float)rand() / (float)RAND_MAX) - 0.5f;
    }
}


int main() {
    const int qkv_elems = N * d;
    const int attention_elems = N * N;

    float* h_q = new float[qkv_elems];
    float* h_k = new float[qkv_elems];
    float* h_v = new float[qkv_elems];
    float* h_o = new float[qkv_elems];
    srand(0);
    
    fill_random(h_q, qkv_elems);
    fill_random(h_k, qkv_elems);
    fill_random(h_v, qkv_elems);
    
    float *d_q, *d_k, *d_v, *d_s, *d_o;
    cudaMalloc(&d_q, qkv_elems * sizeof(float));
    cudaMalloc(&d_k, qkv_elems * sizeof(float));
    cudaMalloc(&d_v, qkv_elems * sizeof(float));
    cudaMalloc(&d_o, qkv_elems * sizeof(float));

    cudaMalloc(&d_s, attention_elems * sizeof(float));

    cudaMemcpy(d_q, h_q, qkv_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k, qkv_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, qkv_elems * sizeof(float), cudaMemcpyHostToDevice);

    dim3 score_block(16,16); dim3 score_grid((N / score_block.x), (N / score_block.y));
    attention_score<<<score_grid, score_block>>>(d_q, d_k, d_s);

    dim3 softmax_block(SOFTMAX_DIM); dim3 softmax_grid(N);
    softmax_s<<<softmax_grid, softmax_block>>>(d_s);

    dim3 output_block(16,16); dim3 output_grid((d / output_block.x), (N / output_block.y));
    attention_output<<<output_grid, output_block>>>(d_s, d_v, d_o);

    cudaMemcpy(h_o, d_o, qkv_elems * sizeof(float), cudaMemcpyDeviceToHost);
}
