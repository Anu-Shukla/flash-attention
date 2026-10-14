#include "attention.cuh"

#include <stdlib.h>
#include <cuda_runtime.h>

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

    launch_naive(d_q, d_k, d_v, d_s, d_o);

    cudaMemcpy(h_o, d_o, qkv_elems * sizeof(float), cudaMemcpyDeviceToHost);
}
