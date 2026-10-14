#include "attention.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

void fill_random(float* data, int count) {
    for (int i = 0; i < count; i++) {
        data[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    }
}

int main() {
    constexpr int qkv_elems = N * d;
    constexpr int score_elems = N * N;

    float* h_q = new float[qkv_elems];
    float* h_k = new float[qkv_elems];
    float* h_v = new float[qkv_elems];
    float* h_naive = new float[qkv_elems];
    float* h_flash = new float[qkv_elems];

    srand(0);
    fill_random(h_q, qkv_elems);
    fill_random(h_k, qkv_elems);
    fill_random(h_v, qkv_elems);

    float *d_q, *d_k, *d_v, *d_s, *d_naive, *d_flash;
    cudaMalloc(&d_q, qkv_elems * sizeof(float));
    cudaMalloc(&d_k, qkv_elems * sizeof(float));
    cudaMalloc(&d_v, qkv_elems * sizeof(float));
    cudaMalloc(&d_s, score_elems * sizeof(float));
    cudaMalloc(&d_naive, qkv_elems * sizeof(float));
    cudaMalloc(&d_flash, qkv_elems * sizeof(float));

    cudaMemcpy(d_q, h_q, qkv_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k, qkv_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, qkv_elems * sizeof(float), cudaMemcpyHostToDevice);

    launch_naive(d_q, d_k, d_v, d_s, d_naive);
    launch_flash(d_q, d_k, d_v, d_flash);

    cudaMemcpy(h_naive, d_naive, qkv_elems * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_flash, d_flash, qkv_elems * sizeof(float), cudaMemcpyDeviceToHost);

    float max_absolute_error = 0.0f;
    float max_relative_error = 0.0f;
    int mismatches = 0;

    for (int i = 0; i < qkv_elems; i++) {
        float absolute_error = fabsf(h_flash[i] - h_naive[i]);
        float relative_error = absolute_error / (fabsf(h_naive[i]) + 1e-6f);

        max_absolute_error = fmaxf(max_absolute_error, absolute_error);
        max_relative_error = fmaxf(max_relative_error, relative_error);

        if (absolute_error > 1e-4f + 1e-4f * fabsf(h_naive[i])) {
            mismatches++;
        }
    }

    printf("Max absolute error: %g\n", max_absolute_error);
    printf("Max relative error: %g\n", max_relative_error);
    printf("Mismatches: %d / %d\n", mismatches, qkv_elems);

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_s);
    cudaFree(d_naive);
    cudaFree(d_flash);

    delete[] h_q;
    delete[] h_k;
    delete[] h_v;
    delete[] h_naive;
    delete[] h_flash;
}
