#include "attention.cuh"

#include <cuda_runtime.h>
#include <cmath>

__global__ void flash_attention_forward(const float* q,
                                        const float* k,
                                        const float* v,
                                        float* o) {
    __shared__ float q_tile[B_r][d];
    __shared__ float k_tile[B_c][d];
    __shared__ float v_tile[B_c][d];
    __shared__ float s_tile[B_r][B_c];
    __shared__ float o_tile[B_r][d];
    __shared__ float running_max[B_r];
    __shared__ float running_sum[B_r];
    __shared__ float rescale[B_r];

    int row = threadIdx.y;
    int global_q_row = blockIdx.x * B_r + row;

    for (int dim = threadIdx.x; dim < d; dim += blockDim.x) {
        q_tile[row][dim] = q[global_q_row * d + dim];
        o_tile[row][dim] = 0.0f;
    }

    if (threadIdx.x == 0) {
        running_max[row] = -INFINITY;
        running_sum[row] = 0.0f;
    }
    __syncthreads();

    for (int tile = 0; tile < T_c; tile++) {
        int global_k_row = tile * B_c + row;

        for (int dim = threadIdx.x; dim < d; dim += blockDim.x) {
            k_tile[row][dim] = k[global_k_row * d + dim];
            v_tile[row][dim] = v[global_k_row * d + dim];
        }
        __syncthreads();

        int col = threadIdx.x;
        global_k_row = tile * B_c + col;

        if (global_k_row <= global_q_row) {
            float score = 0.0f;
            for (int dim = 0; dim < d; dim++) {
                score += q_tile[row][dim] * k_tile[col][dim];
            }
            s_tile[row][col] = score * scale;
        } else {
            s_tile[row][col] = -INFINITY;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            float old_max = running_max[row];
            float tile_max = -INFINITY;

            for (int i = 0; i < B_c; i++) {
                tile_max = fmaxf(tile_max, s_tile[row][i]);
            }

            float new_max = fmaxf(old_max, tile_max);
            rescale[row] = expf(old_max - new_max);

            float tile_sum = 0.0f;
            for (int i = 0; i < B_c; i++) {
                tile_sum += expf(s_tile[row][i] - new_max);
            }

            running_sum[row] = rescale[row] * running_sum[row] + tile_sum;
            running_max[row] = new_max;
        }
        __syncthreads();

        for (int out_col = threadIdx.x; out_col < d; out_col += blockDim.x) {
            float value = 0.0f;
            for (int i = 0; i < B_c; i++) {
                value += expf(s_tile[row][i] - running_max[row])
                       * v_tile[i][out_col];
            }

            o_tile[row][out_col] =
                rescale[row] * o_tile[row][out_col] + value;
        }
        __syncthreads();
    }

    for (int col = threadIdx.x; col < d; col += blockDim.x) {
        o[global_q_row * d + col] =
            o_tile[row][col] / running_sum[row];
    }
}

void launch_flash(const float* q,
                  const float* k,
                  const float* v,
                  float* o) {
    dim3 block(B_c, B_r);
    dim3 grid(T_r);
    flash_attention_forward<<<grid, block>>>(q, k, v, o);
}
