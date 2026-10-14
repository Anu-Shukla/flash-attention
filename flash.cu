#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>
#include <cub/cub.cuh>
#include <cuda_runtime.h>


constexpr int N = 1024;
constexpr int d = 64;
constexpr float scale = 1.0f/8.0f;


constexpr int B_r = 16; //number of query rows handled in 1 block
constexpr int B_c = 16; //number of key/value cols handled in 1 block

//number of Tiles required:
constexpr int T_r = (N + B_r - 1)/B_r;
constexpr int T_c = (N + B_c - 1)/B_c;
//we will use a grid of T_r blocks, each block will handle T_c tiles

__global__ void flash_attention_forward(const float* q, const float* k, const float* v, float* o) {
    int block_id = blockIdx.x;

    //SRAM tiles:
    __shared__ float q_tile[B_r][d];
    __shared__ float k_tile[B_c][d];
    __shared__ float v_tile[B_c][d];
    __shared__ float s_tile[B_r][B_c];
    __shared__ float o_tile[B_r][d];

    __shared__ float running_max[B_r];
    __shared__ float running_sum[B_r];
    __shared__ float rescale[B_r];
    
    //load Q tile:
    int row_in_tile = threadIdx.y;
    int global_row = block_id * B_r + row_in_tile;
    for (int dim = threadIdx.x; dim < d; dim += blockDim.x) {
        q_tile[row_in_tile][dim] = q[global_row * d + dim];
    }
    __syncthreads();

    //initialize running max and running sum values:
    if (threadIdx.x == 0) {
      running_max[threadIdx.y] = -INFINITY;
      running_sum[threadIdx.y] = 0.0f;
    }

    for (int dim = threadIdx.x; dim < d; dim += blockDim.x) {
        o_tile[threadIdx.y][dim] = 0.0f;
    }
    __syncthreads();


    for (int tile_index = 0; tile_index < T_c; tile_index++) {
        //load K/V tiles:
        int global_row = tile_index * B_c + threadIdx.y;
        for (int dim = threadIdx.x; dim < d; dim += blockDim.x) {
            k_tile[threadIdx.y][dim] = k[global_row * d + dim];
            v_tile[threadIdx.y][dim] = v[global_row * d + dim];
        }
        __syncthreads();

        //compute attention tile:
        int col = threadIdx.x;
        int row = threadIdx.y;

        float accumulator = 0.0f;

        int global_q_row = blockIdx.x * B_r + row;
        int global_k_row = tile_index * B_c + col;
        if (global_k_row <= global_q_row) {
           for (int i = 0; i < d; i++) {
                accumulator += q_tile[row][i] * k_tile[col][i];
           } 
           s_tile[row][col] = accumulator * scale;
        } else {
            s_tile[row][col] = -INFINITY;
        }
        __syncthreads();

        //update row max and row sum:
        if (threadIdx.x == 0) {
            float old_max = running_max[row];
            float tile_max = -INFINITY;
            for (int i = 0; i < B_c; i++) {
                tile_max = fmaxf(tile_max, s_tile[row][i]);
            }
            float new_max = fmaxf(tile_max, old_max);
            rescale[row] = expf(old_max - new_max);

            float tile_sum = 0;
            for (int i = 0; i < B_c; i++) {
                tile_sum += expf(s_tile[row][i] - new_max);
            }
            running_sum[row] = rescale[row] * running_sum[row] + tile_sum;
            running_max[row] = new_max;
        }
        __syncthreads();

        //implicitly make the tile for P using s_tile and multiply by v_tile to write to o_tile
        for (int o_tile_col = threadIdx.x; o_tile_col < d; o_tile_col += blockDim.x){
            float accumulator = 0.0f;
            for (int k = 0; k < B_c; k++) {
                accumulator += expf(s_tile[row][k] - running_max[row]) * v_tile[k][o_tile_col];
            }
            o_tile[row][o_tile_col] = rescale[row] * o_tile[row][o_tile_col] + accumulator;
        }
        __syncthreads();
    }
    //write each o_tile to the O in HBM while normalizing in the process:
    int row = threadIdx.y;
    for (int o_col = threadIdx.x; o_col < d; o_col += blockDim.x) {
        o[global_row * d + o_col] = o_tile[row][o_col] / running_sum[row];
    }
}

void fill_random(float* data, int elems) {
    for (int i = 0; i < elems; i++) {
        data[i] = ((float)rand() / (float)RAND_MAX) - 0.5f;
    }
}


int main() {
    const int qkv_elems = N * d;

    float* h_q = new float[qkv_elems];
    float* h_k = new float[qkv_elems];
    float* h_v = new float[qkv_elems];
    float* h_o = new float[qkv_elems];
    srand(0);
    
    fill_random(h_q, qkv_elems);
    fill_random(h_k, qkv_elems);
    fill_random(h_v, qkv_elems);
    
    float *d_q, *d_k, *d_v, *d_o;
    cudaMalloc(&d_q, qkv_elems * sizeof(float));
    cudaMalloc(&d_k, qkv_elems * sizeof(float));
    cudaMalloc(&d_v, qkv_elems * sizeof(float));
    cudaMalloc(&d_o, qkv_elems * sizeof(float));

    cudaMemcpy(d_q, h_q, qkv_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k, qkv_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, qkv_elems * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(B_c, B_r);
    dim3 grid(T_r);
    flash_attention_forward<<<grid, block>>>(d_q, d_k, d_v, d_o);

    cudaMemcpy(h_o, d_o, qkv_elems * sizeof(float), cudaMemcpyDeviceToHost);
}
