#include <cuda.h>
#include "../include/header.h"

const long long size = 1 << 24;
const int threadPerBlock = 1024;

__global__ void reduce3(int* input_data, int* output_data){
    // extend the loop, which lessen half of the calculation scale
    int block_tid = threadIdx.x;
    long long idx = 2 * blockDim.x * blockIdx.x + threadIdx.x;  // just use half grid number of before alg.
    if (idx >= size) return;  // the boundry. we just use half of the grid of before alg.
    if (idx + blockDim.x < size){
        input_data[idx] += input_data[idx + blockDim.x];
    }
    __syncthreads();

    __shared__ int shared_data[threadPerBlock];
    for (int s = threadPerBlock / 2; s >= 1; s /= 2){
        if (block_tid + s < threadPerBlock){
            shared_data[block_tid] += input_data[block_tid + s];
        }
        __syncthreads();
    }

    if (block_tid == 0){
        output_data[blockIdx.x] = shared_data[0];
    }
}

int main(){

    return 0;
}