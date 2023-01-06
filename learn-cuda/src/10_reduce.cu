#include <cuda.h>
#include "../include/header.h"

const int threadsPerBlock = 1024;

__global__ void reduce0(long long* input_data, long long* output_data, const int size){
    __shared__ long long shared_data[threadsPerBlock];
    // for every block, copy the input_data[] to shared_data[]
    int block_tid = threadIdx.x;
    int global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_idx >= size) return;
    shared_data[block_tid] = input_data[global_idx];
    __syncthreads();

    // for every block, calculate the sum by the reduce-alg
    for (int s = 1; s < blockDim.x; s *= 2){
        if (block_tid % (2*s) == 0 && block_tid + s < blockDim.x){
            shared_data[block_tid] += shared_data[block_tid + s];
        }
        __syncthreads();
    }

    // save every block's answer
    if (block_tid == 0){
        output_data[blockIdx.x] = shared_data[0];
    }
}

__global__ void reduce1(long long* input_data, long long* output_data, const int size){
    __shared__ long long shared_data[threadsPerBlock];
    int block_tid = threadIdx.x;
    int global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_idx >= size) return;
    shared_data[block_tid] = input_data[global_idx];
    __syncthreads();

    // for every block, calculate the sum
    for (int s = 1; s <= blockDim.x; s *= 2){
        int index = 2 * s * block_tid;  // 用这种方式把前几个thread映射到整个shared_memory上
        if (index + s >= blockDim.x){
            break;
        }
        shared_data[index] += shared_data[index + s];
        __syncthreads();
    }

    // copy block answer to output_data
    if (block_tid == 0){
        output_data[blockIdx.x] = shared_data[0];
    }
}

__global__ void reduce2(long long* input_data, long long* output_data, const int size){
    __shared__ long long shared_data[threadsPerBlock];
    int block_tid = threadIdx.x;
    int global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_idx >= size) return;
    shared_data[block_tid] = input_data[global_idx];
    __syncthreads();

    // for every block, calculate the sum
    for (int s = blockDim.x / 2; s >= 1; s /= 2){
        // 交错配对
        if (block_tid < s){
            shared_data[block_tid] += shared_data[block_tid + s];
        }
        __syncthreads();
    }

    // copy block answer to output_data
    if (block_tid == 0){
        output_data[blockIdx.x] = shared_data[0];
    }
}

long long cpu_common(long long* input_data, int size){
    long long sum = 0;
    for (int i = 0; i < size; i++){
        sum += input_data[i];
    }
    return sum;
}

int cpu_reduce(long long* input_data, int size){
    for (int step = 1; step < size; step *= 2){
        for (int i = 0; i + step < size; i += 2*step){
            input_data[i] += input_data[i + step];
        }
    }
    if (size % 2 == 1) input_data[0] += input_data[size-1];
    return input_data[0];
}

int main(){
    // excecution configuration
    long long size = 1 << 22;
    long long nbytes = size * sizeof(long long);

    dim3 block(1024);
    dim3 grid((size - 1) / block.x + 1);
    printf("data size: %lld, block(%d, 1, 1), grid(%d, 1, 1)\n", size / 1024 / 1024, block.x, grid.x);

    // allocate space
    long long* input_data_host = (long long*)malloc(nbytes);
    long long* output_data_host = (long long*)malloc(grid.x * sizeof(long long));
    for (int i = 0; i < size; i++){
        input_data_host[i] = 1;
    }

    long long* input_data_dev = nullptr;
    long long* output_data_dev = nullptr;
    cudaMalloc((void**)&input_data_dev, nbytes);
    cudaMalloc((void**)&output_data_dev, grid.x * sizeof(long long));
    cudaMemcpy(input_data_dev, input_data_host, nbytes, cudaMemcpyHostToDevice);

    //cpu-common
    double t = get_time();
    long long ans = cpu_common(input_data_host, size);
    t = get_time() - t;
    printf("cpu-common: ans=%lld, time=%6lf\n", ans, t);

    // cpu-reduce

    // reduce0
    t = get_time();
    reduce0<<<grid, block>>>(input_data_dev, output_data_dev, size);
    check_kernal_error();

    cudaMemcpy(output_data_host, output_data_dev, grid.x*sizeof(long long), cudaMemcpyDeviceToHost);
    long long gpu_sum = 0;
    for (int i = 0; i < grid.x; i++){
        gpu_sum += output_data_host[i];
    }
    t = get_time() - t;
    
    printf("reduce0: ans=%lld, time=%6lf\n", gpu_sum, t);

    // reduece1
    t = get_time();
    reduce1<<<grid, block>>>(input_data_dev, output_data_dev, size);
    check_kernal_error();

    cudaMemcpy(output_data_host, output_data_dev, grid.x*sizeof(long long), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++){
        gpu_sum += output_data_host[i];
    }
    t = get_time() - t;
    
    printf("reduce1: ans=%lld, time=%6lf\n", gpu_sum, t);

    //reduce2
    t = get_time();
    reduce2<<<grid, block>>>(input_data_dev, output_data_dev, size);
    check_kernal_error();

    cudaMemcpy(output_data_host, output_data_dev, grid.x*sizeof(long long), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++){
        gpu_sum += output_data_host[i];
    }
    t = get_time() - t;
    
    printf("reduce2: ans=%lld, time=%6lf\n", gpu_sum, t);

    // free space
    free(input_data_host);
    free(output_data_host);
    cudaFree(input_data_dev);
    cudaFree(output_data_dev);

    return 0;
}