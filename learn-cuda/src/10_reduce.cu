#include <cuda.h>
#include "../include/header.h"

__global__ void reduce0(long long* input_data, long long* output_data, const int size){
    // extern __shared__ long long shared_data[];
    // for every block, copy the input_data[] to shared_data[]
    int block_tid = threadIdx.x;
    int global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_idx >= size) return;
    long long* shared_data = input_data + blockIdx.x * blockDim.x;
    __syncthreads();

    // for every block, calculate the sum by the reduce-alg
    for (int s = 1; s < blockDim.x; s *= 2){
        if (block_tid % (2*s) == 0 && block_tid + 2*s < blockDim.x){
            shared_data[block_tid] += shared_data[block_tid + s];
        }
        __syncthreads();
    }

    // save every block's answer
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
    cudaMalloc((void**)&output_data_dev, grid.x * sizeof(int));
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
    cudaMemcpy(output_data_host, output_data_dev, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
    long long gpu_sum = 0;
    for (int i = 0; i < grid.x; i++){
        gpu_sum += output_data_host[i];
    }
    t = get_time() - t;
    check_kernal_error();
    printf("reduce0: ans=%lld, time=%6lf\n", gpu_sum, t);

    // free space
    free(input_data_host);
    free(output_data_host);
    cudaFree(input_data_dev);
    cudaFree(output_data_dev);

    return 0;
}