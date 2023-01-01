#include <cuda.h>
#include "../include/header.h"

__global__ void branch_kernal_1(float* arr){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float a = 0.f, b = 0.f;
    if (tid % 2 == 0){
        a = 100.0f;
    }
    else{
        b = 100.0f;
    }
    arr[tid] = a + b;
}

__global__ void branch_kernal_2(float* arr){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float a = 0.f, b = 0.f;
    bool flag = tid % 2;
    if (flag){
        a = 100.0f;
    }
    else{
        b = 100.0f;
    }
    arr[tid] = a + b;
}

__global__ void branch_kernal_3(float* arr){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float a = 0.f, b = 0.f;
    if ((tid/warpSize) % 2 == 0){
        a = 100.0f;
    }
    else{
        b = 100.0f;
    }
    arr[tid] = a + b;
}

int main(){
    double t;
    // determine the block and grid size, all 1D.
    int data_size = 64;
    dim3 block(data_size);
    dim3 grid((data_size - 1) / block.x + 1);
    printf("Execution Configuration: grid(%d), block(%d)\n", grid.x, block.x);

    // allocate space
    float* c_host = (float*)malloc(sizeof(float) * data_size);
    memset(c_host, 0, sizeof(c_host));
    printf("initial host array:\n");
    for (int i = 0; i < data_size; i++){
        printf("host[%d]: %f\n", i, c_host[i]);
    }

    float* c_dev = nullptr;
    check(cudaMalloc((void**)&c_dev, sizeof(float) * data_size));
    cudaMemcpy(c_dev, c_host, sizeof(c_dev), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // run kernal 1
    t = get_time();
    branch_kernal_1<<<grid, block>>>(c_dev);
    cudaDeviceSynchronize();
    t = get_time() - t;
    printf("kernal 1: %6lfs\n", t);

    // run kernal 2
    t = get_time();
    branch_kernal_2<<<grid, block>>>(c_dev);
    cudaDeviceSynchronize();
    t = get_time() - t;
    printf("kernal 2: %6lfs\n", t);

    // run kernal 3
    t = get_time();
    branch_kernal_3<<<grid, block>>>(c_dev);
    cudaDeviceSynchronize();
    t = get_time() - t;
    printf("kernal 3: %6lfs\n", t);

    free(c_host);
    cudaFree(c_dev);

    return 0;
}