#include <cuda.h>
#include <sys/time.h>
#include <stdio.h>

void check(cudaError_t err_type){
    if (err_type != cudaSuccess){
        printf("ERROR: %s %d\n", __FILE__, __LINE__);
        printf("code: %d, reason: %s\n", err_type, cudaGetErrorString(err_type));
        exit(1);
    }
}

void init_data(int *arr, int size){
    srand(10);
    for (int i = 0; i < size; i++){
        arr[size] = rand() % 101;
    }
}

