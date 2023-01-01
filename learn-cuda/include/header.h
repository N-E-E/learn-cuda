#include <cuda.h>
#include <sys/time.h>
#include <stdio.h>

bool check(cudaError_t err_type){
    if (err_type != cudaSuccess){
        printf("ERROR: %s %d\n", __FILE__, __LINE__);
        printf("code: %d, reason: %s\n", err_type, cudaGetErrorString(err_type));
        exit(1);
    }
    return true;
}

void check_kernal_error(){
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("cuda error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

void init_data(int *arr, int size){
    srand(10);
    for (int i = 0; i < size; i++){
        arr[size] = rand() % 101;
    }
}

// timing
double get_time(){
    struct timeval tv;
    double t;
    
    gettimeofday(&tv, (struct timezone*)0);
    t = tv.tv_sec + (double)tv.tv_usec * 1e-6;

    return t;
}

