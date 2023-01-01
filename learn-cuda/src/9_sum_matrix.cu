#include <cuda.h>
#include "../include/header.h"

__global__ void matrix_sum(float* A, float* B, float* C, const int nx, const int ny){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    if (ix < nx && iy < ny){
        int idx = ix * nx + iy;
        // if (idx == 0){
        //     printf("A_dev[0]:%2f\n", A[0]);
        // }
        C[idx] = A[idx] + B[idx];
    }
}
int main(int argc, char** argv){
    // paramater configuration
    int nx = 1 << 13;
    int ny = 1 << 13;
    int size = nx * ny;
    int nbytes = size * sizeof(float);
    printf("matrix size: %d*%d, space: %2lf MB\n", nx, ny, nbytes/(1024.0*1024.0));

    int block_dim_x = 32, block_dim_y = 32;
    if (argc > 2){
        block_dim_x = atoi(argv[1]);
        block_dim_y = atoi(argv[2]);
    }
    dim3 block(block_dim_x, block_dim_y);
    dim3 grid((nx - 1) / block.x + 1, (ny - 1) / block.y + 1);
    printf("grid(%d, %d), block(%d, %d)\n", grid.x, grid.y, block.x, block.y);

    // allocate space
    float* A_host = (float*)malloc(nbytes);
    float* B_host = (float*)malloc(nbytes);
    float* C_host = (float*)malloc(nbytes);
    for (int i = 0; i < nx * ny; i++){
        A_host[i] = 1.0;
        B_host[i] = 2.0;
    }
    // printf("A_host[0]:%2f\n", A_host[0]);
    
    float *A_dev = nullptr, *B_dev = nullptr, *C_dev = nullptr;
    cudaMalloc((void**)&A_dev, nbytes);
    cudaMalloc((void**)&B_dev, nbytes);
    cudaMalloc((void**)&C_dev, nbytes);

    // copy data
    cudaMemcpy(A_dev, A_host, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev, B_host, nbytes, cudaMemcpyHostToDevice);

    // run kernal
    double t = get_time();
    matrix_sum<<<grid, block>>>(A_dev, B_dev, C_dev, nx, ny);
    check(cudaDeviceSynchronize());
    t = get_time() - t;
    printf("time: %6lf\n", t);

    cudaMemcpy(C_host, C_dev, nbytes, cudaMemcpyDeviceToHost);
    printf("display some C_host data:\n");
    printf("C_host[0]: %2lf, C_host[nbytes-1]: %2lf\n", C_host[0], C_host[nbytes/sizeof(float)-1]);

    // free space
    free(A_host);
    free(B_host);
    free(C_host);
    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);

    cudaDeviceReset();
    return 0;
}