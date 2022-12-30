#include <cuda.h>
#include <stdio.h>
#include <sys/time.h>

#define N 50000000

// cal the id. 1D block and 2D grid
#define get_tid() (blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x) + threadIdx.x)
#define get_bid() (blockIdx.x + gridDim.x * blockIdx.y)

// timing
double get_time(){
    struct timeval tv;
    double t;
    
    gettimeofday(&tv, (struct timezone*)0);
    t = tv.tv_sec + (double)tv.tv_usec * 1e-6;

    return t;
}

// cpu programme
void vec_add_host(float* a, float* b, float* c, const int size){
    for (int i = 0; i < size; i++){
        c[i] = a[i] + b[i] + c[i];
    }
}

// gpu programme
__global__ void vec_add_dev(float* a, float* b, float* c, const int size){
    int idx = get_tid();
    if (idx < size){
        c[idx] = a[idx] + b[idx] + c[idx];
    }
}

int main(){
    long long nbytes = N * sizeof(float);

    // 1D block
    int block_size = 256;
    // 2D grid
    int s = ceil(sqrt((N + block_size - 1.) / block_size));
    dim3 grid(s, s);

    //allocate cpu size
    float* hx = (float*)malloc(nbytes);
    float* hy = (float*)malloc(nbytes);
    float* hz = (float*)malloc(nbytes);
    if (hx == nullptr || hy == nullptr || hz == nullptr){
        printf("space allocate error!\n");
        return -1;
    }
    else{
        printf("allocate %2f MB on cpu\n", nbytes / (1024.f * 1024.f));
    }

    // allocate gpu size
    float* dx = nullptr;
    float* dy = nullptr;
    float* dz = nullptr;
    
    cudaError_t err1 = cudaMalloc((void**)&dx, nbytes);
    cudaError_t err2 = cudaMalloc((void**)&dy, nbytes);
    cudaError_t err3 = cudaMalloc((void**)&dz, nbytes);
    if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess){
        printf("gpu space allocate error!\n");
        return -1;
    }
    else{
        printf("allocate %2f MB on gpu\n", nbytes / (1024.f * 1024.f));
    }

    // init cpu array
    memset(hx, 1, nbytes);
    memset(hy, 1, nbytes);
    memset(hz, 1, nbytes);

    // copy data to gpu
    cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dz, hz, nbytes, cudaMemcpyHostToDevice);

    cudaThreadSynchronize();

    // gpu timing
    double td = get_time();
    vec_add_dev<<<grid, block_size>>>(dx, dy, dz, N);
    cudaThreadSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("cuda error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    td = get_time() - td;

    // cpu timing
    double th = get_time();
    vec_add_host(hx, hy, hz, N);
    th = get_time() - th;

    printf("GPU time: %e, CPU time: %e, speed up: %g\n", td, th, th/td);

    // free
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);

    free(hx);
    free(hy);
    free(hz);

}