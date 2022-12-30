#include <cuda.h>
#include <include/header.h>

__global__ void disp_matrix(float* mat, const int nx, const int ny){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = ix * nx + iy;
    printf("thread_id(%d, %d); block_id(%d, %d), coordinate(%d, %d)\n",threadIdx.x, threadIdx.y, \
            blockIdx.x, blockIdx.y, ix, iy);
    printf("global_id: %d, val: %2f", idx, mat[idx]);
}

int main(){
    int nx = 8, ny = 6;
    int nxy = nx * ny;
    int nbytes = nx * ny;
}