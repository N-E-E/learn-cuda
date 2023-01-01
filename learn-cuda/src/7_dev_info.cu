#include <cuda.h>
#include "../include/header.h"

int main(){
    // check cuda-supporting device number
    int dev_num = 0;
    if (check(cudaGetDeviceCount(&dev_num))){
        printf("device number: %d\n", dev_num);
    }

    // check device property
    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop, dev);
    printf("thread num of per block: %d\n", dev_prop.maxThreadsPerBlock);
    printf("block size: %d*%d*%d\n", dev_prop.maxThreadsDim[0], dev_prop.maxThreadsDim[1], dev_prop.maxThreadsDim[2]);
    printf("WarpSize: %d\n", dev_prop.warpSize);
    return 0;
}