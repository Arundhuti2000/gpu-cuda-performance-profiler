
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


__global__ void helloKernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from GPU thread %d!\n", idx);
}


int main() {
    // Launch kernel with 1 block of 5 threads
    printf("Hello from CPU!\n");
    helloKernel <<<1, 5 >>> ();
    // Wait for GPU to finish before printing from host
    cudaDeviceSynchronize();
    return 0;
}