#include <cuda_runtime.h>
#include <stdio.h>
#include <device_launch_parameters.h>

__global__ void l1CacheTest(float* input, float* output, int size, int stride) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0.0f;

    // Strided memory access pattern
    for (int i = tid; i < size; i += stride) {
        sum += input[i];
    }

    output[tid] = sum;
}

//int main() {
//    const int SIZE = 1024 * 1024;  // 1M elements
//    const int THREADS = 256;
//    const int BLOCKS = SIZE / THREADS;
//
//    float* h_input = new float[SIZE];
//    float* d_input, * d_output;
//
//    // Initialize data
//    for (int i = 0; i < SIZE; i++) {
//        h_input[i] = 1.0f;
//    }
//
//    // Allocate device memory
//    cudaMalloc(&d_input, SIZE * sizeof(float));
//    cudaMalloc(&d_output, BLOCKS * THREADS * sizeof(float));
//
//    // Copy to device
//    cudaMemcpy(d_input, h_input, SIZE * sizeof(float), cudaMemcpyHostToDevice);
//
//    // Test different strides
//    int strides[] = { 1, 2, 4, 8, 16, 32 };
//
//    cudaEvent_t start, stop;
//    cudaEventCreate(&start);
//    cudaEventCreate(&stop);
//
//    printf("Stride\tTime (ms)\n");
//    for (int stride : strides) {
//        float ms = 0;
//
//        cudaEventRecord(start);
//        l1CacheTest <<<BLOCKS, THREADS >>> (d_input, d_output, SIZE, stride);
//        cudaEventRecord(stop);
//
//        cudaEventSynchronize(stop);
//        cudaEventElapsedTime(&ms, start, stop);
//
//        printf("%d\t%.3f\n", stride, ms);
//    }
//
//    // Cleanup
//    cudaFree(d_input);
//    cudaFree(d_output);
//    delete[] h_input;
//
//    return 0;
//}