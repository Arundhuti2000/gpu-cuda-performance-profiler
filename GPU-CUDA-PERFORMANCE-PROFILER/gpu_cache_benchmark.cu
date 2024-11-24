#include <cuda_runtime.h>
#include <stdio.h>
#include <device_launch_parameters.h>

// L1 Cache Test Kernel
__global__ void l1CacheTest(float* input, float* output, int size, int stride) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0.0f;
    // Strided memory access pattern
    for (int i = tid; i < size; i += stride) {
        sum += input[i];
    }
    output[tid] = sum;
}

// L2 Cache Test Kernel
__global__ void l2CacheTest(float* input, float* output, int size, int stride) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0.0f;

    // Access pattern designed to test L2 cache
    for (int i = tid; i < size; i += blockDim.x * gridDim.x) {
        int idx = (i * stride) % size;
        sum += input[idx];
    }
    output[tid] = sum;
}

void runL1Test(float* d_input, float* d_output, int SIZE, int THREADS, int BLOCKS) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int strides[] = { 1, 2, 4, 8, 16, 32 };
    printf("\nRunning L1 Cache Test\n");
    printf("Stride\tTime (ms)\n");

    for (int stride : strides) {
        float ms = 0;
        cudaEventRecord(start);
        l1CacheTest << <BLOCKS, THREADS >> > (d_input, d_output, SIZE, stride);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        printf("%d\t%.3f\n", stride, ms);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void runL2Test(float* d_input, float* d_output, int SIZE, int THREADS, int BLOCKS) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int strides[] = { 1, 2, 4, 8, 16, 32, 64, 128 };
    printf("\nRunning L2 Cache Test\n");
    printf("Stride\tTime (ms)\tEffective Bandwidth (GB/s)\n");

    for (int stride : strides) {
        float ms = 0;
        cudaEventRecord(start);
        l2CacheTest << <BLOCKS, THREADS >> > (d_input, d_output, SIZE, stride);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);

        float dataSize = SIZE * sizeof(float);
        float bandwidthGB = (dataSize / ms) / 1e6;
        printf("%d\t%.3f\t\t%.2f\n", stride, ms, bandwidthGB);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    const int SIZE = 1024 * 1024;  // 1M elements
    const int THREADS = 256;
    const int BLOCKS = SIZE / THREADS;

    // Allocate and initialize host memory
    float* h_input = new float[SIZE];
    for (int i = 0; i < SIZE; i++) {
        h_input[i] = 1.0f;
    }

    // Allocate device memory
    float* d_input, * d_output;
    cudaMalloc(&d_input, SIZE * sizeof(float));
    cudaMalloc(&d_output, BLOCKS * THREADS * sizeof(float));

    // Copy to device
    cudaMemcpy(d_input, h_input, SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Menu system
    int choice = 0;
    do {
        printf("\nGPU Cache Test Menu:\n");
        printf("1. Run L1 Cache Test\n");
        printf("2. Run L2 Cache Test\n");
        printf("3. Run Both Tests\n");
        printf("0. Exit\n");
        printf("Enter your choice: ");
        scanf("%d", &choice);

        switch (choice) {
        case 1:
            runL1Test(d_input, d_output, SIZE, THREADS, BLOCKS);
            break;
        case 2:
            runL2Test(d_input, d_output, SIZE, THREADS, BLOCKS);
            break;
        case 3:
            runL1Test(d_input, d_output, SIZE, THREADS, BLOCKS);
            runL2Test(d_input, d_output, SIZE, THREADS, BLOCKS);
            break;
        case 0:
            printf("Exiting...\n");
            break;
        default:
            printf("Invalid choice! Please try again.\n");
        }
    } while (choice != 0);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;

    return 0;
}