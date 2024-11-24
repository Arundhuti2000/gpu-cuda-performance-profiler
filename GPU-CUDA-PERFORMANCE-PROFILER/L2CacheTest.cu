#include <cuda_runtime.h>
#include <stdio.h>
#include <device_launch_parameters.h>

// Test 1: L2 Cache Bandwidth Test
__global__ void l2CacheBandwidthTest(float* input, float* output, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0.0f;

    // Access pattern designed to bypass L1 and test L2
#pragma unroll
    for (int i = tid; i < size; i += blockDim.x * gridDim.x) {
        // Large stride to force L2 cache access
        int idx = (i * 32) % size;
        sum += input[idx];
    }
    output[tid] = sum;
}

// Test 2: Memory Coalescing Efficiency Test
__global__ void coalescingTest(float* input, float* output, int size, int pattern) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0.0f;

    if (pattern == 0) {
        // Coalesced access
        for (int i = tid; i < size; i += blockDim.x * gridDim.x) {
            sum += input[i];
        }
    }
    else {
        // Non-coalesced access
        for (int i = 0; i < size / 32; i++) {
            int idx = (tid * 32 + i) % size;
            sum += input[idx];
        }
    }
    output[tid] = sum;
}

// Test 3: Shared Memory Bank Conflict Test
__global__ void sharedMemoryTest(float* input, float* output, int size) {
    __shared__ float sharedMem[1024];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Load data into shared memory
    if (tid < size) {
        sharedMem[threadIdx.x] = input[tid];
    }
    __syncthreads();

    // Test bank conflicts vs no conflicts
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 32; i++) {
        // Test both sequential and strided access
        int idx = (threadIdx.x + i) % blockDim.x;  // No bank conflicts
        //int idx = (threadIdx.x * 32) % blockDim.x;  // Potential bank conflicts
        sum += sharedMem[idx];
    }

    if (tid < size) {
        output[tid] = sum;
    }
}

// Test 4: Cache Hit Rate Test
__global__ void cacheHitRateTest(float* input, float* output, int size, int repetitions) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0.0f;

    // Repeatedly access same data to test cache hit rate
    for (int r = 0; r < repetitions; r++) {
#pragma unroll
        for (int i = 0; i < 16; i++) {
            int idx = (tid + i) % size;
            sum += input[idx];
        }
    }
    output[tid] = sum;
}

int main() {
    const int SIZE = 1024 * 1024;  // 1M elements
    const int THREADS = 256;
    const int BLOCKS = (SIZE + THREADS - 1) / THREADS;

    float* h_input, * h_output;
    float* d_input, * d_output;

    // Allocate host memory
    h_input = new float[SIZE];
    h_output = new float[SIZE];

    // Initialize input data
    for (int i = 0; i < SIZE; i++) {
        h_input[i] = 1.0f;
    }

    // Allocate device memory
    cudaMalloc(&d_input, SIZE * sizeof(float));
    cudaMalloc(&d_output, SIZE * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Create timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("Running GPU Memory Architecture Tests\n");
    printf("====================================\n\n");

    // Test 1: L2 Cache Bandwidth
    {
        float ms = 0;
        cudaEventRecord(start);
        l2CacheBandwidthTest << <BLOCKS, THREADS >> > (d_input, d_output, SIZE);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        printf("L2 Cache Bandwidth Test: %.3f ms\n", ms);
    }

    // Test 2: Memory Coalescing
    {
        float ms_coalesced = 0, ms_noncoalesced = 0;

        // Coalesced access
        cudaEventRecord(start);
        coalescingTest << <BLOCKS, THREADS >> > (d_input, d_output, SIZE, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms_coalesced, start, stop);

        // Non-coalesced access
        cudaEventRecord(start);
        coalescingTest << <BLOCKS, THREADS >> > (d_input, d_output, SIZE, 1);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms_noncoalesced, start, stop);

        printf("Memory Coalescing Test:\n");
        printf("  Coalesced: %.3f ms\n", ms_coalesced);
        printf("  Non-coalesced: %.3f ms\n", ms_noncoalesced);
        printf("  Efficiency ratio: %.2fx\n", ms_noncoalesced / ms_coalesced);
    }

    // Test 3: Shared Memory Bank Conflicts
    {
        float ms = 0;
        cudaEventRecord(start);
        sharedMemoryTest << <BLOCKS, THREADS >> > (d_input, d_output, SIZE);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        printf("Shared Memory Bank Conflict Test: %.3f ms\n", ms);
    }

    // Test 4: Cache Hit Rate
    {
        const int REPETITIONS = 100;
        float ms = 0;
        cudaEventRecord(start);
        cacheHitRateTest << <BLOCKS, THREADS >> > (d_input, d_output, SIZE, REPETITIONS);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        printf("Cache Hit Rate Test: %.3f ms\n", ms);
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}