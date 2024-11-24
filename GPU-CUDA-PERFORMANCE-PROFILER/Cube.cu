#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>

// Constants for cube vertices and fragments
const int NUM_VERTICES = 8;
const int NUM_FRAGMENTS = 8;
const int ITERATIONS = 1000;

// GPU kernel for vertex transformation
__global__ void transformVertices(float* vertices, float* transformed, float angle) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_VERTICES) {
        // Each vertex has 3 components (x,y,z)
        int base = idx * 3;

        // Get vertex coordinates
        float x = vertices[base];
        float y = vertices[base + 1];
        float z = vertices[base + 2];

        // Rotation matrix multiplication
        float cosA = cosf(angle);
        float sinA = sinf(angle);

        transformed[base] = x * cosA - z * sinA;
        transformed[base + 1] = y;
        transformed[base + 2] = x * sinA + z * cosA;
    }
}

// GPU kernel for color processing
__global__ void processColors(float* colors, float* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_FRAGMENTS) {
        // Each color has 4 components (RGBA)
        int base = idx * 4;

        // Simple color modification
        output[base] = colors[base];     // R
        output[base + 1] = colors[base + 1]; // G
        output[base + 2] = colors[base + 2]; // B
        output[base + 3] = colors[base + 3]; // A
    }
}

//int main() {
//    // Cube vertex data (8 vertices)
//    float vertices[NUM_VERTICES * 3] = {
//        -1.0f, -1.0f, -1.0f,  // vertex 0
//         1.0f, -1.0f, -1.0f,  // vertex 1
//         1.0f,  1.0f, -1.0f,  // vertex 2
//        -1.0f,  1.0f, -1.0f,  // vertex 3
//        -1.0f, -1.0f,  1.0f,  // vertex 4
//         1.0f, -1.0f,  1.0f,  // vertex 5
//         1.0f,  1.0f,  1.0f,  // vertex 6
//        -1.0f,  1.0f,  1.0f   // vertex 7
//    };
//
//    // Color data (RGBA)
//    float colors[NUM_FRAGMENTS * 4] = {
//        1.0f, 0.0f, 0.0f, 1.0f,  // Red
//        0.0f, 1.0f, 0.0f, 1.0f,  // Green
//        0.0f, 0.0f, 1.0f, 1.0f,  // Blue
//        1.0f, 1.0f, 0.0f, 1.0f,  // Yellow
//        1.0f, 0.0f, 1.0f, 1.0f,  // Magenta
//        0.0f, 1.0f, 1.0f, 1.0f,  // Cyan
//        1.0f, 1.0f, 1.0f, 1.0f,  // White
//        0.5f, 0.5f, 0.5f, 1.0f   // Gray
//    };
//
//    // Allocate GPU memory
//    float* d_vertices, * d_transformed, * d_colors, * d_output;
//    cudaMalloc(&d_vertices, NUM_VERTICES * 3 * sizeof(float));
//    cudaMalloc(&d_transformed, NUM_VERTICES * 3 * sizeof(float));
//    cudaMalloc(&d_colors, NUM_FRAGMENTS * 4 * sizeof(float));
//    cudaMalloc(&d_output, NUM_FRAGMENTS * 4 * sizeof(float));
//
//    // Copy data to GPU
//    cudaMemcpy(d_vertices, vertices, NUM_VERTICES * 3 * sizeof(float), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_colors, colors, NUM_FRAGMENTS * 4 * sizeof(float), cudaMemcpyHostToDevice);
//
//    // Create CUDA events for timing
//    cudaEvent_t start, stop;
//    cudaEventCreate(&start);
//    cudaEventCreate(&stop);
//
//    // Start timing
//    cudaEventRecord(start);
//
//    // Run the simulation
//    printf("Starting CUDA kernels...\n");
//
//    for (int i = 0; i < ITERATIONS; i++) {
//        float angle = i * 0.01f;
//
//        // Launch kernels with 256 threads per block
//        transformVertices <<<(NUM_VERTICES + 255) / 256, 256 >> > (d_vertices, d_transformed, angle);
//        processColors <<<(NUM_FRAGMENTS + 255) / 256, 256 >> > (d_colors, d_output);
//    }
//
//    // Stop timing
//    cudaEventRecord(stop);
//    cudaEventSynchronize(stop);
//
//    // Calculate elapsed time
//    float milliseconds = 0;
//    cudaEventElapsedTime(&milliseconds, start, stop);
//
//    printf("Processing complete!\n");
//    printf("Time taken: %.2f ms\n", milliseconds);
//    printf("Average time per iteration: %.3f ms\n", milliseconds / ITERATIONS);
//
//    // Check for errors
//    cudaError_t cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        return 1;
//    }
//
//    // Clean up
//    cudaFree(d_vertices);
//    cudaFree(d_transformed);
//    cudaFree(d_colors);
//    cudaFree(d_output);
//    cudaEventDestroy(start);
//    cudaEventDestroy(stop);
//
//    return 0;
////}