#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

// Kernel for vertex transformation
__global__ void transformVertices(float* vertices, float* transformed,
    float angle, int numVertices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numVertices) {
        // Each vertex has 3 components (x,y,z)
        int base = idx * 3;

        // Get vertex coordinates
        float x = vertices[base];
        float y = vertices[base + 1];
        float z = vertices[base + 2];

        // Perform rotation around Y axis
        float cosA = cosf(angle);
        float sinA = sinf(angle);

        transformed[base] = x * cosA + z * sinA;
        transformed[base + 1] = y;
        transformed[base + 2] = -x * sinA + z * cosA;
    }
}

// Kernel for simple fragment processing
__global__ void processFragments(float* vertices, float* colors,
    float* output, int numFragments) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numFragments) {
        int base = idx * 4; // RGBA

        // Simple lighting calculation
        float intensity = 0.8f;

        output[base] = colors[base] * intensity;     // R
        output[base + 1] = colors[base + 1] * intensity; // G
        output[base + 2] = colors[base + 2] * intensity; // B
        output[base + 3] = colors[base + 3];             // A
    }
}

//int main(int argc, char** argv) {
//    // Cube vertex data (8 vertices)
//    const int numVertices = 8;
//    float vertices[] = {
//        -1.0f, -1.0f, -1.0f,  // 0
//         1.0f, -1.0f, -1.0f,  // 1
//         1.0f,  1.0f, -1.0f,  // 2
//        -1.0f,  1.0f, -1.0f,  // 3
//        -1.0f, -1.0f,  1.0f,  // 4
//         1.0f, -1.0f,  1.0f,  // 5
//         1.0f,  1.0f,  1.0f,  // 6
//        -1.0f,  1.0f,  1.0f   // 7
//    };
//
//    // Color data (RGBA for each vertex)
//    const int numFragments = numVertices;
//    float colors[] = {
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
//    // Allocate device memory
//    float* d_vertices, * d_transformed, * d_colors, * d_output;
//    cudaMalloc(&d_vertices, numVertices * 3 * sizeof(float));
//    cudaMalloc(&d_transformed, numVertices * 3 * sizeof(float));
//    cudaMalloc(&d_colors, numFragments * 4 * sizeof(float));
//    cudaMalloc(&d_output, numFragments * 4 * sizeof(float));
//
//    // Copy data to device
//    cudaMemcpy(d_vertices, vertices, numVertices * 3 * sizeof(float), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_colors, colors, numFragments * 4 * sizeof(float), cudaMemcpyHostToDevice);
//
//    // Launch configuration
//    int blockSize = 256;
//    int vertexBlocks = (numVertices + blockSize - 1) / blockSize;
//    int fragmentBlocks = (numFragments + blockSize - 1) / blockSize;
//
//    // Benchmark loop
//    printf("Starting benchmark...\n");
//    const int numIterations = 1000;
//
//    cudaEvent_t start, stop;
//    cudaEventCreate(&start);
//    cudaEventCreate(&stop);
//
//    cudaEventRecord(start);
//
//    for (int i = 0; i < numIterations; i++) {
//        float angle = i * 0.01f;
//
//        transformVertices <<<vertexBlocks, blockSize>>> (d_vertices, d_transformed, angle, numVertices);
//        processFragments <<<fragmentBlocks, blockSize>>> (d_transformed, d_colors, d_output, numFragments);
//    }
//    cudaEventRecord(stop);
//    cudaEventSynchronize(stop);
//
//    float milliseconds = 0;
//    cudaEventElapsedTime(&milliseconds, start, stop);
//
//    printf("Benchmark complete!\n");
//    printf("Total time: %.2f ms\n", milliseconds);
//    printf("Average iteration time: %.3f ms\n", milliseconds / numIterations);
//
//    // Cleanup
//    cudaFree(d_vertices);
//    cudaFree(d_transformed);
//    cudaFree(d_colors);
//    cudaFree(d_output);
//
//    return 0;
//}