// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#include "divergence.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include "helper.cuh"


// Computes divergence
__global__
void computeDivergenceKernel(float *q, const float *v1, const float *v2, int w, int h, int nc)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z;
    int i = x + y * w + w * h * z;
    int dx = (x - 1) + y * w + w * h * z;
    int dy = x + (y - 1) * w + w * h * z;
    if (x < w && y < h && z < nc)
        q[i] = 0;
        if (x > 0)
            q[i] += (v1[i] - v1[dx]);
        if (y > 0)
            q[i] += (v2[i] - v2[dy]);
}


void computeDivergenceCuda(float *q, const float *v1, const float *v2, int w, int h, int nc)
{
    // calculate block and grid size
    dim3 block(32, 10, 3);
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    computeDivergenceKernel <<<grid,block>>> (q, v1, v2, w, h, nc);

    // check for errors
    CUDA_CHECK;
}
