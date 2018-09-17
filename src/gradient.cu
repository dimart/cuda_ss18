// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#include "gradient.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include "helper.cuh"


// Computes gradient in x-direction (u) and y-direction (v)
__global__
void computeGradientKernel(float *u, float *v, const float *imgIn, int w, int h, int nc)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z;
    int i = x + y * w + w * h * z;
    int dx = (x + 1) + y * w + w * h * z;
    int dy = x + (y + 1) * w + w * h * z;
    if (x < w && y < h && z < nc)
        if (x + 1 < w)
            u[i] = imgIn[dx] - imgIn[i];
        if (y + 1 < h)
            v[i] = imgIn[dy] - imgIn[i];
}


void computeGradientCuda(float *u, float *v, const float *imgIn, int w, int h, int nc)
{
    // calculate block and grid size
    dim3 block(32, 10, 3);     // TODO (4.1) specify suitable block size
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    computeGradientKernel <<<grid,block>>> (u, v, imgIn, w, h, nc);

    // check for errors
    CUDA_CHECK;
}
