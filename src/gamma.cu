// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#include "gamma.cuh"

#include <iostream>
#include <math.h>
#include <cuda_runtime.h>
#include "helper.cuh"


__global__
void computeGammaKernel(float *imgOut, const float *imgIn, float gamma, int w, int h, int nc)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z;
    int i = x + y * w + w * h * z;
    if (x < w && y < h && z < nc)
        imgOut[i] = pow(imgIn[i], gamma);
}


void computeGamma(float *imgOut, const float *imgIn, float gamma, size_t w, size_t h, size_t nc)
{
    if (!imgOut || !imgIn)
    {
        std::cerr << "arrays not allocated!" << std::endl;
        return;
    }

    // compute gamma correction on CPU
    for (size_t i = 0; i < h * w * nc; i++)
    {
        imgOut[i] = pow(imgIn[i], gamma);
    }
}


void computeGammaCuda(float *imgOut, const float *imgIn, float gamma, int w, int h, int nc)
{
    if (!imgOut || !imgIn)
    {
        std::cerr << "arrays not allocated!" << std::endl;
        return;
    }

    // calculate block and grid size
    dim3 block(32, 10, 3);
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    computeGammaKernel <<<grid,block>>> (imgOut, imgIn, gamma, w, h, nc);

    // check for errors
    CUDA_CHECK;
}
