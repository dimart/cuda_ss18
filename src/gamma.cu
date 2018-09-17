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
    // TODO (3.2) implement kernel for gamma correction
}


void computeGamma(float *imgOut, const float *imgIn, float gamma, size_t w, size_t h, size_t nc)
{
    if (!imgOut || !imgIn)
    {
        std::cerr << "arrays not allocated!" << std::endl;
        return;
    }

    // TODO (3.1) compute gamma correction on CPU
}


void computeGammaCuda(float *imgOut, const float *imgIn, float gamma, int w, int h, int nc)
{
    if (!imgOut || !imgIn)
    {
        std::cerr << "arrays not allocated!" << std::endl;
        return;
    }

    // calculate block and grid size
    dim3 block(0, 0, 0);     // TODO (3.2) specify suitable block size

    // TODO (3.2) implement computeGrid2D() in helper.cuh etc
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    // TODO (3.2) execute gamma correction kernel

    // check for errors
    // TODO (3.2)
}
