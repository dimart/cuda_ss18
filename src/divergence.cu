// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#include "divergence.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include "helper.cuh"


__global__
void computeDivergenceKernel(float *q, const float *v1, const float *v2, int w, int h, int nc)
{
    // TODO (4.2) compute divergence
}


void computeDivergenceCuda(float *q, const float *v1, const float *v2, int w, int h, int nc)
{
    // calculate block and grid size
    dim3 block(0, 0, 0);     // TODO (4.2) specify suitable block size
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    // TODO (4.2) execute divergence kernel

    // check for errors
    // TODO (4.2)
}
