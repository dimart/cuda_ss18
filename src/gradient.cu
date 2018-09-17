// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#include "gradient.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include "helper.cuh"


__global__
void computeGradientKernel(float *u, float *v, const float *imgIn, int w, int h, int nc)
{
    // TODO (4.1) compute gradient in x-direction (u) and y-direction (v)
}


void computeGradientCuda(float *u, float *v, const float *imgIn, int w, int h, int nc)
{
    // calculate block and grid size
    dim3 block(0, 0, 0);     // TODO (4.1) specify suitable block size
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    // TODO (4.1) execute gradient kernel

    // check for errors
    // TODO (4.1)
}
