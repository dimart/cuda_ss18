// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#include "norm.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include "helper.cuh"


__global__
void computeNormKernel(float *imgOut, const float *u, int w, int h, int nc)
{
    // TODO (4.3) compute norm
}


void computeNormCuda(float *imgOut, const float *u, int w, int h, int nc)
{
    // calculate block and grid size
    dim3 block(0, 0, 0);     // TODO (4.3) specify suitable block size
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    // TODO (4.3) execute divergence kernel

    // check for errors
    // TODO (4.3)

}
