// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#include "structure_tensor.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include "helper.cuh"


__global__
void computeTensorOutputKernel(float *imgOut, const float *lmb1, const float *lmb2, const float *imgIn, int w, int h, int nc, float alpha, float beta)
{
    // TODO (8.3) compute structure tensor output
}


__global__
void computeDetectorKernel(float *lmb1, float *lmb2, const float *tensor11, const float *tensor12, const float *tensor22, int w, int h)
{
    // TODO (8.1) compute eigenvalues
    // TODO (8.2) implement detector
}


__global__
void computeStructureTensorKernel(float *tensor11, float *tensor12, float *tensor22, const float *dx, const float *dy, int w, int h, int nc)
{
    // TODO (7.3) compute structure tensor
}


void computeTensorOutputCuda(float *imgOut, const float *lmb1, const float *lmb2, const float *imgIn, int w, int h, int nc, float alpha, float beta)
{
    // calculate block and grid size
    dim3 block(0, 0, 0);     // TODO (8.3) specify suitable block size
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    // TODO (8.3) execute kernel for computing tensor output

    // check for errors
    // TODO (8.3)
}


void computeDetectorCuda(float *lmb1, float *lmb2, const float *tensor11, const float *tensor12, const float *tensor22, int w, int h)
{
    // calculate block and grid size
    dim3 block(0, 0, 0);     // TODO (8.2) specify suitable block size
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    // TODO (8.2) execute kernel for detector

    // check for errors
    // TODO (8.2)
}


void computeStructureTensorCuda(float *tensor11, float *tensor12, float *tensor22, const float *dx, const float *dy, int w, int h, int nc)
{
    // calculate block and grid size
    dim3 block(0, 0, 0);     // TODO (7.3) specify suitable block size
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    // TODO (7.3) execute structure tensor kernel

    // check for errors
    // TODO (7.3)
}
