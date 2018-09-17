// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#include "energy.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include "helper.cuh"


__global__
void minimizeEnergySorStepKernel(float *uOut, const float *uIn, const float *diffusivity, const float *imgData, int w, int h, int nc, float lambda, float sor_theta, int redOrBlack)
{
    // TODO (11.3) implement SOR update step
}


__global__
void minimizeEnergyJacobiStepKernel(float *uOut, const float *uIn, const float *diffusivity, const float *imgData, int w, int h, int nc, float lambda)
{
    // TODO (11.2) implement Jacobi update step
}


__global__
void computeEnergyKernel(float *d_energy, float *a_in, float *d_imgData,
                        int w, int h, int nc, float lambda, float epsilon)
{
    // TODO (12.2) compute energy
}


void minimizeEnergySorStepCuda(float *uOut, const float *uIn, const float *diffusivity, const float *imgData, int w, int h, int nc, float lambda, float sor_theta, int redOrBlack)
{
    // calculate block and grid size
    dim3 block(0, 0, 0);     // TODO (11.3) specify suitable block size
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    // TODO (11.3) execute kernel for SOR update step

    // check for errors
    // TODO (11.3)
}

void minimizeEnergyJacobiStepCuda(float *uOut, const float *uIn, const float *diffusivity, const float *imgData, int w, int h, int nc, float lambda)
{
    // calculate block and grid size
    dim3 block(0, 0, 0);     // TODO (11.2) specify suitable block size
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    // TODO (11.2) execute kernel for Jacobi update step

    // check for errors
    // TODO (11.2)
}

void computeEnergyCuda(float *d_energy, float *a_in, float *d_imgData,
                       int w, int h, int nc, float lambda, float epsilon)
{
    // calculate block and grid size
    dim3 block(0, 0, 0);     // TODO (12.2) specify suitable block size
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    // TODO (12.2) execute kernel for computing energy

    // check for errors
    // TODO (12.2)
}
