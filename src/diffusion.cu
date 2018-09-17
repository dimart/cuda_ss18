// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#include "diffusion.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include "helper.cuh"


__global__
void updateDiffusivityKernel(float *u, const float *d_div, int w, int h, int nc, float dt)
{
    // TODO (9.5) update diffusivity
}


__global__
void multDiffusivityKernel(float *v1, float *v2, int w, int h, int nc, float epsilon)
{
    // TODO (9.3) multiply diffusivity
}

__global__
void multDiffusivityAnisotropicKernel(float *v1, float *v2, float *g11, float *g12, float *g22, int w, int h, int nc)
{
    // TODO (10.2) multiply diffusivity (anisotropic)
}


__global__
void computeDiffusivityKernel(float *diffusivity, const float *u, int w, int h, int nc, float epsilon)
{
    // TODO (11.2) compute diffusivity
}


__device__ void eigen(float a11, float a12, float a22, float* lmin, float* lmax, float* v11, float* v12, float* v21, float* v22)
{
    // TODO (10.1) compute eigen values and eigen vectors
}


__global__
void computeDiffusionTensorKernel(float *d_difftensor11, float *d_difftensor12, float *d_difftensor22,
                                  float *d_tensor11, float *d_tensor12, float *d_tensor22,
                                  float alpha, float C, int w, int h, int nc)
{
    // TODO (10.1) compute diffusion tensor
}


void updateDiffusivityCuda(float *u, const float *d_div, int w, int h, int nc, float dt)
{
    // calculate block and grid size
    dim3 block(0, 0, 0);     // TODO (9.5) specify suitable block size
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    // TODO (9.5) execute kernel for updating diffusivity

    // check for errors
    // TODO (9.5)
}


void multDiffusivityCuda(float *v1, float *v2, int w, int h, int nc, float epsilon)
{
    // calculate block and grid size
    dim3 block(0, 0, 0);     // TODO (9.3) specify suitable block size
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    // TODO (9.3) execute kernel for multiplying diffusivity

    // check for errors
    // TODO (9.3)
}


void multDiffusivityAnisotropicCuda(float *v1, float *v2, float *g11, float *g12, float *g22, int w, int h, int nc)
{
    // calculate block and grid size
    dim3 block(0, 0, 0);     // TODO (10.2) specify suitable block size
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    // TODO (10.2) execute kernel for multiplying diffusivity (anisotropic)

    // check for errors
    // TODO (10.2)
}


void computeDiffusivityCuda(float *diffusivity, const float *u, int w, int h, int nc, float epsilon)
{
    // calculate block and grid size
    dim3 block(0, 0, 0);     // TODO (11.2) specify suitable block size
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    // TODO (11.2) execute kernel for computing diffusivity

    // check for errors
    // TODO (11.2)
}


void computeDiffusionTensorCuda(float *d_difftensor11, float *d_difftensor12, float *d_difftensor22,
                                float *d_tensor11, float *d_tensor12, float *d_tensor22,
                                float alpha, float C, int w, int h, int nc)
{
    // calculate block and grid size
    dim3 block(0, 0, 0);     // TODO (10.1) specify suitable block size
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    // TODO (10.1) execute kernel for computing diffusion tensor

    // check for errors
    // TODO (10.1)
}
