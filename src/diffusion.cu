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
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= w || y >= h) return;

    for (int ch = 0; ch < nc; ch++) {
        int pos = x + y * w + w * h * ch;
        u[pos] += dt * d_div[pos];
    }
}


__global__
void multDiffusivityKernel(float *v1, float *v2, int w, int h, int nc, float epsilon, size_t diffusivity_mode)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= w || y >= h) return;

    // compute the gradient norm
    float norm = 0.0f;
    for (int ch = 0; ch < nc; ch++) {
        int pos = x + y * w + w * h * ch;
        norm += v1[pos] * v1[pos] + v2[pos] * v2[pos];
//        norm += sqrtf(v1[pos] * v1[pos] + v2[pos] * v2[pos]);
    }
    norm = sqrtf(norm);

    // compute diffusivity using the norm
    float g = funcDiffusivity(norm, epsilon, diffusivity_mode);

    // multiply it and store the result back at v1, v2
    for (int ch = 0; ch < nc; ch++) {
        int pos = x + y * w + w * h * ch;
        v1[pos] *= g;
        v2[pos] *= g;
    }
}

__global__
void multDiffusivityAnisotropicKernel(float *v1, float *v2, float *g11, float *g12, float *g22, int w, int h, int nc)
{
    // TODO (10.2) multiply diffusivity (anisotropic)
}


__global__
void computeDiffusivityKernel(float *diffusivity, const float *u, int w, int h, int nc, float epsilon)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= w || y >= h) return;

    // compute the gradient norm
    float norm = 0.0f;
    for (int ch = 0; ch < nc; ch++) {
        int ch_skip = w * h * ch;
        int pos = x + y * w + ch_skip;
        int dx = (x + 1) + y * w + ch_skip;
        int dy = x + (y + 1) * w + ch_skip;

        // compute gradient at the point
        float v1 = 0.0f;
        float v2 = 0.0f;
        if (x + 1 < w)
            v1 = u[dx] - u[pos];
        if (y + 1 < h)
            v2 = u[dy] - u[pos];

        // add it up for the norm
        norm += v1 * v1 + v2 * v2;
    }
    norm = sqrtf(norm);

    // compute Huber diffusivity (mode = 1) using the norm
    diffusivity[x + y * w] = funcDiffusivity(norm, epsilon, 1);
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
    dim3 block(32, 32, 1);
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    updateDiffusivityKernel<<<grid, block>>>(u, d_div, w, h, nc, dt);

    // check for errors
    CUDA_CHECK;
}


void multDiffusivityCuda(float *v1, float *v2, int w, int h, int nc, float epsilon, size_t diffusivity_mode)
{
    // calculate block and grid size
    dim3 block(32, 32, 1);
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    multDiffusivityKernel<<<grid, block>>>(v1, v2, w, h, nc, epsilon, diffusivity_mode);

    // check for errors
    CUDA_CHECK;
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
    dim3 block(32, 32, 1);
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    computeDiffusivityKernel<<<grid, block>>>(diffusivity, u, w, h, nc, epsilon);

    // check for errors
    CUDA_CHECK;
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
