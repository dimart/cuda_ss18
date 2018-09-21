// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#include "energy.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include "helper.cuh"
#include <stdio.h>

__global__
void minimizeEnergySorStepKernel(float *uOut, const float *uIn, const float *diffusivity, const float *imgData, int w, int h, int nc, float lambda, float sor_theta, int redOrBlack)
{
    // TODO (11.3) implement SOR update step
}


__global__
void minimizeEnergyJacobiStepKernel(float *uOut, const float *uIn, const float *diffusivity, const float *imgData, int w, int h, int nc, float lambda)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= w || y >= h) return;

    int at = x + y * w;
    int xm1 = (x - 1) + y * w;
    int ym1 = x + (y - 1) * w;

    float gr = (x + 1 < w) ? diffusivity[at] : 0;
    float gu = (y + 1 < h) ? diffusivity[at] : 0;
    float gl = (x > 0) ? diffusivity[xm1] : 0;
    float gd = (y > 0) ? diffusivity[ym1] : 0;

    float denom = 2 + lambda * (gr + gu + gl + gd);

    for (int ch = 0; ch < nc; ch++) {
        int ch_skip = w * h * ch;
        int pos = x + y * w + ch_skip;
        int pos_xm = x - 1 + y * w + ch_skip;
        int pos_xp = x + 1 + y * w + ch_skip;
        int pos_ym = x + (y - 1) * w + ch_skip;
        int pos_yp = x + (y + 1) * w + ch_skip;

        float gru = (x + 1 < w) ? (gr * uIn[pos_xp]) : 0;
        float guu = (y + 1 < h) ? (gu * uIn[pos_yp]) : 0;
        float glu = (x > 0)     ? (gl * uIn[pos_xm]) : 0;
        float gdu = (y > 0)     ? (gd * uIn[pos_ym]) : 0;

        float f2 = 2 * imgData[pos];
        float gu_sum = gru + guu + glu + gdu;

        uOut[pos] = (f2 + lambda * gu_sum) / denom;
    }
}


__global__
void computeEnergyKernel(float *d_energy, float *a_in, float *d_imgData,
                        int w, int h, int nc, float lambda, float epsilon)
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
            v1 = a_in[dx] - a_in[pos];
        if (y + 1 < h)
            v2 = a_in[dy] - a_in[pos];

        // add it up for the norm
        norm += v1 * v1 + v2 * v2;
    }
    norm = sqrtf(norm);

    for (int ch = 0; ch < nc; ch++) {
        int ch_skip = w * h * ch;
        int pos = x + y * w + ch_skip;

        // Total energy per pixel
        d_energy[pos] = pow(a_in[pos] - d_imgData[pos], 2) + lambda * huber(norm, epsilon);
    }
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
    dim3 block(32, 32, 1);
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    minimizeEnergyJacobiStepKernel<<<grid, block>>>(uOut, uIn, diffusivity, imgData, w, h, nc, lambda);

    // check for errors
    CUDA_CHECK;
}

void computeEnergyCuda(float *d_energy, float *a_in, float *d_imgData,
                       int w, int h, int nc, float lambda, float epsilon)
{
    // calculate block and grid size
    dim3 block(32, 32, 1);     // TODO (12.2) specify suitable block size
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    computeEnergyKernel<<<grid, block>>>(d_energy, a_in, d_imgData, w, h, nc, lambda, epsilon);

    // check for errors
    CUDA_CHECK;
}
