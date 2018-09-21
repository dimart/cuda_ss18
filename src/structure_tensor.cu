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
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= w || y >= h) return;

    int pos = x + y * w;
    float lambda1 = lmb1[pos];
    float lambda2 = lmb2[pos];

    if (lambda2 >= lambda1 && lambda1 >= alpha) {
        // corner pixel => make it red
        imgOut[pos] = 255;
    } else if (lambda1 <= beta && beta < alpha && alpha <= lambda2) {
        // edge pixel => make it yellow
        imgOut[pos] = 255;
        imgOut[pos + w * h] = 255;
    } else {
        // otherwise make the original pixel darker
        for (int ch = 0; ch < nc; ch++)
            imgOut[pos + w * h * ch] = 0.5 * imgIn[pos + w * h * ch];
    }
}


__device__
void computeEigenValues(float *lmb1, float *lmb2, const float m11, const float m12, const float m22, const int pos) {
    float trace = m11 + m22;
    float det = m11 * m22 - m12 * m12;
    float a = trace / 2.0;
    float b = sqrtf((trace * trace) / 4.0 - det);
    float lambda1 = a + b;
    float lambda2 = a - b;
    // follow convention that l1 <= l2
    if (lambda1 < lambda2) {
        lmb1[pos] = lambda1;
        lmb2[pos] = lambda2;
    } else {
        lmb1[pos] = lambda2;
        lmb2[pos] = lambda1;
    }
}

__global__
void computeDetectorKernel(float *lmb1, float *lmb2, const float *tensor11, const float *tensor12, const float *tensor22, int w, int h)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= w || y >= h) return;

    // compute eigenvalues
    int pos = x + y * w;
    computeEigenValues(lmb1, lmb2, tensor11[pos], tensor12[pos], tensor22[pos], pos);
}


__global__
void computeStructureTensorKernel(float *tensor11, float *tensor12, float *tensor22, const float *dx, const float *dy, int w, int h, int nc)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= w || y >= h) return;

    int i = x + y * w;
    tensor11[i] = 0;
    tensor12[i] = 0;
    tensor22[i] = 0;
    for (int ch = 0; ch < nc; ch++) {
        int ch_skip = w * h * ch;
        tensor11[i] += dx[i + ch_skip] * dx[i + ch_skip];
        tensor12[i] += dx[i + ch_skip] * dy[i + ch_skip];
        tensor22[i] += dy[i + ch_skip] * dy[i + ch_skip];
    }
}


void computeTensorOutputCuda(float *imgOut, const float *lmb1, const float *lmb2, const float *imgIn, int w, int h, int nc, float alpha, float beta)
{
    // calculate block and grid size
    dim3 block(32, 32, 1);
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    computeTensorOutputKernel<<<grid, block>>>(imgOut, lmb1, lmb2, imgIn, w, h, nc, alpha, beta);

    // check for errors
    CUDA_CHECK;
}


void computeDetectorCuda(float *lmb1, float *lmb2, const float *tensor11, const float *tensor12, const float *tensor22, int w, int h)
{
    // calculate block and grid size
    dim3 block(32, 32, 1);
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    computeDetectorKernel<<<grid, block>>>(lmb1, lmb2, tensor11, tensor12, tensor22, w, h);

    // check for errors
    CUDA_CHECK;
}


void computeStructureTensorCuda(float *tensor11, float *tensor12, float *tensor22, const float *dx, const float *dy, int w, int h, int nc)
{
    // calculate block and grid size
    dim3 block(32, 32, 1);
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    computeStructureTensorKernel<<<grid, block>>>(tensor11, tensor12, tensor22, dx, dy, w, h, nc);

    // check for errors
    CUDA_CHECK;
}
