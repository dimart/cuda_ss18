// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#include "convolution.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include "helper.cuh"


// TODO (6.3) define constant memory for convolution kernel

// TODO (6.2) define texture for image


__global__
void computeConvolutionTextureMemKernel(float *imgOut, const float *imgIn, const float *kernel, int kradius, int w, int h, int nc)
{
    // TODO (6.2) compute convolution using texture memory
}


__global__
void computeConvolutionSharedMemKernel(float *imgOut, const float *imgIn, const float *kernel, int kradius, int w, int h, int nc)
{
    // TODO (6.1) compute convolution using shared memory
}


__global__
void computeConvolutionGlobalMemKernel(float *imgOut, const float *imgIn, const float *kernel, int kradius, int w, int h, int nc)
{
    // TODO (5.4) compute convolution using global memory
}


void createConvolutionKernel(float *kernel, int kradius, float sigma)
{
    float sum = 0;
    for (int a = -kradius; a <= kradius; a++) {
        for (int b = -kradius; b <= kradius; b++) {
            int at = kradius + b + (2 * kradius + 1) * (kradius + a);
            float num = exp(-(a * a + b * b) / (2 * sigma * sigma));
            float denum = 2 * M_PI * sigma * sigma;
            kernel[at] = num / denum;
            sum += kernel[at];
//            std::cout << kernel[at] << " ";
        }
//        std::cout << std::endl;
    }
//    std::cout << "normalized:" << std::endl;
    for (int a = -kradius; a <= kradius; a++) {
        for (int b = -kradius; b <= kradius; b++) {
            int at = kradius + b + (2 * kradius + 1)* (kradius + a);
            kernel[at] /= sum;
//            std::cout << kernel[at] << " ";
        }
//        std::cout << std::endl;
    }
}


void computeConvolution(float *imgOut, const float *imgIn, const float *kernel, int kradius, int w, int h, int nc)
{
    if (!imgOut || !imgIn)
    {
        std::cerr << "arrays not allocated!" << std::endl;
        return;
    }

    for (int ch = 0; ch < nc; ch++) {
        int skip = w * h * ch;
    }
}


void computeConvolutionTextureMemCuda(float *imgOut, const float *imgIn, const float *kernel, int kradius, int w, int h, int nc)
{
    if (!imgOut || !imgIn)
    {
        std::cerr << "arrays not allocated!" << std::endl;
        return;
    }

    // calculate block and grid size
    dim3 block(0, 0, 0);     // TODO (6.2) specify suitable block size
    dim3 grid = computeGrid2D(block, w, h);

    // TODO (6.2) bind texture

    // run cuda kernel
    // TODO (6.2) execute kernel for convolution using global memory

    // TODO (6.2) unbind texture

    // check for errors
    // TODO (6.2)
}


void computeConvolutionSharedMemCuda(float *imgOut, const float *imgIn, const float *kernel, int kradius, int w, int h, int nc)
{
    if (!imgOut || !imgIn)
    {
        std::cerr << "arrays not allocated!" << std::endl;
        return;
    }

    // calculate block and grid size
    dim3 block(0, 0, 0);     // TODO (6.1) specify suitable block size
    dim3 grid = computeGrid2D(block, w, h);

    // TODO (6.1) calculate shared memory size

    // run cuda kernel
    // TODO (6.1) execute kernel for convolution using global memory

    // check for errors
    // TODO (6.1)
}


void computeConvolutionGlobalMemCuda(float *imgOut, const float *imgIn, const float *kernel, int kradius, int w, int h, int nc)
{
    if (!imgOut || !imgIn)
    {
        std::cerr << "arrays not allocated!" << std::endl;
        return;
    }

    // calculate block and grid size
    dim3 block(0, 0, 0);     // TODO (5.4) specify suitable block size
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    // TODO (5.4) execute kernel for convolution using global memory

    // check for errors
    // TODO (5.4)
}
