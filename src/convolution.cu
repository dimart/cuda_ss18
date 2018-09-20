// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#include "convolution.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include "helper.cuh"
#include <stdio.h>


// define constant memory for convolution kernel
const uint MAX_KERNEL_RADIUS = 20;
const uint MAX_KERNEL_DIAMETER = 2 * MAX_KERNEL_RADIUS + 1;
__constant__ float constKernel[MAX_KERNEL_DIAMETER * MAX_KERNEL_DIAMETER];

// TODO (6.2) define texture for image


__global__
void computeConvolutionTextureMemKernel(float *imgOut, const float *imgIn, const float *kernel, int kradius, int w, int h, int nc)
{
    // TODO (6.2) compute convolution using texture memory
}


__global__
void computeConvolutionSharedMemKernel(float *imgOut, const float *imgIn, const float *kernel, int kradius, int w, int h, int nc)
{
    extern __shared__ float s_imgIn[];

    /**
     * Fill in s_imgIn.
     * Use all block's threads.
     * */
    int rad2 = kradius * 2;
    int patchDimX = blockDim.x + rad2;
    int patchDimY = blockDim.y + rad2;

    // linearized thread number
    int thread_num = threadIdx.x + threadIdx.y * blockDim.x;

    // how many pixels to copy for each thread:
    int per_thread = ceilf(float(patchDimX * patchDimY) / (blockDim.x * blockDim.y));

    // process channel by channel
    for (int ch = 0; ch < nc; ch++) {
        int ch_skip = w * h * ch;
        int s_ch_skip = patchDimX * patchDimY * ch;

        // origin position on image imgIn
        int origin_x = blockIdx.x * blockDim.x - kradius;
        int origin_y = blockIdx.y * blockDim.y - kradius;

        // linearized start position at s_imgIn
        int s_at = thread_num * per_thread;

        // copy the coalesced chunk of memory until the thread's chunk end (s_at < fill_to)
        // or until we moved out of s_imgIn memory (s_at < patchDimX * patchDimY)
        int fill_to = s_at + per_thread;
        while (s_at < fill_to && s_at < patchDimX * patchDimY) {
            int s_x = s_at % patchDimX;
            int s_y = s_at / patchDimX;

            int img_x = origin_x + s_x;
            int img_y = origin_y + s_y;

            if (img_x < 0) img_x = 0;
            if (img_y < 0) img_y = 0;
            if (img_x >= w) img_x = w - 1;
            if (img_y >= h) img_y = h - 1;

            s_imgIn[s_at + s_ch_skip] = imgIn[img_x + w * img_y + ch_skip];
            s_at++;
        }
        __syncthreads();
    }

    /**
     * Compute convolution using shared memory s_imgIn.
     * Use only `on-image` threads.
     * */
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= w || y >= h) return;

    for (int ch = 0; ch < nc; ch++) {
        int ch_skip = w * h * ch;
        int s_ch_skip = patchDimX * patchDimY * ch;

        float convolved_val = 0;
        for (int a = -kradius; a <= kradius; a++) {
            for (int b = -kradius; b <= kradius; b++) {
                int at_ker = a + kradius + (b + kradius) * (2 * kradius + 1);
                // add kradius and multiply on patchDimX to account for the 'boarder' in s_imgIn
                int posX = threadIdx.x + kradius + a;
                int posY = (threadIdx.y + kradius + b) * patchDimX;
                convolved_val += s_imgIn[posX + posY + s_ch_skip] * constKernel[at_ker];
            }
        }
        imgOut[x + y * w + ch_skip] = convolved_val;
    }
}


// compute convolution using global memory
__global__
void computeConvolutionGlobalMemKernel(float *imgOut, const float *imgIn, const float *kernel, int kradius, int w, int h, int nc)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= w || y >= h) return;

    computeConvolutionAtPixel(imgOut, imgIn, kernel, kradius, w, h, nc, x, y);
}


__host__ __device__
void computeConvolutionAtPixel(float *imgOut, const float *imgIn, const float *kernel, int kradius, int w, int h, int nc, int x, int y) {
    for (int ch = 0; ch < nc; ch++) {
        int ch_skip = w * h * ch;
        float convolved_val = 0;
        int convolved_origin = x + y * w + ch_skip;

        for (int a = -kradius; a <= kradius; a++) {
            for (int b = -kradius; b <= kradius; b++) {
                int at_ker = a + kradius + (b + kradius) * (2 * kradius + 1);
                int at_x = x - a;
                int at_y = y - b;
                if (at_x < 0) at_x = 0;
                if (at_y < 0) at_y = 0;
                if (at_x >= w) at_x = w - 1;
                if (at_y >= h) at_y = h - 1;

                convolved_val += imgIn[at_x + at_y * w + ch_skip] * kernel[at_ker];
            }
        }

        imgOut[convolved_origin] = convolved_val;
    }
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
        }
    }

    for (int a = -kradius; a <= kradius; a++) {
        for (int b = -kradius; b <= kradius; b++) {
            int at = kradius + b + (2 * kradius + 1)* (kradius + a);
            kernel[at] /= sum;
        }
    }
}


void computeConvolution(float *imgOut, const float *imgIn, const float *kernel, int kradius, int w, int h, int nc)
{
    if (!imgOut || !imgIn)
    {
        std::cerr << "arrays not allocated!" << std::endl;
        return;
    }

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            computeConvolutionAtPixel(imgOut, imgIn, kernel, kradius, w, h, nc, x, y);
        }
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
    dim3 block(32, 32, 1);
    dim3 grid = computeGrid2D(block, w, h);

    // calculate shared memory size
    int rad2 = kradius * 2;
    size_t smBytes = (block.x + rad2) * (block.y + rad2) * nc * sizeof(float);

    // fill-in constant memory
    cudaMemcpyToSymbol(constKernel, kernel, (rad2 + 1) * (rad2 + 1) * sizeof(float));
    std::cout << "filled constant memory for kernel" << std::endl;

    // run cuda kernel
    computeConvolutionSharedMemKernel<<<grid,block,smBytes>>>(imgOut, imgIn, kernel, kradius, w, h, nc);

    // check for errors
    CUDA_CHECK;
}


void computeConvolutionGlobalMemCuda(float *imgOut, const float *imgIn, const float *kernel, int kradius, int w, int h, int nc)
{
    if (!imgOut || !imgIn)
    {
        std::cerr << "arrays not allocated!" << std::endl;
        return;
    }

    // calculate block and grid size
    dim3 block(32, 32, 1);
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    computeConvolutionGlobalMemKernel<<<grid, block>>>(imgOut, imgIn, kernel, kradius, w, h, nc);

    // check for errors
    CUDA_CHECK;
}
