// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#include "histogram.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include "helper.cuh"


__global__
void computeHistogramAtomicKernel(int *histogram, float *imgIn, int nbins, int w, int h, int nc)
{
    // TODO (13.1) update histogram using atomic operations
}


__global__
void computeHistogramAtomicSharedMemKernel(int *histogram, float *imgIn, int w, int h, int nc)
{
    // TODO (13.3) update histogram using atomic operations on shared memory
}


void computeHistogramCuda(int *histogram, float *imgIn, int nbins, int w, int h, int nc)
{
    if (!histogram)
    {
        std::cerr << "histogram not allocated!" << std::endl;
        return;
    }

    // calculate block and grid size
    dim3 block(0, 0, 0);     // TODO (13.1) specify suitable block size
    dim3 grid(0, 0, 0);      // TODO (13.1) compute grid dimensions

    // run cuda kernel
    // TODO (13.1) execute kernel for histogram update using atomic operations

    // check for errors
    // TODO (13.1)
}

void computeHistogramCudaShared(int *histogram, float *imgIn, int w, int h, int nc)
{
    if (!histogram)
    {
        std::cerr << "histogram not allocated!" << std::endl;
        return;
    }

    // calculate block and grid size
    dim3 block(0, 0, 0);     // TODO (13.3) specify suitable block size
    dim3 grid(0, 0, 0);      // TODO (13.3) compute grid dimensions

    // run cuda kernel
    // TODO (13.3) execute kernel for histogram update using atomic operations on shared memory

    // check for errors
    // TODO (13.3)
}
