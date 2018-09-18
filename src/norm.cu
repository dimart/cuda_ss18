// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#include "norm.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include "helper.cuh"

//__global__
//void computeSqrtKernel(float *imgOut, int w, int h, int nc)
//{
//    int x = threadIdx.x + blockDim.x * blockIdx.x;
//    int y = threadIdx.y + blockDim.y * blockIdx.y;
//    if (x >= w || y >= h || threadIdx.z != 0) return;

//    int at = x + y * w;
//    imgOut[at] = sqrt(imgOut[at]);
//}

//__global__
//void computeSumSquaresAcrossChannelsKernel(float *imgOut, const float *u, int w, int h, int nc)
//{
//    int x = threadIdx.x + blockDim.x * blockIdx.x;
//    int y = threadIdx.y + blockDim.y * blockIdx.y;
//    int z = threadIdx.z;
//    if (x >= w || y >= h || z >= nc) return;

//    int at = x + y * w;
//    imgOut[at] = 0;
////    __syncthreads();
//    imgOut[at] += pow(u[at + w * h * z], 2);
//}

__global__
void computeNormIterateKernel(float *imgOut, const float *u, int w, int h, int nc)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= w || y >= h) return;

    int at = x + y * w;
    imgOut[at] = 0;
    for (int ch = 0; ch < nc; ch++) {
        imgOut[at] += pow(u[at + w * h * ch], 2);
    }
    imgOut[at] = sqrt(imgOut[at]);
}


void computeNormCuda(float *imgOut, const float *u, int w, int h, int nc)
{
    // calculate block and grid size
    dim3 block(32, 32, 1);
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    computeNormIterateKernel<<<grid, block>>>(imgOut, u, w, h, nc);
//    computeSumSquaresAcrossChannelsKernel<<<grid, block>>>(imgOut, u, w, h, nc);
//    computeSqrtKernel<<<grid, block>>>(imgOut, w, h, nc);

    // check for errors
    CUDA_CHECK;
}
