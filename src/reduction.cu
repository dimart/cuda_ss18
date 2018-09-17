// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#include "reduction.cuh"

#include <iostream>
#include <math.h>
#include <cuda_runtime.h>
#include "helper.cuh"

#include "cublas_v2.h"

__global__ void reduceKernel(float *g_idata, float *g_odata, int n)
{
    // TODO (12.1) implement parallel reduction kernel
}


void runParallelReduction(int n, size_t repeats)
{
    // fill input array
    float *elemns = new float[n];
    int cpu = n;
    for(int i = 0; i < n; ++ i)
    {
        elemns[i] = 1;
    }

    // TODO (12.1) first implement parallel reduction (sum) on CPU (optional) and measure time

    // allocate arrays on GPU
    float *d_input = NULL;
    float *d_output = NULL;
    // TODO alloc cuda memory for device arrays

    Timer timer;
    timer.start();

    for(size_t k = 0; k < repeats; ++k)
    {
        // upload input to GPU
        // TODO (12.1) copy from elemns to d_input

        while(true)
        {
            // TODO (12.1) implement parallel reduction
        }
    }

    timer.end();
    float t = timer.get() / (float)repeats;  // elapsed time in seconds
    std::cout << "reduce0: " << t*1000 << " ms" << std::endl;

    // download result
    float *result = new float[1];
    // TODO (12.1) download result from d_output to result
    int sum = result[0];
    std::cout << "result reduce0: " << (int)sum << " (CPU=" << cpu << ")" << std::endl;

    // create cublas handle
    cublasHandle_t handle;
    // TODO (12.2) create handle using cublasCreate()

    timer.start();
    for(size_t k = 0; k < repeats; ++k)
    {
        // upload input to GPU
        // TODO (12.2) copy from elemns to d_input

        // TODO (12.2) call cublasSasum() and store output in result
    }
    timer.end();
    std::cout << "cublasSasum: " << timer.get() / (float)repeats << " ms" << std::endl;
    sum = result[0];
    std::cout << "result cublas: " << (int)sum << " (CPU=" << cpu << ")" << std::endl;
}
