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
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= n) return;

    __shared__ int partial_sum;

    // thread 0 is responsible for initializing partial_sum
    if(threadIdx.x == 0) partial_sum = 0;
    __syncthreads();

    // each thread updates the partial sum
    atomicAdd(&partial_sum, g_idata[i]);
    __syncthreads();

    // thread 0 updates the total sum
    if(threadIdx.x == 0) atomicAdd(g_odata, partial_sum);
}


void runParallelReduction(int n, size_t repeats)
{
    // initialize CUDA context
    cudaDeviceSynchronize();  CUDA_CHECK;

    // fill input array
    float *elemns = new float[n];
    int cpu = 0;
    for(int i = 0; i < n; ++i)
    {
        elemns[i] = 1;
    }

    /**
     * Sum on CPU
     **/

    Timer timer;
    timer.start();

    for(size_t k = 0; k < repeats; ++k)
    {
        for(int i = 0; i < n; i++) {
            cpu += elemns[i];
        }
    }

    timer.end();
    float t = timer.get() / (float)repeats;  // elapsed time in seconds
    std::cout << "cpu0: " << t*1000 << " ms" << std::endl;

    /**
     * Sum on GPU
     **/
    float *d_input = NULL;
    float *d_output = NULL;
    size_t nbytes = (size_t) n * sizeof(float);
    size_t nbytes_result = (size_t) 1 * sizeof(float);
    cudaMalloc(&d_input, nbytes);
    cudaMalloc(&d_output, nbytes_result);

    timer.start();

    for(size_t k = 0; k < repeats; ++k)
    {
        // CPU => GPU
        cudaMemcpy(d_input, elemns, nbytes, cudaMemcpyHostToDevice); CUDA_CHECK;

        // calculate block and grid size
        dim3 block(1024, 1, 1);
        dim3 grid = computeGrid1D(block, n);

        // run cuda kernel
        reduceKernel<<<grid, block, sizeof(float)>>>(d_input, d_output, n); CUDA_CHECK;

        cudaDeviceSynchronize(); CUDA_CHECK;
    }

    timer.end();
    t = timer.get() / (float)repeats;  // elapsed time in seconds
    std::cout << "reduce0: " << t*1000 << " ms" << std::endl;

    // download result
    float *result = new float[1];
    // GPU => CPU
    cudaMemcpy(result, d_output, nbytes_result, cudaMemcpyDeviceToHost); CUDA_CHECK;

    int sum = result[0];
    std::cout << "result reduce0: " << (int)sum << " (CPU=" << cpu << ")" << std::endl;

    /**
     * Sum on CPU using cuBLAS
     **/
    result[0] = 0.f;

    // create cublas handle
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cout << "CUBLAS initialization failed\n";
        return;
    }

    timer.start();
    for(size_t k = 0; k < repeats; ++k)
    {
        cublasSasum(handle, n, d_input, 1, result);
    }
    timer.end();
    std::cout << "cublasSasum: " << timer.get() / (float)repeats << " ms" << std::endl;
    sum = result[0];
    std::cout << "result cublas: " << (int)sum << " (CPU=" << cpu << ")" << std::endl;
}
