// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################

#include <cuda_runtime.h>
#include <iostream>
using namespace std;



// cuda error checking
#define CUDA_CHECK cuda_check(__FILE__,__LINE__)
void cuda_check(string file, int line)
{
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
    {
        cout << endl << file << ", line " << line << ": " << cudaGetErrorString(e) << " (" << e << ")" << endl;
        exit(1);
    }
}


// perform the actual computation on GPU
__device__
void addArrays(float* d_a, float* d_b, float* d_c, int n)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) d_c[i] = d_a[i] + d_b[i];
}


// kernel to call from the main function
__global__
void addArraysKernel(float* d_a, float* d_b, float* d_c, int n)
{
    addArrays(d_a, d_b, d_c, n);
}


int main(int argc, char **argv)
{
    // alloc and init input arrays on host (CPU)
    int n = 20;
    float *a = new float[n];
    float *b = new float[n];
    float *c = new float[n];
    for(int i=0; i<n; i++)
    {
        a[i] = i;
        b[i] = (i%5)+1;
        c[i] = 0;
    }

    // CPU computation
    for(int i=0; i<n; i++) c[i] = a[i] + b[i];

    // print result
    cout << "CPU:"<<endl;
    for(int i=0; i<n; i++) cout << i << ": " << a[i] << " + " << b[i] << " = " << c[i] << endl;
    cout << endl;
    // init c
    for(int i=0; i<n; i++) c[i] = 0;
    
    
    // GPU computation
    // allocate memory on GPU
    size_t nbytes = (size_t)(n)*sizeof(float);
    float* d_a = NULL;
    float* d_b = NULL;
    float* d_c = NULL;
    cudaMalloc(&d_a, nbytes); CUDA_CHECK;
    cudaMalloc(&d_b, nbytes); CUDA_CHECK;
    cudaMalloc(&d_c, nbytes); CUDA_CHECK;
    
    // CPU => GPU
    cudaMemcpy(d_a, a, (size_t)(n)*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
    cudaMemcpy(d_b, b, (size_t)(n)*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
    cudaMemcpy(d_c, c, (size_t)(n)*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
    
    // launch kernel
    dim3 block = dim3(128,1,1);
    dim3 grid = dim3((n + block.x - 1) / block.x, 1, 1);
    addArraysKernel <<<grid,block>>> (d_a, d_b, d_c, n);
    
    // GPU => CPU
    cudaMemcpy(a, d_a, (size_t)(n)*sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
    cudaMemcpy(b, d_b, (size_t)(n)*sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
    cudaMemcpy(c, d_c, (size_t)(n)*sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
    
    // print result
    cout << "GPU:"<<endl;
    for(int i=0; i<n; i++) cout << i << ": " << a[i] << " + " << b[i] << " = " << c[i] << endl;
    cout << endl;

    // free CPU arrays
    delete[] a;
    delete[] b;
    delete[] c;
    
    // free GPU arrays
    cudaFree(d_a); CUDA_CHECK;
    cudaFree(d_b); CUDA_CHECK;
    cudaFree(d_c); CUDA_CHECK;
}



