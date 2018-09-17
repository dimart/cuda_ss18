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
void squareArray(float* d_a, int n)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) d_a[i] = d_a[i] * d_a[i];
}


// kernel to call from the main function
__global__
void squareArrayKernel(float* d_a, int n)
{
    squareArray(d_a, n);
}


int main(int argc,char **argv)
{
    // alloc and init input arrays on host (CPU)
    int n = 10;
    float *a = new float[n];
    for(int i=0; i<n; i++) a[i] = i;

    // CPU computation
    for(int i=0; i<n; i++)
    {
        float val = a[i];
        val = val*val;
        a[i] = val;
    }

    // print result
    cout << "CPU:"<<endl;
    for(int i=0; i<n; i++) cout << i << ": " << a[i] << endl;
    cout << endl;
    
    
    // GPU computation
    // reinit data on CPU
    for(int i=0; i<n; i++) a[i] = i;
    
    // allocate memory on GPU
    float* d_a = NULL;
    size_t nbytes = (size_t)(n)*sizeof(float);
    cudaMalloc(&d_a, nbytes); CUDA_CHECK;
    
    // copy the array a from CPU => GPU
    cudaMemcpy(d_a, a, (size_t)(n)*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;

    // launch kernel
    dim3 block = dim3(128, 1, 1);
    dim3 grid = dim3((n + block.x - 1) / block.x, 1, 1);
    squareArrayKernel <<<grid,block>>> (d_a, n);

    // copy the array d_a from GPU => CPU
    cudaMemcpy(a, d_a, (size_t)(n)*sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;

    // print result
    cout << "GPU:" << endl;
    for(int i=0; i<n; i++) cout << i << ": " << a[i] << endl;
    cout << endl;

    // free CPU arrays
    delete[] a;
    // free GPU arrays
    cudaFree(d_a); CUDA_CHECK;
}



