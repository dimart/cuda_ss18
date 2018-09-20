// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#ifndef TUM_DIFFUSION_H
#define TUM_DIFFUSION_H

#include <iostream>


__host__ __device__
inline float funcDiffusivity(float x, float eps, int mode)
{
    if (mode == 1)
    {
        // Huber diffusion (non-linear isotropic)
        return 1.0 / max(eps, x);
    }
    else if (mode == 2)
    {
        // exponential diffusivity function (non-linear isotropic)
        return expf(-(x * x) / eps) / eps;
    }
    else
    {
        // Laplace diffusion (linear isotropic)
        return 1.f;
    }
}

void updateDiffusivityCuda(float *u, const float *d_div, int w, int h, int nc, float dt);

void multDiffusivityCuda(float *v1, float *v2, int w, int h, int nc, float epsilon, size_t diffusivity_mode);

void multDiffusivityAnisotropicCuda(float *v1, float *v2, float *g11, float *g12, float *g22, int w, int h, int nc);

void computeDiffusivityCuda(float *diffusivity, const float *u, int w, int h, int nc, float epsilon);

void computeDiffusionTensorCuda(float *d_difftensor11, float *d_difftensor12, float *d_difftensor22,
                                float *d_tensor11, float *d_tensor12, float *d_tensor22,
                                float alpha, float C, int w, int h, int nc);

#endif
