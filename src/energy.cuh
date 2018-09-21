// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#ifndef TUM_ENERGY_H
#define TUM_ENERGY_H

#include <iostream>


// Huber regularization
__device__
inline float huber(float x, float eps)
{
    if (x < eps) {
        return pow(x, 2) / (2 * eps);
    } else {
        return x - eps / 2;
    }
}

void minimizeEnergySorStepCuda(float *uOut, const float *uIn, const float *diffusivity, const float *imgData, int w, int h, int nc, float lambda, float sor_theta, int redOrBlack);

void minimizeEnergyJacobiStepCuda (float *uOut, const float *uIn, const float *diffusivity, const float *imgData, int w, int h, int nc, float lambda);

void computeEnergyCuda(float *d_energy, float *a_in, float *d_imgData,
                       int w, int h, int nc, float lambda, float epsilon);

#endif
