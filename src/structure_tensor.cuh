// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#ifndef TUM_STRUCTURE_TENSOR_H
#define TUM_STRUCTURE_TENSOR_H

#include <iostream>

void computeTensorOutputCuda(float *imgOut, const float *lmb1, const float *lmb2, const float *imgIn, int w, int h, int nc, float alpha, float beta);

void computeDetectorCuda(float *lmb1, float *lmb2, const float *tensor11, const float *tensor12, const float *tensor22, int w, int h);

void computeStructureTensorCuda(float *tensor11, float *tensor12, float *tensor22, const float *dx, const float *dy, int w, int h, int nc);

#endif
