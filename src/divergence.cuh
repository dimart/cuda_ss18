// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#ifndef TUM_DIVERGENCE_H
#define TUM_DIVERGENCE_H

#include <iostream>

void computeDivergenceCuda(float *q, const float *v1, const float *v2, int w, int h, int nc);

#endif
