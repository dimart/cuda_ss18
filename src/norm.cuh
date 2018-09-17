// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#ifndef TUM_NORM_H
#define TUM_NORM_H

#include <iostream>

void computeNormCuda(float *imgOut, const float *u, int w, int h, int nc);

#endif
