// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#ifndef TUM_GRADIENT_H
#define TUM_GRADIENT_H

#include <iostream>

void computeGradientCuda(float *u, float *v, const float *imgIn, int w, int h, int nc);

#endif
